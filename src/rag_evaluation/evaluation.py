"""
Evaluation orchestrator with strategy pattern for No-RAG vs RAG modes.

This is the main entry point for running evaluations.
"""

import os
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from .config import EvaluationConfig, Mode, RetrievalMode
from .chunk_schema import (
    Chunk,
    QueryVariant,
    QueryKind,
    RetrievalResult,
    Context,
    Prediction,
    Content_type,
)
from .index_loader import IndexManager
from .embedding_client import get_embedding_backend
from .query_reformulation import QueryReformulator, sanitize_query_expansions
from .hyde import HyDEGenerator
from .retrieval import RetrievalPipeline
from .fusion import FusionStrategy
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ContextProvider(ABC):
    """Abstract base class for context providers."""

    @abstractmethod
    def get_context(self, question: str, example: Dict[str, Any]) -> Optional[Context]:
        """
        Get context for a question.

        Args:
            question: Question text
            example: Full example dict

        Returns:
            Context object or None
        """
        pass


class NoRAGContextProvider(ContextProvider):
    """No-RAG context provider (returns None)."""

    def get_context(self, question: str, example: Dict[str, Any]) -> Optional[Context]:
        """No context for baseline mode."""
        return None


class RAGContextProvider(ContextProvider):
    """RAG context provider (performs retrieval)."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        query_generator: "QueryGenerator",
        tokenizer: AutoTokenizer,
        config: EvaluationConfig,
    ):
        """
        Initialize RAG context provider.

        Args:
            retrieval_pipeline: Retrieval pipeline
            query_generator: Query generator
            tokenizer: Tokenizer for evidence packing
            config: Evaluation configuration
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.query_generator = query_generator
        self.tokenizer = tokenizer
        self.config = config

    def get_context(self, question: str, example: Dict[str, Any]) -> Optional[Context]:
        """
        Retrieve context for a question.

        Args:
            question: Question text
            example: Full example dict

        Returns:
            Context object with retrieved passages
        """
        # Generate query variants
        queries = self.query_generator.generate_queries(example)

        if not queries:
            return None

        # Retrieve
        results = self.retrieval_pipeline.retrieve(queries, question)

        if not results:
            return None

        # Apply lexical overlap gate
        fusion = FusionStrategy(self.config.retrieval.fusion)
        results = fusion.filter_by_lexical_overlap(
            results,
            question,
            self.config.retrieval.fusion.min_lexical_overlap,
        )

        if not results:
            return None

        # Apply content type prioritization
        if self.config.retrieval.prioritize_content_types:
            results = fusion.filter_by_content_type(
                results,
                self.config.retrieval.prioritize_content_types,
            )

        # Limit to max_passages
        results = results[: self.config.retrieval.fine.max_passages]

        # Pack evidence by token limit
        results = self._pack_evidence(results)

        # Build context
        total_tokens = sum(r.chunk.token_count for r in results)

        metadata = {
            "num_queries": len(queries),
            "num_results": len(results),
            "query_kinds": [q.kind.value for q in queries],
        }

        return Context(passages=results, total_tokens=total_tokens, metadata=metadata)

    def _pack_evidence(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Pack evidence to fit within token limit.

        Args:
            results: Retrieval results

        Returns:
            Packed results
        """
        max_tokens = self.config.retrieval.fine.max_evidence_tokens

        if max_tokens <= 0:
            return results

        kept = []
        used = 0

        for result in results:
            tokens = result.chunk.token_count

            if used + tokens <= max_tokens:
                kept.append(result)
                used += tokens
            else:
                break

        return kept


class QueryGenerator:
    """Generates query variants for retrieval."""

    def __init__(self, config: EvaluationConfig, base_url: str):
        """
        Initialize query generator.

        Args:
            config: Evaluation configuration
            base_url: Ollama base URL
        """
        self.config = config
        self.base_url = base_url

        # Initialize components
        self.reformulator = None
        if config.retrieval.queries.use_reformulation:
            self.reformulator = QueryReformulator(config.retrieval.queries, base_url)

        self.hyde_generator = None
        if config.retrieval.queries.use_hyde:
            self.hyde_generator = HyDEGenerator(config.retrieval.queries, base_url)

    def generate_queries(self, example: Dict[str, Any]) -> List[QueryVariant]:
        """
        Generate all query variants for an example.

        Args:
            example: Example dict

        Returns:
            List of query variants
        """
        queries = []
        qid = 0
        question = (example.get("question") or "").strip()
        options = self._extract_options(example)
        example_id = example.get("id", "unknown")

        # Base query
        if self.config.retrieval.queries.use_base_query:
            queries.append(
                QueryVariant(
                    query_id=f"{example_id}_base_{qid}",
                    text=question,
                    kind=QueryKind.BASE,
                    example_id=example_id,
                )
            )
            qid += 1

        # Option-aware queries
        if self.config.retrieval.queries.use_option_aware and options:
            for opt_key, opt_text in sorted(options.items()):
                if opt_text.strip():
                    queries.append(
                        QueryVariant(
                            query_id=f"{example_id}_opt_{opt_key}_{qid}",
                            text=f"{question}\nOption: {opt_text.strip()}",
                            kind=QueryKind.OPTION,
                            example_id=example_id,
                            metadata={"option": opt_key},
                        )
                    )
                    qid += 1

        # Dataset expansions
        if self.config.retrieval.queries.use_expansions:
            expansions = example.get("query_expansions", [])
            expansions = sanitize_query_expansions(expansions)

            for exp in expansions:
                queries.append(
                    QueryVariant(
                        query_id=f"{example_id}_exp_{qid}",
                        text=f"{question} {exp}",
                        kind=QueryKind.EXPANSION,
                        example_id=example_id,
                        metadata={"expansion": exp},
                    )
                )
                qid += 1

        return queries

    async def generate_llm_queries(self, example: Dict[str, Any]) -> List[QueryVariant]:
        """
        Generate LLM-based query variants (reformulation, HyDE).

        Args:
            example: Example dict

        Returns:
            List of additional query variants
        """
        queries = []
        question = (example.get("question") or "").strip()
        options = self._extract_options(example)
        example_id = example.get("id", "unknown")
        qid = 0

        # Reformulation
        if self.reformulator:
            reformulated = await self.reformulator.reformulate(question)
            queries.append(
                QueryVariant(
                    query_id=f"{example_id}_reform_{qid}",
                    text=reformulated,
                    kind=QueryKind.REFORM,
                    example_id=example_id,
                )
            )
            qid += 1

        # HyDE
        if self.hyde_generator:
            hyde_passages = await self.hyde_generator.generate_for_example(question, options)

            for i, passage in enumerate(hyde_passages):
                queries.append(
                    QueryVariant(
                        query_id=f"{example_id}_hyde_{i}_{qid}",
                        text=passage,
                        kind=QueryKind.HYDE,
                        example_id=example_id,
                        metadata={"hyde_index": i},
                    )
                )
                qid += 1

        return queries

    def _extract_options(self, example: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract options from example dict."""
        # MedQA format
        if "options" in example and isinstance(example["options"], dict):
            options = {}
            for key in sorted(example["options"].keys()):
                if key.upper() in "ABCDE" and example["options"][key]:
                    options[key.upper()] = str(example["options"][key]).strip()
            return options if options else None

        # MedMCQA format
        if any(key in example for key in ["opa", "opb", "opc", "opd"]):
            return {
                "A": (example.get("opa") or "").strip(),
                "B": (example.get("opb") or "").strip(),
                "C": (example.get("opc") or "").strip(),
                "D": (example.get("opd") or "").strip(),
            }

        return None


class LLMScorer:
    """LLM-based answer scorer."""

    def __init__(self, config: EvaluationConfig, base_url: str):
        """
        Initialize LLM scorer.

        Args:
            config: Evaluation configuration
            base_url: Ollama base URL
        """
        self.config = config
        self.base_url = base_url

        # Initialize async client
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package required: pip install ollama")

        self.executor = ThreadPoolExecutor(max_workers=config.llm.num_workers)
        self._client = ollama.Client(host=base_url)

    async def score_batch(
        self,
        prompts: List[str],
    ) -> List[str]:
        """
        Score a batch of prompts.

        Args:
            prompts: List of prompt strings

        Returns:
            List of generated answers
        """
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(
                self.executor,
                lambda p=prompt: self._generate(p),
            )
            for prompt in prompts
        ]

        return await asyncio.gather(*tasks)

    def _generate(self, prompt: str) -> str:
        """Generate single response."""
        import ollama

        client = self._client or ollama.Client(host=self.base_url)

        # Determine max tokens based on model type
        if self.config.llm.reasoning_model:
            max_tokens = self.config.llm.reasoning_max_tokens
        elif self.config.llm.use_cot:
            max_tokens = self.config.llm.cot_max_tokens
        else:
            max_tokens = self.config.llm.max_tokens

        response = client.generate(
            model=self.config.llm.model_name,
            prompt=prompt,
            options={
                "temperature": self.config.llm.temperature,
                "top_p": self.config.llm.top_p,
                "num_predict": max_tokens,
            },
        )

        # Handle reasoning models (e.g., gpt-oss, deepseek-r1)
        if self.config.llm.reasoning_model and "thinking" in response:
            # For reasoning models, extract the final answer from response
            # The thinking field contains the reasoning process
            thinking = response.get("thinking", "")
            final_response = response.get("response", "")
            # Return just the final answer for parsing
            return final_response

        return response.get("response", "")


class EvaluationOrchestrator:
    """
    Main evaluation orchestrator.

    Supports No-RAG and RAG modes via strategy pattern.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation orchestrator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        config.validate()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # Load index manager (corpus name auto-detected from directory)
        self.index_manager = IndexManager(config.index_dir)
        self.index_manager.load()

        # Initialize embedding client
        self.embedding_client = get_embedding_backend(config.embedding)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)

        # Initialize context provider based on mode
        if config.retrieval.mode == Mode.NO_RAG:
            self.context_provider = NoRAGContextProvider()
        else:
            # Initialize retrieval pipeline
            self.retrieval_pipeline = RetrievalPipeline(
                self.index_manager,
                self.embedding_client,
                config.retrieval,
            )

            # Initialize query generator
            self.query_generator = QueryGenerator(config, config.llm.base_url)

            # Initialize RAG context provider
            self.context_provider = RAGContextProvider(
                self.retrieval_pipeline,
                self.query_generator,
                self.tokenizer,
                config,
            )

        # Initialize LLM scorer
        self.llm_scorer = LLMScorer(config, config.llm.base_url)

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()

    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load evaluation examples."""
        path = self.config.eval_data_path

        examples = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except:
                            continue
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data

        # Apply limit
        if self.config.limit:
            examples = examples[: self.config.limit]

        logger.info(f"Loaded {len(examples)} examples from {path}")
        return examples

    def _build_prompt(self, example: Dict[str, Any], context: Optional[Context]) -> str:
        """Build prompt for LLM."""
        question = (example.get("question") or "").strip()
        options = self._extract_options(example)

        # Build options string
        opt_str = "\n".join([f"{k}) {v}" for k, v in sorted(options.items()) if v.strip()])
        letters = sorted(options.keys())
        allowed = ", ".join(letters[:-1]) + f" or {letters[-1]}" if len(letters) > 1 else letters[0]

        if self.config.llm.use_cot:
            # Chain-of-thought prompt
            if context:
                rules = (
                    "You are a medical expert answering a single-best-answer MCQ.\n"
                    "Use the references if they are relevant; if they appear off-topic, ignore them and answer from medical knowledge.\n"
                    "Think step by step and explain your reasoning before choosing an option.\n"
                    f"Output format: First provide your reasoning, then end with 'Answer: X' where X is a single uppercase letter.\n"
                    f"Valid answers: {{ {' '.join(letters)} }}\n"
                    "If unsure, guess.\n"
                )

                # Format context
                ctx_parts = []
                for i, passage in enumerate(context.passages, 1):
                    text = passage.chunk.text.replace("\n", " ")
                    source = passage.chunk.section_metadata.textbook
                    title = passage.chunk.section_metadata.section

                    ctx_parts.append(
                        f"Reference {i}: [{source}] {title}\n{text}"
                    )

                ctx_str = "\n\n".join(ctx_parts)

                prompt = (
                    rules
                    + f"\n{ctx_str}\n\n"
                    + f"Question: {question}\n\nOptions:\n{opt_str}\n\n"
                    + f"Answer ({allowed}): "
                )
            else:
                rules = (
                    "You are a medical expert answering a single-best-answer MCQ.\n"
                    "Choose exactly one option based on medical knowledge.\n"
                    "Think step by step and explain your reasoning before choosing an option.\n"
                    f"Output format: First provide your reasoning, then end with 'Answer: X' where X is a single uppercase letter.\n"
                    f"Valid answers: {{ {' '.join(letters)} }}\n"
                    "If unsure, guess.\n"
                )

                prompt = (
                    rules
                    + f"\nQuestion: {question}\n\nOptions:\n{opt_str}\n\n"
                    + f"Answer ({allowed}): "
                )
        else:
            # Direct answer prompt
            if context:
                rules = (
                    "You are a medical expert answering a single-best-answer MCQ.\n"
                    "Use the references if they are relevant; if they appear off-topic, ignore them and answer from medical knowledge.\n"
                    "Choose exactly one option.\n"
                    "Output format: a single uppercase letter only, with no words, spaces, or punctuation.\n"
                    f"Valid answers: {{ {' '.join(letters)} }}\n"
                    "If unsure, guess.\n"
                )

                # Format context
                ctx_parts = []
                for i, passage in enumerate(context.passages, 1):
                    text = passage.chunk.text.replace("\n", " ")
                    source = passage.chunk.section_metadata.textbook
                    title = passage.chunk.section_metadata.section

                    ctx_parts.append(
                        f"Reference {i}: [{source}] {title}\n{text}"
                    )

                ctx_str = "\n\n".join(ctx_parts)

                prompt = (
                    rules
                    + f"\n{ctx_str}\n\n"
                    + f"Question: {question}\n\nOptions:\n{opt_str}\n\n"
                    + f"Answer ({allowed}): "
                )
            else:
                rules = (
                    "You are a medical expert answering a single-best-answer MCQ.\n"
                    "Choose exactly one option based on medical knowledge.\n"
                    "Output format: a single uppercase letter only, with no words, spaces, or punctuation.\n"
                    f"Valid answers: {{ {' '.join(letters)} }}\n"
                    "If unsure, guess.\n"
                )

                prompt = (
                    rules
                    + f"\nQuestion: {question}\n\nOptions:\n{opt_str}\n\n"
                    + f"Answer ({allowed}): "
                )

        return prompt

    def _extract_options(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Extract options from example."""
        # MedQA format
        if "options" in example and isinstance(example["options"], dict):
            options = {}
            for key in sorted(example["options"].keys()):
                if key.upper() in "ABCDE" and example["options"][key]:
                    options[key.upper()] = str(example["options"][key]).strip()
            return options

        # MedMCQA format
        if any(key in example for key in ["opa", "opb", "opc", "opd"]):
            return {
                "A": (example.get("opa") or "").strip(),
                "B": (example.get("opb") or "").strip(),
                "C": (example.get("opc") or "").strip(),
                "D": (example.get("opd") or "").strip(),
            }

        return {}

    def _parse_answer(self, response: str, allowed_letters: List[str]) -> Optional[str]:
        """Parse answer from LLM response."""
        import re

        # Look for "Answer: X" pattern
        match = re.search(r"Answer\s*[:：]\s*([A-E])", response, re.IGNORECASE)
        if match:
            ans = match.group(1).upper()
            if ans in allowed_letters:
                return ans

        # Look for single letter at end
        lines = response.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            if len(last_line) == 1 and last_line.upper() in allowed_letters:
                return last_line.upper()

        # Look for any single letter
        match = re.search(r"\b([A-E])\b", response)
        if match:
            ans = match.group(1).upper()
            if ans in allowed_letters:
                return ans

        return None

    def _get_gold_answer(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract gold standard answer."""
        # Try various fields
        ans = example.get("answer_idx") or example.get("answer") or example.get("gold")

        if isinstance(ans, str):
            ans = ans.upper()
            if ans in "ABCDE":
                return ans

        # Try cop field (MedMCQA)
        cop = example.get("cop")
        if isinstance(cop, int) and 1 <= cop <= 4:
            return ["A", "B", "C", "D"][cop - 1]

        return None

    async def evaluate_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[Prediction]:
        """
        Evaluate a batch of examples using batch retrieval.

        This is optimized to retrieve for all examples at once instead of per-example.

        Args:
            examples: List of example dicts

        Returns:
            List of predictions
        """
        # Prepare batch retrieval data
        all_queries: List[List[QueryVariant]] = []
        questions: List[str] = []

        # Generate queries for all examples
        if self.config.retrieval.mode == Mode.RAG:
            # Generate LLM queries for all examples (async batch)
            llm_queries_batch = await self._generate_llm_queries_batch(examples)

            for example, llm_queries in zip(examples, llm_queries_batch):
                # Get base queries
                base_queries = self.query_generator.generate_queries(example)

                # Combine with LLM queries
                all_queries.append(base_queries + llm_queries)
                questions.append(example.get("question", ""))

        # Batch retrieve (KEY OPTIMIZATION: retrieve all at once)
        if self.config.retrieval.mode == Mode.RAG and all_queries:
            logger.info(f"Starting batch retrieval for {len(examples)} examples...")
            all_results = self.retrieval_pipeline.retrieve_batch(all_queries, questions)
            logger.info(f"Batch retrieval complete, got {len(all_results)} result lists")
        else:
            all_results = [[] for _ in examples]

        # Build contexts from batch results
        logger.info(f"Building contexts...")
        contexts = []
        for example, results in zip(examples, all_results):
            if self.config.retrieval.mode == Mode.RAG:
                context = self._create_context_from_results(
                    results,
                    example.get("question", ""),
                    example,
                )
            else:
                context = None
            contexts.append(context)

        logger.info(f"Contexts built: {len(contexts)}")

        # Build prompts
        logger.info(f"Building prompts...")
        prompts = []
        metas = []

        for example, context in zip(examples, contexts):
            prompt = self._build_prompt(example, context)

            prompts.append(prompt)
            metas.append(
                {
                    "example": example,
                    "context": context,
                }
            )

        logger.info(f"Prompts built: {len(prompts)}")

        # Score batch
        logger.info(f"Starting LLM generation for {len(prompts)} examples...")
        responses = await self.llm_scorer.score_batch(prompts)
        logger.info(f"LLM generation complete")

        # Parse responses
        predictions = []
        for response, meta in zip(responses, metas):
            example = meta["example"]
            context = meta["context"]

            options = self._extract_options(example)
            allowed_letters = list(options.keys())

            predicted = self._parse_answer(response, allowed_letters)
            gold = self._get_gold_answer(example)

            is_correct = (predicted == gold) if (predicted and gold) else None

            prediction = Prediction(
                example_id=example.get("id", "unknown"),
                question=example.get("question", ""),
                gold_answer=gold,
                predicted_answer=predicted or "",
                is_correct=is_correct,
                context=context,
                raw_output=response,
                metadata={
                    "num_contexts": len(context.passages) if context else 0,
                },
            )

            predictions.append(prediction)

        return predictions

    async def _generate_llm_queries_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[List[QueryVariant]]:
        """
        Generate LLM queries (reformulation, HyDE) for all examples.

        This is optimized to batch the LLM calls.

        Args:
            examples: List of example dicts

        Returns:
            List of query lists (one per example)
        """
        if not self.query_generator.reformulator and not self.query_generator.hyde_generator:
            return [[] for _ in examples]

        # Batch generate reformulations
        reformulations = []
        if self.query_generator.reformulator:
            questions = [ex.get("question", "") for ex in examples]
            reformulations = await self.query_generator.reformulator.reformulate_batch(questions)
        else:
            reformulations = [None] * len(examples)

        # Batch generate HyDE passages (use batch method for better performance)
        hyde_passages = []
        if self.query_generator.hyde_generator:
            hyde_passages = await self.query_generator.hyde_generator.generate_batch(examples)
        else:
            hyde_passages = [[] for _ in examples]

        # Convert to QueryVariant lists
        all_llm_queries = []
        for ex_idx, example in enumerate(examples):
            queries = []
            example_id = example.get("id", "unknown")
            qid = 0

            # Add reformulation
            if reformulations[ex_idx]:
                queries.append(
                    QueryVariant(
                        query_id=f"{example_id}_reform_{qid}",
                        text=reformulations[ex_idx],
                        kind=QueryKind.REFORM,
                        example_id=example_id,
                    )
                )
                qid += 1

            # Add HyDE passages
            for i, passage in enumerate(hyde_passages[ex_idx]):
                queries.append(
                    QueryVariant(
                        query_id=f"{example_id}_hyde_{i}_{qid}",
                        text=passage,
                        kind=QueryKind.HYDE,
                        example_id=example_id,
                        metadata={"hyde_index": i},
                    )
                )
                qid += 1

            all_llm_queries.append(queries)

        return all_llm_queries

    def _create_context_from_results(
        self,
        results: List[RetrievalResult],
        question: str,
        example: Dict[str, Any],
    ) -> Optional[Context]:
        """
        Create Context from retrieval results.

        Args:
            results: Retrieval results
            question: Question text
            example: Example dict

        Returns:
            Context object or None
        """
        if not results:
            return None

        # Apply lexical overlap gate
        fusion = FusionStrategy(self.config.retrieval.fusion)
        filtered_results = fusion.filter_by_lexical_overlap(
            results,
            question,
            self.config.retrieval.fusion.min_lexical_overlap,
        )

        if not filtered_results:
            return None

        # Apply content type prioritization
        if self.config.retrieval.prioritize_content_types:
            filtered_results = fusion.filter_by_content_type(
                filtered_results,
                self.config.retrieval.prioritize_content_types,
            )

        # Limit to max_passages
        filtered_results = filtered_results[: self.config.retrieval.fine.max_passages]

        # Pack evidence by token limit
        filtered_results = self._pack_evidence(filtered_results)

        # Build context
        total_tokens = sum(r.chunk.token_count for r in filtered_results)

        metadata = {
            "num_results": len(filtered_results),
        }

        return Context(passages=filtered_results, total_tokens=total_tokens, metadata=metadata)

    def _pack_evidence(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Pack evidence to fit within token limit.

        Args:
            results: Retrieval results

        Returns:
            Packed results
        """
        max_tokens = self.config.retrieval.fine.max_evidence_tokens

        if max_tokens <= 0:
            return results

        kept = []
        used = 0

        for result in results:
            tokens = result.chunk.token_count

            if used + tokens <= max_tokens:
                kept.append(result)
                used += tokens
            else:
                break

        return kept

    async def evaluate(self) -> Dict[str, Any]:
        """
        Run full evaluation.

        Returns:
            Metrics dictionary
        """
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("MEDICAL RAG EVALUATION")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.config.retrieval.mode.value}")
        logger.info(f"Index: {self.config.index_dir}")
        logger.info(f"Data: {self.config.eval_data_path}")
        logger.info("=" * 70)

        # Load examples
        examples = self._load_examples()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Process in batches
        all_predictions = []

        batch_size = self.config.batch_size

        with open(self.config.output_dir / "predictions.jsonl", "w", encoding="utf-8") as pred_file:
            for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
                batch = examples[i : i + batch_size]
                predictions = await self.evaluate_batch(batch)

                all_predictions.extend(predictions)

                # Write predictions
                for pred in predictions:
                    pred_file.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")

                pred_file.flush()

        # Calculate metrics
        metrics = self.metrics_calculator.calculate(all_predictions)

        elapsed = time.time() - start_time

        metrics["elapsed_seconds"] = elapsed
        metrics["examples_per_second"] = len(all_predictions) / elapsed if elapsed > 0 else 0

        # Save metrics
        metrics_path = self.config.output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Print summary
        self._print_summary(metrics)

        return metrics

    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total examples: {metrics['total']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        logger.info(f"With context: {metrics['with_context_accuracy']:.4f} (n={metrics['with_context_count']})")
        logger.info(f"Without context: {metrics['without_context_accuracy']:.4f} (n={metrics['without_context_count']})")
        logger.info(f"Elapsed: {metrics['elapsed_seconds']:.1f}s ({metrics['examples_per_second']:.2f} ex/s)")
        logger.info("=" * 70)
