"""
HyDE (Hypothetical Document Embeddings) query generation.

Generates hypothetical medical passages to improve retrieval.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from .config import QueryConfig
from .query_reformulation import AsyncLLMClient

logger = logging.getLogger(__name__)


class HyDEMode(str, Enum):
    """HyDE generation modes."""

    QUESTION = "question"  # Generate for the question only
    OPTION = "option"  # Generate for each option
    BOTH = "both"  # Generate for question + each option


# Default HyDE prompts
DEFAULT_HYDE_PROMPT = """You are writing a brief medical reference note to help retrieve textbooks.

Question: {question}

Write a concise 2–4 sentence clinical answer using specific terminology
(diagnosis, mechanism, key pathways, drug classes, hallmark findings).
Avoid option letters and hedging. No step-by-step reasoning.
"""

DEFAULT_HYDE_OPTION_PROMPT = """You are writing a brief medical reference note to help retrieve textbooks.

Question: {question}
Candidate option: {option}

In 2–4 sentences, summarize the key medical facts that would support or refute this option.
Use precise medical terminology (mechanisms, pathways, hallmark findings). No option letters.
"""


class HyDEGenerator:
    """
    Generates hypothetical passages for HyDE retrieval.

    For each question (and optionally each option), generates a brief
    medical passage that represents the ideal retrieved content.
    """

    # Negation detection patterns
    NEGATION_TOKENS = (" EXCEPT", " Except", " NOT ", " FALSE ", " LEAST ", " INCORRECT ")

    def __init__(self, config: QueryConfig, base_url: str):
        """
        Initialize HyDE generator.

        Args:
            config: Query configuration
            base_url: Ollama base URL
        """
        self.config = config
        self.base_url = base_url
        self.mode = HyDEMode(config.hyde_mode)

        # Initialize LLM client
        self.llm_client = AsyncLLMClient(
            model_name=config.hyde_model,
            base_url=base_url,
            max_workers=4,
        )

    def is_negation_question(self, question: str) -> bool:
        """
        Detect if this is a negation question (EXCEPT, NOT, etc.).

        Args:
            question: Question text

        Returns:
            True if negation question detected
        """
        q_l = " " + (question or "")
        return any(tok in q_l for tok in self.NEGATION_TOKENS)

    def build_prompts_for_example(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Build HyDE prompts for a single example.

        Args:
            question: Question text
            options: Optional dict of options (A, B, C, D, E)

        Returns:
            List of prompt strings
        """
        prompts = []
        is_neg = self.is_negation_question(question)

        # Question-level HyDE
        if self.mode in (HyDEMode.QUESTION, HyDEMode.BOTH):
            if is_neg:
                # For negation questions, generate two passages:
                # one for true statements, one for the exception
                prompt_true = DEFAULT_HYDE_PROMPT.format(question)
                prompt_true += "\n\nFocus on statements that are LIKELY TRUE in typical cases."
                prompts.append(prompt_true)

                prompt_false = DEFAULT_HYDE_PROMPT.format(question)
                prompt_false += "\n\nFocus on statements that are LIKELY FALSE or the EXCEPTION."
                prompts.append(prompt_false)
            else:
                prompts.append(DEFAULT_HYDE_PROMPT.format(question=question))

        # Option-level HyDE
        if self.mode in (HyDEMode.OPTION, HyDEMode.BOTH) and options:
            for opt_key, opt_text in sorted(options.items()):
                if opt_text.strip():
                    prompt = DEFAULT_HYDE_OPTION_PROMPT.format(
                        question=question,
                        option=opt_text.strip(),
                    )
                    prompts.append(prompt)

        return prompts

    async def generate_for_example(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Generate HyDE passages for a single example.

        Args:
            question: Question text
            options: Optional dict of options

        Returns:
            List of generated hypothetical passages
        """
        prompts = self.build_prompts_for_example(question, options)

        if not prompts:
            return []

        try:
            passages = await self.llm_client.generate_batch(
                prompts,
                max_tokens=self.config.hyde_max_tokens,
                temperature=self.config.hyde_temperature,
                top_p=1.0,
            )
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return []

        # Clean passages
        cleaned = []
        for p in passages or []:
            p = (p or "").strip()
            if p:
                cleaned.append(p)

        return cleaned

    async def generate_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[List[str]]:
        """
        Generate HyDE passages for a batch of examples.

        Args:
            examples: List of example dicts with 'question' and optionally 'options'

        Returns:
            List of HyDE passage lists (one per example)
        """
        # Collect all prompts
        all_prompts = []
        spans = []  # (example_idx, start, end)

        cursor = 0
        for ex_idx, ex in enumerate(examples):
            question = ex.get("question", "").strip()
            options = self._extract_options(ex)

            prompts = self.build_prompts_for_example(question, options)
            all_prompts.extend(prompts)
            spans.append((ex_idx, cursor, cursor + len(prompts)))
            cursor += len(prompts)

        if not all_prompts:
            return [[] for _ in examples]

        # Generate batch
        try:
            all_passages = await self.llm_client.generate_batch(
                all_prompts,
                max_tokens=self.config.hyde_max_tokens,
                temperature=self.config.hyde_temperature,
                top_p=1.0,
            )
        except Exception as e:
            logger.warning(f"Batch HyDE generation failed: {e}")
            return [[] for _ in examples]

        # Organize back by example
        results = [[] for _ in examples]
        for ex_idx, start, end in spans:
            passages = all_passages[start:end]
            cleaned = [p.strip() for p in passages if p and p.strip()]
            results[ex_idx] = cleaned

        return results

    def _extract_options(self, example: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Extract options from example dict.

        Handles both MedQA and MedMCQA formats.

        Args:
            example: Example dict

        Returns:
            Dict of options or None
        """
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
