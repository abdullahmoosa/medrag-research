"""
Query reformulation using LLM.

Rewrites clinical vignettes into optimized search queries for textbook retrieval.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .config import QueryConfig

logger = logging.getLogger(__name__)


# Default reformulation prompt (medical-domain specific)
DEFAULT_REFORMULATION_PROMPT = """You are a medical information retrieval expert.

Rewrite the following clinical vignette into ONE optimized search query
for retrieving relevant medical textbook passages.

Instructions:
- Remove narrative patient details that do not appear in textbooks
- Preserve clinically meaningful abstractions
  (e.g., pediatric vs adult, acute vs chronic, pregnancy, sex-specific anatomy)
- Convert ages and timelines into medical categories when relevant
- Focus on pathophysiology, anatomy, histology, or mechanism being tested
- Use formal medical terminology used in textbooks
- Avoid full sentences; use keyword-style phrasing
- Target 8–12 words (do not exceed 12)

Example:
Input:
"A 17-year-old female develops pruritic wheals after peanut exposure..."

Output:
"IgE mediated mast cell degranulation superficial dermis urticaria"

Now rewrite the following question:

{question}

Respond with ONLY the rewritten query.
"""


class AsyncLLMClient:
    """Async LLM client for query reformulation."""

    def __init__(self, model_name: str, base_url: str, max_workers: int = 4):
        """
        Initialize async LLM client.

        Args:
            model_name: Model name (Ollama)
            base_url: Ollama base URL
            max_workers: Number of worker threads
        """
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.base_url = base_url

        try:
            import ollama
            self._client = ollama.Client(host=base_url)
        except Exception:
            self._client = None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 120,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate text asynchronously."""
        import ollama

        client = self._client or ollama.Client(host=self.base_url)
        loop = asyncio.get_event_loop()

        resp = await loop.run_in_executor(
            self.executor,
            lambda: client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stop": ["\n", "\r\n"] if max_tokens <= 5 else None,
                },
            ),
        )

        return resp.get("response", "")

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 120,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        """Generate multiple texts asynchronously."""
        tasks = [
            self.generate(p, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)


class QueryReformulator:
    """
    Reformulates queries for better retrieval.

    Uses LLM to convert clinical vignettes into search-optimized queries.
    """

    # Validation patterns
    BAD_TOKENS = [
        "year-old",
        "years old",
        "patient",
        "presents",
        "comes",
        "man",
        "woman",
        "male",
        "female",
    ]

    def __init__(self, config: QueryConfig, base_url: str):
        """
        Initialize query reformulator.

        Args:
            config: Query configuration
            base_url: Ollama base URL
        """
        self.config = config
        self.base_url = base_url

        # Initialize LLM client
        self.llm_client = AsyncLLMClient(
            model_name=config.reformulation_model,
            base_url=base_url,
            max_workers=4,  # Can be made configurable
        )

        # Load prompt template
        self.prompt_template = config.reformulation_prompt_template or DEFAULT_REFORMULATION_PROMPT

    def validate_reformulated_query(self, query: str) -> bool:
        """
        Validate that a reformulated query meets quality criteria.

        Args:
            query: Reformulated query text

        Returns:
            True if query passes validation
        """
        if not query or not query.strip():
            return False

        tokens = [t for t in re.findall(r"\S+", query.strip())]

        # Length check
        if not (5 <= len(tokens) <= 14):
            return False

        # Content check
        lower = query.lower()
        if any(bt in lower for bt in self.BAD_TOKENS):
            return False

        # Must contain at least one multi-char medical token
        if not any(len(re.sub(r"[^a-zA-Z]", "", t)) > 4 for t in tokens):
            return False

        return True

    def heuristic_fallback(self, question: str) -> str:
        """
        Generate a heuristic fallback query by extracting keywords.

        Args:
            question: Original question text

        Returns:
            Fallback query string
        """
        tokens = [w.lower() for w in re.findall(r"[a-zA-Z]+", question) if len(w) > 4]
        return " ".join(tokens[:10]) if tokens else question[:120]

    async def reformulate(self, question: str) -> str:
        """
        Reformulate a single question.

        Args:
            question: Question text

        Returns:
            Reformulated query
        """
        # Build prompt
        prompt = self.prompt_template.format(question=question)

        # Generate
        try:
            reformulated = await self.llm_client.generate(
                prompt,
                max_tokens=self.config.reformulation_max_tokens,
                temperature=self.config.reformulation_temperature,
            )
        except Exception as e:
            logger.warning(f"Reformulation failed: {e}, using fallback")
            return self.heuristic_fallback(question)

        # Clean and validate
        reformulated = reformulated.strip().replace("\n", " ")

        if not self.validate_reformulated_query(reformulated):
            logger.debug(f"Query failed validation, using fallback: {reformulated[:50]}...")
            return self.heuristic_fallback(question)

        return reformulated

    async def reformulate_batch(self, questions: List[str]) -> List[str]:
        """
        Reformulate multiple questions.

        Args:
            questions: List of question texts

        Returns:
            List of reformulated queries
        """
        # Build prompts
        prompts = [self.prompt_template.format(question=q) for q in questions]

        # Generate batch
        try:
            reformulated_list = await self.llm_client.generate_batch(
                prompts,
                max_tokens=self.config.reformulation_max_tokens,
                temperature=self.config.reformulation_temperature,
            )
        except Exception as e:
            logger.warning(f"Batch reformulation failed: {e}, using fallbacks")
            return [self.heuristic_fallback(q) for q in questions]

        # Clean and validate
        results = []
        for q, reformulated in zip(questions, reformulated_list):
            reformulated = (reformulated or "").strip().replace("\n", " ")

            if not self.validate_reformulated_query(reformulated):
                reformulated = self.heuristic_fallback(q)

            results.append(reformulated)

        return results


def sanitize_query_expansions(expansions: List[str]) -> List[str]:
    """
    Sanitize query expansions from dataset.

    Removes invalid or low-quality expansions.

    Args:
        expansions: List of query expansion strings

    Returns:
        Sanitized list of unique expansions
    """
    BAD_QE = ("answer", "true/false", "select one", "all are true", "except")

    out = []
    for e in expansions or []:
        s = (e or "").strip()
        sl = s.lower()

        # Length check
        if not (3 <= len(s) <= 50):
            continue

        # Content check
        if any(b in sl for b in BAD_QE):
            continue

        # Single letters
        if sl in {"a", "b", "c", "d"}:
            continue

        out.append(s)

    # Deduplicate
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    return uniq
