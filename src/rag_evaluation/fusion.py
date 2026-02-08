"""
Fusion strategies for combining multiple retrieval results.

Supports RRF (Reciprocal Rank Fusion), scoring, and filtering.
"""

import logging
from typing import List, Tuple, Dict, Any, Set, Optional, TYPE_CHECKING
from collections import defaultdict
from functools import lru_cache

import numpy as np

from .config import FusionConfig
from .chunk_schema import RetrievalResult, Chunk

if TYPE_CHECKING:
    from .chunk_schema import Content_type

logger = logging.getLogger(__name__)


class FusionStrategy:
    """
    Fusion strategy for combining multiple retrieval result lists.

    Supports:
    - RRF (Reciprocal Rank Fusion)
    - Score normalization and weighting
    - Lexical overlap filtering
    - Deduplication
    """

    # Stopwords for lexical overlap calculation
    STOPWORDS = set(
        """\
a an the and or for to of in on at from by with without into about as is are was were be been being that which who
whom whose this these those it its they them their do does did done doing have has had having not no yes can could
will would should may might must more most many few some any each every other another such than then when where
while how why patient patients month months year years day days week weeks male female man woman boy girl
""".split()
    )

    def __init__(self, config: FusionConfig):
        """
        Initialize fusion strategy.

        Args:
            config: Fusion configuration
        """
        self.config = config

    def fuse_rrf(
        self,
        rank_lists: List[List[Tuple[int, float]]],
        rrf_k: int,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple result lists using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank + 1)) for each list containing the item

        Args:
            rank_lists: List of (chunk_idx, score) tuples
            rrf_k: RRF constant
            top_k: Number of results to return

        Returns:
            Fused list of (chunk_idx, fused_score) tuples
        """
        scores: Dict[int, float] = {}
        votes: Dict[int, int] = {}

        for lst in rank_lists:
            for rank, (chunk_idx, _orig_score) in enumerate(lst):
                scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
                votes[chunk_idx] = votes.get(chunk_idx, 0) + 1

        # Convert to list and sort
        fused = [(idx, scores[idx], votes[idx]) for idx in scores]
        fused.sort(key=lambda x: x[1], reverse=True)

        return [(idx, score) for idx, score, _votes in fused[:top_k]]

    def _minmax_normalize(self, values: List[float]) -> List[float]:
        """
        Min-max normalize values to [0, 1].

        Args:
            values: List of values

        Returns:
            Normalized values
        """
        if not values:
            return []

        lo, hi = min(values), max(values)
        if hi == lo:
            return [0.0 for _ in values]

        return [(v - lo) / (hi - lo) for v in values]

    @lru_cache(maxsize=200_000)
    def _lexical_overlap(self, query: str, text: str) -> float:
        """
        Calculate lexical overlap between query and text.

        Overlap = |query_tokens ∩ text_tokens| / |query_tokens|

        Args:
            query: Query text
            text: Passage text

        Returns:
            Overlap ratio [0, 1]
        """
        import re

        def tokenize(s):
            return [
                w
                for w in re.findall(r"[a-zA-Z]+", (s or "").lower())
                if w not in self.STOPWORDS and len(w) > 2
            ]

        query_tokens = set(tokenize(query))
        text_tokens = set(tokenize(text))

        if not query_tokens:
            return 0.0

        return len(query_tokens & text_tokens) / len(query_tokens)

    def fuse_and_rescore(
        self,
        rank_lists: List[List[Tuple[int, float]]],
        chunks: List[Chunk],
        query: str,
    ) -> List[RetrievalResult]:
        """
        Fuse multiple result lists with scoring, filtering, and deduplication.

        Args:
            rank_lists: List of (chunk_idx, score) tuples from different retrievers
            chunks: List of all chunks (for looking up by index)
            query: Query text for lexical overlap

        Returns:
            List of fused and scored RetrievalResult objects
        """
        # Stage 1: RRF fusion
        fused_with_votes = self.fuse_rrf(
            rank_lists,
            rrf_k=self.config.rrf_k,
            top_k=max(len(chunks), 100),  # Get plenty for filtering
        )

        # Stage 2: Build candidate list with all scores
        candidates = []
        for chunk_idx, rrf_score, votes in fused_with_votes:
            chunk = chunks[chunk_idx]

            # Calculate individual scores (if available)
            scores = {"rrf_score": rrf_score, "fusion_votes": votes}

            # Add raw scores from individual retrievers
            for lst in rank_lists:
                for idx, raw_score in lst:
                    if idx == chunk_idx:
                        # Determine which retriever this came from
                        scores["raw_score"] = raw_score
                        break

            # Calculate lexical overlap
            lexical_overlap = self._lexical_overlap(query, chunk.text)
            scores["lexical_overlap"] = lexical_overlap

            candidates.append(
                {
                    "chunk_idx": chunk_idx,
                    "chunk": chunk,
                    "rrf_score": rrf_score,
                    "fusion_votes": votes,
                    "lexical_overlap": lexical_overlap,
                    "all_scores": scores,
                }
            )

        if not candidates:
            return []

        # Stage 3: Normalize scores
        rrf_values = [c["rrf_score"] for c in candidates]
        overlap_values = [c["lexical_overlap"] for c in candidates]

        rrf_norm = self._minmax_normalize(rrf_values)
        overlap_norm = self._minmax_normalize(overlap_values)

        # Stage 4: Calculate combined score
        total_weight = (
            self.config.alpha_rrf
            + self.config.beta_overlap
            + self.config.bm25_weight
            + self.config.dense_weight
        )

        if total_weight == 0:
            total_weight = 1.0

        for i, c in enumerate(candidates):
            combined = (
                self.config.alpha_rrf * rrf_norm[i]
                + self.config.beta_overlap * overlap_norm[i]
            ) / total_weight

            c["final_score"] = combined

        # Stage 5: Filter by final score threshold
        filtered = [c for c in candidates if c["final_score"] >= self.config.min_final_threshold]

        # Stage 6: Deduplication
        if self.config.dedupe_by_doc or self.config.dedupe_by_section:
            filtered = self._dedupe(filtered)

        # Stage 7: Sort by final score
        filtered.sort(key=lambda x: x["final_score"], reverse=True)

        # Stage 8: Convert to RetrievalResult objects
        results = []
        for rank, c in enumerate(filtered):
            result = RetrievalResult(
                chunk=c["chunk"],
                score=c["final_score"],
                rank=rank,
                scores=c["all_scores"],
            )
            results.append(result)

        return results

    def _dedupe(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate candidates by doc_id and/or section_id.

        Keeps the highest-scoring candidate from each group.

        Args:
            candidates: List of candidate dicts

        Returns:
            Deduplicated list
        """
        if not (self.config.dedupe_by_doc or self.config.dedupe_by_section):
            return candidates

        # Group by key
        groups: Dict[str, List[Dict]] = defaultdict(list)

        for c in candidates:
            if self.config.dedupe_by_section:
                key = c["chunk"].section_id
            else:
                key = c["chunk"].doc_id

            groups[key].append(c)

        # Keep best from each group
        deduped = []
        for group in groups.values():
            best = max(group, key=lambda x: x["final_score"])
            deduped.append(best)

        return deduped

    def filter_by_lexical_overlap(
        self,
        results: List[RetrievalResult],
        query: str,
        threshold: float,
    ) -> List[RetrievalResult]:
        """
        Filter results by lexical overlap threshold.

        If top result is below threshold, returns empty list.

        Args:
            results: Retrieval results
            query: Query text
            threshold: Overlap threshold

        Returns:
            Filtered results
        """
        if not results:
            return []

        # Only apply filter if lexical_overlap is present in scores
        if "lexical_overlap" not in results[0].scores:
            # No lexical overlap score calculated, skip filtering
            return results

        top_overlap = results[0].scores.get("lexical_overlap", 0.0)

        if top_overlap < threshold:
            logger.debug(f"Top lexical overlap {top_overlap:.3f} < {threshold}, dropping all results")
            return []

        return results

    def filter_by_content_type(
        self,
        results: List[RetrievalResult],
        prioritize_types: List["Content_type"],
    ) -> List[RetrievalResult]:
        """
        Re-rank results to prioritize specific content types.

        Args:
            results: Retrieval results
            prioritize_types: Content types to prioritize

        Returns:
            Re-ranked results
        """
        if not results or not prioritize_types:
            return results

        # Boost scores for prioritized types
        for result in results:
            content_type = result.chunk.section_metadata.content_type
            if content_type in prioritize_types:
                result.score *= 1.5  # Boost by 50%

        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)

        # Reassign ranks
        for i, result in enumerate(results):
            result.rank = i

        return results
