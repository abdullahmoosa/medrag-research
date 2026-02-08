"""
Core retrieval logic with multi-stage architecture.

Supports coarse (section-level) and fine (chunk-level) retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import faiss

from .index_loader import IndexManager
from .chunk_schema import Chunk, QueryVariant, QueryKind, RetrievalResult, Content_type
from .config import RetrievalConfig, RetrievalMode
from .fusion import FusionStrategy

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(
        self,
        queries: List[QueryVariant],
        top_k: int,
        filter_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k chunks.

        Args:
            queries: List of query variants
            top_k: Number of results to return
            filter_ids: Optional set of chunk IDs to restrict search space

        Returns:
            List of (chunk_idx, score) tuples
        """
        pass


class Denseretriever(Retriever):
    """Dense retrieval using FAISS."""

    def __init__(
        self,
        index_manager: IndexManager,
        embedding_client,
        config: RetrievalConfig,
    ):
        """
        Initialize dense retriever.

        Args:
            index_manager: Index manager
            embedding_client: Embedding backend
            config: Retrieval configuration
        """
        self.index_manager = index_manager
        self.embedding_client = embedding_client
        self.config = config

        self.faiss_index = index_manager.faiss_index
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded")

    def retrieve(
        self,
        queries: List[QueryVariant],
        top_k: int,
        filter_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Dense retrieval using FAISS.

        Args:
            queries: List of query variants
            top_k: Number of results
            filter_ids: Optional ID filter for coarse retrieval

        Returns:
            List of (chunk_idx, score) tuples
        """
        if not queries:
            return []

        # Encode queries
        query_texts = [q.text for q in queries]
        q_embs = self.embedding_client.encode(query_texts, is_query=True)

        # Normalize for cosine similarity
        q_embs = q_embs.astype(np.float32)
        faiss.normalize_L2(q_embs)

        # Search
        # When filtering is active, request more results to ensure we get enough after filtering
        search_k = top_k
        if filter_ids is not None:
            # Calculate filter ratio to request appropriate number of results
            filter_ratio = len(filter_ids) / self.faiss_index.ntotal  # e.g., 3000/64416 = 0.046
            # Request enough to get top_k after filtering
            # Add 50% safety margin
            search_k = int(top_k / filter_ratio * 1.5) if filter_ratio > 0 else top_k * 10
            # Cap at reasonable maximum (don't request more than 500)
            search_k = min(search_k, 500)
            search_k = max(search_k, top_k * 2)  # At least 2x

        all_results = []
        for q_emb in q_embs:
            scores, ids = self.faiss_index.search(q_emb.reshape(1, -1), search_k)

            # Convert to results
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                if filter_ids is not None and idx not in filter_ids:
                    continue
                results.append((int(idx), float(score)))

            all_results.extend(results)

        # Deduplicate and sort
        seen = set()
        deduped = []
        for idx, score in all_results:
            if idx not in seen:
                seen.add(idx)
                deduped.append((idx, score))

        deduped.sort(key=lambda x: x[1], reverse=True)
        return deduped[:top_k]


class BM25Retriever(Retriever):
    """Sparse retrieval using BM25."""

    def __init__(
        self,
        index_manager: IndexManager,
        config: RetrievalConfig,
    ):
        """
        Initialize BM25 retriever.

        Args:
            index_manager: Index manager
            config: Retrieval configuration
        """
        self.index_manager = index_manager
        self.config = config

        self.bm25_index = index_manager.bm25_index
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded")

    @lru_cache(maxsize=200_000)
    def _tokenize(self, query: str) -> tuple:
        """Tokenize query with caching."""
        return tuple(query.lower().split())

    def retrieve(
        self,
        queries: List[QueryVariant],
        top_k: int,
        filter_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Sparse retrieval using BM25.

        Args:
            queries: List of query variants
            top_k: Number of results
            filter_ids: Optional ID filter

        Returns:
            List of (chunk_idx, score) tuples
        """
        if not queries:
            return []

        all_results = []

        for query in queries:
            tokens = list(self._tokenize(query.text))

            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokens)

            # When filtering is active, request more results to ensure we get enough after filtering
            search_k = top_k
            if filter_ids is not None:
                # Calculate filter ratio to request appropriate number of results
                filter_ratio = len(filter_ids) / len(scores)  # e.g., 3000/64416 = 0.046
                # Request enough to get top_k after filtering
                # Add 50% safety margin
                search_k = int(top_k / filter_ratio * 1.5) if filter_ratio > 0 else top_k * 10
                # Cap at reasonable maximum (don't request more than 1000 for BM25)
                search_k = min(search_k, 1000)
                search_k = max(search_k, top_k * 2)  # At least 2x

            # Get top-k
            top_k_indices = np.argpartition(-scores, search_k)[:search_k]
            top_k_scores = scores[top_k_indices]

            # Sort by score
            order = np.argsort(-top_k_scores)

            for rank, idx in enumerate(order):
                chunk_idx = int(top_k_indices[idx])
                score = float(top_k_scores[idx])

                if filter_ids is not None and chunk_idx not in filter_ids:
                    continue

                all_results.append((chunk_idx, score))

        # Deduplicate by RRF-style voting
        from collections import defaultdict

        vote_counts = defaultdict(int)
        best_scores = {}

        for chunk_idx, score in all_results:
            vote_counts[chunk_idx] += 1
            if chunk_idx not in best_scores or score > best_scores[chunk_idx][1]:
                best_scores[chunk_idx] = (chunk_idx, score)

        # Sort by votes then by score
        results = list(best_scores.values())
        results.sort(key=lambda x: (vote_counts[x[0]], x[1]), reverse=True)

        return results[:top_k]


class HybridRetriever(Retriever):
    """Hybrid retrieval combining dense and BM25."""

    def __init__(
        self,
        index_manager: IndexManager,
        embedding_client,
        config: RetrievalConfig,
    ):
        """
        Initialize hybrid retriever.

        Args:
            index_manager: Index manager
            embedding_client: Embedding backend
            config: Retrieval configuration
        """
        self.index_manager = index_manager
        self.embedding_client = embedding_client
        self.config = config

        self.dense_retriever = Denseretriever(index_manager, embedding_client, config)
        self.bm25_retriever = BM25Retriever(index_manager, config)
        self.fusion = FusionStrategy(config.fusion)

    def retrieve(
        self,
        queries: List[QueryVariant],
        top_k: int,
        filter_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Hybrid retrieval with fusion.

        Args:
            queries: List of query variants
            top_k: Number of results
            filter_ids: Optional ID filter

        Returns:
            List of (chunk_idx, score) tuples
        """
        if not queries:
            return []

        # Retrieve from both dense and BM25
        dense_results = self.dense_retriever.retrieve(queries, top_k=self.config.fine.dense_k, filter_ids=filter_ids)
        bm25_results = self.bm25_retriever.retrieve(queries, top_k=self.config.fine.bm25_k, filter_ids=filter_ids)

        # Fuse with RRF
        fused = self.fusion.fuse_rrf(
            rank_lists=[dense_results, bm25_results],
            rrf_k=self.config.fusion.rrf_k,
            top_k=top_k,
        )

        return fused


class RetrievalPipeline:
    """
    Multi-stage retrieval pipeline.

    Stage 1 (optional): Coarse retrieval at section level
    Stage 2: Fine retrieval at chunk level
    Stage 3 (optional): Reranking
    """

    def __init__(
        self,
        index_manager: IndexManager,
        embedding_client,
        config: RetrievalConfig,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            index_manager: Index manager
            embedding_client: Embedding backend
            config: Retrieval configuration
        """
        self.index_manager = index_manager
        self.embedding_client = embedding_client
        self.config = config

        # Store FAISS and BM25 indexes for batch operations
        self.faiss_index = index_manager.faiss_index
        self.bm25_index = index_manager.bm25_index

        # Initialize retriever based on mode
        mode = config.fine.mode

        if mode == RetrievalMode.DENSE:
            self.retriever = Denseretriever(index_manager, embedding_client, config)
        elif mode == RetrievalMode.BM25:
            self.retriever = BM25Retriever(index_manager, config)
        elif mode == RetrievalMode.HYBRID:
            self.retriever = HybridRetriever(index_manager, embedding_client, config)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        # Reranker (optional)
        self.reranker = None
        if config.reranker.enabled:
            self._init_reranker()

        # Build section map for coarse retrieval
        self._build_section_map()

    def _build_section_map(self) -> None:
        """Build mapping from section to chunk indices."""
        self.section_to_chunks: Dict[str, Set[int]] = {}

        for idx, chunk in enumerate(self.index_manager.chunks):
            section_id = chunk.section_id
            if section_id not in self.section_to_chunks:
                self.section_to_chunks[section_id] = set()
            self.section_to_chunks[section_id].add(idx)

        logger.info(f"Built section map: {len(self.section_to_chunks)} sections")

    def _init_reranker(self) -> None:
        """Initialize cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.reranker = CrossEncoder(
                self.config.reranker.model_name,
                device=device,
                max_length=self.config.reranker.max_length,
            )

            if self.config.reranker.use_fp16 and device == "cuda":
                self.reranker.model.half()

            logger.info(f"Loaded reranker: {self.config.reranker.model_name} on {device}")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            self.reranker = None

    @lru_cache(maxsize=200_000)
    def _tokenize(self, query: str) -> tuple:
        """Tokenize query with caching (for BM25)."""
        return tuple(query.lower().split())

    def _coarse_retrieval(
        self,
        queries: List[QueryVariant],
    ) -> Set[str]:
        """
        Stage 1: Coarse section-level retrieval.

        Args:
            queries: Query variants

        Returns:
            Set of section IDs to search in fine stage
        """
        if not self.config.coarse.enabled:
            # No coarse stage - search all sections
            return set(self.section_to_chunks.keys())

        # Retrieve at section level
        # For now, we'll use the same retriever but aggregate results by section
        results = self.retriever.retrieve(
            queries,
            top_k=self.config.coarse.dense_k if self.config.fine.mode != RetrievalMode.BM25 else self.config.coarse.bm25_k,
        )

        # Map chunks to sections
        section_scores: Dict[str, float] = {}

        for chunk_idx, score in results:
            chunk = self.index_manager.chunks[chunk_idx]
            section_id = chunk.section_id

            if section_id not in section_scores:
                section_scores[section_id] = score
            else:
                section_scores[section_id] = max(section_scores[section_id], score)

        # Get top-k sections
        sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        top_sections = set([sid for sid, _ in sorted_sections[: self.config.coarse.top_k_sections]])

        logger.debug(f"Coarse retrieval: {len(top_sections)} sections from {len(section_scores)} total")

        return top_sections

    def _fine_retrieval(
        self,
        queries: List[QueryVariant],
        section_filter: Optional[Set[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Stage 2: Fine-grained chunk-level retrieval.

        Args:
            queries: Query variants
            section_filter: Optional set of sections to search within

        Returns:
            List of retrieval results
        """
        # Build chunk ID filter from section filter
        filter_ids = None
        if section_filter:
            filter_ids = set()
            for section_id in section_filter:
                filter_ids.update(self.section_to_chunks.get(section_id, set()))

            logger.debug(f"Restricting search to {len(filter_ids)} chunks in {len(section_filter)} sections")

        # Retrieve
        results = self.retriever.retrieve(
            queries,
            top_k=self.config.fine.top_k,
            filter_ids=filter_ids,
        )

        # Convert to RetrievalResult objects
        retrieval_results = []
        for rank, (chunk_idx, score) in enumerate(results):
            chunk = self.index_manager.chunks[chunk_idx]
            retrieval_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    scores={"final": score},
                )
            )

        return retrieval_results

    def _rerank(self, results: List[RetrievalResult], query_text: str) -> List[RetrievalResult]:
        """
        Stage 3: Rerank top-k results.

        Args:
            results: Retrieval results
            query_text: Query text for reranking

        Returns:
            Reranked results
        """
        if not self.reranker or not results:
            return results

        # Prepare pairs
        top_k = min(self.config.reranker.top_k, len(results))
        pairs = [(query_text, r.chunk.text) for r in results[:top_k]]

        try:
            scores = self.reranker.predict(
                pairs,
                batch_size=self.config.reranker.batch_size,
                show_progress_bar=False,
            )

            # Create new RetrievalResult objects with updated scores (frozen dataclass)
            reranked = []
            for i, result in enumerate(results[:top_k]):
                # Create updated scores dict
                updated_scores = dict(result.scores)
                updated_scores["ce_score"] = float(scores[i])

                # Create new RetrievalResult with updated score
                reranked.append(
                    RetrievalResult(
                        chunk=result.chunk,
                        score=float(scores[i]),
                        rank=i,  # Will be updated after sorting
                        scores=updated_scores,
                        metadata=result.metadata,
                    )
                )

            # Keep non-reranked results as-is
            reranked.extend(results[top_k:])

            # Re-sort by new score
            reranked = sorted(reranked, key=lambda x: x.score, reverse=True)

            # Reassign ranks
            reranked = [
                RetrievalResult(
                    chunk=r.chunk,
                    score=r.score,
                    rank=i,
                    scores=r.scores,
                    metadata=r.metadata,
                )
                for i, r in enumerate(reranked)
            ]

            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")

        return results

    def retrieve(self, queries: List[QueryVariant], question: str) -> List[RetrievalResult]:
        """
        Full retrieval pipeline.

        Args:
            queries: Query variants
            question: Original question text (for reranking)

        Returns:
            List of retrieval results
        """
        # Stage 1: Coarse retrieval (optional)
        section_filter = self._coarse_retrieval(queries)

        # Stage 2: Fine retrieval
        results = self._fine_retrieval(queries, section_filter)

        # Stage 3: Reranking (optional)
        if self.config.reranker.enabled:
            results = self._rerank(results, question)

        return results

    def retrieve_batch(
        self,
        all_queries: List[List[QueryVariant]],
        questions: List[str],
    ) -> List[List[RetrievalResult]]:
        """
        Batch retrieval for multiple examples.

        This is the key optimization: encode all queries at once instead of per-example.

        Args:
            all_queries: List of query lists (one per example)
            questions: List of question texts (one per example, for reranking)

        Returns:
            List of retrieval results (one per example)
        """
        if not all_queries:
            return []

        # Flatten all queries with metadata tracking
        flat_queries: List[QueryVariant] = []
        query_to_example: List[int] = []  # Which example each query belongs to

        for ex_idx, queries in enumerate(all_queries):
            for query in queries:
                flat_queries.append(query)
                query_to_example.append(ex_idx)

        if not flat_queries:
            # No queries for any example
            return [[] for _ in all_queries]

        logger.info(f"Batch retrieval: {len(flat_queries)} queries from {len(all_queries)} examples")

        # Batch encode all queries
        q_texts = [q.text for q in flat_queries]
        q_embs = self.embedding_client.encode(q_texts, is_query=True)

        # Normalize for cosine similarity
        q_embs = q_embs.astype(np.float32)
        faiss.normalize_L2(q_embs)

        logger.info(f"  ✓ Encoded {len(q_embs)} queries, shape: {q_embs.shape}")

        # Batch search FAISS (if dense or hybrid mode)
        mode = self.config.fine.mode

        # Collect all results per example
        example_results: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(all_queries))}

        if (mode == RetrievalMode.DENSE or mode == RetrievalMode.HYBRID) and self.faiss_index is not None:
            # Dense retrieval
            logger.info(f"  Starting dense retrieval...")
            k = self.config.fine.dense_k
            scores, ids = self.faiss_index.search(q_embs, k)

            # Distribute results to examples
            for q_idx, (query, score_list, id_list) in enumerate(zip(flat_queries, scores, ids)):
                ex_idx = query_to_example[q_idx]

                for score, idx in zip(score_list, id_list):
                    if idx == -1:  # FAISS returns -1 for missing results
                        continue
                    example_results[ex_idx].append((int(idx), float(score)))

            logger.info(f"  ✓ Dense retrieval complete")

        if (mode == RetrievalMode.BM25 or mode == RetrievalMode.HYBRID) and self.bm25_index is not None:
            # BM25 retrieval
            logger.info(f"  Starting BM25 retrieval for {len(flat_queries)} queries...")
            from collections import defaultdict

            for q_idx, query in enumerate(flat_queries):
                ex_idx = query_to_example[q_idx]
                tokens = list(self._tokenize(query.text))

                # Get BM25 scores
                scores = self.bm25_index.get_scores(tokens)

                # Get top-k
                k = self.config.fine.bm25_k
                top_k_indices = np.argpartition(-scores, k)[:k]
                top_k_scores = scores[top_k_indices]

                # Add to example results
                for chunk_idx, score in zip(top_k_indices, top_k_scores):
                    example_results[ex_idx].append((int(chunk_idx), float(score)))

            logger.info(f"  ✓ BM25 retrieval complete")

        # Fuse and deduplicate within each example
        logger.info(f"  Fusing results...")
        final_results = []

        for ex_idx in range(len(all_queries)):
            results_list = example_results[ex_idx]

            if mode == RetrievalMode.HYBRID:
                # Deduplicate and RRF within this example
                vote_counts = defaultdict(int)
                best_scores = {}

                for chunk_idx, score in results_list:
                    vote_counts[chunk_idx] += 1
                    if chunk_idx not in best_scores or score > best_scores[chunk_idx][1]:
                        best_scores[chunk_idx] = (chunk_idx, score)

                # Sort by votes then by score
                deduped = list(best_scores.values())
                deduped.sort(key=lambda x: (vote_counts[x[0]], x[1]), reverse=True)

                # Take top-k
                top_k_results = deduped[:self.config.fine.top_k]
            else:
                # Just deduplicate
                seen = set()
                deduped = []
                for idx, score in results_list:
                    if idx not in seen:
                        seen.add(idx)
                        deduped.append((idx, score))

                deduped.sort(key=lambda x: x[1], reverse=True)
                top_k_results = deduped[:self.config.fine.top_k]

            # Convert to RetrievalResult objects
            retrieval_results = []
            for rank, (chunk_idx, score) in enumerate(top_k_results):
                chunk = self.index_manager.chunks[chunk_idx]
                retrieval_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=score,
                        rank=rank,
                        scores={"final": score},
                    )
                )

            # Rerank if enabled
            if self.config.reranker.enabled and retrieval_results:
                question = questions[ex_idx]
                retrieval_results = self._rerank(retrieval_results, question)

            final_results.append(retrieval_results)

        logger.info(f"  ✓ Batch retrieval complete, returning {len(final_results)} results")
        return final_results
