"""
Index building module for BM25 and FAISS indexes.

Supports pluggable embedding backends (sentence_transformers, ollama, medembed).
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

from .config import IndexConfig, EmbeddingConfig
from .semantic_chunker import SemanticChunk


logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Backend Interface
# =============================================================================

class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerBackend(EmbeddingBackend):
    """SentenceTransformers embedding backend."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize SentenceTransformers backend.

        Args:
            config: Embedding configuration
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(
            config.model,
            device=config.device
        )
        self.normalize = config.normalize
        self.batch_size = config.batch_size
        logger.info(f"Loaded SentenceTransformer model: {config.model}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using SentenceTransformers."""
        import time
        logger.info(f"  Encoding {len(texts)} texts with batch_size={self.batch_size}...")
        start = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 1000,
        )

        elapsed = time.time() - start
        throughput = len(texts) / elapsed
        logger.info(f"  Encoding completed: {len(texts)} texts in {elapsed:.1f}s ({throughput:.0f} texts/sec)")

        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class OllamaBackend(EmbeddingBackend):
    """Ollama embedding backend."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize Ollama backend.

        Args:
            config: Embedding configuration
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package required. Install with: pip install ollama"
            )

        self.client = ollama.Client()
        self.model = config.model
        logger.info(f"Using Ollama model: {config.model}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Ollama."""
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(response['embedding'])

        arr = np.array(embeddings, dtype=np.float32)

        # Normalize if requested
        # Normalize if requested
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-8)

        return arr

    def get_dimension(self) -> int:
        """Get embedding dimension (needs to be inferred)."""
        # Encode a dummy text to get dimension
        emb = self.client.embeddings(model=self.model, prompt="test")
        return len(emb['embedding'])


class MedEmbedBackend(EmbeddingBackend):
    """
    MedEmbed backend for medical-domain embeddings.

    Placeholder for MedEmbed integration.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize MedEmbed backend.

        Args:
            config: Embedding configuration
        """
        self.model = config.model
        # TODO: Implement MedEmbed loading logic
        logger.warning("MedEmbed backend not fully implemented, using placeholder")
        raise NotImplementedError(
            "MedEmbed backend requires specific implementation. "
            "Please update this class with your MedEmbed integration."
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using MedEmbed."""
        raise NotImplementedError("MedEmbed encode not implemented")

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        raise NotImplementedError("MedEmbed get_dimension not implemented")


def get_embedding_backend(config: EmbeddingConfig) -> EmbeddingBackend:
    """
    Get embedding backend based on configuration.

    Args:
        config: Embedding configuration

    Returns:
        EmbeddingBackend instance
    """
    backend_map = {
        'sentence_transformers': SentenceTransformerBackend,
        'ollama': OllamaBackend,
        'medembed': MedEmbedBackend,
    }

    backend_class = backend_map.get(config.backend)
    if backend_class is None:
        raise ValueError(
            f"Unknown embedding backend: {config.backend}. "
            f"Choose from: {list(backend_map.keys())}"
        )

    return backend_class(config)


# =============================================================================
# BM25 Index
# =============================================================================

class BM25Index:
    """
    BM25 sparse index for retrieval.

    Uses rank_bm25.BM25Okapi for efficient scoring.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None  # Will hold BM25Okapi instance

    def index(self, documents: List[str]):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
        """
        import time
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 package required for efficient BM25. "
                "Install with: pip install rank_bm25"
            )

        logger.info(f"Tokenizing {len(documents)} documents for BM25...")
        start = time.time()

        self.corpus = documents

        # Tokenize corpus for BM25Okapi
        self.tokenized_corpus = [doc.lower().split() for doc in documents]

        # Build BM25Okapi index (this is fast and efficient)
        logger.info("  Building BM25Okapi index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b, epsilon=0.25)

        elapsed = time.time() - start
        logger.info(f"✓ BM25 index built in {elapsed:.1f}s with {len(self.bm25.idf)} unique terms")

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for a query.

        Args:
            query: Query string

        Returns:
            Scores array of shape (num_documents,)
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call index() first.")

        # Tokenize query
        tokens = query.lower().split()

        # Get scores using BM25Okapi (fast!)
        return self.bm25.get_scores(tokens)

    def save(self, path: str):
        """Save BM25 index to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'k1': self.k1,
                'b': self.b,
                'corpus': self.corpus,
                'tokenized_corpus': self.tokenized_corpus,
                # Save BM25Okapi instance directly
                'bm25': self.bm25,
            }, f)
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, path: str) -> 'BM25Index':
        """Load BM25 index from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls(k1=data['k1'], b=data['b'])
        index.corpus = data['corpus']
        index.tokenized_corpus = data.get('tokenized_corpus', [])
        index.bm25 = data.get('bm25')

        # If bm25 object not available (old format), rebuild it
        if index.bm25 is None and index.tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi
                logger.info("  Rebuilding BM25Okapi from tokenized corpus...")
                index.bm25 = BM25Okapi(index.tokenized_corpus, k1=index.k1, b=index.b)
                logger.info("  ✓ BM25Okapi rebuilt successfully")
            except ImportError:
                logger.warning("  rank_bm25 not available, BM25 scoring will be slow")

        return index


# =============================================================================
# FAISS Index
# =============================================================================

class FAISSIndex:
    """
    FAISS dense index for retrieval.
    """

    def __init__(self, dimension: int, metric: str = "cosine", index_type: str = "flat"):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            metric: Distance metric ("cosine", "l2", "inner_product")
            index_type: Index type ("flat", "ivf", "hnsw")
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu or faiss-gpu package required. "
                "Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.index = None
        self.embeddings = None

        self._build_index()

    def _build_index(self):
        """Build FAISS index based on configuration."""
        import faiss

        if self.index_type == "flat":
            if self.metric == "cosine":
                # For cosine similarity, use inner product on normalized vectors
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.metric == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.metric == "inner_product":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

        elif self.index_type == "ivf":
            # IVF index requires training
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

        elif self.index_type == "hnsw":
            # HNSW index
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            if self.metric == "l2":
                self.index.hnsw.metric_type = faiss.METRIC_L2
            else:
                self.index.hnsw.metric_type = faiss.METRIC_INNER_PRODUCT

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        logger.info(f"Built FAISS index: {self.index_type}, metric: {self.metric}, dim: {self.dimension}")

    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to index.

        Args:
            embeddings: Embeddings array of shape (n, dimension)
        """
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        # Train if using IVF
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.embeddings = embeddings

        logger.info(f"Added {embeddings.shape[0]} embeddings to FAISS index")

    def save(self, path: str):
        """Save FAISS index to file."""
        import faiss

        # Save index
        faiss.write_index(self.index, path)

        # Save embeddings separately
        emb_path = path.replace('.index', '_embeddings.npy')
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)

        logger.info(f"Saved FAISS index to {path}")

    @classmethod
    def load(cls, path: str, metric: str = "cosine") -> 'FAISSIndex':
        """Load FAISS index from file."""
        import faiss

        index = faiss.read_index(path)
        dimension = index.d

        faiss_index = cls(dimension=dimension, metric=metric)
        faiss_index.index = index

        # Try loading embeddings
        emb_path = path.replace('.index', '_embeddings.npy')
        if os.path.exists(emb_path):
            faiss_index.embeddings = np.load(emb_path)

        return faiss_index


# =============================================================================
# Index Builder
# =============================================================================

class IndexBuilder:
    """
    Build BM25 and FAISS indexes from chunks.
    """

    def __init__(self, index_config: IndexConfig, embed_config: EmbeddingConfig):
        """
        Initialize index builder.

        Args:
            index_config: Index configuration
            embed_config: Embedding configuration
        """
        self.index_config = index_config
        self.embed_config = embed_config
        self.embedding_backend = get_embedding_backend(embed_config)

        # Indexes
        self.bm25_index = None
        self.faiss_index = None

    def build_indexes(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """
        Build all configured indexes.

        Args:
            chunks: List of chunks to index

        Returns:
            Dictionary with build statistics
        """
        texts = [chunk.text for chunk in chunks]
        stats = {
            'num_chunks': len(chunks),
        }

        # Build BM25
        if self.index_config.bm25_enabled:
            logger.info("Building BM25 index...")
            self.bm25_index = BM25Index()
            self.bm25_index.index(texts)
            # Note: unique terms already logged by BM25Index.index()
            stats['bm25_built'] = True

        # Build FAISS
        if self.index_config.faiss_enabled:
            logger.info("Building FAISS index...")
            logger.info(f"Generating embeddings for {len(texts)} chunks (batch_size={self.embedding_backend.batch_size})...")
            import time
            start_embed = time.time()
            embeddings = self.embedding_backend.encode(texts)
            embed_time = time.time() - start_embed
            logger.info(f"✓ Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}d) in {embed_time:.1f}s ({embed_time/60:.1f} minutes)")

            logger.info(f"Building FAISS {self.index_config.faiss_index_type} index ({self.index_config.faiss_metric} metric)...")
            self.faiss_index = FAISSIndex(
                dimension=self.embedding_backend.get_dimension(),
                metric=self.index_config.faiss_metric,
                index_type=self.index_config.faiss_index_type
            )
            self.faiss_index.add(embeddings)
            logger.info(f"✓ FAISS index built with {embeddings.shape[0]} vectors")
            stats['faiss_built'] = True
            stats['embedding_dim'] = embeddings.shape[1]

        return stats

    def save_indexes(self, output_dir: str, corpus_name: str):
        """
        Save all indexes to disk.

        Args:
            output_dir: Output directory
            corpus_name: Name of the corpus
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.bm25_index:
            bm25_path = os.path.join(output_dir, f"{corpus_name}_bm25.pkl")
            self.bm25_index.save(bm25_path)

        if self.faiss_index:
            faiss_path = os.path.join(output_dir, f"{corpus_name}_faiss.index")
            self.faiss_index.save(faiss_path)

        logger.info(f"Saved indexes to {output_dir}")

    @classmethod
    def load_indexes(cls, output_dir: str, corpus_name: str) -> tuple:
        """
        Load indexes from disk.

        Args:
            output_dir: Output directory
            corpus_name: Name of the corpus

        Returns:
            Tuple of (bm25_index, faiss_index)
        """
        bm25_path = os.path.join(output_dir, f"{corpus_name}_bm25.pkl")
        faiss_path = os.path.join(output_dir, f"{corpus_name}_faiss.index")

        bm25_index = None
        faiss_index = None

        if os.path.exists(bm25_path):
            bm25_index = BM25Index.load(bm25_path)
            logger.info(f"Loaded BM25 index from {bm25_path}")

        if os.path.exists(faiss_path):
            faiss_index = FAISSIndex.load(faiss_path)
            logger.info(f"Loaded FAISS index from {faiss_path}")

        return bm25_index, faiss_index
