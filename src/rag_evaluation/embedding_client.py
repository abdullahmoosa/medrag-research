"""
Embedding client with pluggable backends.

Supports SentenceTransformers, Ollama, and MedEmbed backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

import numpy as np

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings
            is_query: Whether these are queries (affects instruction prepending)

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
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )

        # Determine device
        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device

        self.model = SentenceTransformer(config.model_name, device=device)
        self.normalize = config.normalize
        self.batch_size = config.batch_size
        self.use_instruction = config.use_instruction
        self.query_instruction = config.query_instruction

        # Detect if this is a BGE model
        self.is_bge = "bge" in config.model_name.lower()

        logger.info(f"Loaded SentenceTransformer model: {config.model_name} on {device}")

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts using SentenceTransformers."""
        import time

        logger.info(f"  Encoding {len(texts)} texts with batch_size={self.batch_size}...")
        start = time.time()

        # Add instruction for BGE queries
        if self.is_bge and is_query and self.use_instruction:
            texts = [self.query_instruction + " " + (t or "") for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 1000,
        )

        elapsed = time.time() - start
        throughput = len(texts) / elapsed
        logger.info(
            f"  ✓ Encoding completed: {len(texts)} texts in {elapsed:.1f}s ({throughput:.0f} texts/sec)"
        )

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
            raise ImportError("ollama package required. Install with: pip install ollama")

        base_url = config.ollama_base_url or "http://localhost:11434"
        self.client = ollama.Client(host=base_url)
        self.model_name = config.model_name
        self.normalize = config.normalize
        logger.info(f"Using Ollama model: {config.model_name} at {base_url}")

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts using Ollama."""
        import time

        logger.info(f"  Encoding {len(texts)} texts with Ollama...")
        start = time.time()

        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])

        arr = np.array(embeddings, dtype=np.float32)

        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / (norms + 1e-8)

        elapsed = time.time() - start
        logger.info(f"  ✓ Encoding completed: {len(texts)} texts in {elapsed:.1f}s")

        return arr

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        # Encode a dummy text to get dimension
        emb = self.client.embeddings(model=self.model_name, prompt="test")
        return len(emb["embedding"])


class MedEmbedBackend(EmbeddingBackend):
    """
    MedEmbed backend for medical-domain embeddings.

    Placeholder implementation - should be customized based on your MedEmbed setup.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize MedEmbed backend.

        Args:
            config: Embedding configuration
        """
        self.model_name = config.model_name
        self.normalize = config.normalize
        self.batch_size = config.batch_size

        # TODO: Implement actual MedEmbed loading logic
        # This depends on how MedEmbed is served (Ollama, custom API, local model, etc.)
        logger.warning(
            "MedEmbed backend using placeholder implementation. "
            "Please update this class with your MedEmbed integration."
        )

        # For now, raise an error to force explicit implementation
        raise NotImplementedError(
            "MedEmbed backend requires specific implementation. "
            "Please update MedEmbedBackend class with your MedEmbed integration."
        )

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
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
        "sentence_transformers": SentenceTransformerBackend,
        "ollama": OllamaBackend,
        "medembed": MedEmbedBackend,
    }

    backend_class = backend_map.get(config.backend)
    if backend_class is None:
        raise ValueError(
            f"Unknown embedding backend: {config.backend}. " f"Choose from: {list(backend_map.keys())}"
        )

    return backend_class(config)
