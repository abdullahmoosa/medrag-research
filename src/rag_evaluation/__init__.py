"""
Medical RAG Evaluation System

A modular, production-grade retrieval-augmented generation evaluation framework
for medical question answering.

Supports:
- No-RAG baseline evaluation
- Multi-stage retrieval (section-level → chunk-level)
- Pluggable query variants (base, reformulation, HyDE, option-aware)
- Multiple retrieval backends (dense, BM25, hybrid)
- Configurable fusion and reranking strategies
"""

__version__ = "1.0.0"

from .config import (
    EvaluationConfig,
    RetrievalConfig,
    QueryConfig,
    FusionConfig,
    RerankerConfig,
    Mode,
)
from .chunk_schema import Chunk, SectionMetadata, QueryVariant
from .evaluation import EvaluationOrchestrator
from .metrics import MetricsCalculator

__all__ = [
    "EvaluationConfig",
    "RetrievalConfig",
    "QueryConfig",
    "FusionConfig",
    "RerankerConfig",
    "Mode",
    "Chunk",
    "SectionMetadata",
    "QueryVariant",
    "EvaluationOrchestrator",
    "MetricsCalculator",
]
