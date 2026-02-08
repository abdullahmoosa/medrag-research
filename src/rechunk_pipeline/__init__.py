"""
Medical Textbook Rechunking Pipeline

A modular, configurable pipeline for processing medical textbooks
into high-quality semantic chunks for RAG systems.

Example usage:
    from src.rechunk_pipeline import RechunkingPipeline, PipelineConfig

    config = PipelineConfig()
    pipeline = RechunkingPipeline(config)
    stats = pipeline.run()
"""

from .config import PipelineConfig, EmbeddingConfig, ChunkingConfig, CleaningConfig, MetadataConfig, IndexConfig, get_config
from .clean_text import TextCleaner
from .split_structure import StructureSplitter, TextSection
from .semantic_chunker import SemanticChunker, SemanticChunk, TokenCounter
from .metadata import MetadataExtractor, create_chunk_metadata, validate_metadata
from .index_builder import IndexBuilder, get_embedding_backend
from .run_experiment import RechunkingPipeline

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "CleaningConfig",
    "MetadataConfig",
    "IndexConfig",
    "get_config",
    "TextCleaner",
    "StructureSplitter",
    "TextSection",
    "SemanticChunker",
    "SemanticChunk",
    "TokenCounter",
    "MetadataExtractor",
    "create_chunk_metadata",
    "validate_metadata",
    "IndexBuilder",
    "get_embedding_backend",
    "RechunkingPipeline",
]
