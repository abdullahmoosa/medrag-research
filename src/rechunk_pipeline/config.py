"""
Configuration for Medical Textbook Rechunking Pipeline.

All experiments should be driven by modifying values in this file.
No hardcoded values should exist in other modules.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "BAAI/bge-m3"
    backend: str = "sentence_transformers"  # Options: "sentence_transformers", "ollama", "medembed"
    batch_size: int = 32
    device: str = "cuda"  # Options: "cuda", "cpu", "mps"
    normalize: bool = True  # Normalize embeddings for cosine similarity


@dataclass
class ChunkingConfig:
    """Semantic chunking configuration."""

    # Token limits (using approximate tokenization)
    min_tokens: int = 120
    target_tokens: int = 220
    max_tokens: int = 350

    # Structural constraints
    allow_cross_section: bool = False  # Whether chunks can cross section boundaries
    preserve_structure: bool = True  # Maintain chapter/section hierarchy

    # Overlap between chunks (optional, for better retrieval at boundaries)
    enable_overlap: bool = False  # Enable overlap between consecutive chunks
    overlap_ratio: float = 0.10  # Overlap 10% of content (0.0 to 0.5)
    overlap_strategy: str = "tokens"  # Options: "tokens", "sentences"

    # Semantic grouping
    semantic_similarity_threshold: float = 0.75  # For merging related chunks
    max_sentences_per_chunk: int = 15  # Safety limit

    # Filtering (optional)
    filter_min_tokens: int = 0  # Remove chunks below this size (0 = disabled)


@dataclass
class CleaningConfig:
    """Text cleaning configuration."""

    remove_figures: bool = True
    remove_references: bool = True
    remove_page_numbers: bool = True
    remove_table_continuations: bool = True

    # Custom patterns
    figure_patterns: List[str] = field(default_factory=lambda: [
        r"Fig\.\s+\d+\.\d+",
        r"Figure\s+\d+",
        r"Fig\.\s+\d+[A-Za-z]?",
    ])

    page_patterns: List[str] = field(default_factory=lambda: [
        r"^\s*\d+\s*$",  # Standalone numbers
        r"Page\s+\d+",
        r"^\d+\s*\|\s*.*",  # Number | content format
    ])

    reference_patterns: List[str] = field(default_factory=lambda: [
        r"^References?$",
        r"^Bibliography$",
        r"^Further Reading$",
        r"^Suggested Reading$",
    ])


@dataclass
class MetadataConfig:
    """Metadata extraction configuration."""

    detect_content_type: bool = True  # Classify as anatomy/pathology/physiology
    detect_medical_system: bool = True  # Detect cardiovascular, nervous, etc.
    extract_keywords: bool = True  # Extract medical keywords
    max_keywords: int = 10


@dataclass
class IndexConfig:
    """Index building configuration."""

    bm25_enabled: bool = True
    faiss_enabled: bool = True

    # FAISS settings
    faiss_metric: str = "cosine"  # Options: "cosine", "l2", "inner_product"
    faiss_index_type: str = "flat"  # Options: "flat", "ivf", "hnsw"

    # Document-level indexing
    build_doc_index: bool = True  # Index at document level
    build_chunk_index: bool = True  # Index at chunk level

    # Index storage
    save_index: bool = True
    index_precision: str = "float32"  # Options: "float32", "float16", "uint8"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Input/Output paths
    input_dir: str = "/home/ser/medrag/data/medQA USMLE/textbooks/en"
    output_dir: str = "/home/ser/medrag/processed_corpus"
    corpus_name: str = "medtextbooks_rechunked"

    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    # Processing options
    num_workers: int = 4
    log_level: str = "INFO"
    validate_chunks: bool = True
    save_intermediate: bool = True

    # Output formats
    output_formats: List[str] = field(default_factory=lambda: ["jsonl"])

    def get_output_path(self, filename: str) -> str:
        """Get full path for output file."""
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, filename)

    def get_index_path(self, index_name: str) -> str:
        """Get full path for index file."""
        index_dir = os.path.join(self.output_dir, "indexes")
        os.makedirs(index_dir, exist_ok=True)
        return os.path.join(index_dir, index_name)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()


def get_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from file or return default.

    Args:
        config_path: Optional path to config Python file

    Returns:
        PipelineConfig instance
    """
    if config_path is None:
        return DEFAULT_CONFIG

    # Import config from specified file
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.CONFIG if hasattr(config_module, 'CONFIG') else config_module.DEFAULT_CONFIG
