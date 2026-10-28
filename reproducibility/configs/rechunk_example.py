"""Portable example based on config_match_previous.py; not a recovered run manifest."""
import os

from src.rechunk_pipeline.config import PipelineConfig, EmbeddingConfig, ChunkingConfig

CONFIG = PipelineConfig(
    input_dir=os.environ.get("MEDRAG_TEXTBOOK_DIR", "data/textbooks"),
    output_dir=os.environ.get("MEDRAG_INDEX_DIR", "processed_corpus"),
    corpus_name="medtextbooks_v1",
    embedding=EmbeddingConfig(
        model=os.environ.get("MEDRAG_EMBEDDING", "BAAI/bge-large-en-v1.5"),
        backend="sentence_transformers",
        batch_size=32,
        device=os.environ.get("MEDRAG_DEVICE", "cpu"),
        normalize=True,
    ),
    chunking=ChunkingConfig(
        min_tokens=150, target_tokens=250, max_tokens=400,
        allow_cross_section=False, preserve_structure=True, enable_overlap=False,
    ),
)
