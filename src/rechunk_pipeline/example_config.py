"""
Example configuration for the rechunking pipeline.

Modify this file to create custom experiments.
"""

from src.rechunk_pipeline.config import PipelineConfig, EmbeddingConfig, ChunkingConfig, CleaningConfig, MetadataConfig, IndexConfig

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
# Choose embedding model and backend
# Options: "sentence_transformers", "ollama", "medembed"

# For sentence_transformers (recommended for most use cases):
# - "BAAI/bge-m3" - Multi-lingual, high quality
# - "BAAI/bge-large-en-v1.5" - English only, very high quality
# - "medicalai/ClinicalBERT" - Medical domain (if using sentence-transformers compatible wrapper)

embedding = EmbeddingConfig(
    model="BAAI/bge-m3",
    backend="sentence_transformers",
    batch_size=32,
    device="cuda",  # Use "cpu" if no GPU available
    normalize=True,
)

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

chunking = ChunkingConfig(
    min_tokens=120,
    target_tokens=220,
    max_tokens=350,
    allow_cross_section=False,  # Set to True to allow chunks across sections
    preserve_structure=True,
    semantic_similarity_threshold=0.75,
    max_sentences_per_chunk=15,

    # OPTIONAL: Overlap between chunks (disabled by default)
    # Enable to improve retrieval at chunk boundaries
    enable_overlap=False,  # Set to True to enable overlap
    overlap_ratio=0.10,  # 10% overlap between consecutive chunks
    overlap_strategy="tokens",  # Options: "tokens", "sentences"

    # OPTIONAL: Filter tiny chunks (disabled by default)
    # Remove chunks that are too small (likely fragments/headers)
    filter_min_tokens=0,  # Set to 100 to filter chunks < 100 tokens
)

# =============================================================================
# CLEANING CONFIGURATION
# =============================================================================

cleaning = CleaningConfig(
    remove_figures=True,
    remove_references=True,
    remove_page_numbers=True,
    remove_table_continuations=True,
)

# =============================================================================
# METADATA CONFIGURATION
# =============================================================================

metadata = MetadataConfig(
    detect_content_type=True,  # Classify as anatomy/pathology/physiology/etc.
    detect_medical_system=True,  # Detect cardiovascular, nervous, etc.
    extract_keywords=True,
    max_keywords=10,
)

# =============================================================================
# INDEX CONFIGURATION
# =============================================================================

index = IndexConfig(
    bm25_enabled=True,
    faiss_enabled=True,
    faiss_metric="cosine",  # Options: "cosine", "l2", "inner_product"
    faiss_index_type="flat",  # Options: "flat", "ivf", "hnsw"
    build_doc_index=True,
    build_chunk_index=True,
    save_index=True,
    index_precision="float32",
)

# =============================================================================
# MAIN PIPELINE CONFIGURATION
# =============================================================================

CONFIG = PipelineConfig(
    # Input/Output
    input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
    output_dir="/home/ser/medrag/processed_corpus",
    corpus_name="medtextbooks_rechunked",

    # Sub-configurations
    embedding=embedding,
    chunking=chunking,
    cleaning=cleaning,
    metadata=metadata,
    index=index,

    # Processing
    num_workers=4,
    log_level="INFO",
    validate_chunks=True,
    save_intermediate=True,
)

# =============================================================================
# OTHER EXAMPLE CONFIGURATIONS
# =============================================================================

# Example 1: Smaller chunks for dense retrieval
# CONFIG_SMALL_CHUNKS = PipelineConfig(
#     input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
#     output_dir="/home/ser/medrag/processed_corpus_small_chunks",
#     corpus_name="medtextbooks_small_chunks",
#     chunking=ChunkingConfig(min_tokens=80, target_tokens=150, max_tokens=200),
#     embedding=EmbeddingConfig(model="BAAI/bge-large-en-v1.5"),
#     cleaning=cleaning,
#     metadata=metadata,
#     index=index,
# )

# Example 2: Using Ollama for local embeddings
# CONFIG_OLLAMA = PipelineConfig(
#     input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
#     output_dir="/home/ser/medrag/processed_corpus_ollama",
#     corpus_name="medtextbooks_ollama",
#     embedding=EmbeddingConfig(
#         model="nomic-embed-text",  # Ollama model
#         backend="ollama",
#         device="cpu",
#     ),
#     chunking=chunking,
#     cleaning=cleaning,
#     metadata=metadata,
#     index=index,
# )
