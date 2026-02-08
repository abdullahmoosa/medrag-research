"""
Configuration to match previous processed_corpus chunk settings.
Use this to build a comparable MedEmbed index.
"""
from src.rechunk_pipeline.config import PipelineConfig, EmbeddingConfig, ChunkingConfig, CleaningConfig, MetadataConfig, IndexConfig

# Use MedEmbed for medical domain
embedding_medembed = EmbeddingConfig(
    model="abhinand/MedEmbed-large-v0.1",
    backend="sentence_transformers",
    batch_size=32,
    device="cuda",
    normalize=True,
)

# Match previous chunking settings
chunking_match_previous = ChunkingConfig(
    min_tokens=150,      # Previous: 150 (was 120 in default)
    target_tokens=250,   # Previous: 250 (was 220 in default)
    max_tokens=400,      # Previous: 400 (was 350 in default)
    allow_cross_section=False,
    preserve_structure=True,
    enable_overlap=False,
)

CONFIG = PipelineConfig(
    input_dir="/home/ser/medrag/data/medQA USMLE/textbooks/en",
    output_dir="/home/ser/medrag/processed_corpus_medembed",
    corpus_name="medtextbooks_v1_medembed",
    embedding=embedding_medembed,
    chunking=chunking_match_previous,
    # Rest uses defaults from PipelineConfig
)
