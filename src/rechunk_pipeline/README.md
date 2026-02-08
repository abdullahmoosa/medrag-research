# Medical Textbook Rechunking Pipeline

A modular, configurable Python pipeline for processing medical textbooks into high-quality semantic chunks for RAG (Retrieval-Augmented Generation) systems.

## Features

- **Semantic Chunking**: Creates mechanism-consistent chunks that respect section boundaries
- **Rich Metadata**: Automatically extracts content type (anatomy/pathology/physiology), medical systems, and keywords
- **Multiple Embedding Backends**: Support for SentenceTransformers, Ollama, and custom MedEmbed
- **Dual Indexing**: BM25 (sparse) and FAISS (dense) indexes
- **Configurable**: All experiments driven by a single configuration file
- **Modular Design**: Each component can be used independently or as part of the full pipeline

## Installation

```bash
# Core dependencies
pip install tiktoken

# For SentenceTransformers backend (recommended)
pip install sentence-transformers

# For FAISS indexing
pip install faiss-cpu  # or faiss-gpu

# For Ollama backend (optional)
pip install ollama
```

## Quick Start

### Option 1: Use the example config

```bash
cd rechunk_pipeline
python -m run_experiment --config example_config.py
```

### Option 2: Use as a library

```python
from src.rechunk_pipeline import RechunkingPipeline, PipelineConfig

# Create config
config = PipelineConfig(
    input_dir="/path/to/textbooks",
    output_dir="./processed_corpus",
    corpus_name="my_corpus",
)

# Run pipeline
pipeline = RechunkingPipeline(config)
stats = pipeline.run()

print(f"Created {stats['total_chunks']} chunks")
```

## Configuration

All experiments are driven by modifying `config.py` or creating a custom config file.

### Key Configuration Options

```python
from src.rechunk_pipeline.config import PipelineConfig, ChunkingConfig, EmbeddingConfig

config = PipelineConfig(
    # Paths
    input_dir="/path/to/textbooks",
    output_dir="./output",
    corpus_name="my_corpus",

    # Chunking
    chunking=ChunkingConfig(
        min_tokens=120,
        target_tokens=220,
        max_tokens=350,
        allow_cross_section=False,  # Respect section boundaries
    ),

    # Embeddings
    embedding=EmbeddingConfig(
        model="BAAI/bge-m3",
        backend="sentence_transformers",
        device="cuda",  # or "cpu"
    ),
)
```

## Architecture

```
rechunk_pipeline/
├── config.py              # Configuration classes
├── clean_text.py          # Text cleaning (remove figures, page numbers, etc.)
├── split_structure.py     # Split into chapters/sections
├── semantic_chunker.py    # Create semantic chunks
├── metadata.py            # Extract medical metadata
├── index_builder.py       # Build BM25 and FAISS indexes
├── run_experiment.py      # Main pipeline runner
├── example_config.py      # Example configuration
├── example_usage.py       # Usage examples
└── README.md
```

## Module Details

### 1. Text Cleaning (`clean_text.py`)

Removes:
- Figure references and captions
- Page numbers
- Reference sections
- URLs and email addresses
- Copyright notices

### 2. Structural Splitting (`split_structure.py`)

Splits textbooks into:
- Chapters (detected via headers like "Chapter 1", "1. Title", etc.)
- Sections (sub-headings within chapters)

### 3. Semantic Chunking (`semantic_chunker.py`)

Creates chunks with:
- Target size: 120-350 tokens (configurable)
- Never crosses section boundaries (unless allowed)
- Maintains semantic coherence
- Validates chunk sizes

### 4. Metadata Extraction (`metadata.py`)

Automatically detects:
- Content type: anatomy, physiology, pathology, pharmacology, clinical
- Medical system: cardiovascular, nervous, respiratory, etc.
- Keywords: Top medical terms from chunk

### 5. Index Building (`index_builder.py`)

Supports:
- **BM25**: Sparse retrieval index
- **FAISS**: Dense retrieval index with multiple metrics (cosine, L2, inner-product)
- **Multiple backends**: SentenceTransformers, Ollama, MedEmbed

## Output Format

The pipeline generates:

1. **Corpus JSONL** (`corpus_name.jsonl`):
   ```json
   {
     "chunk_id": "doc_Anatomy_Gray_000001",
     "doc_id": "doc_Anatomy_Gray_000001",
     "textbook": "Anatomy_Gray",
     "chapter": "The Skeletal System",
     "section": "Cartilage",
     "text": "...",
     "token_count": 220,
     "metadata": {
       "content_type": "anatomy",
       "system": "musculoskeletal",
       "keywords": ["Cartilage", "Bone", "Tissue", "Matrix"]
     }
   }
   ```

2. **Metadata** (`corpus_name_meta.json`):
   - Corpus statistics
   - Configuration used
   - Content type distribution
   - System distribution

3. **Statistics** (`corpus_name_stats.json`):
   - Processing time
   - Number of chunks
   - Average chunk size
   - Validation results

4. **Indexes**:
   - `corpus_name_bm25.pkl`: BM25 sparse index
   - `corpus_name_faiss.index`: FAISS dense index
   - `corpus_name_faiss_embeddings.npy`: Embeddings array

## Usage Examples

### Example 1: Process all textbooks

```python
from src.rechunk_pipeline import RechunkingPipeline, PipelineConfig

config = PipelineConfig()
pipeline = RechunkingPipeline(config)
stats = pipeline.run()
```

### Example 2: Custom chunk size

```python
from src.rechunk_pipeline.config import ChunkingConfig

config = PipelineConfig(
    chunking=ChunkingConfig(
        min_tokens=80,
        target_tokens=150,
        max_tokens=200,
    )
)
```

### Example 3: Different embedding model

```python
from src.rechunk_pipeline.config import EmbeddingConfig

config = PipelineConfig(
    embedding=EmbeddingConfig(
        model="BAAI/bge-large-en-v1.5",
        backend="sentence_transformers",
    )
)
```

### Example 4: Use individual components

```python
from src.rechunk_pipeline import TextCleaner, StructureSplitter, SemanticChunker
from src.rechunk_pipeline.config import CleaningConfig, ChunkingConfig

# Clean text
cleaner = TextCleaner(CleaningConfig())
cleaned = cleaner.clean_text(raw_text)

# Split structure
splitter = StructureSplitter()
sections = splitter.split_textbook(cleaned, "Anatomy_Gray")

# Create chunks
chunker = SemanticChunker(ChunkingConfig())
chunks = chunker.chunk_sections(sections, "doc_id")
```

### Example 5: Load existing indexes

```python
from src.rechunk_pipeline.index_builder import IndexBuilder

bm25_index, faiss_index = IndexBuilder.load_indexes(
    output_dir="/path/to/output",
    corpus_name="my_corpus"
)
```

## Running from Command Line

```bash
# Use default config
python -m src.rechunk_pipeline.run_experiment

# Use custom config
python -m src.rechunk_pipeline.run_experiment --config my_config.py

# Override settings
python -m src.rechunk_pipeline.run_experiment \
    --input_dir /path/to/textbooks \
    --output_dir ./output \
    --corpus_name my_corpus \
    --embedding_model BAAI/bge-large-en-v1.5
```

## Validation

The pipeline includes built-in validation:

- Checks chunk size limits
- Validates no section boundary violations
- Reports chunks outside target range
- Calculates metadata coverage statistics

## Performance Considerations

- **GPU**: Use `device="cuda"` for faster embedding generation
- **Batch size**: Increase `batch_size` for faster embedding (if memory allows)
- **Workers**: Set `num_workers` for parallel processing
- **Index type**: Use `faiss_index_type="ivf"` or `"hnsw"` for large corpora

## Integration with Existing RAG Pipeline

The output JSONL format integrates seamlessly with most RAG systems:

```python
import json

# Load chunks
chunks = []
with open('processed_corpus/medtextbooks_rechunked.jsonl', 'r') as f:
    for line in f:
        chunks.append(json.loads(line))

# Use in your RAG system
for chunk in chunks:
    print(f"{chunk['textbook']} - {chunk['section']}")
    print(chunk['text'])
    print()
```

## Troubleshooting

### Issue: "tiktoken not installed"
**Solution**: `pip install tiktoken`

### Issue: "CUDA out of memory"
**Solution**: Use `device="cpu"` or reduce `batch_size`

### Issue: "No sections found"
**Solution**: Check textbook formatting. The splitter expects clear section headers.

### Issue: "Chunks too small/large"
**Solution**: Adjust `min_tokens`, `target_tokens`, `max_tokens` in config

## License

This pipeline is part of the MedRAG project.

## Contributing

To extend the pipeline:

1. Add new embedding backend: Extend `EmbeddingBackend` in `index_builder.py`
2. Add new metadata type: Extend `MetadataExtractor` in `metadata.py`
3. Add new cleaning rules: Extend `TextCleaner` in `clean_text.py`

## Citation

If you use this pipeline in your research, please cite the MedRAG project.
