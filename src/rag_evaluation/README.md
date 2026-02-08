# Medical RAG Evaluation System

A production-grade, modular retrieval-augmented generation evaluation framework for medical question answering.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Orchestrator                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │   Evaluation Mode    │
           │  (Strategy Pattern)   │
           └───────────┬───────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
   ┌────▼────┐                   ┌────▼────┐
   │ No-RAG  │                   │   RAG   │
   │  Mode   │                   │  Mode   │
   └────┬────┘                   └────┬────┘
        │                             │
        │                     ┌───────┴────────┐
        │                     │  Context       │
        │                     │  Provider      │
        │                     └───────┬────────┘
        │                             │
        │                     ┌───────┴──────────────────┐
        │                     │                          │
        │              ┌──────▼─────┐          ┌────────▼─────┐
        │              │  Coarse    │          │    Fine      │
        │              │ Retrieval  │─────────▶│ Retrieval    │
        │              │ (Section)  │          │  (Chunk)     │
        │              └──────┬─────┘          └──────┬───────┐
        │                     │                      │
        │                     │               ┌──────┴───────┐
        │                     │               │              │
        │                     │         ┌─────▼────┐  ┌────▼────┐
        │                     │         │  Dense   │  │  BM25   │
        │                     │         └─────┬────┘  └────┬────┘
        │                     │               │             │
        │                     │               └──────┬──────┘
        │                     │                      │
        │                     │               ┌──────▼──────┐
        │                     │               │    Fusion   │
        │                     │               │    (RRF)    │
        │                     │               └──────┬──────┘
        │                     │                      │
        │                     │               ┌──────▼──────┐
        │                     │               │  Reranker   │
        │                     │               │  (Optional) │
        │                     │               └──────┬──────┘
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                      │
               ┌──────▼──────┐
               │   Metrics   │
               └─────────────┘
```

## Key Features

### 1. Dual Evaluation Modes

**No-RAG Mode (Baseline)**
- LLM-only evaluation without retrieval
- Establishes upper bound for LLM capability
- Measures RAG value-add

**RAG Mode**
- Full retrieval pipeline
- Multiple query variants
- Multi-stage retrieval
- Pluggable fusion and reranking

### 2. Multi-Stage Retrieval

**Stage 1: Coarse Retrieval (Optional)**
- Section-level retrieval
- Reduces search space for fine stage
- Configurable top-k sections

**Stage 2: Fine Retrieval**
- Chunk-level retrieval within sections
- Dense (FAISS), BM25, or Hybrid
- RRF fusion

**Stage 3: Reranking (Optional)**
- Cross-encoder reranker
- Top-K candidate reranking
- Configurable gating

### 3. Query Variants

Supports multiple query types:

- **Base**: Original question text
- **Reformulation**: LLM-rewritten query for medical retrieval
- **HyDE**: Hypothetical passage generation
- **Option-aware**: Separate queries per MCQ option
- **Expansions**: Dataset-provided query variations

Each query variant is tagged and contributes independently to retrieval results.

### 4. Metadata-Aware Filtering

- Filter by content_type (anatomy, physiology, pathology, etc.)
- Filter by medical_system (cardiovascular, nervous, etc.)
- Deduplication by document_id or section_id
- Prioritization of specific content types

## Module Structure

```
src/rag_evaluation/
├── __init__.py              # Package exports
├── chunk_schema.py          # Data models (Chunk, Query, etc.)
├── config.py                # Configuration dataclasses
├── index_loader.py          # Index loading (BM25, FAISS, chunks)
├── embedding_client.py      # Embedding backends
├── query_reformulation.py   # LLM query reformulation
├── hyde.py                  # HyDE generation
├── retrieval.py             # Core retrieval pipeline
├── fusion.py                # RRF and fusion strategies
├── evaluation.py            # Evaluation orchestrator
├── metrics.py               # Metrics calculation
└── cli.py                   # Command-line interface
```

## Installation

```bash
# Install dependencies
pip install sentence-transformers faiss-cpu transformers numpy tqdm

# For GPU support (optional)
pip install faiss-gpu
```

## Usage

### Basic No-RAG Baseline

```bash
python -m src.rag_evaluation.cli \
    --mode no_rag \
    --eval-path /path/to/test.jsonl \
    --output-dir ./results/no_rag_baseline
```

### Basic RAG Evaluation

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --index-dir /home/ser/medrag/processed_corpus \
    --eval-path /home/ser/medrag/data/medQA\ USMLE/questions/US/test.jsonl \
    --output-dir ./results/rag_hybrid \
    --retrieval-mode hybrid \
    --top-k 12
```

### With Query Reformulation

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --use-reformulation \
    --reformulation-temperature 0.0 \
    --reformulation-max-tokens 120
```

### With HyDE

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --use-hyde \
    --hyde-mode question \
    --hyde-temperature 0.2
```

### With Reranking

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --use-reranker \
    --reranker-model BAAI/bge-reranker-large \
    --reranker-top-k 150 \
    --reranker-fp16
```

### Full Configuration

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --index-dir /home/ser/medrag/processed_corpus \
    --eval-path /home/ser/medrag/data/medQA\ USMLE/questions/US/test.jsonl \
    --output-dir ./results/full_experiment \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --embedding-batch-size 128 \
    --retrieval-mode hybrid \
    --coarse-retrieval \
    --coarse-top-k 20 \
    --top-k 12 \
    --dense-k 80 \
    --bm25-k 400 \
    --max-passages 6 \
    --max-evidence-tokens 1200 \
    --use-reformulation \
    --use-hyde \
    --hyde-mode both \
    --use-option-aware \
    --rrf-k 60 \
    --alpha-rrf 0.6 \
    --beta-overlap 0.4 \
    --use-reranker \
    --reranker-model BAAI/bge-reranker-large \
    --llm-model llama3-med42-8b \
    --use-cot \
    --batch-size 32 \
    --log-level INFO
```

## Configuration Files

You can also use YAML/JSON config files instead of CLI flags:

```python
from src.rag_evaluation import EvaluationConfig

config = EvaluationConfig.from_dict({
    "index_dir": "/home/ser/medrag/processed_corpus",
    "eval_data_path": "/path/to/test.jsonl",
    "output_dir": "./results",
    "retrieval": {
        "mode": "rag",
        "coarse": {
            "enabled": True,
            "top_k_sections": 20,
        },
        "fine": {
            "top_k": 12,
            "mode": "hybrid",
        },
        "queries": {
            "use_reformulation": True,
            "use_hyde": True,
            "hyde_mode": "both",
        },
    },
    # ... etc
})
```

## Programmatic Usage

```python
import asyncio
from src.rag_evaluation import EvaluationConfig, EvaluationOrchestrator

# Build config
config = EvaluationConfig()
config.retrieval.mode = Mode.RAG
config.retrieval.fine.top_k = 12
config.retrieval.queries.use_reformulation = True

# Run evaluation
async def run():
    orchestrator = EvaluationOrchestrator(config)
    metrics = await orchestrator.evaluate()
    print(f"Accuracy: {metrics['accuracy']:.4f}")

asyncio.run(run())
```

## Experiment Examples

### Experiment 1: No-RAG vs RAG Comparison

```bash
# No-RAG baseline
python -m src.rag_evaluation.cli \
    --mode no_rag \
    --output-dir ./exp1/no_rag

# RAG with hybrid retrieval
python -m src.rag_evaluation.cli \
    --mode rag \
    --retrieval-mode hybrid \
    --output-dir ./exp1/rag_hybrid

# Compare results
```

### Experiment 2: Ablation Study

```bash
# Base only
python -m src.rag_evaluation.cli --mode rag --output-dir ./exp2/base

# Base + reformulation
python -m src.rag_evaluation.cli --mode rag --use-reformulation --output-dir ./exp2/reform

# Base + HyDE
python -m src.rag_evaluation.cli --mode rag --use-hyde --output-dir ./exp2/hyde

# Base + reformulation + HyDE
python -m src.rag_evaluation.cli --mode rag --use-reformulation --use-hyde --output-dir ./exp2/both
```

### Experiment 3: Retrieval Mode Comparison

```bash
# Dense only
python -m src.rag_evaluation.cli --mode rag --retrieval-mode dense --output-dir ./exp3/dense

# BM25 only
python -m src.rag_evaluation.cli --mode rag --retrieval-mode bm25 --output-dir ./exp3/bm25

# Hybrid
python -m src.rag_evaluation.cli --mode rag --retrieval-mode hybrid --output-dir ./exp3/hybrid
```

## Output Format

### predictions.jsonl

```json
{
  "example_id": "123",
  "question": "A 45-year-old male presents with...",
  "gold_answer": "A",
  "predicted_answer": "A",
  "is_correct": true,
  "context": {
    "passages": [
      {
        "chunk_id": "Harrison:20:3:42",
        "text": "Myocardial infarction occurs when...",
        "score": 0.89,
        "rank": 0
      }
    ],
    "total_tokens": 856,
    "metadata": {
      "num_queries": 3,
      "query_kinds": ["base", "reform", "hyde"]
    }
  },
  "raw_output": "A",
  "timestamp": "2026-01-26T15:30:00"
}
```

### metrics.json

```json
{
  "total": 1000,
  "correct": 650,
  "accuracy": 0.65,
  "with_context_count": 850,
  "with_context_accuracy": 0.68,
  "without_context_count": 150,
  "without_context_accuracy": 0.55,
  "by_subject": {
    "Cardiology": {"count": 100, "accuracy": 0.72},
    "Neurology": {"count": 80, "accuracy": 0.65}
  },
  "confusion_matrix": {
    "labels": ["A", "B", "C", "D"],
    "matrix": [[...]]
  },
  "coverage": {
    "mean": 5.2,
    "min": 0,
    "max": 6,
    "median": 6
  }
}
```

## Design Principles

1. **Modularity**: Each component is independent and testable
2. **Type Safety**: Strong typing throughout with dataclasses and enums
3. **Configuration-Driven**: No hardcoded values
4. **Extensibility**: Easy to add new retrievers, rerankers, query generators
5. **Performance**: Batched processing, async I/O, GPU support
6. **Research-Friendly**: Built for serious experimentation, not demos

## Extension Points

### Adding a New Retriever

```python
class CustomRetriever(Retriever):
    def retrieve(self, queries, top_k, filter_ids=None):
        # Your retrieval logic
        return results
```

### Adding a New Query Variant

```python
class CustomQueryGenerator:
    def generate_queries(self, example):
        # Your query generation logic
        return query_variants
```

### Adding a New Fusion Strategy

```python
class CustomFusion:
    def fuse(self, rank_lists, config):
        # Your fusion logic
        return fused_results
```

## Performance Tips

1. **Use GPU for embeddings**: Set `--embedding-device cuda`
2. **Increase batch sizes**: `--embedding-batch-size 128`, `--batch-size 32`
3. **Use FP16 reranker**: `--reranker-fp16` (faster, less VRAM)
4. **Enable coarse retrieval**: Reduces fine-stage search space
5. **Use retrieval batching**: `--retrieval-batch-size 256`

## Troubleshooting

### CUDA Out of Memory
- Reduce `--embedding-batch-size`
- Reduce `--batch-size`
- Use CPU for embeddings: `--embedding-device cpu`

### Slow Retrieval
- Enable coarse retrieval: `--coarse-retrieval`
- Reduce BM25 k: `--bm25-k 200`
- Reduce dense k: `--dense-k 50`

### Low Accuracy
- Check query reformulation is producing valid queries
- Adjust fusion weights: `--alpha-rrf`, `--beta-overlap`
- Enable reranking: `--use-reranker`
- Try different retrieval modes

## License

MIT License - See LICENSE file for details.
