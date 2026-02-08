# Quick Start Guide

Get started with the Medical RAG Evaluation System in 5 minutes.

## Prerequisites

1. **Processed corpus** already built in `/home/ser/medrag/processed_corpus/`
2. **Evaluation data** (MedQA test set)
3. **Ollama** running with medical LLM models
4. **Python 3.8+** with required packages

## Installation

```bash
# Navigate to project
cd /home/ser/medrag

# Install dependencies
pip install sentence-transformers faiss-cpu transformers numpy tqdm ollama

# Optional: GPU support
pip install faiss-gpu
```

## Verify Setup

```bash
# Check processed corpus exists
ls /home/ser/medrag/processed_corpus/

# Should see:
# medtextbooks_v1.jsonl
# medtextbooks_v1_bm25.pkl
# medtextbooks_v1_faiss.index
# medtextbooks_v1_faiss_embeddings.npy
# medtextbooks_v1_meta.json
# medtextbooks_v1_stats.json

# Check Ollama is running
ollama list

# Should see your models (e.g., llama3-med42-8b)
```

## Run Your First Evaluation

### 1. No-RAG Baseline (2 minutes)

Establish the LLM-only performance:

```bash
python -m src.rag_evaluation.cli \
    --mode no_rag \
    --eval-path "/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl" \
    --output-dir ./quickstart/no_rag_baseline \
    --limit 50 \
    --llm-model llama3-med42-8b
```

Expected output:
```
======================================================================
EVALUATION COMPLETE
======================================================================
Total examples: 50
Accuracy: 0.4200 (21/50)
With context: 0.0000 (n=0)
Without context: 0.4200 (n=50)
```

### 2. Basic RAG (5 minutes)

Now add retrieval:

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --index-dir /home/ser/medrag/processed_corpus \
    --eval-path "/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl" \
    --output-dir ./quickstart/basic_rag \
    --limit 50 \
    --retrieval-mode hybrid \
    --top-k 12 \
    --llm-model llama3-med42-8b \
    --embedding-model BAAI/bge-large-en-v1.5
```

Expected output:
```
======================================================================
EVALUATION COMPLETE
======================================================================
Total examples: 50
Accuracy: 0.5800 (29/50)
With context: 0.5800 (n=48)
Without context: 0.0000 (n=2)
```

### 3. RAG + Query Reformulation (10 minutes)

Add LLM-based query reformulation:

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --index-dir /home/ser/medrag/processed_corpus \
    --eval-path "/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl" \
    --output-dir ./quickstart/rag_reform \
    --limit 50 \
    --retrieval-mode hybrid \
    --use-reformulation \
    --top-k 12 \
    --llm-model llama3-med42-8b
```

### 4. Full Pipeline (15 minutes)

All features enabled:

```bash
python -m src.rag_evaluation.cli \
    --mode rag \
    --index-dir /home/ser/medrag/processed_corpus \
    --eval-path "/home/ser/medrag/data/medQA USMLE/questions/US/test.jsonl" \
    --output-dir ./quickstart/full \
    --limit 50 \
    --retrieval-mode hybrid \
    --use-reformulation \
    --use-hyde \
    --hyde-mode question \
    --use-option-aware \
    --top-k 12 \
    --max-passages 6 \
    --rrf-k 60 \
    --alpha-rrf 0.6 \
    --beta-overlap 0.4 \
    --llm-model llama3-med42-8b
```

## Compare Results

```bash
# Check accuracy across runs
echo "No-RAG: $(cat quickstart/no_rag_baseline/metrics.json | jq -r '.accuracy')"
echo "Basic RAG: $(cat quickstart/basic_rag/metrics.json | jq -r '.accuracy')"
echo "RAG + Reform: $(cat quickstart/rag_reform/metrics.json | jq -r '.accuracy')"
echo "Full Pipeline: $(cat quickstart/full/metrics.json | jq -r '.accuracy')"
```

Expected improvement:
- No-RAG: ~40-45%
- Basic RAG: ~55-60%
- With reformulation: ~60-65%
- Full pipeline: ~65-70%

## Troubleshooting

### "Index not found" error

```bash
# Verify index directory
ls -la /home/ser/medrag/processed_corpus/

# Rebuild if needed using rechunking pipeline
python run_production.py
```

### "CUDA out of memory" error

```bash
# Reduce batch sizes
--embedding-batch-size 64 \
--batch-size 16

# Or use CPU for embeddings
--embedding-device cpu
```

### "Ollama connection refused" error

```bash
# Check Ollama is running
ollama list

# Start Ollama service
ollama serve

# Or specify different URL
--llm-base-url http://localhost:11434
```

### Low retrieval quality

```bash
# Try different retrieval modes
--retrieval-mode dense  # Semantic only
--retrieval-mode bm25   # Keyword only
--retrieval-mode hybrid # Combined (recommended)

# Adjust fusion weights
--alpha-rrf 0.7 \
--beta-overlap 0.3

# Enable reranking
--use-reranker \
--reranker-model BAAI/bge-reranker-large
```

## Next Steps

1. **Run full evaluation** (remove `--limit 50`)
2. **Experiment with configurations** using different flags
3. **Analyze results** in `metrics.json` and `predictions.jsonl`
4. **Compare to baselines** in your research
5. **Extend the system** with custom retrievers or query generators

See `README.md` for full documentation and advanced usage.
