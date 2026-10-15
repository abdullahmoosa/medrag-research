# Project Rules

## Web Search

- **Never use the built-in WebSearch tool.**
- All web searches are performed with the local script:
  ```bash
  python3 ddg_search.py "search query"
  ```
  (options: `-n N`, `--json`, `-t d|w|m|y`, `-r region`, `-b backend` — see `.claude/skills/ddg-search/SKILL.md`)

# Codebase

Medical RAG system for MedQA/MedMCQA-style questions: chunk medical textbooks, build indexes, retrieve evidence, and evaluate LLM answers.

## Layout

- `src/rag_evaluation/` — modular RAG evaluation framework (the main evaluation entry point)
- `src/rechunk_pipeline/` — semantic rechunking + index-building pipeline for textbooks
- `src/scripts/` — older/one-off scripts (corpus building, fine-tuning, reasoning generation, legacy evals)
- `src/app/` — demo app / RAG service
- `data/` — datasets (`data/medQA USMLE/` questions + textbooks)
- `processed_corpus/` — chunked corpus + FAISS/BM25 indexes (output of rechunk pipeline)
- `evaluation_results/` — eval outputs (`final_output/` holds the organized run tree)

## Running things

Use the `medrag` conda env — the system python3 has no numpy. All packages use relative imports, so run as modules from the repo root, not as scripts.

```bash
PY=/home/ser/mambaforge/envs/medrag/bin/python   # or: conda activate medrag

# RAG evaluation (main CLI)
$PY -m src.rag_evaluation.cli --mode rag --retrieval-mode hybrid --use-reranker

# Rechunking pipeline
$PY -m src.rechunk_pipeline.run_experiment --config src/rechunk_pipeline/config_match_previous.py
$PY -m src.rechunk_pipeline.test_pipeline   # smoke test with sample data

# Unit tests
$PY -m pytest src/rag_evaluation/tests.py -v
```

## rag_evaluation

Config-driven via dataclasses in `config.py` (`EvaluationConfig` → `retrieval`/`embedding`/`llm` sub-configs); `cli.py` maps flags onto them. Pipeline: `index_loader` → `retrieval` (coarse section-level → fine chunk-level, dense/BM25/hybrid) → `fusion` (RRF + lexical overlap) → optional `reranker` (cross-encoder, default `NeuML/biomedbert-base-reranker`) → evidence packing → Ollama LLM. Query variants are toggleable: base, LLM `query_reformulation`, `hyde`, option-aware, dataset expansions. `metrics.py` computes scores; `evaluation.py` orchestrates no_rag vs rag modes.

- Output dirs are auto-organized under `evaluation_results/final_output/` by index/retrieval mode/reranker/reformulation/model, with `v2`, `v3`… subdirs on re-runs.
- Embedding backend default `sentence_transformers` with `BAAI/bge-large-en-v1.5`; LLM default is Ollama (`--llm-base-url`, WSL host IP by default).

## rechunk_pipeline

Everything is driven by a `PipelineConfig` dataclass (`config.py`); experiments = edit a config file or use CLI overrides (`--input_dir`, `--embedding_model`, …). Stages, each its own module: `clean_text` → `split_structure` (chapter/section hierarchy) → `semantic_chunker` (token-targeted semantic chunks, optional overlap) → `metadata` (content type / medical system / keywords) → `index_builder` (BM25 + FAISS, chunk- and doc-level) written to `processed_corpus/`.

- `config_match_previous.py` reproduces the previous `processed_corpus` chunk settings (for comparable MedEmbed indexes).
