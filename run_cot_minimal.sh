#!/usr/bin/env bash
# Minimal paired CoT+RAG completion set (supersedes the 36-run grid).
# 1. Wait for the in-flight base run (bge/dense/coarse_k20/reranker_off/no_reformulation/cot)
# 2. Run its med42 twin + the medembed coarse_k20/reranker_on/reformulation CoT pair.
set -u
PY=/home/ser/mambaforge/envs/medrag/bin/python
cd /home/ser/medrag

is_done() { $PY -c "import json,glob,sys; h=glob.glob(sys.argv[1]+'/**/metrics.json',recursive=True); sys.exit(0 if any(json.load(open(m)).get('total')==1273 for m in h) else 1)" "$1"; }

echo "=== waiting for in-flight base run to finish ==="
while pgrep -f "src.rag_evaluation.cli" > /dev/null; do sleep 30; done
echo "=== in-flight run finished ==="

BGE="--mode rag --index-dir /home/ser/medrag/processed_corpus --index-name index_1 --retrieval-mode dense --use-cot --coarse-retrieval --coarse-top-k 20"
MED="--mode rag --index-dir /home/ser/medrag/processed_corpus_medembed --index-name medembed --embedding-model abhinand/MedEmbed-large-v0.1 --retrieval-mode dense --use-cot --coarse-retrieval --coarse-top-k 20 --use-reranker --use-reformulation"

echo "=== [1/3] index_1/dense/coarse_k20/reranker_off/no_reformulation/cot [llama3_med42_8b] ==="
if is_done "evaluation_results/final_output/RAG/index_1/dense/coarse_k20/reranker_off/no_reformulation/llama3_med42_8b/cot"; then echo "SKIP (done)"
else $PY -m src.rag_evaluation.cli $BGE --llm-model thewindmom/llama3-med42-8b:latest || echo 'FAILED: bge k20 rr_off no_reform cot med42'; fi

echo "=== [2/3] medembed/dense/coarse_k20/reranker_on/reformulation/cot [llama3_8b] ==="
if is_done "evaluation_results/final_output/RAG/medembed/dense/coarse_k20/reranker_on/reformulation/llama3_8b/cot"; then echo "SKIP (done)"
else $PY -m src.rag_evaluation.cli $MED --llm-model llama3:8b || echo 'FAILED: medembed k20 rr_on reform cot base'; fi

echo "=== [3/3] medembed/dense/coarse_k20/reranker_on/reformulation/cot [llama3_med42_8b] ==="
if is_done "evaluation_results/final_output/RAG/medembed/dense/coarse_k20/reranker_on/reformulation/llama3_med42_8b/cot"; then echo "SKIP (done)"
else $PY -m src.rag_evaluation.cli $MED --llm-model thewindmom/llama3-med42-8b:latest || echo 'FAILED: medembed k20 rr_on reform cot med42'; fi

echo "=== MINIMAL COT SET COMPLETE ==="
