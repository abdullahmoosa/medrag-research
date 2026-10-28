# Manuscript reproduction package

This directory supports the revised results for **A Systematic Study of Retrieval
Pipeline Design for Retrieval-Augmented Medical Question Answering**.

## What is included

- Current source in `src/rechunk_pipeline/` and `src/rag_evaluation/`.
- `data/correctness.csv`: 1,273 aligned question hashes and binary correctness
  for 73 selected evaluations (61 zero-shot and 12 CoT).
- `data/selected_runs.csv`: settings, counts, runtime, original paths and file
  hashes for those evaluations. Paths refer to excluded raw files and are
  provenance references, not files shipped in the source archive.
- `data/source_manifest.csv`: inventory of all 107 retained metrics files,
  including incomplete runs, superseded versions and copied aliases.
- Analysis scripts for Tables 2–5, Appendices A1–A2, Supplementary Table S1,
  and the data used by Figures 4–5.
- Portable dependency specifications, a preprocessing example, and example
  generation commands with the author-confirmed reranker set explicitly.

`src/scripts/` contains legacy experiments and is excluded from the source
archive. The chat application in `src/app/` has a separate setup guide and is
not needed for manuscript evaluation. Older `paper/repro/` scripts generated
historical manuscript artifacts; use this directory for the revised analysis.

## 1. Install the analysis environment

Run commands from the repository or extracted source-archive root. Python 3.10
is the reference setup used in the retained environment files.

```bash
python3.10 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproducibility/requirements-analysis.txt
python reproducibility/scripts/analyze_results.py
```

This analysis needs neither a GPU nor Ollama. It writes CSV tables and figure
data into `reproducibility/generated/`. Outputs are generated locally and are
not prerequisites for running the analysis. The bundled analysis inputs have
checksums in `data/SHA256SUMS`.

`reference_results/` contains the release-preparation outputs for comparison,
with its own checksum list. These were generated with the bundled Python
environment (NumPy 2.3.5); the analysis requirement pins NumPy 1.26.4 from the
retained research environment. The portable runtime installation and fresh
model generation were not repeated during package preparation.

| Output | Contents |
|---|---|
| `table2_baselines.csv` | All retained no-RAG baselines |
| `table3_selected_rag.csv` | Six explicitly selected RAG rows |
| `table4_rag_vs_no_rag.csv` | Matched zero-shot and CoT Med42 comparisons |
| `table5_component_summary.csv` | Matched zero-shot contrasts |
| `table5_matched_pairs.csv` | Exact pair membership and weighting |
| `appendix_A1_zero_shot.csv` | 61 selected zero-shot evaluations |
| `appendix_A2_cot.csv` | 12 selected CoT evaluations; no Gemma CoT |
| `supplementary_S1_model_comparisons.csv` | 28 matched Llama comparisons |
| `figure4_data.csv`, `figure5_data.csv` | Embedding comparison and Med42 runtime tradeoffs |

Accuracy intervals in the generated run summaries use Wilson's method.
Paired-difference intervals use 100,000 question-level bootstrap replicates
and seed 42. Exact two-sided McNemar tests use discordant correctness counts.
Component intervals resample question rows while keeping the matched
configuration pairs fixed. Runtime differences are descriptive.

All screening and reporting used the same 1,273-question MedQA-USMLE test
split. These are exploratory comparisons with nominal p-values; the selected
maximum and tests do not correct for configuration selection or multiplicity.

## 2. Source provenance and historical limits

Read [PROVENANCE.md](PROVENANCE.md) and `source_provenance.json` before claiming
an exact historical execution snapshot. The source and saved outputs are
available, but the metrics did not store the execution commit, immutable model
digests, Ollama runtime version, or complete precision settings. The source
record identifies what can be verified and leaves missing metadata explicit.

Selection takes the highest numeric **complete** version for each recorded
configuration; an unversioned directory is v1. Completeness requires 1,273
unique aligned questions and agreement between prediction correctness and
metrics. `bge_large_en_v1.5` is a copied alias of `index_1`; equal-version
copies are checked and the original `index_1` path is preferred. Other repeated
observations remain separate, including the original coarse directory strata.
They retain their weight in component summaries. Run IDs map back to all
original paths in the manifests.

The CLI evaluator calls the batch retrieval implementation. In that path:

- Coarse section filtering is inactive; `coarse_off`, `coarse_k20`, and
  `coarse_k30` are recorded directory strata, not active filtering conditions.
- Hybrid aggregation ranks deduplicated candidates by occurrence count and
  then maximum raw retrieval score. It does not use the scalar-path RRF or
  weighted-fusion helpers.
- The lexical gate receives no lexical-overlap field and does not filter the
  batch results. Configured RRF 60, weights 0.6/0.4 and thresholds 0.35/0.01
  must not be interpreted as active batch fusion settings.
- Retrieval requests up to 80 dense and 400 BM25 candidates per query,
  consolidates at most 12, and optionally reranks those candidates. The configured
  reranker ceiling of 150 does not increase that input pool.
- Final context uses at most six passages and a nominal budget of 1,200 stored
  chunk tokens, not a strict generation-model-tokenizer limit. No controlled
  passage-count sensitivity experiment is included.

The last row in `table5_component_summary.csv` compares recorded strata for
historical traceability; it is **not** an estimated effect of active coarse
filtering. See pair membership rather than treating repeated strata as
independent systems.

## 3. Install the generation environment

For a new evaluation, use a separate Python 3.10 environment, preferably Linux
x86-64:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, from the repository root:

```bash
conda env create -f environment.yml
conda activate medrag
```

`requirements.txt` selects the dependencies needed by the active pipelines.
`reproducibility/environments/historical-*` preserves the original broad exports
for provenance. They contain host-local package paths and are not portable
installation files. The portable specification is derived from those exports,
not an exact recovered execution lockfile. The release preparation did not
install this runtime environment or repeat model generation.

The default portable specification uses CPU FAISS. Embedding inference can use
CUDA with an appropriate PyTorch installation; choose a PyTorch build matching
the host. A different FAISS/device setup cannot reproduce the historical RTX
3090 timing measurements directly. Do not install CPU and GPU FAISS together.

Install the Ollama server separately, start it, and obtain the model releases
under their respective terms. The Python `ollama` package is only the client.
The evaluator also loads the tokenizer named in `EvaluationConfig`
(`deepseek-ai/DeepSeek-R1` in this source), so initial execution can require
Hugging Face access even when a different LLM tag is selected.

## 4. Obtain inputs and build indexes

Obtain the MedQA-USMLE test split and the 18 textbook inputs through the
original providers under their applicable terms. They are not redistributed in
the source archive. Required evaluation JSONL fields are `question`, an
`options` mapping such as `{"A": "...", "B": "..."}`, and `answer_idx` (the
correct option letter). Optional fields can carry subject metadata.

Place textbook plain-text files in `data/textbooks/`, then run:

```bash
python -m src.rechunk_pipeline.run_experiment \
  --config reproducibility/configs/rechunk_example.py
```

This example builds BGE indexes in `processed_corpus/`. To build a separate
MedEmbed index:

```bash
MEDRAG_EMBEDDING=abhinand/MedEmbed-large-v0.1 \
MEDRAG_INDEX_DIR=processed_corpus_medembed \
python -m src.rechunk_pipeline.run_experiment \
  --config reproducibility/configs/rechunk_example.py
```

Set `MEDRAG_DEVICE=cuda` for embedding inference on a suitable GPU.
The example uses the 150/250/400 chunk settings recorded in
`config_match_previous.py`; it is not evidence that every historical index was
rebuilt with that exact source or input text. Keep the embedding model used to
build each index identical to the query embedding model used in evaluation.

## 5. Run an evaluation

Set local paths and the Ollama address:

```bash
export MEDRAG_TEST_FILE="$PWD/data/medqa_test.jsonl"
export MEDRAG_OLLAMA_URL="http://localhost:11434"
```

No-RAG Med42 zero-shot baseline:

```bash
python -m src.rag_evaluation.cli \
  --mode no_rag --eval-path "$MEDRAG_TEST_FILE" \
  --index-dir processed_corpus --embedding-model BAAI/bge-large-en-v1.5 \
  --embedding-device cpu \
  --output-dir evaluation_results \
  --llm-base-url "$MEDRAG_OLLAMA_URL" \
  --llm-model thewindmom/llama3-med42-8b:latest
```

This source initializes the index manager, embedding backend and tokenizer even
in no-RAG mode. An accessible processed index is therefore required for this
baseline command, although its passages are not used in the answer prompt.

Selected BGE dense, reformulation and reranking configuration:

```bash
python -m src.rag_evaluation.cli \
  --mode rag --eval-path "$MEDRAG_TEST_FILE" \
  --index-dir processed_corpus --index-name index_1 \
  --embedding-model BAAI/bge-large-en-v1.5 --embedding-device cpu \
  --retrieval-mode dense --dense-k 80 --bm25-k 400 --top-k 12 \
  --max-passages 6 --max-evidence-tokens 1200 \
  --use-reformulation --use-reranker \
  --reranker-model BAAI/bge-reranker-large --reranker-top-k 150 \
  --coarse-retrieval --coarse-top-k 20 \
  --batch-size 32 --retrieval-batch-size 256 --llm-num-workers 8 \
  --llm-model thewindmom/llama3-med42-8b:latest \
  --llm-base-url "$MEDRAG_OLLAMA_URL" --output-dir evaluation_results
```

The coarse flag above reproduces the selected directory label; it has no
filtering effect in this batch implementation. Use `--embedding-device cuda`
on a suitable host. Changing hardware, concurrency, models, or indexes can
change results and runtime.

| Variant | Change to the example command |
|---|---|
| CoT | Add `--use-cot` to either baseline or RAG |
| Base Llama | Use `--llm-model llama3:8b` |
| Gemma zero-shot | Use `--llm-model gemma3:12b-it-qat` |
| MedEmbed | Use `--index-dir processed_corpus_medembed --index-name medembed --embedding-model abhinand/MedEmbed-large-v0.1` |
| Hybrid | Use `--retrieval-mode hybrid` |
| No reranker | Omit `--use-reranker` |
| No reformulation | Omit `--use-reformulation` |

The CoT experiments are a selected subset, not a full factorial grid. Historical
scripts `run_*.sh` and `scripts_reviewer/generate_*.py` retain original host paths
and scheduling logic; do not execute them unchanged on another host. They do
not override the current CLI reranker default. Use the explicit commands above.

Direct answers use temperature 0, top-p 1 and at most two generated tokens;
CoT uses at most 512. Query reformulation uses the Med42 tag, temperature 0,
and a 120-token maximum. No generation seed was recorded. Prompt construction
and answer extraction are in `src/rag_evaluation/evaluation.py`, and the rewrite
template is in `query_reformulation.py`. Unparseable answers, including saved
null correctness values, count as incorrect.

Recorded elapsed time begins inside `evaluate()` and includes data loading,
retrieval, generation, prediction writing, and metric calculation. Earlier
model and index initialization is excluded.

For new runs, save the source commit, `python -m pip freeze`, `ollama --version`,
model digests and `/api/show` metadata before generation. Current mutable tags
cannot recover historical model builds retrospectively.

## 6. Rebuild analysis inputs from retained raw outputs

This step is optional and needs the original output tree, which is excluded
from the public source archive because it contains question and textbook text.

```bash
python reproducibility/scripts/export_analysis_inputs.py \
  --results-dir evaluation_results/final_output
python reproducibility/scripts/analyze_results.py
```

The export checks full question alignment, duplicate questions, correctness
counts and equal-version alias copies. It retains raw file SHA-256 hashes and
replaces question text with hashes. The matrix keeps the original question-text
sort order so the bootstrap sampling sequence remains reproducible.

For a direct comparison using raw outputs instead of the bundle:

```bash
python scripts_reviewer/compare_models.py \
  --results-dir evaluation_results/final_output \
  --csv reproducibility/generated/direct_base_vs_med42.csv
```

## 7. Prepare the release archive

After committing the package, create the source archive from the intended
release commit:

```bash
mkdir -p dist
git archive --format=tar.gz --prefix=medrag-research/ \
  --output=dist/medrag-research-source.tar.gz HEAD
```

The `.gitattributes` export rules exclude manuscript drafts, reviewer material,
raw predictions, source datasets and textbook text, local conversations, and
legacy experimental code. These rules affect source archives; **they do not
remove anything already tracked from Git history or from a Git clone**. Historical
public files require a separate cleanup decision if their removal is wanted.

The archive includes original MIT-licensed software and text-free derived
analysis inputs. Third-party code, data, textbooks and model weights retain
their respective terms. No Zenodo DOI exists in this package until a release
has actually been archived successfully. Cite that specific archived release.
