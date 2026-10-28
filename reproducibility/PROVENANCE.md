# Source and experimental provenance

Prepared 28 October 2026. This describes the evidence retained in the repository;
it does not retrospectively assign missing execution metadata.

## Source version

The source reference commit before package preparation is
`2e4a707` (the full identifier and per-file SHA-256 values are in
`source_provenance.json`). All 27 Python files under the two active pipeline
packages were compared with initial source commit `f0855e4`. Twenty-six are
byte-identical. The only changed file is `src/rag_evaluation/cli.py`:

1. Commit `cc7a7f4` changed the CLI reranker default from
   `BAAI/bge-reranker-large` to `NeuML/biomedbert-base-reranker`.
2. The later base-model evaluation change preserves meaningful model tags in
   output directory names, such as `llama3:8b` becoming `llama3_8b`.

The retrieval, evaluation, fusion, configuration and rechunking implementation
files are unchanged from the initial tracked source. The retained generation
scripts call `python -m src.rag_evaluation.cli`. The saved metrics do not record
an execution commit or a source snapshot. Consequently, this package provides
the available source with file hashes and a traceable history; an exact source
identity for every historical run cannot be certified from the metrics alone.

Package preparation changes the installation specifications, analysis scripts,
documentation and source-archive rules. It does not change pipeline behavior.

## Models and metadata

| Item | Retained evidence |
|---|---|
| Llama base tag | `llama3:8b`, present in retained run scripts |
| Med42 tag | `thewindmom/llama3-med42-8b:latest`, present in retained run scripts |
| Gemma tag | `gemma3:12b-it-qat`, confirmed by the author |
| Experimental reranker | `BAAI/bge-reranker-large`, confirmed by the author |
| Embeddings | BGE `BAAI/bge-large-en-v1.5`; MedEmbed `abhinand/MedEmbed-large-v0.1` |
| Immutable model digests | Not retained in the available metrics |
| Historical Ollama server version | Not retained in the available metrics |
| Complete precision/quantization metadata | Not retained; tag text is not a recovered model-build manifest |
| Generation seed | Not recorded |

The current CLI default does not establish the reranker used historically.
The reproduction example explicitly overrides it with the author-confirmed
model. Retained historical command files that omit the override are provided
as historical records, not as ready-to-run reproduction wrappers.

## Retained results

The source inventory has 107 metric files, 104 reporting 1,273 questions.
After validating prediction counts and selecting versions, the analysis bundle
contains 73 entries: 61 zero-shot and 12 selected CoT. Copied BGE directory aliases
are collapsed; original coarse-directory strata and other repeated observations
remain separate. Their names describe provenance, not active coarse filtering.

The highest numeric complete version is selected for each recorded configuration.
Unversioned paths count as v1. Equal-version BGE alias copies are checked for
identical correctness, then `index_1` is preferred. Correctness is aligned by
question text because the saved example IDs are not unique. The exported matrix
contains SHA-256 question identifiers and correctness only. Null correctness
values are counted as incorrect, matching metric calculation.

Every included row must have 1,273 unique questions and a correctness total
matching its metrics. Raw metrics and prediction file hashes are in the
manifests. Context hashes permit checking whole-run context identity without
distributing retrieved passages; differing hashes do not quantify which
passages differ or establish equivalent evidence.

## Effective retrieval and interpretation

The CLI calls `evaluate()`, then `evaluate_batch()`, then `retrieve_batch()`.
That path does not execute the scalar coarse-section stage, scalar RRF or
weighted-fusion helper. Hybrid aggregation uses candidate occurrence counts
and maximum raw scores. The lexical gate lacks the required overlap field
for these batch results. Configured but inactive parameters are distinguished
from executed operations in the reproduction guide.

The recorded coarse-stratum contrast is retained in the analysis for traceability,
but it cannot measure the causal effect of coarse filtering. Repeated strata
also contribute repeated observations to the other matched component summaries.

All reported screening and evaluation used the same MedQA-USMLE test split;
there was no separate preliminary or validation dataset. Comparisons are
exploratory. No independent benchmark, passage-count sensitivity experiment,
retrieval relevance assessment or evidence-faithfulness measurement is created
by this package preparation.

## Archive boundary

The release source archive excludes the entire `paper/` tree, raw prediction
outputs containing third-party text, legacy experiments, local conversations
and internal session notes. The `.gitattributes` rules apply to source archives.
They do not delete existing files or erase repository history. A Git clone can
still contain material already tracked in the repository.

The text-free bundle supports recomputing accuracy and paired statistics without
redistributing question or textbook content. It does not support re-evaluating
retrieval relevance, generated reasoning or context quality without obtaining
the original inputs and permitted access to raw outputs.
