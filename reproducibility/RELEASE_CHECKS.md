# Package preparation checks

Checked 28 October 2026 against the retained repository files.

- Inventoried 107 metrics files; 104 report 1,273 evaluated questions.
- Validated complete raw prediction files for unique questions, shared question
  sets and correctness totals, counting null answers as incorrect.
- Verified that all 21 equal-version BGE alias groups have identical metrics
  and prediction file hashes.
- Exported 73 selected entries: 61 zero-shot and 12 CoT, without raw question,
  answer, reasoning or textbook text.
- Regenerated all 28 matched model comparisons with paired bootstrap intervals
  (100,000 replicates, seed 42) and exact two-sided McNemar tests.
- Compared all 28 paired correctness and discordance counts with the existing
  supplementary comparison table: agreement.
- Compared direct raw-output analysis with the bundled correctness-matrix
  analysis: agreement to the direct CSV's rounding precision.
- Regenerated Table 4: +4.63 percentage points for zero-shot RAG versus no-RAG
  (95% CI +2.20 to +7.07; p=0.0002655), and +0.63 for matched CoT
  (95% CI -2.04 to +3.30; p=0.6842).
- Regenerated the component summary with 24 reranker pairs, 24 reformulation
  pairs, 8 dense/hybrid pairs and 26 recorded-stratum pairs. The last contrast
  does not represent active coarse filtering.
- Checked syntax of added analysis/configuration files and the updated direct
  comparison script, and verified the bundled CSV/reference checksum lists.
- Compared 27 pipeline Python files with initial tracked source: 26 unchanged;
  the CLI differences are documented in `PROVENANCE.md`.

Analysis ran using the bundled NumPy 2.3.5 environment. No new model inference,
training, benchmark evaluation or passage-count experiment was performed.
The portable generation environment was specified but not freshly installed.
Historical runtime versions, model digests and complete precision metadata
remain unavailable in the retained metrics.

Source-archive content inspection is performed after committing the package;
the archive exclusions and their history limitations are described in the guide.
