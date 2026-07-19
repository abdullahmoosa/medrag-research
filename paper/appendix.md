# Appendix

## A. Artifact Inventory
### A.1 Required Contracts
- `paper/artifacts/tables/run_manifest.csv`
- `paper/artifacts/tables/pairwise_comparisons.csv`
- `paper/artifacts/tables/ablation_effects.csv`
- `paper/artifacts/tables/error_analysis.csv`
- `paper/artifacts/tables/technique_pair_details.csv`
- `paper/artifacts/tables/technique_effect_by_dimension.csv`
- `paper/artifacts/tables/technique_highlights.json`

### A.2 Manuscript Tables
- Table 1: `paper/artifacts/tables/table1_run_matrix_summary.csv`
- Table 2: `paper/artifacts/tables/table2_top10_completed_configurations.csv`
- Table 3: `paper/artifacts/tables/table3_core_head_to_head.csv`
- Table 4: `paper/artifacts/tables/table4_factor_deltas.csv`

### A.3 Figures
- Figure 1: `paper/artifacts/figures/figure1_accuracy_by_family.png`
- Figure 2: `paper/artifacts/figures/figure2_ablation_effect_sizes.png`
- Figure 3: `paper/artifacts/figures/figure3_dense_vs_hybrid_paired_deltas.png`
- Figure 4: `paper/artifacts/figures/figure4_error_transition_chart.png`

## B. Run-Selection Policy
1. Discover all runs from `evaluation_results/final_output/**/metrics.json`.
2. Include only completed runs where `total == 1273`.
3. Retain reruns (`v2/v3/v4`) for stability analysis.
4. Use best-of rerun group for headline ranking; report mean/std across reruns where applicable.

## C. Pairwise Statistical Protocol
For each head-to-head pair:
1. Align per-example correctness by deterministic row order.
2. Compute paired accuracy delta.
3. Compute 95% bootstrap CI (3000 samples, seed=42).
4. Compute McNemar exact p-value from discordant pairs.

## D. Validation Checklist (Implemented)
1. Completeness test: all Table 1 rows satisfy `total == 1273`.
2. Consistency test: `correct/total` equals `accuracy` within tolerance.
3. Traceability test: each run row references existing metrics/predictions files.
4. Pairing sanity test: ablation pair counts are non-negative.
5. Core-comparison sanity: p-values in [0,1], finite deltas.

Run:

```bash
python3 paper/repro/scripts/validate_paper_artifacts.py
```

## E. Detailed Core Comparison Values
From `table3_core_head_to_head.csv`:
- Zero-shot RAG vs zero-shot NO_RAG: delta = +0.045561665357, p=0.00348701
- CoT RAG vs CoT NO_RAG: delta = +0.006284367636, p=0.68417649
- Best MedEmbed vs Best BGE(index_1): delta = +0.001571091909, p=0.94500624

## F. Notes on Prediction Pairing
Some prediction files store `example_id="unknown"` for all rows. To preserve valid paired evaluation across runs, comparisons are aligned by row order, which is stable for the same MedQA test split and evaluation pipeline.
