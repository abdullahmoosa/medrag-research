# Reproducibility Package

This package regenerates all manuscript artifacts directly from saved evaluation outputs in `evaluation_results/final_output`.

## Scope
- Dataset: MedQA USMLE test set
- Completed-run rule: `total == 1273`
- No new model training or inference runs
- Local execution context: RTX 3090 (as reported in study metadata)

## Generate Artifacts
From repo root:

```bash
python3 paper/repro/scripts/generate_paper_artifacts.py
```

Outputs:
- Tables: `paper/artifacts/tables/*.csv`
- Figures: `paper/artifacts/figures/*.png`
- Figure specs: `paper/artifacts/figures/*.json`

## Validate Artifacts

```bash
python3 paper/repro/scripts/validate_paper_artifacts.py
```

Validation checks:
1. Completeness: all table-1 runs satisfy `total == 1273`
2. Consistency: `correct / total` equals `accuracy` within tolerance
3. Traceability: each run row points to existing `metrics.json` and `predictions.jsonl`
4. Pairing sanity: ablation table has non-negative pair counts
5. Core comparison sanity: table-3 deltas are finite and reproducible from run-level accuracies

## File Contracts
Required table contracts are materialized at:
- `paper/artifacts/tables/run_manifest.csv`
- `paper/artifacts/tables/pairwise_comparisons.csv`
- `paper/artifacts/tables/ablation_effects.csv`
- `paper/artifacts/tables/error_analysis.csv`

Figure spec contract is materialized as one JSON per figure under:
- `paper/artifacts/figures/figure*.json`

## Determinism
The analysis uses fixed random seed (`42`) for bootstrap confidence intervals.
