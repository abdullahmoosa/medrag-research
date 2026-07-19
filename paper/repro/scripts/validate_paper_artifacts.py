#!/usr/bin/env python3
"""Validate generated paper artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path


EPS = 1e-9


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    tdir = repo / "paper" / "artifacts" / "tables"

    required = [
        tdir / "run_manifest.csv",
        tdir / "table1_run_matrix_summary.csv",
        tdir / "table3_core_head_to_head.csv",
        tdir / "ablation_effects.csv",
        tdir / "pairwise_comparisons.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required artifact files: {missing}")

    # 1) Completeness and 2) consistency + 5) traceability
    n_rows = 0
    for row in read_csv(tdir / "table1_run_matrix_summary.csv"):
        n_rows += 1
        total = int(row["total"])
        correct = int(row["correct"])
        acc = float(row["accuracy"])
        if total != 1273:
            raise SystemExit(f"Completeness check failed: total={total} for run_id={row['run_id']}")
        calc = correct / total
        if abs(calc - acc) > EPS:
            raise SystemExit(
                f"Accuracy consistency failed for run_id={row['run_id']}: expected {calc}, found {acc}"
            )

        metrics_path = repo / row["metrics_path"]
        preds_path = repo / row["predictions_path"]
        if not metrics_path.exists() or not preds_path.exists():
            raise SystemExit(
                f"Traceability failed for run_id={row['run_id']}: "
                f"metrics exists={metrics_path.exists()} preds exists={preds_path.exists()}"
            )

    if n_rows == 0:
        raise SystemExit("table1_run_matrix_summary.csv is empty")

    # 4) Pairing sanity on ablation effects
    for row in read_csv(tdir / "ablation_effects.csv"):
        n_pairs = int(row["n_pairs"])
        if n_pairs < 0:
            raise SystemExit(f"Ablation pairing failed for factor={row['factor']}: n_pairs={n_pairs}")

    # 5) Core comparison sanity
    for row in read_csv(tdir / "table3_core_head_to_head.csv"):
        delta = float(row["delta"])
        p = float(row["p_value"])
        if math.isnan(delta) or math.isinf(delta):
            raise SystemExit(f"Core comparison delta invalid for comparison_id={row['comparison_id']}")
        if not (0.0 <= p <= 1.0):
            raise SystemExit(f"Core comparison p-value invalid for comparison_id={row['comparison_id']}: {p}")

    print("Validation passed:")
    print("  - completeness")
    print("  - accuracy consistency")
    print("  - traceability")
    print("  - ablation pairing sanity")
    print("  - core comparison sanity")


if __name__ == "__main__":
    main()
