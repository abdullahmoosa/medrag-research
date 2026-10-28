#!/usr/bin/env python3
"""
Compare evaluation results between two (or more) LLMs across the same config tree.

Reads evaluation_results/final_output trees, joins runs by config path (model
and version dirs stripped), and reports per-config accuracy with paired
significance tests computed from predictions.jsonl:

  - McNemar's exact test (both models answer the same questions)
  - Bootstrap 95% CI of the accuracy delta (paired percentile, 100k resamples)

Runs join on question text (example_id is "unknown" in this dataset).

Usage:
    python scripts_reviewer/compare_models.py                # llama3_8b vs llama3_med42_8b
    python scripts_reviewer/compare_models.py --model-a llama3_8b --model-b gemma3
    python scripts_reviewer/compare_models.py --csv out.csv  # also write CSV
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

DEFAULT_BASE = Path(__file__).resolve().parents[1] / "evaluation_results" / "final_output"
VERSION_RE = re.compile(r"^v\d+$")


def find_runs(model: str) -> dict[tuple, Path]:
    """
    Collect the latest run per config for one model.

    Returns {config_path_tuple: run_dir}. Config path = directory parts with the
    model dir and any version dirs removed. Later versions win (v3 > v2 > base).
    """
    runs: dict[tuple, Path] = {}
    versions: dict[tuple, tuple[int, bool]] = {}
    for metrics in sorted(BASE.rglob("metrics.json")):
        if json.loads(metrics.read_text()).get("total") != 1273 or not metrics.with_name("predictions.jsonl").is_file():
            continue
        rel = metrics.relative_to(BASE)
        if model not in rel.parts:
            continue
        parts = list(rel.parts)
        i = parts.index(model)
        version = 1
        tail = []
        for p in parts[i + 1:]:
            if p == "metrics.json":
                continue
            m = VERSION_RE.match(p)
            if m:
                version = max(version, int(p[1:]))
            else:
                tail.append(p)
        config = tuple("index_1" if p == "bge_large_en_v1.5" else p for p in parts[:i] + tail)
        rank = (version, "index_1" in parts)
        if rank > versions.get(config, (0, False)):
            versions[config] = rank
            runs[config] = metrics.parent
    return runs


def load_predictions(run_dir: Path) -> dict[str, bool]:
    """{question_text: is_correct} for one run."""
    preds = {}
    with open(run_dir / "predictions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["question"] in preds:
                raise ValueError(f"Duplicate question in {run_dir}")
            preds[d["question"]] = bool(d["is_correct"])
    if len(preds) != 1273:
        raise ValueError(f"Expected 1,273 unique predictions in {run_dir}")
    return preds


def paired_stats(pred_a: dict[str, bool], pred_b: dict[str, bool], n_boot: int = 100_000, seed: int = 42):
    """McNemar exact p-value and bootstrap CI for accuracy(B) - accuracy(A)."""
    if set(pred_a) != set(pred_b) or not pred_a:
        raise ValueError("Paired comparisons require identical, nonempty question sets")
    questions = sorted(pred_a)
    a = np.array([pred_a[q] for q in questions], dtype=bool)
    b = np.array([pred_b[q] for q in questions], dtype=bool)

    b01 = int((b & ~a).sum())  # B right, A wrong
    c10 = int((a & ~b).sum())  # A right, B wrong
    if b01 + c10 > 0:
        # Exact two-sided McNemar p-value for a Binomial(discordant, 0.5).
        discordant = b01 + c10
        lower_tail_numerator = sum(math.comb(discordant, i) for i in range(min(b01, c10) + 1))
        p_mcnemar = min(1.0, 2 * lower_tail_numerator / (2 ** discordant))
    else:
        p_mcnemar = 1.0

    rng = np.random.default_rng(seed)
    n = len(questions)
    a_f, b_f = a.astype(float), b.astype(float)
    # Resample question indices once so each bootstrap replicate preserves the
    # paired design (both systems are evaluated on the same sampled questions).
    bootstrap_deltas = []
    for start in range(0, n_boot, 256):
        size = min(256, n_boot - start)
        indices = rng.integers(0, n, (size, n))
        bootstrap_deltas.append((b_f[indices] - a_f[indices]).mean(axis=1))
    deltas = np.concatenate(bootstrap_deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n": n,
        "acc_a": a.mean(),
        "acc_b": b.mean(),
        "delta": b.mean() - a.mean(),
        "ci": (lo, hi),
        "p": p_mcnemar,
        "discordant": (b01, c10),
    }


def fmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-a", default="llama3_8b", help="Baseline model dir name")
    ap.add_argument("--model-b", default="llama3_med42_8b", help="Comparison model dir name")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_BASE,
        help="Root directory containing the NO_RAG and RAG result trees",
    )
    args = ap.parse_args()

    global BASE
    BASE = args.results_dir.expanduser().resolve()
    runs_a, runs_b = find_runs(args.model_a), find_runs(args.model_b)
    both = sorted(set(runs_a) & set(runs_b), key=lambda c: (len(c), c))
    only_a = set(runs_a) - set(runs_b)
    only_b = set(runs_b) - set(runs_a)

    if only_a:
        print(f"configs only in {args.model_a} (skipped): {len(only_a)}")
    if only_b:
        print(f"configs only in {args.model_b} (skipped): {len(only_b)}")

    name_a, name_b = args.model_a, args.model_b
    header = f"| Config | n | {name_a} acc | {name_b} acc | delta | 95% CI | McNemar p | sig |"
    print(f"\nComparing {name_b} vs {name_a} (delta = {name_b} - {name_a})\n")
    print(header)
    print("|:---|---:|---:|---:|---:|:---|---:|:---:|")

    rows = []
    for config in both:
        preds_a = load_predictions(runs_a[config])
        preds_b = load_predictions(runs_b[config])
        s = paired_stats(preds_a, preds_b)
        sig = "yes" if s["p"] < args.alpha else "no"
        cfg_str = "/".join(config)
        print(
            f"| {cfg_str} | {s['n']} | {s['acc_a']:.3f} | {s['acc_b']:.3f} "
            f"| {s['delta']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {fmt(s['p'])} | {sig} |"
        )
        rows.append((cfg_str, s))

    if rows:
        deltas = np.array([s["delta"] for _, s in rows])
        sig_n = sum(1 for _, s in rows if s["p"] < args.alpha)
        print(f"\n{len(rows)} shared configs | mean delta {deltas.mean():+.3f} "
              f"| range [{deltas.min():+.3f}, {deltas.max():+.3f}] | {sig_n} significant at alpha={args.alpha}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "n", f"acc_{name_a}", f"acc_{name_b}", "delta",
                        "ci_low", "ci_high", "mcnemar_p", "significant"])
            for cfg_str, s in rows:
                w.writerow([cfg_str, s["n"], f"{s['acc_a']:.4f}", f"{s['acc_b']:.4f}",
                            f"{s['delta']:+.4f}", f"{s['ci'][0]:.4f}", f"{s['ci'][1]:.4f}",
                            f"{s['p']:.4g}", s["p"] < args.alpha])
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
