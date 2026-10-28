#!/usr/bin/env python3
"""Regenerate revised tables and figure data from text-free, aligned correctness.

All tests are nominal and exploratory. No inference calls or raw text required.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("llama3_8b", "llama3_med42_8b", "gemma3")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def bootstrap(delta, samples, seed):
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        indices = rng.integers(0, len(delta), (size, len(delta)))
        estimates[start:start + size] = delta[indices].mean(axis=1)
    return np.percentile(estimates, [2.5, 97.5]) * 100


def paired(a, b, values, samples, seed):
    x, y = values[a["run_id"]], values[b["run_id"]]
    b_only, a_only = int(((y == 1) & (x == 0)).sum()), int(((x == 1) & (y == 0)).sum())
    lo, hi = bootstrap(y - x, samples, seed)
    discordant = b_only + a_only
    # Exact binomial tail using Python integers avoids 2**(-n) underflow.
    p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(b_only, a_only) + 1)) / (2 ** discordant)) if discordant else 1.
    return dict(
        run_a=a["run_id"], run_b=b["run_id"], n=len(x),
        correct_a=int(x.sum()), correct_b=int(y.sum()),
        accuracy_a_percent=100 * x.mean(), accuracy_b_percent=100 * y.mean(),
        difference_pp=100 * (y - x).mean(), ci_low_pp=lo, ci_high_pp=hi,
        b_only_correct=b_only, a_only_correct=a_only, exact_mcnemar_p=p,
        contexts_identical=a["context_sha256"] == b["context_sha256"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "reproducibility/data")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reproducibility/generated")
    ap.add_argument("--bootstrap-samples", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.bootstrap_samples < 1:
        ap.error("--bootstrap-samples must be positive")
    rows = read_csv(args.data_dir / "selected_runs.csv")
    matrix = read_csv(args.data_dir / "correctness.csv")
    if len(matrix) != 1273 or len({r["question_sha256"] for r in matrix}) != 1273:
        raise ValueError("Expected 1,273 unique aligned question hashes")
    values = {r["run_id"]: np.array([int(q[r["run_id"]]) for q in matrix], dtype=float) for r in rows}
    lookup = {r["canonical_configuration"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Repeated canonical configuration in selected input")
    for r in rows:
        a = values[r["run_id"]]
        if not np.isin(a, [0, 1]).all() or int(a.sum()) != int(r["correct"]):
            raise ValueError(f"Incorrect correctness input for {r['run_id']}")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stat = lambda a, b: paired(a, b, values, args.bootstrap_samples, args.seed)
    baseline = lambda prompt, model: lookup[f"NO_RAG/{prompt}/{model}"]
    rag = lambda model="llama3_med42_8b", prompt="zero_shot", retrieval="dense", reform="reformulation", rerank="reranker_on", stratum="coarse_k20", embedding="index_1": lookup[f"RAG/{embedding}/{retrieval}/{stratum}/{rerank}/{reform}/{model}/{prompt}"]

    def summary(r):
        n, k = int(r["total"]), int(r["correct"])
        p, z = k / n, 1.959963984540054
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * ((p*(1-p)/n + z*z/(4*n*n))**.5) / (1 + z*z/n)
        return dict(run_id=r["run_id"], model=r["model"], prompt=r["prompt"], embedding=r["embedding"], retrieval=r["retrieval"], reranker=r["reranker"], reformulation=r["reformulation"], correct=k, n=n, accuracy_percent=100*p, wilson_ci_low_percent=100*(center-half), wilson_ci_high_percent=100*(center+half), elapsed_seconds=r["elapsed_seconds"])

    for prompt, name in [("zero_shot", "appendix_A1_zero_shot.csv"), ("cot", "appendix_A2_cot.csv")]:
        write_csv(out / name, [summary(r) for r in rows if r["prompt"] == prompt])
    write_csv(out / "table2_baselines.csv", [summary(r) for r in rows if r["family"] == "NO_RAG"])
    table3 = [rag(reform="no_reformulation", rerank="reranker_off"), rag(rerank="reranker_off"), rag(), rag(retrieval="hybrid"), rag(model="llama3_8b"), rag(model="gemma3")]
    write_csv(out / "table3_selected_rag.csv", [summary(r) for r in table3])
    write_csv(out / "table4_rag_vs_no_rag.csv", [dict(prompt=p, **stat(baseline(p, "llama3_med42_8b"), rag(prompt=p))) for p in ("zero_shot", "cot")])

    comparisons = []
    for a in rows:
        if a["model"] != "llama3_8b":
            continue
        key = a["canonical_configuration"].replace("/llama3_8b", "/llama3_med42_8b")
        if key in lookup:
            comparisons.append(dict(configuration=a["canonical_configuration"].replace("/llama3_8b", ""), prompt=a["prompt"], **stat(a, lookup[key])))
    write_csv(out / "supplementary_S1_model_comparisons.csv", comparisons)

    factors = [
        ("reranker", "reranker_off", "reranker_on", "Reranker off to on"),
        ("reformulation", "no_reformulation", "reformulation", "Query reformulation off to on"),
        ("retrieval", "dense", "hybrid", "Dense to hybrid"),
        ("recorded_stratum", "coarse_off", "coarse_k20", "Recorded stratum off to k20 (coarse filtering inactive)"),
    ]
    components, pair_details = [], []
    match_fields = ["embedding", "retrieval", "recorded_stratum", "reranker", "reformulation", "model", "prompt"]
    zero = [r for r in rows if r["family"] == "RAG" and r["prompt"] == "zero_shot"]
    for field, before, after, label in factors:
        groups = defaultdict(dict)
        for r in zero:
            key = tuple(r[f] for f in match_fields if f != field)
            groups[key][r[field]] = r
        pairs = [(g[before], g[after]) for g in groups.values() if before in g and after in g]
        deltas = np.stack([values[b["run_id"]] - values[a["run_id"]] for a, b in pairs])
        lo, hi = bootstrap(deltas.mean(axis=0), args.bootstrap_samples, args.seed)
        components.append(dict(contrast=label, matched_configuration_pairs=len(pairs), mean_difference_pp=100*deltas.mean(), median_difference_pp=100*np.median(deltas.mean(axis=1)), ci_low_pp=lo, ci_high_pp=hi, mean_runtime_difference_seconds=np.mean([float(b["elapsed_seconds"]) - float(a["elapsed_seconds"]) for a,b in pairs])))
        pair_details.extend(dict(contrast=label, run_a=a["run_id"], run_b=b["run_id"], difference_pp=100*(values[b["run_id"]]-values[a["run_id"]]).mean()) for a,b in pairs)
    write_csv(out / "table5_component_summary.csv", components)
    write_csv(out / "table5_matched_pairs.csv", pair_details)

    figure4 = []
    for group, stratum, rr in [("A", "coarse_off", "reranker_off"), ("B", "coarse_off", "reranker_on"), ("C", "coarse_k20", "reranker_off"), ("D", "coarse_k20", "reranker_on")]:
        for embed in ("index_1", "medembed"):
            figure4.append(dict(group=group, **summary(rag(stratum=stratum, rerank=rr, embedding=embed))))
    write_csv(out / "figure4_data.csv", figure4)
    figure5 = [
        ("Zero-shot without RAG", baseline("zero_shot", "llama3_med42_8b")),
        ("Zero-shot dense", table3[0]), ("Zero-shot dense + reformulation", table3[1]),
        ("Zero-shot dense + reformulation + reranker", table3[2]),
        ("Zero-shot hybrid + reformulation + reranker", table3[3]),
        ("CoT without RAG", baseline("cot", "llama3_med42_8b")),
        ("CoT dense + reformulation + reranker", rag(prompt="cot")),
    ]
    write_csv(out / "figure5_data.csv", [dict(condition=label, **summary(r)) for label,r in figure5])
    (out / "analysis_settings.txt").write_text(f"Paired question bootstrap replicates: {args.bootstrap_samples}\nSeed: {args.seed}\nExact two-sided McNemar tests; nominal exploratory p-values.\nComponent intervals hold matched configuration pairs fixed.\nRecorded coarse strata are not active components; repeated observations retain their weight.\n", encoding="utf-8")
    print(f"Generated tables and figure data: {len(rows)} runs, {len(comparisons)} model pairs, {len(pair_details)} component/stratum pairs.")


if __name__ == "__main__":
    main()
