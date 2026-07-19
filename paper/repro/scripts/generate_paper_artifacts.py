#!/usr/bin/env python3
"""Generate paper-ready artifacts from saved evaluation runs.

This script is intentionally self-contained and deterministic. It reads
`evaluation_results/final_output/**/{metrics.json,predictions.jsonl}` and writes
machine-generated tables/figures for manuscript use.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


COMPLETE_TOTAL = 1273
BOOTSTRAP_SAMPLES = 3000
RANDOM_SEED = 42

INDEX_ALIASES = {
    "index_1": "bge_large_en_v1.5",
}


@dataclass
class RunRecord:
    run_id: str
    family: str
    index_name: str
    embedding_model: str
    retrieval_mode: str
    coarse_mode: str
    reranker: str
    reformulation: str
    prompt_mode: str
    llm_model: str
    total: int
    correct: int
    accuracy: float
    metrics_path: str
    predictions_path: str
    is_complete: bool
    version: str
    group_key: str


class Canvas:
    """Minimal RGB canvas with basic drawing primitives and PNG export."""

    def __init__(self, width: int, height: int, bg: Tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 3)
        self.clear(bg)

    def clear(self, color: Tuple[int, int, int]) -> None:
        r, g, b = color
        row = bytes([r, g, b]) * self.width
        for y in range(self.height):
            start = y * self.width * 3
            self.pixels[start : start + self.width * 3] = row

    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        idx = (y * self.width + x) * 3
        self.pixels[idx : idx + 3] = bytes(color)

    def draw_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: Tuple[int, int, int],
        fill: bool = True,
    ) -> None:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if fill:
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    self.set_pixel(x, y, color)
        else:
            for x in range(x0, x1 + 1):
                self.set_pixel(x, y0, color)
                self.set_pixel(x, y1, color)
            for y in range(y0, y1 + 1):
                self.set_pixel(x0, y, color)
                self.set_pixel(x1, y, color)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_point(self, x: int, y: int, color: Tuple[int, int, int], size: int = 2) -> None:
        self.draw_rect(x - size, y - size, x + size, y + size, color, fill=True)

    def save_png(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)  # filter type 0
            start = y * stride
            raw.extend(self.pixels[start : start + stride])
        compressed = zlib.compress(bytes(raw), level=9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        png_bytes = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
        out_path.write_bytes(png_bytes)


def canonical_index_name(index_name: str) -> str:
    return INDEX_ALIASES.get(index_name, index_name)


def infer_embedding_model(index_name: str, family: str) -> str:
    if family == "NO_RAG":
        return "none"
    canonical = canonical_index_name(index_name)
    mapping = {
        "index_1": "BAAI/bge-large-en-v1.5",
        "bge_large_en_v1.5": "BAAI/bge-large-en-v1.5",
        "medembed": "abhinand/MedEmbed-large-v0.1",
    }
    return mapping.get(canonical, canonical)


def parse_run_from_metrics(metrics_path: Path, repo_root: Path) -> Optional[RunRecord]:
    rel = metrics_path.relative_to(repo_root)
    parts = rel.parts
    # Expected prefix: evaluation_results/final_output
    if len(parts) < 5 or parts[0] != "evaluation_results" or parts[1] != "final_output":
        return None

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    total = int(metrics.get("total", 0) or 0)
    correct = int(metrics.get("correct", 0) or 0)
    accuracy = float(metrics.get("accuracy", 0.0) or 0.0)

    section = parts[2]

    if section == "NO_RAG":
        # evaluation_results/final_output/NO_RAG/{prompt}/{model}/[vN]/metrics.json
        prompt_mode = parts[3]
        llm_model = parts[4]
        version = "base"
        if len(parts) >= 7 and parts[5].startswith("v"):
            version = parts[5]

        group_key = "|".join(["NO_RAG", prompt_mode, llm_model])
        run_id = f"{group_key}|{version}"
        pred_path = metrics_path.with_name("predictions.jsonl")
        return RunRecord(
            run_id=run_id,
            family="NO_RAG",
            index_name="none",
            embedding_model="none",
            retrieval_mode="none",
            coarse_mode="none",
            reranker="none",
            reformulation="none",
            prompt_mode=prompt_mode,
            llm_model=llm_model,
            total=total,
            correct=correct,
            accuracy=accuracy,
            metrics_path=str(rel),
            predictions_path=str(pred_path.relative_to(repo_root)),
            is_complete=(total == COMPLETE_TOTAL),
            version=version,
            group_key=group_key,
        )

    if section == "RAG":
        # evaluation_results/final_output/RAG/{index}/{retrieval}/{coarse}/{reranker}/{reform}/{model}/{prompt}/[vN]/metrics.json
        if len(parts) < 11:
            return None

        index_name = canonical_index_name(parts[3])
        retrieval_mode = parts[4]
        coarse_mode = parts[5]
        reranker = parts[6]
        reformulation = parts[7]
        llm_model = parts[8]
        prompt_mode = parts[9]

        version = "base"
        if len(parts) >= 12 and parts[10].startswith("v"):
            version = parts[10]

        group_key = "|".join(
            [
                "RAG",
                index_name,
                retrieval_mode,
                coarse_mode,
                reranker,
                reformulation,
                llm_model,
                prompt_mode,
            ]
        )
        run_id = f"{group_key}|{version}"
        pred_path = metrics_path.with_name("predictions.jsonl")

        return RunRecord(
            run_id=run_id,
            family="RAG",
            index_name=index_name,
            embedding_model=infer_embedding_model(index_name, "RAG"),
            retrieval_mode=retrieval_mode,
            coarse_mode=coarse_mode,
            reranker=reranker,
            reformulation=reformulation,
            prompt_mode=prompt_mode,
            llm_model=llm_model,
            total=total,
            correct=correct,
            accuracy=accuracy,
            metrics_path=str(rel),
            predictions_path=str(pred_path.relative_to(repo_root)),
            is_complete=(total == COMPLETE_TOTAL),
            version=version,
            group_key=group_key,
        )

    return None


def read_predictions(path: Path) -> List[Dict[str, object]]:
    data: List[Dict[str, object]] = []
    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            ex_id = str(obj.get("example_id", "")).strip()
            gold = obj.get("gold_answer")
            pred = obj.get("predicted_answer")
            is_correct = obj.get("is_correct")
            if is_correct is None and isinstance(gold, str) and isinstance(pred, str):
                is_correct = (gold == pred)
            if not isinstance(is_correct, bool):
                is_correct = bool(is_correct) if is_correct is not None else False
            data.append(
                {
                    "row_id": idx + 1,
                    "example_id": ex_id if ex_id else f"row_{idx + 1}",
                    "gold": gold,
                    "pred": pred,
                    "is_correct": is_correct,
                    "question": obj.get("question", ""),
                }
            )
    return data


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bootstrap_delta_ci(a: List[int], b: List[int], seed: int = RANDOM_SEED) -> Tuple[float, float]:
    if not a or not b or len(a) != len(b):
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(a)
    deltas: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idxs = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in idxs) / n
        db = sum(b[i] for i in idxs) / n
        deltas.append(db - da)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return (lo, hi)


def mcnemar_exact_p(a: List[int], b: List[int]) -> float:
    # a/b are paired correctness indicators for run A and run B.
    n01 = 0  # A wrong, B right
    n10 = 0  # A right, B wrong
    for xa, xb in zip(a, b):
        if xa == 0 and xb == 1:
            n01 += 1
        elif xa == 1 and xb == 0:
            n10 += 1
    n = n01 + n10
    if n == 0:
        return 1.0

    k = min(n01, n10)
    # Two-sided exact binomial test with p=0.5
    tail = 0
    for i in range(0, k + 1):
        tail += math.comb(n, i)
    p = min(1.0, 2.0 * (tail / (2 ** n)))
    return p


def question_type(question: str) -> str:
    q = (question or "").lower()
    if any(tok in q for tok in [" except", " not ", "incorrect", "false"]):
        return "negation"
    if any(tok in q for tok in ["most likely diagnosis", "diagnosis"]):
        return "diagnosis"
    if any(tok in q for tok in ["mechanism", "pathophysiology", "pathway"]):
        return "mechanism"
    if any(tok in q for tok in ["drug", "treatment", "therapy"]):
        return "treatment"
    if any(tok in q for tok in ["side effect", "adverse"]):
        return "adverse_effect"
    return "other"


def paired_correctness(
    pred_a: List[Dict[str, object]], pred_b: List[Dict[str, object]]
) -> Tuple[List[int], List[int], List[str]]:
    # Most runs in this corpus use non-unique example_id=\"unknown\".
    # We default to positional alignment, which is valid because all runs share
    # the same MedQA test ordering and length when complete.
    n = min(len(pred_a), len(pred_b))
    a = [1 if pred_a[i].get("is_correct") else 0 for i in range(n)]
    b = [1 if pred_b[i].get("is_correct") else 0 for i in range(n)]
    ids = [str(pred_a[i].get("example_id") or f"row_{i+1}") for i in range(n)]
    return a, b, ids


def build_run_manifest(repo_root: Path) -> List[RunRecord]:
    metrics_files = sorted((repo_root / "evaluation_results").glob("**/metrics.json"))
    runs: List[RunRecord] = []
    for m in metrics_files:
        rr = parse_run_from_metrics(m, repo_root)
        if rr is not None:
            runs.append(rr)
    return runs


def best_run(records: List[RunRecord], predicate) -> Optional[RunRecord]:
    subset = [r for r in records if predicate(r)]
    if not subset:
        return None
    return max(subset, key=lambda x: x.accuracy)


def collapse_best_of(records: List[RunRecord]) -> List[RunRecord]:
    best: Dict[str, RunRecord] = {}
    for r in records:
        current = best.get(r.group_key)
        if current is None or r.accuracy > current.accuracy:
            best[r.group_key] = r
    return list(best.values())


def rerun_stability(records: List[RunRecord]) -> List[Dict[str, object]]:
    groups: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in records:
        groups[r.group_key].append(r)
    rows: List[Dict[str, object]] = []
    for gk, items in sorted(groups.items()):
        accs = [x.accuracy for x in items]
        best_item = max(items, key=lambda x: x.accuracy)
        rows.append(
            {
                "group_key": gk,
                "n_runs": len(items),
                "mean_accuracy": f"{statistics.mean(accs):.12f}",
                "std_accuracy": f"{statistics.pstdev(accs):.12f}",
                "best_accuracy": f"{best_item.accuracy:.12f}",
                "best_run_id": best_item.run_id,
                "best_metrics_path": best_item.metrics_path,
            }
        )
    return rows


def make_pairwise_row(
    comparison_id: str,
    slice_definition: str,
    run_a: RunRecord,
    run_b: RunRecord,
    pred_cache: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    pred_a = pred_cache.get(run_a.run_id, {})
    pred_b = pred_cache.get(run_b.run_id, {})
    a_vec, b_vec, _ids = paired_correctness(pred_a, pred_b)

    if not a_vec:
        return {
            "comparison_id": comparison_id,
            "slice_definition": slice_definition,
            "run_a_id": run_a.run_id,
            "run_b_id": run_b.run_id,
            "metric": "accuracy",
            "delta": "",
            "relative_delta_pct": "",
            "p_value": "",
            "ci_low": "",
            "ci_high": "",
            "test_name": "",
        }

    acc_a = sum(a_vec) / len(a_vec)
    acc_b = sum(b_vec) / len(b_vec)
    delta = acc_b - acc_a
    rel = (delta / acc_a * 100.0) if acc_a > 0 else float("nan")
    ci_low, ci_high = bootstrap_delta_ci(a_vec, b_vec)
    p_value = mcnemar_exact_p(a_vec, b_vec)

    return {
        "comparison_id": comparison_id,
        "slice_definition": slice_definition,
        "run_a_id": run_a.run_id,
        "run_b_id": run_b.run_id,
        "metric": "accuracy",
        "delta": f"{delta:.12f}",
        "relative_delta_pct": f"{rel:.6f}",
        "p_value": f"{p_value:.8f}",
        "ci_low": f"{ci_low:.12f}",
        "ci_high": f"{ci_high:.12f}",
        "test_name": "McNemar exact + paired bootstrap CI",
    }


def matched_factor_deltas(best_completed: List[RunRecord]) -> Tuple[List[Dict[str, object]], Dict[str, List[float]]]:
    # Delta orientation is level_b - level_a.
    factor_specs = [
        ("reranker", "reranker_off", "reranker_on"),
        ("reformulation", "no_reformulation", "reformulation"),
        ("retrieval_mode", "dense", "hybrid"),
        ("coarse_mode", "coarse_off", "coarse_k20"),
    ]

    rows: List[Dict[str, object]] = []
    detail: Dict[str, List[float]] = {}

    # Determine comparable dimensions.
    dims = [
        "family",
        "index_name",
        "retrieval_mode",
        "coarse_mode",
        "reranker",
        "reformulation",
        "prompt_mode",
        "llm_model",
    ]

    for factor, level_a, level_b in factor_specs:
        other_dims = [d for d in dims if d != factor]
        grouped: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)

        for r in best_completed:
            if r.family != "RAG":
                continue
            key = tuple(str(getattr(r, d)) for d in other_dims)
            grouped[key][str(getattr(r, factor))] = r.accuracy

        deltas: List[float] = []
        for key, vals in grouped.items():
            if level_a in vals and level_b in vals:
                deltas.append(vals[level_b] - vals[level_a])

        detail[factor] = deltas

        if deltas:
            rows.append(
                {
                    "factor": factor,
                    "level_a": level_a,
                    "level_b": level_b,
                    "n_pairs": len(deltas),
                    "mean_delta": f"{statistics.mean(deltas):.12f}",
                    "median_delta": f"{statistics.median(deltas):.12f}",
                    "positive_pairs": sum(1 for x in deltas if x > 0),
                    "negative_pairs": sum(1 for x in deltas if x < 0),
                }
            )
        else:
            rows.append(
                {
                    "factor": factor,
                    "level_a": level_a,
                    "level_b": level_b,
                    "n_pairs": 0,
                    "mean_delta": "",
                    "median_delta": "",
                    "positive_pairs": 0,
                    "negative_pairs": 0,
                }
            )

    return rows, detail


def build_factor_pair_details(best_completed: List[RunRecord]) -> List[Dict[str, object]]:
    # Delta is level_b - level_a.
    factor_specs = [
        ("reranker", "reranker_off", "reranker_on"),
        ("reformulation", "no_reformulation", "reformulation"),
        ("retrieval_mode", "dense", "hybrid"),
        ("coarse_mode", "coarse_off", "coarse_k20"),
    ]
    dims = [
        "family",
        "index_name",
        "retrieval_mode",
        "coarse_mode",
        "reranker",
        "reformulation",
        "prompt_mode",
        "llm_model",
    ]

    rows: List[Dict[str, object]] = []
    pair_id = 0
    for factor, level_a, level_b in factor_specs:
        other_dims = [d for d in dims if d != factor]
        grouped: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
        for r in best_completed:
            if r.family != "RAG":
                continue
            key = tuple(str(getattr(r, d)) for d in other_dims)
            grouped[key][str(getattr(r, factor))] = r.accuracy

        for key, vals in sorted(grouped.items(), key=lambda x: x[0]):
            if level_a in vals and level_b in vals:
                pair_id += 1
                row = {d: v for d, v in zip(other_dims, key)}
                row.update(
                    {
                        "pair_id": pair_id,
                        "factor": factor,
                        "level_a": level_a,
                        "level_b": level_b,
                        "delta": f"{(vals[level_b] - vals[level_a]):.12f}",
                        "comparison_key": "|".join(key),
                    }
                )
                rows.append(row)
    return rows


def summarize_pair_details_by_dimension(
    pair_rows: List[Dict[str, object]], dimension: str
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in pair_rows:
        factor = str(r["factor"])
        dim_val = str(r.get(dimension, ""))
        grouped[(factor, dim_val)].append(float(r["delta"]))

    out: List[Dict[str, object]] = []
    for (factor, dim_val), deltas in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out.append(
            {
                "factor": factor,
                "dimension": dimension,
                "dimension_value": dim_val,
                "n_pairs": len(deltas),
                "mean_delta": f"{statistics.mean(deltas):.12f}",
                "median_delta": f"{statistics.median(deltas):.12f}",
                "positive_pairs": sum(1 for x in deltas if x > 0),
                "negative_pairs": sum(1 for x in deltas if x < 0),
                "min_delta": f"{min(deltas):.12f}",
                "max_delta": f"{max(deltas):.12f}",
            }
        )
    return out


def technique_highlights(pair_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    by_factor: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in pair_rows:
        by_factor[str(r["factor"])].append(r)

    out: Dict[str, Dict[str, object]] = {}
    for factor, rows in by_factor.items():
        rows_sorted = sorted(rows, key=lambda x: float(x["delta"]))
        out[factor] = {
            "strongest_negative": rows_sorted[0],
            "strongest_positive": rows_sorted[-1],
        }
    return out


def generate_error_analysis(
    best_rag: RunRecord,
    baseline: RunRecord,
    pred_cache: Dict[str, List[Dict[str, object]]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    pred_r = pred_cache[best_rag.run_id]
    pred_b = pred_cache[baseline.run_id]
    n = min(len(pred_r), len(pred_b))

    rows: List[Dict[str, object]] = []
    summary_counts = defaultdict(int)

    for i in range(n):
        r = pred_r[i]
        b = pred_b[i]
        r_ok = 1 if r.get("is_correct") else 0
        b_ok = 1 if b.get("is_correct") else 0

        if b_ok == 0 and r_ok == 1:
            status = "wrong_to_correct"
            note = "RAG corrected baseline error"
        elif b_ok == 1 and r_ok == 0:
            status = "correct_to_wrong"
            note = "RAG introduced error"
        elif b_ok == 1 and r_ok == 1:
            status = "unchanged_correct"
            note = "Both correct"
        else:
            status = "unchanged_wrong"
            note = "Both incorrect"

        summary_counts[status] += 1

        rows.append(
            {
                "example_id": str(r.get("example_id") or f"row_{i+1}"),
                "gold": r.get("gold") or b.get("gold") or "",
                "pred_best_rag": r.get("pred") or "",
                "pred_baseline": b.get("pred") or "",
                "changed_correctness": status,
                "question_type": question_type(str(r.get("question") or b.get("question") or "")),
                "notes": note,
            }
        )

    summary = [
        {"transition": k, "count": v}
        for k, v in sorted(summary_counts.items(), key=lambda x: x[0])
    ]
    return rows, summary


def save_figure_specs(fig_dir: Path) -> None:
    specs = [
        {
            "figure_id": "figure1_accuracy_by_family",
            "input_table": "paper/artifacts/tables/table1_run_matrix_summary.csv",
            "x": "family_group",
            "y": "accuracy",
            "group_by": "family_group",
            "filter": "is_complete==True",
            "caption_short": "Accuracy by evaluation family.",
            "caption_long": "Distribution of completed-run accuracy across NO_RAG, RAG-BGE, and RAG-MedEmbed families.",
        },
        {
            "figure_id": "figure2_ablation_effect_sizes",
            "input_table": "paper/artifacts/tables/ablation_effects.csv",
            "x": "factor",
            "y": "mean_delta",
            "group_by": "factor",
            "filter": "n_pairs>0",
            "caption_short": "Ablation effect sizes.",
            "caption_long": "Mean paired accuracy deltas for key factors with confidence intervals where applicable.",
        },
        {
            "figure_id": "figure3_dense_vs_hybrid_paired_deltas",
            "input_table": "paper/artifacts/tables/table4_factor_deltas.csv",
            "x": "pair_id",
            "y": "delta_hybrid_minus_dense",
            "group_by": "comparison_key",
            "filter": "comparison=='hybrid_vs_dense'",
            "caption_short": "Dense vs hybrid paired deltas.",
            "caption_long": "Per-slice paired difference (hybrid minus dense). Negative values indicate dense outperforms hybrid.",
        },
        {
            "figure_id": "figure4_error_transition_chart",
            "input_table": "paper/artifacts/tables/error_transition_summary.csv",
            "x": "transition",
            "y": "count",
            "group_by": "comparison",
            "filter": "comparison in ['zero_shot','cot']",
            "caption_short": "Error transition counts.",
            "caption_long": "Transition counts (wrong->correct, correct->wrong, unchanged) for key head-to-head comparisons.",
        },
    ]

    fig_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        out = fig_dir / f"{spec['figure_id']}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)


def draw_figures(
    fig_dir: Path,
    table1_rows: List[Dict[str, object]],
    ablation_rows: List[Dict[str, object]],
    hybrid_detail_rows: List[Dict[str, object]],
    transition_rows: List[Dict[str, object]],
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    axis_color = (20, 20, 20)
    grid_color = (225, 225, 225)

    def map_y(v: float, vmin: float, vmax: float, top: int, bottom: int) -> int:
        if vmax <= vmin:
            return (top + bottom) // 2
        t = (v - vmin) / (vmax - vmin)
        return int(bottom - t * (bottom - top))

    # Figure 1: Accuracy by family (bars + points)
    fam_map = {"NO_RAG": [], "RAG-BGE": [], "RAG-MedEmbed": []}
    for row in table1_rows:
        fam = row.get("family")
        idx = row.get("index_name")
        acc = float(row.get("accuracy", 0.0))
        if fam == "NO_RAG":
            fam_map["NO_RAG"].append(acc)
        elif fam == "RAG" and idx in {"index_1", "bge_large_en_v1.5"}:
            fam_map["RAG-BGE"].append(acc)
        elif fam == "RAG" and idx == "medembed":
            fam_map["RAG-MedEmbed"].append(acc)

    c = Canvas(1000, 600, bg=(255, 255, 255))
    left, right, top, bottom = 80, 950, 60, 540
    c.draw_rect(left, top, right, bottom, axis_color, fill=False)
    for i in range(6):
        gy = top + int(i * (bottom - top) / 5)
        c.draw_line(left, gy, right, gy, grid_color)
    vals = [v for x in fam_map.values() for v in x] or [0.0, 1.0]
    vmin, vmax = min(vals), max(vals)
    pad = max(0.01, 0.1 * (vmax - vmin if vmax > vmin else 0.1))
    vmin -= pad
    vmax += pad
    colors = [(62, 114, 196), (203, 98, 76), (77, 166, 107)]
    labels = list(fam_map.keys())
    for i, label in enumerate(labels):
        xs = left + int((i + 0.5) * (right - left) / len(labels))
        series = fam_map[label]
        if not series:
            continue
        mean_v = statistics.mean(series)
        y0 = map_y(0.0, vmin, vmax, top, bottom)
        ym = map_y(mean_v, vmin, vmax, top, bottom)
        c.draw_rect(xs - 35, y0, xs + 35, ym, colors[i], fill=True)
        c.draw_rect(xs - 35, y0, xs + 35, ym, axis_color, fill=False)
        for j, yv in enumerate(series):
            py = map_y(yv, vmin, vmax, top, bottom)
            px = xs + ((j % 9) - 4) * 4
            c.draw_point(px, py, (35, 35, 35), size=2)
    c.save_png(fig_dir / "figure1_accuracy_by_family.png")

    # Figure 2: Ablation effect sizes (bar chart)
    rows = [r for r in ablation_rows if str(r.get("n_pairs", "0")) != "0"]
    c = Canvas(900, 500, bg=(255, 255, 255))
    left, right, top, bottom = 80, 860, 50, 450
    c.draw_rect(left, top, right, bottom, axis_color, fill=False)
    means = [float(r["mean_delta"]) for r in rows] or [0.0]
    vmin = min(means + [0.0])
    vmax = max(means + [0.0])
    pad = max(0.002, 0.15 * (vmax - vmin if vmax > vmin else 0.01))
    vmin -= pad
    vmax += pad
    zero_y = map_y(0.0, vmin, vmax, top, bottom)
    c.draw_line(left, zero_y, right, zero_y, axis_color)
    for i, r in enumerate(rows):
        x0 = left + int(i * (right - left) / max(1, len(rows))) + 30
        x1 = left + int((i + 1) * (right - left) / max(1, len(rows))) - 30
        yv = map_y(float(r["mean_delta"]), vmin, vmax, top, bottom)
        color = (70, 135, 220) if float(r["mean_delta"]) >= 0 else (220, 95, 95)
        c.draw_rect(x0, zero_y, x1, yv, color, fill=True)
        c.draw_rect(x0, zero_y, x1, yv, axis_color, fill=False)
    c.save_png(fig_dir / "figure2_ablation_effect_sizes.png")

    # Figure 3: Dense vs Hybrid paired deltas (lollipop chart)
    hd = [r for r in hybrid_detail_rows if r.get("comparison") == "hybrid_vs_dense"]
    c = Canvas(900, 500, bg=(255, 255, 255))
    left, right, top, bottom = 80, 860, 50, 450
    c.draw_rect(left, top, right, bottom, axis_color, fill=False)
    ys = [float(r["delta_hybrid_minus_dense"]) for r in hd] or [0.0]
    vmin = min(ys + [0.0])
    vmax = max(ys + [0.0])
    pad = max(0.002, 0.15 * (vmax - vmin if vmax > vmin else 0.01))
    vmin -= pad
    vmax += pad
    zero_y = map_y(0.0, vmin, vmax, top, bottom)
    c.draw_line(left, zero_y, right, zero_y, axis_color)
    for i, row in enumerate(hd):
        x = left + int((i + 1) * (right - left) / (len(hd) + 1))
        y = map_y(float(row["delta_hybrid_minus_dense"]), vmin, vmax, top, bottom)
        c.draw_line(x, zero_y, x, y, (120, 120, 120))
        c.draw_point(x, y, (45, 120, 200), size=4)
    c.save_png(fig_dir / "figure3_dense_vs_hybrid_paired_deltas.png")

    # Figure 4: Error transitions (grouped bars)
    comp_groups: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in transition_rows:
        comp_groups[str(r["comparison"])][str(r["transition"])] = int(r["count"])
    comparisons = ["zero_shot", "cot"]
    categories = ["wrong_to_correct", "correct_to_wrong", "unchanged_correct", "unchanged_wrong"]
    c = Canvas(1000, 550, bg=(255, 255, 255))
    left, right, top, bottom = 80, 960, 50, 500
    c.draw_rect(left, top, right, bottom, axis_color, fill=False)
    vals = []
    for comp in comparisons:
        for cat in categories:
            vals.append(comp_groups.get(comp, {}).get(cat, 0))
    vmax = max(vals) if vals else 1
    vmax = max(1, int(vmax * 1.2))
    for ci, cat in enumerate(categories):
        group_center = left + int((ci + 0.5) * (right - left) / len(categories))
        for j, comp in enumerate(comparisons):
            count = comp_groups.get(comp, {}).get(cat, 0)
            x0 = group_center - 42 + j * 42
            x1 = x0 + 30
            y = map_y(float(count), 0.0, float(vmax), top, bottom)
            color = (58, 121, 189) if comp == "zero_shot" else (214, 125, 61)
            c.draw_rect(x0, bottom, x1, y, color, fill=True)
            c.draw_rect(x0, bottom, x1, y, axis_color, fill=False)
    c.save_png(fig_dir / "figure4_error_transition_chart.png")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    table_dir = repo_root / "paper" / "artifacts" / "tables"
    fig_dir = repo_root / "paper" / "artifacts" / "figures"

    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = build_run_manifest(repo_root)
    if not runs:
        raise RuntimeError("No runs found under evaluation_results/final_output")

    # Cache predictions for all runs
    pred_cache: Dict[str, List[Dict[str, object]]] = {}
    for r in runs:
        pred_path = repo_root / r.predictions_path
        pred_cache[r.run_id] = read_predictions(pred_path)

    # Manifest
    manifest_rows: List[Dict[str, object]] = []
    for r in runs:
        manifest_rows.append(
            {
                "run_id": r.run_id,
                "family": r.family,
                "index_name": r.index_name,
                "embedding_model": r.embedding_model,
                "retrieval_mode": r.retrieval_mode,
                "coarse_mode": r.coarse_mode,
                "reranker": r.reranker,
                "reformulation": r.reformulation,
                "prompt_mode": r.prompt_mode,
                "llm_model": r.llm_model,
                "total": r.total,
                "correct": r.correct,
                "accuracy": f"{r.accuracy:.12f}",
                "metrics_path": r.metrics_path,
                "predictions_path": r.predictions_path,
                "is_complete": r.is_complete,
                "version": r.version,
                "group_key": r.group_key,
            }
        )

    write_csv(
        table_dir / "run_manifest.csv",
        manifest_rows,
        [
            "run_id",
            "family",
            "index_name",
            "embedding_model",
            "retrieval_mode",
            "coarse_mode",
            "reranker",
            "reformulation",
            "prompt_mode",
            "llm_model",
            "total",
            "correct",
            "accuracy",
            "metrics_path",
            "predictions_path",
            "is_complete",
            "version",
            "group_key",
        ],
    )

    completed = [r for r in runs if r.is_complete]
    completed_best = collapse_best_of(completed)

    # Table 1: full run matrix summary (completed runs only)
    table1_rows = [
        {
            "run_id": r.run_id,
            "family": r.family,
            "index_name": r.index_name,
            "embedding_model": r.embedding_model,
            "retrieval_mode": r.retrieval_mode,
            "coarse_mode": r.coarse_mode,
            "reranker": r.reranker,
            "reformulation": r.reformulation,
            "prompt_mode": r.prompt_mode,
            "llm_model": r.llm_model,
            "total": r.total,
            "correct": r.correct,
            "accuracy": f"{r.accuracy:.12f}",
            "metrics_path": r.metrics_path,
            "predictions_path": r.predictions_path,
            "is_complete": r.is_complete,
        }
        for r in sorted(completed, key=lambda x: x.accuracy, reverse=True)
    ]
    write_csv(
        table_dir / "table1_run_matrix_summary.csv",
        table1_rows,
        [
            "run_id",
            "family",
            "index_name",
            "embedding_model",
            "retrieval_mode",
            "coarse_mode",
            "reranker",
            "reformulation",
            "prompt_mode",
            "llm_model",
            "total",
            "correct",
            "accuracy",
            "metrics_path",
            "predictions_path",
            "is_complete",
        ],
    )

    # Table 2: top-10 completed configurations (best-of groups)
    top10 = sorted(completed_best, key=lambda x: x.accuracy, reverse=True)[:10]
    groups = defaultdict(list)
    for r in completed:
        groups[r.group_key].append(r)

    table2_rows = []
    for r in top10:
        accs = [x.accuracy for x in groups[r.group_key]]
        table2_rows.append(
            {
                "run_id": r.run_id,
                "family": r.family,
                "index_name": r.index_name,
                "embedding_model": r.embedding_model,
                "retrieval_mode": r.retrieval_mode,
                "coarse_mode": r.coarse_mode,
                "reranker": r.reranker,
                "reformulation": r.reformulation,
                "prompt_mode": r.prompt_mode,
                "llm_model": r.llm_model,
                "accuracy": f"{r.accuracy:.12f}",
                "correct": r.correct,
                "total": r.total,
                "n_reruns": len(accs),
                "mean_accuracy_reruns": f"{statistics.mean(accs):.12f}",
                "std_accuracy_reruns": f"{statistics.pstdev(accs):.12f}",
                "metrics_path": r.metrics_path,
            }
        )
    write_csv(
        table_dir / "table2_top10_completed_configurations.csv",
        table2_rows,
        [
            "run_id",
            "family",
            "index_name",
            "embedding_model",
            "retrieval_mode",
            "coarse_mode",
            "reranker",
            "reformulation",
            "prompt_mode",
            "llm_model",
            "accuracy",
            "correct",
            "total",
            "n_reruns",
            "mean_accuracy_reruns",
            "std_accuracy_reruns",
            "metrics_path",
        ],
    )

    # Rerun stability table
    write_csv(
        table_dir / "rerun_stability.csv",
        rerun_stability(completed),
        [
            "group_key",
            "n_runs",
            "mean_accuracy",
            "std_accuracy",
            "best_accuracy",
            "best_run_id",
            "best_metrics_path",
        ],
    )

    # Core comparisons
    best_zero_rag = best_run(completed, lambda r: r.family == "RAG" and r.prompt_mode == "zero_shot")
    best_zero_norag = best_run(completed, lambda r: r.family == "NO_RAG" and r.prompt_mode == "zero_shot")
    best_cot_rag = best_run(completed, lambda r: r.family == "RAG" and r.prompt_mode == "cot")
    best_cot_norag = best_run(completed, lambda r: r.family == "NO_RAG" and r.prompt_mode == "cot")
    best_bge = best_run(
        completed,
        lambda r: r.family == "RAG" and r.index_name in {"index_1", "bge_large_en_v1.5"},
    )
    best_medembed = best_run(completed, lambda r: r.family == "RAG" and r.index_name == "medembed")

    pairwise_rows: List[Dict[str, object]] = []
    table3_rows: List[Dict[str, object]] = []

    if best_zero_norag and best_zero_rag:
        row = make_pairwise_row(
            "zero_shot_rag_vs_no_rag",
            "Best completed zero-shot RAG vs best completed zero-shot NO_RAG",
            best_zero_norag,
            best_zero_rag,
            pred_cache,
        )
        pairwise_rows.append(row)
        table3_rows.append(row)

    if best_cot_norag and best_cot_rag:
        row = make_pairwise_row(
            "cot_rag_vs_no_rag",
            "Best completed CoT RAG vs best completed CoT NO_RAG",
            best_cot_norag,
            best_cot_rag,
            pred_cache,
        )
        pairwise_rows.append(row)
        table3_rows.append(row)

    if best_bge and best_medembed:
        row = make_pairwise_row(
            "best_medembed_vs_best_bge",
            "Best MedEmbed-family RAG vs best BGE-family RAG",
            best_bge,
            best_medembed,
            pred_cache,
        )
        pairwise_rows.append(row)
        table3_rows.append(row)

    write_csv(
        table_dir / "pairwise_comparisons.csv",
        pairwise_rows,
        [
            "comparison_id",
            "slice_definition",
            "run_a_id",
            "run_b_id",
            "metric",
            "delta",
            "relative_delta_pct",
            "p_value",
            "ci_low",
            "ci_high",
            "test_name",
        ],
    )

    write_csv(
        table_dir / "table3_core_head_to_head.csv",
        table3_rows,
        [
            "comparison_id",
            "slice_definition",
            "run_a_id",
            "run_b_id",
            "metric",
            "delta",
            "relative_delta_pct",
            "p_value",
            "ci_low",
            "ci_high",
            "test_name",
        ],
    )

    # Ablation effects and Table 4 details
    ablation_rows, factor_detail = matched_factor_deltas(completed_best)
    write_csv(
        table_dir / "ablation_effects.csv",
        ablation_rows,
        [
            "factor",
            "level_a",
            "level_b",
            "n_pairs",
            "mean_delta",
            "median_delta",
            "positive_pairs",
            "negative_pairs",
        ],
    )

    # Technique-level comprehensive analysis tables
    pair_detail_rows = build_factor_pair_details(completed_best)
    write_csv(
        table_dir / "technique_pair_details.csv",
        pair_detail_rows,
        [
            "pair_id",
            "factor",
            "level_a",
            "level_b",
            "family",
            "index_name",
            "retrieval_mode",
            "coarse_mode",
            "reranker",
            "reformulation",
            "prompt_mode",
            "llm_model",
            "delta",
            "comparison_key",
        ],
    )

    technique_dim_rows: List[Dict[str, object]] = []
    for dim in ["index_name", "llm_model", "prompt_mode", "retrieval_mode"]:
        technique_dim_rows.extend(summarize_pair_details_by_dimension(pair_detail_rows, dim))
    write_csv(
        table_dir / "technique_effect_by_dimension.csv",
        technique_dim_rows,
        [
            "factor",
            "dimension",
            "dimension_value",
            "n_pairs",
            "mean_delta",
            "median_delta",
            "positive_pairs",
            "negative_pairs",
            "min_delta",
            "max_delta",
        ],
    )

    with (table_dir / "technique_highlights.json").open("w", encoding="utf-8") as f:
        json.dump(technique_highlights(pair_detail_rows), f, indent=2)

    # Detailed pair rows for Table 4 (focus on dense vs hybrid and all factors)
    detail_rows: List[Dict[str, object]] = []

    # Dense vs hybrid explicit pair listing
    dims = [
        "family",
        "index_name",
        "coarse_mode",
        "reranker",
        "reformulation",
        "prompt_mode",
        "llm_model",
    ]
    grouped = defaultdict(dict)
    for r in completed_best:
        if r.family != "RAG":
            continue
        key = tuple(getattr(r, d) for d in dims)
        grouped[key][r.retrieval_mode] = r.accuracy

    pair_id = 0
    for key, vals in grouped.items():
        if "dense" in vals and "hybrid" in vals:
            pair_id += 1
            detail_rows.append(
                {
                    "comparison": "hybrid_vs_dense",
                    "pair_id": pair_id,
                    "comparison_key": "|".join(str(x) for x in key),
                    "delta_hybrid_minus_dense": f"{(vals['hybrid'] - vals['dense']):.12f}",
                }
            )

    write_csv(
        table_dir / "table4_factor_deltas.csv",
        detail_rows,
        ["comparison", "pair_id", "comparison_key", "delta_hybrid_minus_dense"],
    )

    # Error analysis for best zero-shot comparison
    error_rows: List[Dict[str, object]] = []
    transition_rows: List[Dict[str, object]] = []

    if best_zero_rag and best_zero_norag:
        rows, summary = generate_error_analysis(best_zero_rag, best_zero_norag, pred_cache)
        error_rows.extend(rows)
        for s in summary:
            transition_rows.append(
                {
                    "comparison": "zero_shot",
                    "transition": s["transition"],
                    "count": s["count"],
                }
            )

    if best_cot_rag and best_cot_norag:
        rows, summary = generate_error_analysis(best_cot_rag, best_cot_norag, pred_cache)
        for s in summary:
            transition_rows.append(
                {
                    "comparison": "cot",
                    "transition": s["transition"],
                    "count": s["count"],
                }
            )

    write_csv(
        table_dir / "error_analysis.csv",
        error_rows,
        [
            "example_id",
            "gold",
            "pred_best_rag",
            "pred_baseline",
            "changed_correctness",
            "question_type",
            "notes",
        ],
    )

    write_csv(
        table_dir / "error_transition_summary.csv",
        transition_rows,
        ["comparison", "transition", "count"],
    )

    # Summary JSON for manuscript templating.
    summary_json = {
        "complete_total_required": COMPLETE_TOTAL,
        "n_runs_all": len(runs),
        "n_runs_complete": len(completed),
        "best_overall_run_id": max(completed, key=lambda x: x.accuracy).run_id,
        "best_overall_accuracy": max(completed, key=lambda x: x.accuracy).accuracy,
        "best_zero_shot_rag_run_id": best_zero_rag.run_id if best_zero_rag else None,
        "best_zero_shot_rag_accuracy": best_zero_rag.accuracy if best_zero_rag else None,
        "best_zero_shot_no_rag_run_id": best_zero_norag.run_id if best_zero_norag else None,
        "best_zero_shot_no_rag_accuracy": best_zero_norag.accuracy if best_zero_norag else None,
        "best_cot_rag_run_id": best_cot_rag.run_id if best_cot_rag else None,
        "best_cot_rag_accuracy": best_cot_rag.accuracy if best_cot_rag else None,
        "best_cot_no_rag_run_id": best_cot_norag.run_id if best_cot_norag else None,
        "best_cot_no_rag_accuracy": best_cot_norag.accuracy if best_cot_norag else None,
    }
    with (table_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # Figure specs + figures
    save_figure_specs(fig_dir)
    draw_figures(fig_dir, table1_rows, ablation_rows, detail_rows, transition_rows)

    print("Generated paper artifacts:")
    print(f"  Tables: {table_dir}")
    print(f"  Figures: {fig_dir}")


if __name__ == "__main__":
    main()
