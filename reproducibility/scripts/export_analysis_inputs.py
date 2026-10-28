#!/usr/bin/env python3
"""Export text-free analysis inputs from locally retained metrics and predictions.

Does not modify the original outputs. Requires access to excluded raw files.
Only hashes, correctness flags, settings inferred from paths, and summary metrics
are exported. No questions, answer text, reasoning, or retrieved text is copied.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = re.compile(r"v(\d+)")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "evaluation_results/final_output")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproducibility/data")
    args = parser.parse_args()
    base, out = args.results_dir.resolve(), args.output_dir.resolve()
    if out == base or base in out.parents:
        raise ValueError("Export into a separate directory, outside the raw results tree")
    out.mkdir(parents=True, exist_ok=True)
    runs, predictions, groups = [], {}, defaultdict(list)
    reference = None
    for number, path in enumerate(sorted(base.rglob("metrics.json")), 1):
        rel = path.relative_to(base)
        parts = [p for p in rel.parts[:-1] if not VERSION.fullmatch(p)]
        version = max([int(p[1:]) for p in rel.parts if VERSION.fullmatch(p)] + [1])
        canonical = ["index_1" if p == "bge_large_en_v1.5" else p for p in parts]
        if parts[0] == "NO_RAG":
            _, prompt, model = parts
            index = retrieval = stratum = reranker = reformulation = "none"
        elif parts[0] == "RAG":
            _, index, retrieval, stratum, reranker, reformulation, model, prompt = parts
            index = "index_1" if index == "bge_large_en_v1.5" else index
        else:
            raise ValueError(f"Unrecognized results path: {rel}")
        metrics = json.loads(path.read_text())
        pred_path = path.with_name("predictions.jsonl")
        row = dict(
            run_id=f"R{number:03d}", canonical_configuration="/".join(canonical),
            family=parts[0], embedding=index, retrieval=retrieval,
            recorded_stratum=stratum, reranker=reranker, reformulation=reformulation,
            model=model, prompt=prompt, version=version,
            total=metrics["total"], correct=metrics["correct"],
            accuracy=metrics["correct"] / metrics["total"] if metrics["total"] else 0,
            elapsed_seconds=metrics.get("elapsed_seconds", ""),
            source_metrics_path="evaluation_results/final_output/" + rel.as_posix(),
            source_predictions_path="evaluation_results/final_output/" + pred_path.relative_to(base).as_posix(),
            metrics_sha256=sha256(path), predictions_sha256="",
            context_sha256="",
            status="incomplete", selected=False,
        )
        runs.append(row)
        if row["total"] != 1273 or not pred_path.is_file():
            continue
        values, contexts = {}, {}
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                question = d["question"]
                if question in values:
                    raise ValueError(f"Duplicate question in {pred_path}")
                correctness = d["is_correct"]
                # The evaluator writes null for unparseable answers; these are incorrect.
                if correctness is None:
                    correctness = False
                if not isinstance(correctness, (bool, int)) or correctness not in (0, 1):
                    raise ValueError(f"Nonboolean correctness in {pred_path}")
                values[question] = int(correctness)
                contexts[question] = hashlib.sha256(json.dumps(d.get("context"), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if len(values) != 1273 or sum(values.values()) != row["correct"]:
            raise ValueError(f"Metrics/prediction count mismatch: {pred_path}")
        if reference is None:
            reference = sorted(values)
        if set(values) != set(reference):
            raise ValueError(f"Question sets differ: {pred_path}")
        predictions[row["run_id"]] = [values[q] for q in reference]
        row["context_sha256"] = hashlib.sha256("".join(contexts[q] for q in reference).encode("ascii")).hexdigest()
        row["predictions_sha256"] = sha256(pred_path)
        row["status"] = "complete"
        groups[row["canonical_configuration"]].append(row)
    selected = []
    for group in groups.values():
        # Prefer the original index_1 path over its copied alias for a version tie.
        chosen = max(group, key=lambda r: (r["version"], "/RAG/index_1/" in r["source_metrics_path"], r["source_metrics_path"]))
        for other in group:
            if other["version"] == chosen["version"] and (
                predictions[other["run_id"]] != predictions[chosen["run_id"]]
                or other["predictions_sha256"] != chosen["predictions_sha256"]
                or other["metrics_sha256"] != chosen["metrics_sha256"]
            ):
                raise ValueError("Conflicting equal-version alias copies; choose explicitly")
        chosen["selected"] = True
        selected.append(chosen)
    selected.sort(key=lambda r: r["run_id"])
    if not selected:
        raise ValueError("No complete runs found")
    write_csv(out / "source_manifest.csv", runs)
    write_csv(out / "selected_runs.csv", selected)
    matrix_rows = []
    for i, q in enumerate(reference):
        # Retain question-text sort order to reproduce the historical bootstrap RNG mapping.
        row = {"question_sha256": hashlib.sha256(q.encode("utf-8")).hexdigest()}
        row.update({r["run_id"]: predictions[r["run_id"]][i] for r in selected})
        matrix_rows.append(row)
    write_csv(out / "correctness.csv", matrix_rows)
    print(f"Exported {len(selected)} selected runs and {len(reference)} aligned questions; {len(runs)} source metric files.")


if __name__ == "__main__":
    main()
