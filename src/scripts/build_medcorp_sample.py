#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a *sampled* MedCorp folder from MedRAG sources (streaming):

Sources (Hugging Face):
  - MedRAG/pubmed     (split=train)
  - MedRAG/textbooks  (split=train)
  - MedRAG/wikipedia  (split=train)

What this script does
- Streams each source with HF datasets (no full download required)
- Sampling:
    * Head sampling (fast) or streaming shuffle (approx. random) with adaptive buffer
    * De-duplication:
        - PubMed: by text-hash + optional per-title-cap (to avoid article over-representation)
        - Textbooks/Wikipedia: by text-hash ONLY (no title-based dedup; many chunks share titles)
- Optional length filtering (min/max chars); disable max by passing --max-chars 0
- Writes per-source JSONL + merged medcorp_sample.jsonl (+ optional Parquet)
- Emits counts + a meta.json with your choices

Examples
--------
# 5k each, head sampling, JSONL only
python build_medcorp_sample.py --per-source 5000

# Asymmetric counts, approx. random with smaller buffer, Parquet export
python build_medcorp_sample.py \
  --n-pubmed 4000 --n-textbooks 8000 --n-wikipedia 6000 \
  --random --buffer 10000 --parquet

# PubMed per-title cap (at most 3 chunks per article title), disable max length
python build_medcorp_sample.py --per-source 6000 --random --per-title-cap 3 --max-chars 0
"""

import os
import json
import argparse
import hashlib
from typing import Dict, Any, Optional, Iterable, List
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
SOURCES = [
    ("MedRAG/pubmed",    "train", "pubmed.jsonl",    "pubmed"),
    ("MedRAG/textbooks", "train", "textbooks.jsonl", "textbooks"),
    ("MedRAG/wikipedia", "train", "wikipedia.jsonl", "wikipedia"),
]


# -----------------------------
# Normalization & Filters
# -----------------------------
def normalize_row(row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    """
    Normalize HF example dict -> unified record:
      {id, source, title, text, url, meta}
    """
    title   = row.get("title") or ""
    content = row.get("content") or row.get("contents") or ""
    rid     = row.get("id")

    out = {
        "id": f"{source}:{rid}" if rid is not None else None,
        "source": source,
        "title": title,
        "text": content,
        "url": "",
        "meta": {},
    }

    if source == "pubmed":
        pmid = row.get("PMID") or row.get("pmid")
        # best-effort year extraction (optional fields vary)
        year = row.get("year") or row.get("Year") or row.get("pub_year") or None
        if pmid:
            out["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        out["meta"] = {"PMID": pmid, "year": year}

    elif source == "wikipedia":
        wiki_id = row.get("wiki_id") or row.get("id")
        if wiki_id:
            out["url"] = f"https://en.wikipedia.org/?curid={wiki_id}"
        out["meta"] = {"wiki_id": wiki_id}

    # textbooks: metadata may vary across books; keep minimal
    if not out["text"]:
        return None
    return out


def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    # cheap normalization; okay for caps/comparisons
    return " ".join(s.split())


def _text_hash(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def pass_length(rec: Dict[str, Any], min_chars: int, max_chars: int) -> bool:
    txt = rec.get("text") or ""
    L = len(txt)
    if L < max(0, min_chars):
        return False
    if max_chars and max_chars > 0 and L > max_chars:
        return False
    return True


def pass_pubmed_year(rec: Dict[str, Any], year_min: int) -> bool:
    """Return True if no year info, otherwise check >= year_min."""
    if rec.get("source") != "pubmed":
        return True
    meta = rec.get("meta") or {}
    y = meta.get("year")
    try:
        if y is None:
            return True  # don't filter if missing
        return int(y) >= int(year_min)
    except Exception:
        return True  # keep if unparsable


# -----------------------------
# Streaming Sampler
# -----------------------------
def stream_and_filter(
    ds,
    target_n: int,
    source: str,
    random_mode: bool,
    seed: int,
    buffer_size: int,
    min_chars: int,
    max_chars: int,
    pubmed_year_min: int,
    per_title_cap: int = 0,   # 0 = no cap; used only for pubmed
) -> Iterable[Dict[str, Any]]:
    """
    Streaming-friendly sampler/cleaner:
      - optional shuffle with adaptive buffer
      - keeps only length-valid rows
      - PubMed: de-dup by text-hash + cap per normalized title
      - Textbooks/Wikipedia: de-dup by text-hash only (NO title-based dedup)
    """
    # Adaptive buffer: keep it sane for streaming so we don't stall forever
    if random_mode:
        buf = min(buffer_size, max(2000, target_n * 5))
        ds = ds.shuffle(seed=seed, buffer_size=buf)

    seen_txt = set()
    title_counts = defaultdict(int)
    kept = 0

    pbar = tqdm(total=target_n, desc=f"Sampling {source}", unit="rows")
    for row in ds:
        rec = normalize_row(row, source)
        if rec is None:
            continue
        if not pass_length(rec, min_chars=min_chars, max_chars=max_chars):
            continue
        if not pass_pubmed_year(rec, year_min=pubmed_year_min):
            continue

        # --- De-dup strategy ---
        # All sources: drop exact text duplicates
        t_hash = _text_hash(rec.get("text", ""))
        if t_hash in seen_txt:
            continue

        if source == "pubmed" and per_title_cap > 0:
            t_norm = _norm_title(rec.get("title", ""))
            if t_norm:
                if title_counts[t_norm] >= per_title_cap:
                    continue
                title_counts[t_norm] += 1
        # For textbooks/wikipedia: skip title caps entirely

        seen_txt.add(t_hash)
        yield rec
        kept += 1
        pbar.update(1)
        if kept >= target_n:
            break
    pbar.close()


# -----------------------------
# Writers
# -----------------------------
def write_jsonl_from_iter(rows: Iterable[Dict[str, Any]], out_path: str, source_key: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def concat_jsonls(paths: List[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for p in paths:
            with open(p, "r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)


def jsonl_to_parquet(jsonl_path: str, parquet_path: str):
    import pandas as pd
    df = pd.read_json(jsonl_path, lines=True)
    df.to_parquet(parquet_path, index=False)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="medcorp_sample", help="Output folder.")

    # Counts: either use --per-source for all, or set per-source individually
    ap.add_argument("--per-source", type=int, default=None,
                    help="If set, use this N for each source (overrides per-source defaults).")
    ap.add_argument("--n-pubmed", type=int, default=8000)
    ap.add_argument("--n-textbooks", type=int, default=10000)
    ap.add_argument("--n-wikipedia", type=int, default=2000)

    # Sampling mode
    ap.add_argument("--random", action="store_true", help="Use streaming shuffle (approx random).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer", type=int, default=10000,
                    help="Max shuffle buffer (adaptive: min(buffer, max(2000, N*5))).")

    # Filters
    ap.add_argument("--min-chars", type=int, default=200, help="Drop very short passages.")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="Drop very long passages (0 disables).")
    ap.add_argument("--pubmed-year-min", type=int, default=0,
                    help="Filter PubMed by year >= this (0 keeps all).")
    ap.add_argument("--per-title-cap", type=int, default=0,
                    help="For PubMed only: max passages kept per normalized title (0 disables).")

    # Outputs
    ap.add_argument("--parquet", action="store_true", help="Also write merged Parquet.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve Ns
    if args.per_source is not None:
        n_pubmed = n_textbooks = n_wikipedia = int(args.per_source)
    else:
        n_pubmed, n_textbooks, n_wikipedia = args.n_pubmed, args.n_textbooks, args.n_wikipedia

    target_map = {
        "pubmed": n_pubmed,
        "textbooks": n_textbooks,
        "wikipedia": n_wikipedia,
    }

    produced_paths = []
    counts = {}

    for hf_name, split, filename, key in SOURCES:
        need_n = int(target_map.get(key, 0))
        if need_n <= 0:
            continue

        out_path = os.path.join(args.out_dir, filename)
        # Load streaming dataset (shows its own "Resolving data files" progress)
        ds_stream = load_dataset(hf_name, split=split, streaming=True)

        # Build sampler with our filters/dedup
        rows = stream_and_filter(
            ds_stream,
            target_n=need_n,
            source=key,
            random_mode=args.random,
            seed=args.seed,
            buffer_size=args.buffer,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            pubmed_year_min=args.pubmed_year_min,
            per_title_cap=args.per_title_cap if key == "pubmed" else 0,
        )

        # Write JSONL
        n_written = write_jsonl_from_iter(rows, out_path, key)
        counts[key] = n_written
        produced_paths.append(out_path)

    # Merge
    merged = os.path.join(args.out_dir, "medcorp_sample.jsonl")
    concat_jsonls(produced_paths, merged)

    # Optional Parquet
    if args.parquet:
        merged_parquet = os.path.join(args.out_dir, "medcorp_sample.parquet")
        jsonl_to_parquet(merged, merged_parquet)

    # Save meta & counts
    meta = {
        "counts": counts,
        "settings": {
            "random": bool(args.random),
            "seed": int(args.seed),
            "buffer": int(args.buffer),
            "min_chars": int(args.min_chars),
            "max_chars": int(args.max_chars),
            "pubmed_year_min": int(args.pubmed_year_min),
            "per_title_cap_pubmed": int(args.per_title_cap),
        }
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(args.out_dir, "source_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    print("Done. Files in:", args.out_dir)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
