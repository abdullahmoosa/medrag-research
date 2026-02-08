#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json
from typing import Dict, Any, Iterable, Optional
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

OUT_DIR = os.path.join("data", "medcorp")
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = [
    # (HF dataset name, split, out filename)
    ("MedRAG/pubmed",     "train", os.path.join(OUT_DIR, "pubmed.jsonl")),
    ("MedRAG/textbooks",  "train", os.path.join(OUT_DIR, "textbooks.jsonl")),
    ("MedRAG/wikipedia",  "train", os.path.join(OUT_DIR, "wikipedia.jsonl")),
]

def normalize_row(row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    """Map MedRAG rows to {id, source, title, text, url, meta}."""
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
        pmid = row.get("PMID")
        if pmid: out["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        out["meta"] = {"PMID": pmid}
    elif source == "wikipedia":
        wiki_id = row.get("wiki_id") or row.get("id")  # wiki_id added per dataset notes
        if wiki_id: out["url"] = f"https://en.wikipedia.org/?curid={wiki_id}"
        out["meta"] = {"wiki_id": wiki_id}
    elif source == "textbooks":
        # Leave URL empty; keep anything useful in meta if present
        pass
    # guard
    if not out["text"]:
        return None
    return out

def stream_hf_to_jsonl(hf_name: str, split: str, out_path: str, source_key: str, limit: Optional[int] = None):
    ds = load_dataset(hf_name, split=split, streaming=True)  # stream = memory-safe
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"Writing {source_key}", unit="rows"):
            rec = normalize_row(row, source_key)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    return n

def concat_jsonls(paths, out_path):
    with open(out_path, "w", encoding="utf-8") as w:
        for p in paths:
            with open(p, "r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)

def jsonl_to_parquet(jsonl_path: str, parquet_path: str):
    # Use chunks to handle large files if needed
    df = pd.read_json(jsonl_path, lines=True)
    df.to_parquet(parquet_path, index=False)

def main(limit_per_source: Optional[int] = None):
    produced = []
    counts = {}
    for hf_name, split, out_path in SOURCES:
        key = os.path.basename(out_path).split(".")[0]  # pubmed / textbooks / wikipedia
        n = stream_hf_to_jsonl(hf_name, split, out_path, key, limit=limit_per_source)
        counts[key] = n
        produced.append(out_path)
    # merge
    all_jsonl = os.path.join(OUT_DIR, "medcorp.jsonl")
    concat_jsonls(produced, all_jsonl)
    # parquet
    all_parquet = os.path.join(OUT_DIR, "medcorp.parquet")
    jsonl_to_parquet(all_jsonl, all_parquet)
    # stats
    with open(os.path.join(OUT_DIR, "source_counts.json"), "w") as f:
        json.dump(counts, f, indent=2)
    print("Done. Files in:", OUT_DIR)
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    # Set limit_per_source=None to pull everything.
    # For a quick smoke test, set limit_per_source=5000.
    main(limit_per_source=None)
