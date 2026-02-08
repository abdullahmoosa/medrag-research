#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample and build a textbook-only corpus from MedRAG for medical RAG.

This script:
1. Loads the full MedRAG textbook dataset (125,847 passages)
2. Samples by category/textbook to ensure coverage
3. Filters by length and quality
4. Saves to JSONL format for indexing
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm

from datasets import load_dataset


def normalize_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalize a MedRAG textbook row to standard format."""
    title = row.get("title", "")
    content = row.get("content", row.get("text", ""))

    # Extract source from title (e.g., "Anatomy_Gray" -> "Anatomy", "Gray")
    parts = title.split("_") if "_" in title else [title]

    return {
        "id": f"textbook_{idx}",
        "source": "textbooks",
        "title": title,
        "text": content,
        "url": "",
        "meta": {
            "textbook": parts[0] if len(parts) > 0 else title,
            "original_id": row.get("id", idx),
        }
    }


def load_and_sample_textbooks(
    sample_size: int = 10000,
    seed: int = 42,
    min_chars: int = 500,
    max_chars: int = 8000,
    target_n_per_source: int = None
) -> List[Dict[str, Any]]:
    """
    Load and sample from MedRAG textbook dataset.

    Args:
        sample_size: Target number of passages (None for all)
        seed: Random seed
        min_chars: Minimum character count
        max_chars: Maximum character count
        target_n_per_source: Max per source (None for no limit)

    Returns:
        List of sampled textbook passages
    """
    print(f"Loading MedRAG textbooks...")
    ds = load_dataset("MedRAG/textbooks", split="train")
    print(f"Loaded {len(ds)} passages")

    # Filter by length first
    print(f"\nFiltering by length ({min_chars}-{max_chars} chars)...")
    filtered = []
    for idx, row in enumerate(tqdm(ds, desc="Filtering")):
        title = row.get("title", "")
        content = row.get("content", row.get("text", ""))

        if not content:
            continue

        text_len = len(content)
        if text_len < min_chars or (max_chars > 0 and text_len > max_chars):
            continue

        filtered.append((row, idx))

    print(f"After filtering: {len(filtered)} passages")

    # Sample per source
    random.seed(seed)

    # Group by source (first part of title)
    by_source = defaultdict(list)
    for row, idx in filtered:
        title = row.get("title", "")
        source = title.split("_")[0] if "_" in title else "General"
        by_source[source].append((row, idx))

    print(f"\nFound {len(by_source)} sources:")
    for source, items in sorted(by_source.items()):
        print(f"  {source}: {len(items)} passages")

    # Sample from each source
    sampled = []
    if target_n_per_source:
        print(f"\nSampling max {target_n_per_source} per source...")
        for source, items in by_source.items():
            if len(items) > target_n_per_source:
                items = random.sample(items, target_n_per_source)
            sampled.extend(items)
    elif sample_size and len(filtered) > sample_size:
        print(f"\nSampling {sample_size} passages total...")
        sampled = random.sample(filtered, sample_size)
    else:
        sampled = filtered

    print(f"Selected: {len(sampled)} passages")

    # Normalize
    print("\nNormalizing format...")
    normalized = []
    for row, idx in tqdm(sampled, desc="Normalizing"):
        rec = normalize_row(row, idx)
        if rec and rec.get("text"):
            normalized.append(rec)

    return normalized


def save_textbooks(textbooks: List[Dict[str, Any]], output_path: str):
    """Save textbooks to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for tb in textbooks:
            f.write(json.dumps(tb, ensure_ascii=False) + "\n")

    print(f"Saved {len(textbooks)} passages to {output_path}")


def create_manifest(textbooks: List[Dict[str, Any]], output_dir: str, params: Dict[str, Any]):
    """Create manifest file."""
    # Count by source
    by_source = defaultdict(int)
    for tb in textbooks:
        source = tb.get("meta", {}).get("textbook", "Unknown")
        by_source[source] += 1

    manifest = {
        "counts": {
            "textbooks": len(textbooks),
            "sources": dict(by_source)
        },
        "total": len(textbooks),
        "params": params
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build textbook-only corpus from MedRAG"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="medcorp_textbooks",
        help="Output directory"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Target number of passages (0 for all)"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum character count"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum character count (0 for no limit)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Building Textbook-Only Corpus from MedRAG")
    print("=" * 60)

    # Load and sample
    textbooks = load_and_sample_textbooks(
        sample_size=args.sample_size if args.sample_size > 0 else None,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars
    )

    # Save
    output_path = os.path.join(args.output_dir, "textbooks.jsonl")
    save_textbooks(textbooks, output_path)

    # Manifest
    params = {
        "sample_size": args.sample_size,
        "seed": args.seed,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars
    }
    create_manifest(textbooks, args.output_dir, params)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
