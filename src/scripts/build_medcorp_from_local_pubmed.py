#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build MedCorp folder from local PubMed data with sampling.

This script reads the local PubMed data from data/pubmed/ directory,
samples from each medical specialty category, and converts it to the
same format as build_medcorp_sample_async.py.

The script:
1. Reads all JSONL files from data/pubmed/ (organized by specialty)
2. Samples from each specialty file (can specify sample size per specialty)
3. Converts to the standard MedCorp format with deduplication
4. Applies filters (min/max chars, year filter)
5. Outputs to the target directory with manifest
"""

import os
import json
import argparse
import hashlib
import random
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ---------------------- Configuration ----------------------

PUBMED_DATA_DIR = "/home/ser/medrag/data/pubmed"

# Output format matching build_medcorp_sample_async.py
OUTPUT_FILENAME = "pubmed.jsonl"
MANIFEST_FILENAME = "manifest.json"


# ---------------------- Normalization & Filters ----------------------

def _norm_title(x: str) -> str:
    """Normalize title for deduplication."""
    return (" ".join((x or "").strip().lower().split()))[:512]

def _text_hash(x: str) -> str:
    """Generate hash for text deduplication."""
    return hashlib.md5((x or "").strip().encode("utf-8")).hexdigest()

def normalize_pubmed_row(row: Dict[str, Any], specialty: str) -> Optional[Dict[str, Any]]:
    """
    Normalize a PubMed row to the standard MedCorp format.

    Input format (from data/pubmed/*.jsonl):
    {
        "pmid": "31636002",
        "title": "Artificial Intelligence in Medicine",
        "abstract": "Abstract text here...",
        "journal": "Academic radiology",
        "pub_year": "2020"
    }

    Output format (matching medcorp_sample_20k/pubmed.jsonl):
    {
        "id": "pubmed:31636002",
        "source": "pubmed",
        "title": "Artificial Intelligence in Medicine",
        "text": "Abstract text here...",
        "url": "https://pubmed.ncbi.nlm.nih.gov/31636002/",
        "meta": {"PMID": 31636002, "year": 2020}
    }
    """
    pmid = row.get("pmid")
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    pub_year = row.get("pub_year")

    # Build the record
    rec = {
        "id": f"pubmed:{pmid}" if pmid else None,
        "source": "pubmed",
        "title": title,
        "text": abstract,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "meta": {
            "PMID": int(pmid) if pmid else None,
            "year": int(pub_year) if pub_year else None,
            "specialty": specialty,
            "journal": row.get("journal", "")
        }
    }

    # Filter out empty records
    if not rec["text"]:
        return None

    return rec

def apply_filters(rec: Dict[str, Any], min_chars: int, max_chars: int, year_min: int) -> bool:
    """Apply all filters to a record."""
    if rec is None:
        return False

    # Length filter
    text_len = len(rec.get("text", ""))
    if not ((text_len >= min_chars) and (text_len <= max_chars if max_chars > 0 else True)):
        return False

    # Year filter
    year = rec.get("meta", {}).get("year")
    if year is not None:
        try:
            year = int(year)
            if year < year_min:
                return False
        except (ValueError, TypeError):
            pass

    return True


# ---------------------- Deduplication ----------------------

class Deduplicator:
    """Track duplicates by title hash and content hash."""

    def __init__(self):
        self.seen_titles = set()
        self.seen_hashes = set()

    def is_duplicate(self, rec: Dict[str, Any]) -> bool:
        """Check if record is duplicate."""
        tnorm = _norm_title(rec.get("title", ""))
        thash = _text_hash(rec.get("text", ""))

        # Check for duplicates
        if tnorm and tnorm in self.seen_titles:
            return True
        if thash in self.seen_hashes:
            return True

        # Add to seen sets
        if tnorm:
            self.seen_titles.add(tnorm)
        self.seen_hashes.add(thash)
        return False


# ---------------------- Sampling & Processing ----------------------

def load_and_sample_specialty(
    filepath: str,
    specialty: str,
    sample_size: int,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Load and sample from a specialty JSONL file.

    Args:
        filepath: Path to the specialty JSONL file
        specialty: Name of the medical specialty
        sample_size: Maximum number of records to sample
        seed: Random seed for sampling

    Returns:
        List of raw rows (as dictionaries)
    """
    rows = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    rows.append(row)
                except json.JSONDecodeError:
                    continue

        # Sample if we have more than sample_size
        if len(rows) > sample_size:
            random.seed(seed)
            rows = random.sample(rows, sample_size)

    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return rows


def process_specialty_file(
    filepath: str,
    specialty: str,
    sample_size: int,
    deduplicator: Deduplicator,
    min_chars: int,
    max_chars: int,
    year_min: int,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Process a single specialty file.

    Args:
        filepath: Path to the specialty JSONL file
        specialty: Name of the medical specialty
        sample_size: Maximum number of records to sample from this file
        deduplicator: Deduplicator instance
        min_chars: Minimum character count
        max_chars: Maximum character count (0 = no limit)
        year_min: Minimum publication year
        seed: Random seed

    Returns:
        List of processed, non-duplicate records
    """
    # Load and sample
    raw_rows = load_and_sample_specialty(filepath, specialty, sample_size, seed)

    # Normalize and filter
    processed = []
    for row in raw_rows:
        rec = normalize_pubmed_row(row, specialty)
        if rec and apply_filters(rec, min_chars, max_chars, year_min):
            # Check for duplicates
            if not deduplicator.is_duplicate(rec):
                processed.append(rec)

    return processed


def process_all_specialties(
    pubmed_dir: str,
    sample_per_specialty: int,
    min_chars: int,
    max_chars: int,
    year_min: int,
    seed: int,
    skip_specialties: List[str] = None
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process all specialty files in the PubMed directory.

    Args:
        pubmed_dir: Path to directory containing specialty JSONL files
        sample_per_specialty: Maximum records to sample per specialty
        min_chars: Minimum character count
        max_chars: Maximum character count (0 = no limit)
        year_min: Minimum publication year
        seed: Random seed for reproducibility
        skip_specialties: List of specialties to skip (e.g., ["Unknown"])

    Returns:
        Tuple of (all_records, specialty_counts)
    """
    if skip_specialties is None:
        skip_specialties = ["Unknown"]

    pubmed_path = Path(pubmed_dir)
    if not pubmed_path.exists():
        raise FileNotFoundError(f"PubMed data directory not found: {pubmed_dir}")

    # Find all JSONL files
    jsonl_files = list(pubmed_path.glob("*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {pubmed_dir}")

    print(f"Found {len(jsonl_files)} specialty files")

    deduplicator = Deduplicator()
    all_records = []
    specialty_counts = defaultdict(int)

    # Process each specialty file
    for filepath in tqdm(jsonl_files, desc="Processing specialties"):
        specialty = filepath.stem  # Filename without extension

        # Skip specified specialties
        if specialty in skip_specialties:
            print(f"Skipping {specialty}")
            continue

        records = process_specialty_file(
            filepath=str(filepath),
            specialty=specialty,
            sample_size=sample_per_specialty,
            deduplicator=deduplicator,
            min_chars=min_chars,
            max_chars=max_chars,
            year_min=year_min,
            seed=seed
        )

        specialty_counts[specialty] = len(records)
        all_records.extend(records)

    return all_records, dict(specialty_counts)


# ---------------------- Output ----------------------

def write_jsonl(records: List[Dict[str, Any]], output_path: str) -> int:
    """Write records to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            f.write(line)

    return len(records)


def write_manifest(
    output_dir: str,
    total_count: int,
    specialty_counts: Dict[str, int],
    params: Dict[str, Any]
):
    """Write manifest file."""
    manifest = {
        "counts": {
            "pubmed": total_count,
            "specialties": specialty_counts
        },
        "total": total_count,
        "params": params
    }

    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build MedCorp from local PubMed data with sampling"
    )

    # Input/Output
    parser.add_argument(
        "--pubmed-dir",
        type=str,
        default=PUBMED_DATA_DIR,
        help="Path to directory containing PubMed specialty JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="medcorp_from_local_pubmed",
        help="Output directory for processed data"
    )

    # Sampling
    parser.add_argument(
        "--sample-per-specialty",
        type=int,
        default=100,
        help="Maximum number of records to sample per specialty (use -1 for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Filters
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Minimum character count for abstracts"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Maximum character count for abstracts (0 = no limit)"
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=2000,
        help="Minimum publication year (0 = no filter)"
    )

    # Specialties to skip
    parser.add_argument(
        "--skip-specialties",
        type=str,
        nargs="*",
        default=["Unknown"],
        help="List of specialties to skip (e.g., Unknown)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Building MedCorp from Local PubMed Data")
    print("=" * 60)
    print(f"Input directory: {args.pubmed_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample per specialty: {args.sample_per_specialty if args.sample_per_specialty > 0 else 'all'}")
    print(f"Min chars: {args.min_chars}")
    print(f"Max chars: {args.max_chars if args.max_chars > 0 else 'no limit'}")
    print(f"Min year: {args.year_min if args.year_min > 0 else 'no filter'}")
    print(f"Skipping specialties: {args.skip_specialties}")
    print("=" * 60)

    # Process all specialties
    records, specialty_counts = process_all_specialties(
        pubmed_dir=args.pubmed_dir,
        sample_per_specialty=args.sample_per_specialty if args.sample_per_specialty > 0 else float('inf'),
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        year_min=args.year_min,
        seed=args.seed,
        skip_specialties=args.skip_specialties
    )

    # Write output
    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    print(f"\nWriting {len(records)} records to {output_path}")
    written = write_jsonl(records, output_path)

    # Write manifest
    params = {
        "sample_per_specialty": args.sample_per_specialty,
        "seed": args.seed,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "year_min": args.year_min,
        "skip_specialties": args.skip_specialties
    }

    manifest_path = write_manifest(
        output_dir=args.output_dir,
        total_count=written,
        specialty_counts=specialty_counts,
        params=params
    )

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total records written: {written}")
    print(f"Number of specialties: {len(specialty_counts)}")
    print("\nRecords per specialty:")
    for specialty, count in sorted(specialty_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {specialty}: {count}")
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
