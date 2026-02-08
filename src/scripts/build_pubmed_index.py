#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build FAISS + BM25 hybrid index from local PubMed data.

This script creates a searchable index from PubMed JSONL data generated
by build_medcorp_from_local_pubmed.py. It includes:
- Character length filtering (min_chars, max_chars)
- Support for MedCPT (via Ollama) or MedEmbed embeddings
- FAISS dense retrieval index
- BM25 sparse retrieval index
- Title-boosted tokenization

Usage:
    # Build with MedCPT (default)
    python src/scripts/build_pubmed_index.py \\
        --input medcorp_local_pubmed/pubmed.jsonl \\
        --index-dir indexes/pubmed_medcpt \\
        --min-chars 300 \\
        --max-chars 6000

    # Build with MedEmbed
    python src/scripts/build_pubmed_index.py \\
        --input medcorp_local_pubmed/pubmed.jsonl \\
        --index-dir indexes/pubmed_medembed \\
        --medembed \\
        --min-chars 300 \\
        --max-chars 6000

    # Build with BGE
    python src/scripts/build_pubmed_index.py \\
        --input medcorp_local_pubmed/pubmed.jsonl \\
        --index-dir indexes/pubmed_bge \\
        --embed-model BAAI/bge-m3 \\
        --min-chars 300 \\
        --max-chars 6000
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

# FAISS
import faiss

# BM25
from rank_bm25 import BM25Okapi

# ---------------------- Utilities ----------------------

def norm_text(s: str) -> str:
    """Normalize text (NFC Unicode, strip whitespace)."""
    import unicodedata
    return unicodedata.normalize("NFC", (s or "").strip())

def default_tokenizer(s: str) -> List[str]:
    """Simple whitespace tokenizer with alphanumeric filtering."""
    s = s.lower()
    return [t for t in s.split() if t.isalpha() or any(ch.isalnum() for ch in t)]

def ensure_dir(p: str):
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file, skipping malformed lines."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------- Embedding Clients ----------------------

@dataclass
class OllamaClient:
    """Ollama-based embedding client (for MedCPT models)."""
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout: float = 120.0
    model_doc: str = "oscardp96/medcpt-article:latest"
    max_workers: int = 4  # Reduced from 8 to avoid overwhelming the server
    max_retries: int = 3
    retry_delay: float = 2.0

    def embed_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings using Ollama API with retry logic."""
        import requests
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        model = self.model_doc
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        results: List[np.ndarray] = []

        def _parse(data: Dict[str, Any]) -> np.ndarray:
            if "embedding" in data and isinstance(data["embedding"], list):
                return np.array(data["embedding"], dtype=np.float32)
            raise ValueError(f"Unexpected response: {list(data.keys())}")

        def _one(t: str):
            payload = {"model": model, "prompt": t}

            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    r = requests.post(url, json=payload, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    return _parse(data)
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"  [Retry {attempt + 1}/{self.max_retries}] text_len={len(t)} error={e}")
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise RuntimeError(f"Failed after {self.max_retries} retries for text_len={len(t)}: {e}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_one, t): (i, t) for i, t in enumerate(texts)}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    idx, t = futures[future]
                    raise RuntimeError(f"Failed for text [{idx}] len={len(t)}: {e}")

        if not results:
            raise RuntimeError("No embeddings returned")

        dim_set = {r.shape[0] for r in results if r.ndim == 1}
        if len(dim_set) != 1:
            raise RuntimeError(f"Inconsistent dims: {dim_set}")

        return np.vstack([r if r.ndim == 2 else r.reshape(1, -1) for r in results]).astype(np.float32)


@dataclass
class STEmbeddingClient:
    """SentenceTransformers client (for BGE and other ST models)."""
    model_name: str
    device: str = "auto"
    normalize: bool = True

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer
        import torch

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"✅ Model loaded")

    def embed_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings using SentenceTransformers."""
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        return vecs.astype(np.float32)


@dataclass
class MedEmbedClient:
    """MedEmbed client (specialized for medical text)."""
    model_name: str = "abhinand/MedEmbed-large-v0.1"
    device: str = "auto"

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer
        import torch

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading MedEmbed {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"✅ MedEmbed loaded on GPU. Memory: {allocated:.2f} GB")
        else:
            print(f"✅ MedEmbed loaded on CPU")

    def embed_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings using MedEmbed."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


# ---------------------- Index Builder ----------------------

class PubMedIndexBuilder:
    """Build FAISS + BM25 hybrid index from PubMed data."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        ensure_dir(index_dir)

        # File paths
        self.faiss_path = os.path.join(index_dir, "faiss.index")
        self.docstore_path = os.path.join(index_dir, "docstore.parquet")
        self.idmap_path = os.path.join(index_dir, "idmap.jsonl")
        self.bm25_path = os.path.join(index_dir, "bm25.pkl")
        self.meta_path = os.path.join(index_dir, "meta.json")

        self.faiss_index = None
        self.dim = None
        self.df = None
        self.bm25 = None
        self.bm25_ids = None

    def build(
        self,
        data_path: str,
        embed_client,
        min_chars: int = 300,
        max_chars: int = 6000,
        year_min: int = 2000,
        batch_size: int = 64,
        index_type: str = "flat",
        build_bm25: bool = True,
    ):
        """
        Build the hybrid index.

        Args:
            data_path: Path to PubMed JSONL file
            embed_client: Embedding client instance (OllamaClient, MedEmbedClient, etc.)
            min_chars: Minimum character count for abstracts
            max_chars: Maximum character count for abstracts (0 = no limit)
            year_min: Minimum publication year (0 = no filter)
            batch_size: Batch size for embedding generation
            index_type: FAISS index type ("flat" or "hnsw")
            build_bm25: Whether to build BM25 index
        """
        # 1. Load and filter data
        print(f"Loading data from {data_path}...")
        raw_records = read_jsonl(data_path)
        print(f"Loaded {len(raw_records)} raw records")

        # Filter records
        filtered_records = []
        for rec in tqdm(raw_records, desc="Filtering records"):
            text = rec.get("text", "")
            year = rec.get("meta", {}).get("year")

            # Length filter
            text_len = len(text)
            if text_len < min_chars:
                continue
            if max_chars > 0 and text_len > max_chars:
                continue

            # Year filter
            if year is not None and year_min > 0:
                try:
                    year_int = int(year)
                    if year_int < year_min:
                        continue
                except (ValueError, TypeError):
                    pass

            filtered_records.append(rec)

        print(f"After filtering: {len(filtered_records)} records")
        print(f"  - Filtered out: {len(raw_records) - len(filtered_records)} records")

        if not filtered_records:
            raise ValueError("No records passed the filters!")

        # 2. Prepare DataFrame for indexing
        records_for_index = []
        for rec in filtered_records:
            text = norm_text(rec.get("text", ""))
            if not text:
                continue

            records_for_index.append({
                "doc_id": rec.get("id", ""),
                "source": rec.get("source", "pubmed"),
                "title": rec.get("title", ""),
                "url": rec.get("url", ""),
                "text": text,
            })

        self.df = pd.DataFrame.from_records(records_for_index)

        if self.df.empty:
            raise ValueError("No valid records to index")

        print(f"Prepared {len(self.df)} documents for indexing")

        # 3. Generate embeddings and build FAISS index
        texts = self.df["text"].tolist()

        # Detect client type
        client_type = embed_client.__class__.__name__
        print(f"Generating embeddings using {client_type}...")

        vecs_list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            emb = embed_client.embed_batch(batch, is_query=False)

            if self.dim is None:
                self.dim = int(emb.shape[1])
            vecs_list.append(emb)

        all_vecs = np.vstack(vecs_list).astype(np.float32)

        # Validate dimension
        if self.dim <= 0 or all_vecs.shape[1] != self.dim:
            raise RuntimeError(f"Invalid dimension: {self.dim}")

        # Normalize for cosine similarity
        faiss.normalize_L2(all_vecs)

        # Create FAISS index
        if index_type == "hnsw":
            m = 32
            idx = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 200
        else:
            idx = faiss.IndexFlatIP(self.dim)

        idx.add(all_vecs)
        self.faiss_index = idx

        print(f"✅ FAISS index built: {self.faiss_index.ntotal} vectors, dim={self.dim}")

        # 4. Build BM25 index
        if build_bm25:
            print("Building BM25 index (title-boosted)...")

            def bm25_tokens(row):
                tx = default_tokenizer(row["text"])
                tt = default_tokenizer(row["title"])
                return tt + tt + tt + tx  # 3x title weight

            tokenized = [bm25_tokens(r) for _, r in self.df.iterrows()]
            self.bm25 = BM25Okapi(tokenized)
            self.bm25_ids = list(range(len(tokenized)))

            print("✅ BM25 index built")

        # 5. Save all artifacts
        print("Saving index artifacts...")
        self._save_metadata(embed_client, index_type, build_bm25, min_chars, max_chars, year_min)

        # Save FAISS index
        faiss.write_index(self.faiss_index, self.faiss_path)
        print(f"  - Saved FAISS index: {self.faiss_path}")

        # Save docstore
        self.df.to_parquet(self.docstore_path, index=False)
        print(f"  - Saved docstore: {self.docstore_path}")

        # Save ID map
        with open(self.idmap_path, "w", encoding="utf-8") as f:
            for i, row in self.df.iterrows():
                out = {
                    "row": int(i),
                    "doc_id": row["doc_id"],
                    "source": row["source"],
                    "title": row["title"],
                    "url": row["url"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  - Saved ID map: {self.idmap_path}")

        # Save BM25
        if self.bm25 is not None:
            with open(self.bm25_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "ids": self.bm25_ids}, f)
            print(f"  - Saved BM25: {self.bm25_path}")

        print(f"\n✅ Index successfully built at: {self.index_dir}")

    def _save_metadata(self, embed_client, index_type: str, build_bm25: bool,
                      min_chars: int, max_chars: int, year_min: int):
        """Save metadata about the index."""
        # Detect model info
        if isinstance(embed_client, MedEmbedClient):
            model_type = "medembed"
            model_doc = embed_client.model_name
        elif isinstance(embed_client, STEmbeddingClient):
            model_type = "sentence_transformers"
            model_doc = embed_client.model_name
        else:
            model_type = "ollama"
            model_doc = embed_client.model_doc

        meta = {
            "dim": self.dim,
            "n_docs": int(self.faiss_index.ntotal) if self.faiss_index else 0,
            "index_type": index_type,
            "embed_model_type": model_type,
            "embed_model_doc": model_doc,
            "bm25": build_bm25,
            "filters": {
                "min_chars": min_chars,
                "max_chars": max_chars,
                "year_min": year_min,
            }
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 hybrid index from PubMed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with MedCPT (default)
  python %(prog)s --input medcorp_local_pubmed/pubmed.jsonl --index-dir indexes/pubmed_medcpt

  # Build with MedEmbed
  python %(prog)s --input medcorp_local_pubmed/pubmed.jsonl --index-dir indexes/pubmed_medembed --medembed

  # Build with BGE
  python %(prog)s --input medcorp_local_pubmed/pubmed.jsonl --index-dir indexes/pubmed_bge --embed-model BAAI/bge-m3

  # With custom filters
  python %(prog)s --input medcorp_local_pubmed/pubmed.jsonl --index-dir indexes/pubmed_custom \\
                  --min-chars 500 --max-chars 4000 --year-min 2010
        """
    )

    # Input/output
    parser.add_argument("--input", type=str, required=True,
                       help="Path to PubMed JSONL file")
    parser.add_argument("--index-dir", type=str, required=True,
                       help="Output directory for index")

    # Filtering
    parser.add_argument("--min-chars", type=int, default=300,
                       help="Minimum character count for abstracts (default: 300)")
    parser.add_argument("--max-chars", type=int, default=6000,
                       help="Maximum character count for abstracts (0 = no limit, default: 6000)")
    parser.add_argument("--year-min", type=int, default=2000,
                       help="Minimum publication year (0 = no filter, default: 2000)")

    # Embedding model
    parser.add_argument("--embed-model", type=str, default="oscardp96/medcpt-article:latest",
                       help="Embedding model: MedCPT, MedEmbed, or BGE")
    parser.add_argument("--medembed", action="store_true",
                       help="Use MedEmbed-large-v0.1 (shortcut)")
    parser.add_argument("--model-preset", type=str, choices=["medcpt", "medembed", "bge"],
                       help="Use a model preset")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device for local models (auto, cpu, cuda)")

    # Ollama settings
    parser.add_argument("--ollama-base-url", type=str,
                       default=os.environ.get("OLLAMA_BASE_URL", "http://172.25.208.1:11434"),
                       help="Ollama base URL for MedCPT")

    # Index settings
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding generation (lower for Ollama: 16-32)")
    parser.add_argument("--embed-workers", type=int, default=4,
                       help="Number of parallel workers for Ollama embedding (default: 4)")
    parser.add_argument("--index-type", type=str, default="flat", choices=["flat", "hnsw"],
                       help="FAISS index type")
    parser.add_argument("--no-bm25", action="store_true",
                       help="Skip BM25 index construction")

    # Test query
    parser.add_argument("--smoke-query", type=str, default=None,
                       help="Test query to verify index (optional)")
    parser.add_argument("--smoke-k", type=int, default=5,
                       help="Number of results for smoke test")

    args = parser.parse_args()

    # Handle model presets
    if args.model_preset:
        if args.model_preset == "medembed":
            args.embed_model = "abhinand/MedEmbed-large-v0.1"
        elif args.model_preset == "medcpt":
            args.embed_model = "oscardp96/medcpt-article:latest"
        elif args.model_preset == "bge":
            args.embed_model = "BAAI/bge-m3"

    if args.medembed:
        args.embed_model = "abhinand/MedEmbed-large-v0.1"

    # Create embedding client
    print("=" * 60)
    print("PubMed Index Builder")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.index_dir}")
    print(f"Model: {args.embed_model}")
    print(f"Filters: min_chars={args.min_chars}, max_chars={args.max_chars}, year_min={args.year_min}")
    print("=" * 60)

    if args.embed_model == "abhinand/MedEmbed-large-v0.1":
        embed_client = MedEmbedClient(model_name=args.embed_model, device=args.device)
    elif "bge" in args.embed_model.lower() or args.embed_model.startswith("BAAI/"):
        embed_client = STEmbeddingClient(model_name=args.embed_model, device=args.device, normalize=True)
    else:
        # Use Ollama for MedCPT
        embed_client = OllamaClient(
            base_url=args.ollama_base_url,
            model_doc=args.embed_model,
            max_workers=args.embed_workers,
        )

    # Build index
    builder = PubMedIndexBuilder(index_dir=args.index_dir)
    builder.build(
        data_path=args.input,
        embed_client=embed_client,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        year_min=args.year_min,
        batch_size=args.batch_size,
        index_type=args.index_type,
        build_bm25=(not args.no_bm25),
    )

    # Smoke test
    if args.smoke_query:
        print("\n" + "=" * 60)
        print("Smoke Test")
        print("=" * 60)

        # Load index and test
        from src.scripts.build_medcorp_index import HybridIndex

        idx = HybridIndex(index_dir=args.index_dir)
        idx.load()

        # Create query client
        if args.embed_model == "abhinand/MedEmbed-large-v0.1":
            query_client = MedEmbedClient(model_name=args.embed_model, device=args.device)
        elif "bge" in args.embed_model.lower():
            query_client = STEmbeddingClient(model_name=args.embed_model, device=args.device, normalize=True)
        else:
            query_client = OllamaClient(
                base_url=args.ollama_base_url,
                model_doc=args.embed_model,
            )

        hits = idx.search(args.smoke_query, query_client, k=args.smoke_k, mode="hybrid")

        print(f"\nQuery: {args.smoke_query}\n")
        for i, h in enumerate(hits, 1):
            print(f"{i}. [{h['source']}] {h['title'][:90]}")
            if h.get('url'):
                print(f"   {h['url']}")
            print(f"   Score: {h['score']:.4f}")
            print(f"   {h['text'][:200].replace(chr(10), ' ')}...")
            print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
