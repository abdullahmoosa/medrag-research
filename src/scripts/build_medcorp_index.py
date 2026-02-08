#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid retriever for MedCorp samples using Ollama embeddings + FAISS + BM25.

Kept:
- Safe JSONL reader; directory scan helper
- Title-boosted BM25 tokenization
- Token-based chunking at build time (--chunk-tokens/--overlap-tokens/--hf-tokenizer)
- Evidence packing to a token budget at search time (--max-evidence-tokens)
- Option-aware retrieval for MCQ: `search-mcq` subcommand (question + A/B/C/D)

NOTE: Query expansion has been removed from this file.
If you want to use expansions, construct the expanded query in your eval script
and pass the final string to --query (or to --question/--A..--D for search-mcq).

Examples:
    # Build with default MedCPT models
    python src/scripts/build_medcorp_index.py build --data medcorp_sample_20k --index-dir indexes/medcorp_hybrid
    
    # Build with MedEmbed model
    python src/scripts/build_medcorp_index.py build --data medcorp_sample_20k --index-dir indexes/medcorp_medembed --medembed
    
    # Search with MedEmbed
    python src/scripts/build_medcorp_index.py search --index-dir indexes/medcorp_medembed --query "treatment for diabetes" --medembed
"""

import os, sys, json, pickle, argparse, unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests

try:
    import orjson as fastjson
    def dumps(obj): return fastjson.dumps(obj).decode("utf-8")
    def loads(s):  return fastjson.loads(s)
except Exception:
    def dumps(obj): return json.dumps(obj, ensure_ascii=False)
    def loads(s):  return json.loads(s)

# FAISS
import faiss

# BM25
from rank_bm25 import BM25Okapi

# Tokenizer for token-based chunking + budget packing
from transformers import AutoTokenizer

# -----------------------------
# Utilities
# -----------------------------
def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())

def default_tokenizer(s: str) -> List[str]:
    s = s.lower()
    return [t for t in s.split() if t.isalpha() or any(ch.isalnum() for ch in t)]

def chunk_text_chars(txt: str, chunk_chars: int, overlap: int) -> List[str]:
    if chunk_chars <= 0:
        return [txt]
    out = []
    n = len(txt)
    i = 0
    step = max(1, chunk_chars - overlap)
    while i < n:
        out.append(txt[i:i+chunk_chars])
        i += step
    return out

def chunk_by_tokens(text: str, tokenizer, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    if chunk_tokens <= 0:
        return [text]
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    step = max(1, chunk_tokens - overlap_tokens)
    for i in range(0, len(ids), step):
        piece = ids[i:i+chunk_tokens]
        if not piece: break
        out.append(tokenizer.decode(piece))
        if i + chunk_tokens >= len(ids):
            break
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def rrf_fuse(rank_lists: List[List[Tuple[int, float]]], rrf_k: int = 60, topk: int = 10) -> List[Tuple[int, float]]:
    scores = {}
    for lst in rank_lists:
        for rank, (doc_id, _) in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]

def count_tokens(s: str, tokenizer) -> int:
    return len(tokenizer.encode(s, add_special_tokens=False))

def pack_passages(passages: List[Dict[str, Any]], tokenizer, max_evidence_tokens: int) -> List[Dict[str, Any]]:
    if max_evidence_tokens <= 0:
        return passages
    kept, used = [], 0
    for p in passages:
        t = count_tokens(p.get("text", ""), tokenizer)
        if used + t <= max_evidence_tokens:
            kept.append(p); used += t
        if used >= max_evidence_tokens:
            break
    return kept

def read_jsonl_to_df(path: str) -> pd.DataFrame:
    """Robust JSONL reader; skips malformed lines."""
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                recs.append(loads(s))
            except Exception:
                continue
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame.from_records(recs)

def collect_data_files(data_arg: str) -> List[str]:
    files = []
    if os.path.isdir(data_arg):
        # Only look for medcorp_sample.jsonl specifically
        target_file = os.path.join(data_arg, "medcorp_sample.jsonl")
        if os.path.isfile(target_file):
            files.append(target_file)
    else:
        if os.path.isfile(data_arg):
            files = [data_arg]
    if not files:
        raise FileNotFoundError(f"No medcorp_sample.jsonl file found at: {data_arg}")
    return files

# -----------------------------
# Embedding Clients
# -----------------------------
@dataclass
class OllamaClient:
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout: float = 120.0
    model_doc: str = "oscardp96/medcpt-article:latest"   # doc encoder
    model_query: Optional[str] = None                    # query encoder (defaults to model_doc)
    max_workers: int = 8                                 # concurrent HTTP requests

    def embed_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Robust embedding fetch.
        Ollama embeddings endpoint currently returns a single 'embedding' for a single prompt.
        Some versions may support 'embeddings' for batch; if not, fall back to per-text calls.
        """
        model = self.model_query if (is_query and self.model_query) else self.model_doc
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        results: List[np.ndarray] = []

        def _parse(data: Dict[str, Any]) -> np.ndarray:
            if "embedding" in data and isinstance(data["embedding"], list):
                return np.array(data["embedding"], dtype=np.float32)
            if "embeddings" in data and isinstance(data["embeddings"], list):
                # Newer multi-return format: take each item
                if len(data["embeddings"]) == 1 and "embedding" in data["embeddings"][0]:
                    return np.array(data["embeddings"][0]["embedding"], dtype=np.float32)
                # If truly batched, we'll handle outside
            if "data" in data and isinstance(data["data"], list):
                if data["data"] and "embedding" in data["data"][0]:
                    return np.array(data["data"][0]["embedding"], dtype=np.float32)
            raise ValueError(f"Unexpected embedding response keys: {list(data.keys())}")

        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Try single batched call first (if server supports list input via 'prompt' concatenation) ONLY when len==1
        if len(texts) == 1:
            payload = {"model": model, "prompt": texts[0]}
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            try:
                emb = _parse(data)
            except Exception as e:
                raise RuntimeError(f"Failed to parse embedding response (single) for model={model}: {data}") from e
            results.append(emb)
        else:
            # Parallel per-text requests
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _one(t: str):
                payload = {"model": model, "prompt": t}
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return _parse(data)
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                fut_map = {ex.submit(_one, t): t for t in texts}
                for fut in as_completed(fut_map):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        t = fut_map[fut]
                        raise RuntimeError(f"Embedding fetch failed for model={model}: text_len={len(t)} error={e}") from e

        if not results:
            raise RuntimeError("No embeddings returned (empty results).")
        dim_set = {r.shape[0] for r in results if r.ndim == 1}
        if len(dim_set) != 1:
            raise RuntimeError(f"Inconsistent embedding dims: {dim_set}")
        return np.vstack([r if r.ndim == 2 else r.reshape(1, -1) for r in results]).astype(np.float32)

@dataclass
class MedEmbedClient:
    model_name: str = "abhinand/MedEmbed-large-v0.1"
    device: str = "auto"  # auto, cpu, cuda
    
    def __post_init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Auto-detect device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"Loading MedEmbed model on GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                else:
                    self.device = "cpu"
                    print("CUDA not available, loading MedEmbed model on CPU...")
            else:
                print(f"Loading MedEmbed model on {self.device}...")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"✅ MedEmbed model loaded on GPU. Memory allocated: {allocated:.2f} GB")
            else:
                print(f"✅ MedEmbed model loaded on CPU")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for MedEmbed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings for a batch of texts using MedEmbed."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Important for cosine similarity
        )
        return embeddings.astype(np.float32)

# -----------------------------
# Index Builder / Searcher
# -----------------------------
class HybridIndex:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        ensure_dir(index_dir)
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

    # ---------- Build ----------
    def build(self,
              data_paths: List[str],
              embed_client: OllamaClient,
              chunk_chars: int = 0,
              chunk_overlap: int = 0,
              chunk_tokens: int = 0,
              overlap_tokens: int = 50,
              hf_tokenizer: str = "deepseek-ai/DeepSeek-R1",
              batch_size: int = 64,
              index_type: str = "flat",   # flat | hnsw
              build_bm25: bool = True):
        # 1) Load rows into dataframe
        dfs = []
        for p in data_paths:
            if p.endswith(".parquet"):
                dfs.append(pd.read_parquet(p))
            elif p.endswith(".jsonl"):
                dfs.append(read_jsonl_to_df(p))
        if not dfs:
            raise FileNotFoundError("No valid inputs after scanning.")
        df = pd.concat(dfs, ignore_index=True)

        for col in ["id", "source", "title", "text", "url"]:
            if col not in df.columns:
                df[col] = ""
        df = df.astype({"id":"string","source":"string","title":"string","url":"string"})
        df["text"] = df["text"].astype(str)

        # Tokenizer (for token-based chunking)
        tok = None
        if chunk_tokens > 0:
            tok = AutoTokenizer.from_pretrained(hf_tokenizer, use_fast=True)

        # Optional (re)chunk
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing chunks"):
            text = norm_text(row["text"])
            if not text:
                continue
            if chunk_tokens > 0:
                chunks = chunk_by_tokens(text, tok, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            elif chunk_chars > 0:
                chunks = chunk_text_chars(text, chunk_chars, chunk_overlap)
            else:
                chunks = [text]
            for idx, ch in enumerate(chunks):
                records.append({
                    "doc_id": row["id"] if idx == 0 else f'{row["id"]}#{idx}',
                    "source": row["source"],
                    "title": row["title"],
                    "url": row["url"],
                    "text": ch,
                })

        self.df = pd.DataFrame.from_records(records)
        if self.df.empty:
            raise ValueError("No records to index.")

        # 2) Embeddings -> FAISS
        texts = self.df["text"].tolist()
        vecs_list = []
        
        # Detect client type and show appropriate message
        if isinstance(embed_client, MedEmbedClient):
            print(f"Embedding {len(texts)} passages with MedEmbed model: {embed_client.model_name}")
        else:
            print(f"Embedding {len(texts)} passages with Ollama model: {embed_client.model_doc}")
            
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            
            # For Ollama, split large batches into smaller concurrent requests
            if isinstance(embed_client, OllamaClient) and len(batch) > 32:
                # Split into sub-batches for concurrent processing
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                async def process_ollama_batch_concurrent(batch_texts):
                    sub_batch_size = 32  # Optimal for Ollama API
                    tasks = []
                    
                    def embed_sub_batch(sub_batch):
                        return embed_client.embed_batch(sub_batch, is_query=False)
                    
                    with ThreadPoolExecutor(max_workers=8) as executor:  # Use 8 workers for your 16-core CPU
                        loop = asyncio.get_event_loop()
                        for j in range(0, len(batch_texts), sub_batch_size):
                            sub_batch = batch_texts[j:j+sub_batch_size]
                            task = loop.run_in_executor(executor, embed_sub_batch, sub_batch)
                            tasks.append(task)
                        
                        results = await asyncio.gather(*tasks)
                    
                    return np.vstack(results)
                
                # Run concurrent processing
                emb = asyncio.run(process_ollama_batch_concurrent(batch))
            else:
                # Normal processing for MedEmbed or small batches
                emb = embed_client.embed_batch(batch, is_query=False)
                
            if self.dim is None:
                self.dim = int(emb.shape[1])
            vecs_list.append(emb)
        all_vecs = np.vstack(vecs_list).astype(np.float32)
        # Validate dimension before proceeding
        if self.dim is None or self.dim <= 0 or all_vecs.shape[1] != self.dim:
            raise RuntimeError(f"Invalid embedding dimension detected (self.dim={self.dim}, all_vecs.shape={all_vecs.shape}). Aborting index save.")
        
        # Normalize for cosine similarity (MedEmbed already normalizes, but safe to do again)
        faiss.normalize_L2(all_vecs)

        if index_type == "hnsw":
            m = 32
            idx = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 200
        else:
            idx = faiss.IndexFlatIP(self.dim)
        idx.add(all_vecs)
        self.faiss_index = idx

        # 3) BM25 with title boost
        if build_bm25:
            print("Building BM25 (rank_bm25, title-boost x3) ...")
            def bm25_tokens(row):
                tx = default_tokenizer(row["text"])
                tt = default_tokenizer(row["title"])
                return tt + tt + tt + tx  # 3x title weight
            tokenized = [bm25_tokens(r) for _, r in self.df.iterrows()]
            self.bm25 = BM25Okapi(tokenized)
            self.bm25_ids = list(range(len(tokenized)))

        # 4) Persist
        print("Saving artifacts ...")
        faiss.write_index(self.faiss_index, self.faiss_path)
        self.df.to_parquet(self.docstore_path, index=False)
        with open(self.idmap_path, "w", encoding="utf-8") as f:
            for i, row in self.df.iterrows():
                out = {
                    "row": int(i),
                    "doc_id": row["doc_id"],
                    "source": row["source"],
                    "title": row["title"],
                    "url": row["url"],
                }
                f.write(dumps(out) + "\n")
        # Save metadata with client-specific information
        if isinstance(embed_client, MedEmbedClient):
            meta = {
                "dim": self.dim,
                "n_docs": int(self.faiss_index.ntotal),
                "index_type": index_type,
                "embed_model_type": "medembed",
                "embed_model_doc": embed_client.model_name,
                "embed_model_query": embed_client.model_name,
                "bm25": bool(build_bm25),
                "chunk_tokens": chunk_tokens,
                "overlap_tokens": overlap_tokens,
                "chunk_chars": chunk_chars,
                "chunk_overlap": chunk_overlap,
                "hf_tokenizer": hf_tokenizer,
            }
        else:  # OllamaClient
            meta = {
                "dim": self.dim,
                "n_docs": int(self.faiss_index.ntotal),
                "index_type": index_type,
                "embed_model_type": "ollama",
                "embed_model_doc": embed_client.model_doc,
                "embed_model_query": embed_client.model_query or embed_client.model_doc,
                "bm25": bool(build_bm25),
                "chunk_tokens": chunk_tokens,
                "overlap_tokens": overlap_tokens,
                "chunk_chars": chunk_chars,
                "chunk_overlap": chunk_overlap,
                "hf_tokenizer": hf_tokenizer,
            }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            f.write(dumps(meta))
        if self.bm25 is not None:
            with open(self.bm25_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "ids": self.bm25_ids}, f)
        print("Done.")

    # ---------- Load ----------
    def load(self):
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Missing meta at {self.meta_path}")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = loads(f.read())
        self.dim = int(meta["dim"])
        self.faiss_index = faiss.read_index(self.faiss_path)
        self.df = pd.read_parquet(self.docstore_path)
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                obj = pickle.load(f)
                self.bm25 = obj["bm25"]
                self.bm25_ids = obj["ids"]

    # ---------- Search ----------
    def search(self,
               query: str,
               embed_client,  # Can be OllamaClient or MedEmbedClient
               k: int = 10,
               mode: str = "hybrid",          # hybrid|dense|bm25
               rrf_k: int = 60,
               dense_k: int = 50,
               bm25_k: int = 200) -> List[Dict[str, Any]]:
        q = norm_text(query)

        dense_list = []
        if mode in ("hybrid", "dense"):
            q_emb = embed_client.embed_batch([q], is_query=True)[0].astype(np.float32)
            faiss.normalize_L2(q_emb.reshape(1, -1))
            sims, ids = self.faiss_index.search(q_emb.reshape(1, -1), dense_k)
            sims = sims[0]; ids = ids[0]
            dense_list = [(int(i), float(s)) for i, s in zip(ids, sims) if i != -1]

        bm25_list = []
        if mode in ("hybrid", "bm25"):
            if self.bm25 is None:
                raise RuntimeError("BM25 index not available; rebuild without --no-bm25")
            scores = self.bm25.get_scores(default_tokenizer(q))
            top_idx = np.argsort(-scores)[:bm25_k]
            bm25_list = [(int(i), float(scores[i])) for i in top_idx]

        if mode == "dense":
            results = [(i, s) for (i, s) in dense_list[:k]]
        elif mode == "bm25":
            results = sorted(bm25_list, key=lambda x: x[1], reverse=True)[:k]
        else:
            fused = rrf_fuse([dense_list, bm25_list], rrf_k=rrf_k, topk=k)
            results = fused

        out = []
        for row_idx, score in results:
            rec = self.df.iloc[row_idx].to_dict()
            rec["score"] = float(score)
            out.append(rec)
        return out

# -----------------------------
# Option-aware retrieval (MCQ)
# -----------------------------
def option_aware_search(idx: HybridIndex, question: str, options: List[str],
                        embed_client, k_per: int = 40, fuse_topk: int = 12,  # Can be OllamaClient or MedEmbedClient
                        rrf_k: int = 60, dense_k: int = 80, bm25_k: int = 400) -> List[Dict[str, Any]]:
    base = idx.search(question, embed_client, k=k_per, mode="hybrid",
                      rrf_k=rrf_k, dense_k=dense_k, bm25_k=bm25_k)

    def rows(lst):
        out = []
        for r in lst:
            row = int(idx.df.index[idx.df["doc_id"] == r["doc_id"]][0])
            out.append((row, r.get("score", 0.0)))
        return out

    all_rank_lists = [rows(base)]
    for opt in options:
        q = f"{question}\nOption: {opt}"
        lst = idx.search(q, embed_client, k=k_per, mode="hybrid",
                         rrf_k=rrf_k, dense_k=dense_k, bm25_k=bm25_k)
        all_rank_lists.append(rows(lst))

    fused = rrf_fuse(all_rank_lists, rrf_k=rrf_k, topk=fuse_topk)
    out = []
    for row_idx, score in fused:
        rec = idx.df.iloc[row_idx].to_dict()
        rec["score"] = float(score)
        out.append(rec)
    return out

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(prog="build_medcorp_index.py")
    sub = ap.add_subparsers(dest="cmd", required=True, help="Commands")

    # Build
    b = sub.add_parser("build", help="Build FAISS + optional BM25 indexes from MedCorp sample")
    b.add_argument("--data", type=str, required=True,
                   help="Path to medcorp_sample.jsonl/.parquet or a directory with such files.")
    b.add_argument("--index-dir", type=str, required=True)
    b.add_argument("--embed-model-doc", type=str, default="oscardp96/medcpt-article:latest",
                   help="Document embedding model. Options: oscardp96/medcpt-article:latest, abhinand/MedEmbed-large-v0.1")
    b.add_argument("--embed-model-query", type=str, default="oscardp96/medcpt-query:latest",
                   help="Query embedding model. Options: oscardp96/medcpt-query:latest, abhinand/MedEmbed-large-v0.1. Stored only in meta; searching uses the query flag in search commands.")
    b.add_argument("--ollama-base-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    b.add_argument("--medembed", action="store_true", 
                   help="Use MedEmbed-large-v0.1 for both doc and query embeddings (shortcut for setting both embed models)")
    b.add_argument("--model-preset", type=str, choices=["medcpt", "medembed"], 
                   help="Use a model preset: 'medcpt' (default MedCPT models) or 'medembed' (MedEmbed-large-v0.1)")
    # chunking
    b.add_argument("--chunk-chars", type=int, default=0, help="Character window; 0 keeps original chunks.")
    b.add_argument("--chunk-overlap", type=int, default=100)
    b.add_argument("--chunk-tokens", type=int, default=0, help="Token window size for token-based chunking.")
    b.add_argument("--overlap-tokens", type=int, default=50)
    b.add_argument("--hf-tokenizer", type=str, default="deepseek-ai/DeepSeek-R1")
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--embed-workers", type=int, default=8, help="Concurrent embedding HTTP requests for Ollama models")
    b.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                   help="Device for MedEmbed model: auto (GPU if available), cpu, or cuda")
    b.add_argument("--index-type", type=str, default="flat", choices=["flat", "hnsw"])
    b.add_argument("--no-bm25", action="store_true", help="Skip BM25 build.")
    b.add_argument("--smoke-query", type=str, default=None)
    b.add_argument("--smoke-k", type=int, default=5)

    # Search
    s = sub.add_parser("search", help="Query an existing index (hybrid/dense/bm25)")
    s.add_argument("--index-dir", type=str, required=True)
    s.add_argument("--query", type=str, required=True)
    s.add_argument("--embed-model-query", type=str, default="oscardp96/medcpt-query:latest",
                   help="Query embedding model. Options: oscardp96/medcpt-query:latest, abhinand/MedEmbed-large-v0.1")
    s.add_argument("--ollama-base-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    s.add_argument("--medembed", action="store_true", 
                   help="Use MedEmbed-large-v0.1 for query embeddings (shortcut)")
    s.add_argument("--model-preset", type=str, choices=["medcpt", "medembed"], 
                   help="Use a model preset: 'medcpt' (default) or 'medembed'")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--mode", type=str, default="hybrid", choices=["hybrid","dense","bm25"])
    s.add_argument("--rrf-k", type=int, default=60)
    s.add_argument("--dense-k", type=int, default=60)
    s.add_argument("--bm25-k", type=int, default=300)
    s.add_argument("--max-evidence-tokens", type=int, default=0, help="Pack evidence to this token budget (0 disables).")
    s.add_argument("--hf-tokenizer", type=str, default="deepseek-ai/DeepSeek-R1")
    s.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                   help="Device for MedEmbed model: auto (GPU if available), cpu, or cuda")

    # Search MCQ (option-aware)
    m = sub.add_parser("search-mcq", help="Option-aware retrieval for MCQ")
    m.add_argument("--index-dir", type=str, required=True)
    m.add_argument("--question", type=str, required=True)
    m.add_argument("--A", type=str, required=True)
    m.add_argument("--B", type=str, required=True)
    m.add_argument("--C", type=str, required=True)
    m.add_argument("--D", type=str, required=True)
    m.add_argument("--embed-model-query", type=str, default="oscardp96/medcpt-query:latest",
                   help="Query embedding model. Options: oscardp96/medcpt-query:latest, abhinand/MedEmbed-large-v0.1")
    m.add_argument("--ollama-base-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    m.add_argument("--medembed", action="store_true", 
                   help="Use MedEmbed-large-v0.1 for query embeddings (shortcut)")
    m.add_argument("--model-preset", type=str, choices=["medcpt", "medembed"], 
                   help="Use a model preset: 'medcpt' (default) or 'medembed'")
    m.add_argument("--k", type=int, default=12, help="Returned fused top-k after RRF.")
    m.add_argument("--rrf-k", type=int, default=60)
    m.add_argument("--dense-k", type=int, default=80)
    m.add_argument("--bm25-k", type=int, default=400)
    m.add_argument("--max-evidence-tokens", type=int, default=1200)
    m.add_argument("--hf-tokenizer", type=str, default="deepseek-ai/DeepSeek-R1")
    m.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                   help="Device for MedEmbed model: auto (GPU if available), cpu, or cuda")

    args = ap.parse_args()

    # Handle model presets and MedEmbed shortcut flag
    if hasattr(args, 'model_preset') and args.model_preset:
        if args.model_preset == "medembed":
            if args.cmd == "build":
                args.embed_model_doc = "abhinand/MedEmbed-large-v0.1"
                args.embed_model_query = "abhinand/MedEmbed-large-v0.1"
            else:  # search or search-mcq
                args.embed_model_query = "abhinand/MedEmbed-large-v0.1"
        elif args.model_preset == "medcpt":
            if args.cmd == "build":
                args.embed_model_doc = "oscardp96/medcpt-article:latest"
                args.embed_model_query = "oscardp96/medcpt-query:latest"
            else:  # search or search-mcq
                args.embed_model_query = "oscardp96/medcpt-query:latest"
    
    # Handle MedEmbed shortcut flag (takes precedence over presets)
    if hasattr(args, 'medembed') and args.medembed:
        if args.cmd == "build":
            args.embed_model_doc = "abhinand/MedEmbed-large-v0.1"
            args.embed_model_query = "abhinand/MedEmbed-large-v0.1"
        else:  # search or search-mcq
            args.embed_model_query = "abhinand/MedEmbed-large-v0.1"

    # Build paths as absolute (Windows-safe)
    if args.cmd == "build":
        if not os.path.isabs(args.data):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
            args.data = os.path.join(project_root, args.data)
        if not os.path.isabs(args.index_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
            args.index_dir = os.path.join(project_root, args.index_dir)

        data_files = collect_data_files(args.data)
        
        # Create appropriate embedding client based on model
        if args.embed_model_doc == "abhinand/MedEmbed-large-v0.1":
            embed_client = MedEmbedClient(model_name=args.embed_model_doc, device=args.device)
        else:
            embed_client = OllamaClient(
                base_url=args.ollama_base_url,
                model_doc=args.embed_model_doc,
                max_workers=getattr(args, 'embed_workers', 8),
            )
            
        idx = HybridIndex(index_dir=args.index_dir)
        idx.build(
            data_paths=data_files,
            embed_client=embed_client,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            hf_tokenizer=args.hf_tokenizer,
            batch_size=args.batch_size,
            index_type=args.index_type,
            build_bm25=(not args.no_bm25),
        )
        if args.smoke_query:
            idx.load()
            # Create query client for smoke test
            if args.embed_model_query == "abhinand/MedEmbed-large-v0.1":
                embed_client_q = MedEmbedClient(model_name=args.embed_model_query, device=args.device)
            else:
                embed_client_q = OllamaClient(
                    base_url=args.ollama_base_url,
                    model_doc=args.embed_model_query,
                    model_query=args.embed_model_query,
                )
            hits = idx.search(args.smoke_query, embed_client_q, k=args.smoke_k, mode="hybrid")
            print("\n--- Smoke results ---")
            for h in hits:
                print(f"[{h['source']}] {h['title'][:90]}")
                if h.get('url'): print(h['url'])
                print(h['text'][:220].replace("\n"," ") + " ...")
                print(f"score={h['score']:.4f}\n")

    elif args.cmd == "search":
        if not os.path.isabs(args.index_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
            args.index_dir = os.path.join(project_root, args.index_dir)

        idx = HybridIndex(index_dir=args.index_dir)
        idx.load()
        
        # Create appropriate embedding client
        if args.embed_model_query == "abhinand/MedEmbed-large-v0.1":
            embed_client = MedEmbedClient(model_name=args.embed_model_query, device=args.device)
        else:
            embed_client = OllamaClient(
                base_url=args.ollama_base_url,
                model_doc=args.embed_model_query,
                model_query=args.embed_model_query,
            )
        hits = idx.search(
            args.query, embed_client,
            k=args.k, mode=args.mode,
            rrf_k=args.rrf_k, dense_k=args.dense_k, bm25_k=args.bm25_k,
        )
        if args.max_evidence_tokens > 0:
            tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, use_fast=True)
            hits = pack_passages(hits, tok, args.max_evidence_tokens)

        for i, h in enumerate(hits, 1):
            print(f"{i:02d}. [{h['source']}] {h['title'][:100]}  (score={h['score']:.4f})")
            if h.get('url'): print("    " + h['url'])
            print("    " + h['text'][:240].replace("\n"," ") + " ...")

    elif args.cmd == "search-mcq":
        if not os.path.isabs(args.index_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
            args.index_dir = os.path.join(project_root, args.index_dir)

        idx = HybridIndex(index_dir=args.index_dir)
        idx.load()
        
        # Create appropriate embedding client
        if args.embed_model_query == "abhinand/MedEmbed-large-v0.1":
            embed_client = MedEmbedClient(model_name=args.embed_model_query, device=args.device)
        else:
            embed_client = OllamaClient(
                base_url=args.ollama_base_url,
                model_doc=args.embed_model_query,
                model_query=args.embed_model_query,
            )
        options = [args.A, args.B, args.C, args.D]
        fused = option_aware_search(
            idx, args.question, options, embed_client,
            k_per=max(args.dense_k, args.k), fuse_topk=args.k,
            rrf_k=args.rrf_k, dense_k=args.dense_k, bm25_k=args.bm25_k,
        )
        tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, use_fast=True)
        fused = pack_passages(fused, tok, args.max_evidence_tokens)

        print("\n--- Option-aware fused results ---")
        for i, h in enumerate(fused, 1):
            print(f"{i:02d}. [{h['source']}] {h['title'][:100]}  (score={h['score']:.4f})")
            if h.get('url'): print("    " + h['url'])
            print("    " + h['text'][:240].replace("\n"," ") + " ...")

if __name__ == "__main__":
    main()
