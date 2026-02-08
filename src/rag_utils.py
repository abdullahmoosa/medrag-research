#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared RAG utilities for the medrag-research project.

Includes:
- Data loading and chunking helpers
- SentenceTransformer embedder wrapper
- FAISS/Numpy vector index with save/load
- BM25 and Hybrid retrievers
"""
import os
import re
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:
    BM25Okapi = None
    _HAS_BM25 = False

import torch
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEFAULT_KB_DIR = os.path.join(REPO_ROOT, "data", "pubmed")
DEFAULT_INDEX_DIR = os.path.join(REPO_ROOT, "models", "knowledge_base", "faiss_medcpt_pubmed")


# -----------------------------
# General helpers
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _normalize_model_id(model_id: str) -> str:
    # drop optional ":tag" (e.g., ":latest") for HF
    if ":" in model_id:
        return model_id.split(":", 1)[0]
    return model_id


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
        if not txt:
            return []
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass
        items = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items


def extract_text_fields(rec: Dict[str, Any]) -> str:
    # Heuristic: concatenate likely fields
    fields = []
    for k in [
        "title", "Title",
        "abstract", "Abstract",
        "text", "body", "content", "passage", "Body", "full_text", "article"
    ]:
        v = rec.get(k)
        if isinstance(v, str):
            fields.append(v)
        elif isinstance(v, list):
            fields.extend([x for x in v if isinstance(x, str)])
    if not fields:
        try:
            fields.append(json.dumps(rec, ensure_ascii=False))
        except Exception:
            pass
    text = "\n".join(fields)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_split(text: str) -> List[str]:
    sents = re.split(r"(?<=[\.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def chunk_text(text: str, target_chars: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]
    sents = sentence_split(text)
    chunks: List[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= target_chars:
            buf = buf + " " + s
        else:
            chunks.append(buf)
            if overlap > 0:
                tail = buf[-overlap:]
                tail = tail.split(" ", 1)
                tail = tail[-1] if len(tail) == 2 else tail[0]
                buf = (tail + " " + s).strip()
            else:
                buf = s
    if buf:
        chunks.append(buf)
    return chunks


# -----------------------------
# Embeddings + Vector Index
# -----------------------------

@dataclass
class Passage:
    text: str
    meta: Dict[str, Any]


class EmbeddingModel:
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = _normalize_model_id(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Map of known Ollama models
        ollama_models = {
            "oscardp96/medcpt-article": "oscardp96/medcpt-article:latest",
            "oscardp96/medcpt-query": "oscardp96/medcpt-query:latest",
            "kronos483/MedEmbed-large-v0.1": "kronos483/MedEmbed-large-v0.1:latest",
            "mxbai-embed-large": "mxbai-embed-large:latest",
            "nomic-embed-text": "nomic-embed-text:latest",
            "thewindmom/llama3-med42-8b": "thewindmom/llama3-med42-8b:latest"
        }
        
        # Check if this is an Ollama model
        if self.model_id in ollama_models or any(self.model_id.startswith(k) for k in ollama_models):
            try:
                import ollama
                self.use_ollama = True
                
                # Get the actual Ollama model name
                if self.model_id in ollama_models:
                    self.ollama_model = ollama_models[self.model_id]
                else:
                    # Try to match by prefix
                    for prefix, full_name in ollama_models.items():
                        if self.model_id.startswith(prefix):
                            self.ollama_model = full_name
                            break
                    else:
                        # If not found in mapping, use as-is and assume it exists in Ollama
                        self.ollama_model = self.model_id
                
                print(f"Using Ollama for embeddings with model: {self.ollama_model}")
                
                # Test connection to Ollama
                ollama.embeddings(model=self.ollama_model, prompt="test")
            except Exception as e:
                raise RuntimeError(
                    f"Error connecting to Ollama: {e}. Make sure Ollama is running and model '{self.ollama_model}' is available."
                )
        else:
            # Use SentenceTransformer
            self.use_ollama = False
            try:
                self.model = SentenceTransformer(self.model_id, device=self.device)
            except Exception as e:
                raise RuntimeError(f"Error loading SentenceTransformer model {self.model_id}: {e}")
            
        self.dim = self._detect_dim()

    def _detect_dim(self) -> int:
        emb = self.embed(["test"])
        return int(emb.shape[1])

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
            
        if self.use_ollama:
            import ollama
            # Ollama doesn't support batching internally, so handle it manually
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                for text in batch:
                    response = ollama.embeddings(model=self.ollama_model, prompt=text)
                    embedding = np.array(response['embedding'], dtype=np.float32)
                    # Normalize the embedding to unit length
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
            embs = np.stack(embeddings)
        else:
            # Use SentenceTransformer
            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32)
        return embs


class VectorIndex:
    def __init__(self, dim: int, use_faiss: bool = _HAS_FAISS):
        self.dim = dim
        self.use_faiss = use_faiss
        self.passages: List[Passage] = []
        self.embeddings: Optional[np.ndarray] = None
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dim)  # cosine if normalized
        else:
            self.index = None

    def add(self, passages: List[Passage], embs: np.ndarray):
        if not passages:
            return
        assert embs.shape[0] == len(passages), "Embeddings count must match passages"
        self.passages.extend(passages)
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])
        if self.use_faiss and self.index is not None:
            self.index.add(embs)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[float, Passage]]:
        if self.embeddings is None or (self.use_faiss and self.index is None):
            return []
        if self.use_faiss and self.index is not None:
            D, I = self.index.search(query_emb, top_k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
        else:
            vecs = self.embeddings  # (N, D)
            q = query_emb[0]
            sims = vecs @ q
            idxs = np.argsort(-sims)[:top_k].tolist()
            scores = [float(sims[i]) for i in idxs]
        res: List[Tuple[float, Passage]] = []
        for s, i in zip(scores, idxs):
            if i < 0 or i >= len(self.passages):
                continue
            res.append((s, self.passages[i]))
        return res

    def save(self, path: str):
        ensure_dir(path)
        meta = {
            "dim": self.dim,
            "use_faiss": self.use_faiss,
            "count": len(self.passages),
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(os.path.join(path, "passages.jsonl"), "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps({"text": p.text, "meta": p.meta}, ensure_ascii=False) + "\n")
        if self.embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        if self.use_faiss and self.index is not None and _HAS_FAISS:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))

    @staticmethod
    def load(path: str) -> "VectorIndex":
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        idx = VectorIndex(dim=meta.get("dim", 768), use_faiss=meta.get("use_faiss", False) and _HAS_FAISS)
        passages: List[Passage] = []
        with open(os.path.join(path, "passages.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    passages.append(Passage(text=rec.get("text", ""), meta=rec.get("meta", {})))
                except Exception:
                    continue
        idx.passages = passages
        emb_path = os.path.join(path, "embeddings.npy")
        if os.path.exists(emb_path):
            idx.embeddings = np.load(emb_path)
        faiss_path = os.path.join(path, "index.faiss")
        if idx.use_faiss and os.path.exists(faiss_path):
            idx.index = faiss.read_index(faiss_path)
        elif idx.use_faiss and idx.embeddings is not None and _HAS_FAISS:
            idx.index = faiss.IndexFlatIP(idx.dim)
            idx.index.add(idx.embeddings)
        return idx


# -----------------------------
# Index building from PubMed
# -----------------------------

def build_or_load_index(
    kb_dir: str,
    index_dir: str,
    embed_model_id: str,
    device: Optional[str] = None,
    chunk_chars: int = 800,
    chunk_overlap: int = 100,
    batch_size: int = 64,
) -> Tuple[VectorIndex, "EmbeddingModel"]:
    ensure_dir(index_dir)
    meta_path = os.path.join(index_dir, "meta.json")
    if os.path.exists(meta_path):
        emb_model = EmbeddingModel(embed_model_id, device=device)
        idx = VectorIndex.load(index_dir)
        if idx.dim != emb_model.dim:
            print("[RAG] Embedding dim changed; rebuilding index...")
        else:
            print(f"[RAG] Loaded index from {index_dir} with {len(idx.passages)} passages")
            return idx, emb_model

    emb_model = EmbeddingModel(embed_model_id, device=device)
    idx = VectorIndex(dim=emb_model.dim, use_faiss=_HAS_FAISS)

    files = sorted(glob.glob(os.path.join(kb_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {kb_dir}")

    passages_batch: List[Passage] = []
    texts_batch: List[str] = []
    total_chunks = 0

    for fp in files:
        data = load_json_or_jsonl(fp)
        for rec in data:
            text = extract_text_fields(rec)
            if not text:
                continue
            chunks = chunk_text(text, target_chars=chunk_chars, overlap=chunk_overlap)
            for ch in chunks:
                meta = {"source": os.path.basename(fp)}
                passages_batch.append(Passage(text=ch, meta=meta))
                texts_batch.append(ch)
            if len(texts_batch) >= 4096:
                embs = emb_model.embed(texts_batch, batch_size=batch_size)
                idx.add(passages_batch, embs)
                total_chunks += len(texts_batch)
                print(f"[RAG] Indexed {total_chunks} chunks...")
                passages_batch, texts_batch = [], []

    if texts_batch:
        embs = emb_model.embed(texts_batch, batch_size=batch_size)
        idx.add(passages_batch, embs)
        total_chunks += len(texts_batch)
        print(f"[RAG] Indexed {total_chunks} chunks total.")

    idx.save(index_dir)
    print(f"[RAG] Saved index to {index_dir}")
    return idx, emb_model


# -----------------------------
# Lexical + Hybrid Retrieval
# -----------------------------

def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, passages: List[Passage]):
        if not _HAS_BM25:
            raise ImportError("rank_bm25 is required. Please `pip install rank-bm25`." )
        self.passages = passages
        self.tokenized_corpus: List[List[str]] = [_simple_tokenize(p.text) for p in passages]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Passage]]:
        tokens = _simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(-scores)[:top_k].tolist()
        return [(float(scores[i]), self.passages[i]) for i in idxs]


class HybridRetriever:
    """
    Combines vector and BM25 results via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vector_index: VectorIndex, embedder: EmbeddingModel, bm25: Optional[BM25Retriever] = None):
        self.vector_index = vector_index
        self.embedder = embedder
        self.bm25 = bm25 if bm25 is not None else BM25Retriever(vector_index.passages)

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Tuple[float, Passage]]:
        # Vector results
        q_emb = self.embedder.embed([query])
        vec_hits = self.vector_index.search(q_emb, top_k=top_k)
        # BM25 results
        bm25_hits = self.bm25.search(query, top_k=top_k)

        # Build RRF scores
        pool: Dict[int, float] = {}
        # map passage id by identity index in vector_index.passages
        pid_map = {id(p): i for i, p in enumerate(self.vector_index.passages)}

        def apply_rrf(hits: List[Tuple[float, Passage]]):
            for rank, (_score, passage) in enumerate(hits, start=1):
                pid = pid_map.get(id(passage))
                if pid is None:
                    continue
                pool[pid] = pool.get(pid, 0.0) + 1.0 / (rrf_k + rank)

        apply_rrf(vec_hits)
        apply_rrf(bm25_hits)

        # Sort by fused score
        ranked = sorted(pool.items(), key=lambda x: -x[1])[:top_k]
        return [(score, self.vector_index.passages[pid]) for pid, score in ranked]
