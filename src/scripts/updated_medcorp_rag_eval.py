#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the current MedCorp RAG pipeline (Hybrid FAISS+BM25 with Ollama or MedEmbed query embeddings).

Changes in this version (speed-focused):
- Batched dense retrieval across MANY queries with one FAISS call per batch
- Optional FAISS GPU offload (index_cpu_to_all_gpus)
- BM25 top-k via argpartition (O(N) partial sort)
- BM25 scope control: base | all | none
- Cached tokenization for BM25
- Batched per-example fusion & evidence packing
- Avoid DataFrame.to_dict() on hot path
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import defaultdict
import re

import numpy as np
from functools import lru_cache

from tqdm import tqdm
from transformers import AutoTokenizer  # for token counting / evidence packing

# FAISS (CPU/GPU)
try:
    import faiss  # faiss-gpu if installed
except Exception:
    faiss = None

# Make project importable when run as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your existing hybrid index + embedding clients
from src.scripts.build_medcorp_index import (
    HybridIndex,
    OllamaClient,
    MedEmbedClient,   # <-- enable MedEmbed query embeddings
)

# ----------------------------- Small utils -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json_or_jsonl(p: str) -> List[Dict[str, Any]]:
    out = []
    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(json.loads(s))
                except Exception:
                    continue
    else:
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    out = obj
                else:
                    raise ValueError("JSON must be a list of examples")
        except json.JSONDecodeError:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        out.append(json.loads(s))
                    except Exception:
                        continue
    return out

LETTER_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}

def detect_dataset_format(ex: Dict[str, Any]) -> str:
    """Detect whether this is MedMCQA or MedQA format."""
    if "options" in ex and isinstance(ex["options"], dict):
        return "medqa"
    elif any(key in ex for key in ["opa", "opb", "opc", "opd"]):
        return "medmcqa"
    else:
        # Default fallback
        return "medmcqa"

def gold_letter(ex: Dict[str, Any]) -> Optional[str]:
    """Extract correct answer letter, supporting both MedMCQA and MedQA formats."""
    # MedQA format: answer_idx field
    answer_idx = ex.get("answer_idx")
    if answer_idx and isinstance(answer_idx, str) and answer_idx.upper() in "ABCDE":
        return answer_idx.upper()
    # MedMCQA format: cop field (1-4 mapping)
    cop = ex.get("cop")
    if isinstance(cop, int) and cop in (1,2,3,4):
        return LETTER_MAP[cop]
    # Generic fallbacks
    ans = ex.get("answer") or ex.get("gold")
    if isinstance(ans, str) and ans.upper() in "ABCDE":
        return ans.upper()
    return None

def make_options(ex: Dict[str, Any]) -> Dict[str, str]:
    """Extract options dict, supporting both MedMCQA and MedQA formats."""
    # MedQA format: options dict
    if "options" in ex and isinstance(ex["options"], dict):
        options_dict = ex["options"]
        normalized = {}
        for key in sorted(options_dict.keys()):
            if key.upper() in "ABCDE" and options_dict[key]:
                normalized[key.upper()] = str(options_dict[key]).strip()
        return normalized
    # MedMCQA format: opa, opb, opc, opd fields
    return {
        "A": (ex.get("opa") or "").strip(),
        "B": (ex.get("opb") or "").strip(),
        "C": (ex.get("opc") or "").strip(),
        "D": (ex.get("opd") or "").strip(),
    }

ANSWER_RE = re.compile(r"Answer\s*[:：]\s*([A-E])", re.I)
LETTER_RE  = re.compile(r"\b([A-E])\b")

def parse_letter(text: str, allowed_letters: List[str] = None) -> Optional[str]:
    """Parse answer letter from text, optionally restricting to allowed letters."""
    if allowed_letters is None:
        allowed_letters = ["A", "B", "C", "D", "E"]
    allowed_set = set(l.upper() for l in allowed_letters)

    t = (text or "").strip()

    # Look for "Answer: X" pattern
    m = ANSWER_RE.search(t)
    if m and m.group(1).upper() in allowed_set:
        return m.group(1).upper()

    # First character
    if t and t[0].upper() in allowed_set:
        return t[0].upper()

    # Any standalone letter
    m2 = LETTER_RE.search(t)
    if m2 and m2.group(1).upper() in allowed_set:
        return m2.group(1).upper()

    # Pattern like "C)" or "D."
    m3 = re.search(r"\b([A-E])[)\.]", t, re.I)
    if m3 and m3.group(1).upper() in allowed_set:
        return m3.group(1).upper()

    return None

def format_refs(passages: List[Dict[str, Any]], max_len_each: int = 320) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        title = (p.get("title") or "").strip()
        source = (p.get("source") or "").strip()
        url = (p.get("url") or "").strip()
        txt = (p.get("text") or "").replace("\n", " ").strip()
        if max_len_each and len(txt) > max_len_each:
            txt = txt[:max_len_each] + "..."
        head = f"[{source}] {title}" if title or source else ""
        tail = f" (URL: {url})" if url else ""
        lines.append(f"Reference {i}: {head}{tail}\n{txt}")
    return "\n\n".join(lines)

def build_prompt(question: str, options: Dict[str, str], refs: List[Dict[str, Any]]) -> str:
    opt_str = "\n".join([f"{k}) {v}" for k, v in sorted(options.items()) if v.strip()])
    letters = sorted(options.keys())
    allowed_letters = ", ".join(letters[:-1]) + f" or {letters[-1]}" if len(letters) > 1 else letters[0]

    rules_with_refs = (
        "You are a medical expert answering a single-best-answer MCQ.\n"
        "Use the references if they are relevant; if they appear off-topic or low-relevance, ignore them and answer from medical knowledge.\n"
        "Choose exactly one option.\n"
        "Output format: a single uppercase letter only, with no words, spaces, or punctuation.\n"
        f"Valid answers: {{{' '.join(letters)}}}\n"
        "If unsure, guess.\n"
    )

    rules_without_refs = (
        "You are a medical expert answering a single-best-answer MCQ.\n"
        "Choose exactly one option based on medical knowledge.\n"
        "Output format: a single uppercase letter only, with no words, spaces, or punctuation.\n"
        f"Valid answers: {{{' '.join(letters)}}}\n"
        "If unsure, guess.\n"
    )

    if refs:
        ctx = format_refs(refs)
        return (
            rules_with_refs
            + f"\n{ctx}\n\n"
            + f"Question: {question}\n\nOptions:\n{opt_str}\n\n"
            + f"Answer ({allowed_letters}): "
        )
    else:
        return (
            rules_without_refs
            + f"\nQuestion: {question}\n\nOptions:\n{opt_str}\n\n"
            + f"Answer ({allowed_letters}): "
        )

# ---------------------- Evidence packing + RRF + rescoring ----------------------

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

def rrf_fuse_with_votes(rank_lists: List[List[Tuple[int, float]]], rrf_k: int = 60, topk: int = 10) -> List[Tuple[int, float, int]]:
    """
    RRF fuse a list of rank lists (each list is [(row_idx, orig_score), ...]).
    Returns a list of (row_idx, rrf_score, vote_count) sorted by rrf_score desc and truncated to topk.
    """
    scores: Dict[int, float] = {}
    votes: Dict[int, int] = {}
    for lst in rank_lists:
        for rank, (row_idx, _orig) in enumerate(lst):
            scores[row_idx] = scores.get(row_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            votes[row_idx] = votes.get(row_idx, 0) + 1
    fused = [(rid, scores[rid], votes[rid]) for rid in scores]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:topk]

def rows_for_rerank(idx: HybridIndex, hits: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    out = []
    for h in hits:
        row = int(idx.df.index[idx.df["doc_id"] == h["doc_id"]][0])
        out.append((row, float(h.get("score", 0.0))))
    return out

_STOPWORDS = set("""
a an the and or for to of in on at from by with without into about as is are was were be been being that which who
whom whose this these those it its they them their do does did done doing have has had having not no yes can could
will would should may might must more most many few some any each every other another such than then when where
while how why patient patients month months year years day days week weeks male female man woman boy girl
""".split())

def _lex_overlap(q: str, t: str) -> float:
    import re
    def toks(x):
        return [w for w in re.findall(r"[a-zA-Z]+", (x or "").lower()) if w not in _STOPWORDS and len(w) > 2]
    qt, tt = set(toks(q)), set(toks(t))
    return 0.0 if not qt else len(qt & tt) / len(qt)

def _minmax(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

def fuse_rescore_filter(
    idx: HybridIndex,
    rank_lists: List[List[Tuple[int, float]]],
    question: str,
    topk: int,
    rrf_k: int,
    alpha_rrf: float,
    beta_overlap: float,
    bm25_weight: float = 0.0,
    dense_weight: float = 0.0,
    min_final_threshold: float = 0.35,
    max_passages: int = 6,
) -> List[Dict[str, Any]]:
    """
    Fuse (RRF) -> compute lexical overlap -> optional bm25/dense blending if available
    -> final score -> filter by threshold -> keep top max_passages.
    """
    fused = rrf_fuse_with_votes(rank_lists, rrf_k=rrf_k, topk=max(topk, 40))

    # Build candidate records with features (avoid huge dicts)
    df = idx.df
    cands: List[Dict[str, Any]] = []
    for row_idx, rrf_score, votes in fused:
        rec_row = df.iloc[row_idx]
        rec = {
            "doc_id": rec_row.get("doc_id"),
            "title":  rec_row.get("title", ""),
            "text":   rec_row.get("text", ""),
            "source": rec_row.get("source", ""),
            "url":    rec_row.get("url", ""),
        }
        rec["_row_idx"] = row_idx
        rec["rrf_score"] = float(rrf_score)
        rec["fusion_votes"] = int(votes)
        rec["lexical_overlap"] = _lexical_overlap_cached(question, rec["text"])

        if "bm25" in df.columns:
            rec["bm25"] = float(rec_row.get("bm25", 0.0))
        if "dense" in df.columns:
            rec["dense"] = float(rec_row.get("dense", 0.0))
        cands.append(rec)

    if not cands:
        return []

    # Normalize features
    rrf_norm = _minmax([c["rrf_score"] for c in cands])
    ovlp_vals = [c["lexical_overlap"] for c in cands]
    bm25_norm = _minmax([c.get("bm25", 0.0) for c in cands]) if any("bm25" in c for c in cands) else [0.0]*len(cands)
    dense_norm = _minmax([c.get("dense", 0.0) for c in cands]) if any("dense" in c for c in cands) else [0.0]*len(cands)

    total_weight = max(1e-8, alpha_rrf + beta_overlap + bm25_weight + dense_weight)

    for i, c in enumerate(cands):
        combined = (
            alpha_rrf     * rrf_norm[i] +
            beta_overlap  * ovlp_vals[i] +
            bm25_weight   * bm25_norm[i] +
            dense_weight  * dense_norm[i]
        ) / total_weight
        c["final_score"] = float(combined)

    # Filter by threshold and keep the strongest
    cands = [c for c in cands if c["final_score"] >= min_final_threshold]
    cands.sort(key=lambda x: x["final_score"], reverse=True)
    if max_passages > 0:
        cands = cands[:max_passages]
    return cands

def dedupe_by_docid(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for p in passages:
        d = p.get("doc_id")
        if d in seen:
            continue
        seen.add(d); out.append(p)
    return out

BAD_QE = ("answer", "true/false", "select one", "all are true", "except")

def sanitize_expansions(exps: List[str]) -> List[str]:
    out = []
    for e in exps or []:
        s = (e or "").strip()
        sl = s.lower()
        if not (3 <= len(s) <= 50):
            continue
        if any(b in sl for b in BAD_QE):
            continue
        if sl in {"a","b","c","d"}:
            continue
        out.append(s)
    # unique (preserve order)
    seen = set(); uniq = []
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

# Cache lexical overlap (it’s pure function of (q, t))
@lru_cache(maxsize=200_000)
def _lexical_overlap_cached(q: str, t: str) -> float:
    return _lex_overlap(q, t)

# ----------------------------- Async Ollama LLM -----------------------------

class AsyncOllamaClient:
    def __init__(self, model_name: str, max_workers: int = 4, base_url: Optional[str] = None):
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.base_url = base_url  # override host if needed

    async def generate(self, prompt: str, max_tokens: int = 2, temperature: float = 0.0, top_p: float = 1.0) -> str:
        import ollama
        client = ollama.Client(host=self.base_url) if self.base_url else ollama
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            self.executor,
            lambda: client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,  # keep tiny for single letter
                    "stop": ["\n", "\r\n"],     # stop at newline
                },
            )
        )
        return resp.get("response", "")

    async def generate_batch(self, prompts: List[str], max_tokens: int = 2, temperature: float = 0.0, top_p: float = 1.0) -> List[str]:
        tasks = [self.generate(p, max_tokens=max_tokens, temperature=temperature, top_p=top_p) for p in prompts]
        return await asyncio.gather(*tasks)

# ----------------------------- Fast Embedding Helper -----------------------------

def embed_concurrent(embed_client, texts: List[str], is_query: bool = False, max_workers: int = 8) -> Any:
    """Fast concurrent embedding for Ollama clients during retrieval."""
    if hasattr(embed_client, 'model_name'):  # MedEmbedClient
        return embed_client.embed_batch(texts, is_query)
    
    # OllamaClient - use concurrent processing
    if len(texts) <= 4:  # Small batch, use normal processing
        return embed_client.embed_batch(texts, is_query)
    
    # Split into sub-batches for concurrent processing
    import asyncio
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    
    async def process_concurrent():
        sub_batch_size = max(1, len(texts) // max_workers)
        tasks = []
        
        def embed_sub_batch(sub_batch):
            return embed_client.embed_batch(sub_batch, is_query)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()
            for i in range(0, len(texts), sub_batch_size):
                sub_batch = texts[i:i+sub_batch_size]
                if sub_batch:
                    task = loop.run_in_executor(executor, embed_sub_batch, sub_batch)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        return np.vstack(results)
    
    return asyncio.run(process_concurrent())

# ----------------------------- Tokenizer cache for BM25 -----------------------------

@lru_cache(maxsize=200_000)
def _tok_cached(q: str):
    from src.scripts.build_medcorp_index import default_tokenizer
    return tuple(default_tokenizer(q))

# ----------------------------- Batched dense FAISS helpers -----------------------------

def _normalize_rows_inplace(mat: np.ndarray):
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat /= norms

def dense_search_batch(idx: HybridIndex, q_embs: np.ndarray, dense_k: int) -> Tuple[np.ndarray, np.ndarray]:
    _normalize_rows_inplace(q_embs)
    sims, ids = idx.faiss_index.search(q_embs, dense_k)
    return sims, ids

# ----------------------------- Batch Retrieval Optimization (NEW) -----------------------------

def _collect_queries_for_examples(
    examples: List[Dict[str, Any]], option_aware: bool, use_expansions: bool
):
    """
    Returns:
      q_texts: List[str]
      q_meta:  List[Dict] with fields:
               {"ex_idx": int, "kind": "base"|"opt"|"exp", "opt_key": Optional[str]}
      per_ex_allowed_letters: List[List[str]] for parsing
    """
    q_texts, q_meta = [], []
    per_ex_allowed_letters = []
    for ex_idx, ex in enumerate(examples):
        question = (ex.get("question") or "").strip()
        options  = make_options(ex)
        per_ex_allowed_letters.append(sorted(options.keys()))

        # base
        q_texts.append(question)
        q_meta.append({"ex_idx": ex_idx, "kind": "base", "opt_key": None})

        # option-aware
        if option_aware:
            for k in sorted(options.keys()):
                ot = options[k].strip()
                if ot:
                    q_texts.append(f"{question}\nOption: {ot}")
                    q_meta.append({"ex_idx": ex_idx, "kind": "opt", "opt_key": k})

        # expansions
        if use_expansions:
            for qe in sanitize_expansions(ex.get("query_expansions") or []):
                q_texts.append(f"{question} {qe}")
                q_meta.append({"ex_idx": ex_idx, "kind": "exp", "opt_key": None})

    return q_texts, q_meta, per_ex_allowed_letters

def retrieve_batch_fast(
    batch_examples: List[Dict[str, Any]],
    idx: HybridIndex,
    embed_client: Any,
    mode: str,
    k: int,
    dense_k: int,
    bm25_k: int,
    rrf_k: int,
    use_expansions: bool,
    option_aware: bool,
    max_evidence_tokens: int,
    tokenizer: Optional[AutoTokenizer],
    # post-fusion knobs
    alpha_rrf: float,
    beta_overlap: float,
    bm25_weight: float,
    dense_weight: float,
    min_final_threshold: float,
    min_lexical_overlap_fallback: float,
    max_passages: int,
    bm25_scope: str = "base",   # "base"|"all"|"none"
) -> List[List[Dict[str, Any]]]:
    """
    Returns per-example packed refs (list of dicts) computed in one batched pass.
    """
    # 1) Collect queries
    q_texts, q_meta, _ = _collect_queries_for_examples(
        batch_examples, option_aware, use_expansions
    )

    # 2) Embed ALL queries at once
    if mode in ("hybrid", "dense") and q_texts:
        q_embs = embed_client.embed_batch(q_texts, is_query=True)
        q_embs = np.asarray(q_embs, dtype=np.float32)
    else:
        q_embs = None

    # 3) Dense batched search (one FAISS call)
    dense_lists: List[List[Tuple[int, float]]] = [[] for _ in q_texts]
    if mode in ("hybrid", "dense") and q_embs is not None and len(q_embs) > 0:
        sims, ids = dense_search_batch(idx, q_embs, dense_k)
        for i in range(len(q_texts)):
            dense_lists[i] = [(int(ids[i, j]), float(sims[i, j])) for j in range(dense_k) if ids[i, j] != -1]

    # 4) BM25 (optional) — do only what’s asked
    bm25_lists: List[List[Tuple[int, float]]] = [[] for _ in q_texts]
    if mode in ("hybrid", "bm25") and bm25_scope != "none":
        if bm25_scope == "base":
            q_indices = [i for i, m in enumerate(q_meta) if m["kind"] == "base"]
        else:
            q_indices = list(range(len(q_texts)))

        for i in q_indices:
            q = q_texts[i]
            # O(N) partial sort for top-k
            scores = idx.bm25.get_scores(list(_tok_cached(q)))
            top_idx = np.argpartition(-scores, bm25_k)[:bm25_k]
            top_scores = scores[top_idx]
            order = np.argsort(-top_scores)
            bm25_lists[i] = [(int(top_idx[j]), float(top_scores[j])) for j in order]

    # 5) Group per example + fuse
    rank_lists_per_ex: List[List[List[Tuple[int, float]]]] = [[] for _ in batch_examples]
    for qi, meta in enumerate(q_meta):
        if mode == "dense":
            pairs = dense_lists[qi]
            rank_lists_per_ex[meta["ex_idx"]].append(pairs)
        elif mode == "bm25":
            pairs = bm25_lists[qi]
            rank_lists_per_ex[meta["ex_idx"]].append(pairs)
        else:
            # hybrid: add dense list
            if dense_lists[qi]:
                rank_lists_per_ex[meta["ex_idx"]].append(dense_lists[qi])
            # hybrid: add bm25 list (if present for this query)
            if bm25_lists[qi]:
                rank_lists_per_ex[meta["ex_idx"]].append(bm25_lists[qi])

    fused_refs = []
    for ex_idx, ex in enumerate(batch_examples):
        question = (ex.get("question") or "").strip()
        rank_lists = rank_lists_per_ex[ex_idx]

        if not rank_lists:
            fused_refs.append([])
            continue

        fused = fuse_rescore_filter(
            idx=idx,
            rank_lists=rank_lists,
            question=question,
            topk=k,
            rrf_k=rrf_k,
            alpha_rrf=alpha_rrf,
            beta_overlap=beta_overlap,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            min_final_threshold=min_final_threshold,
            max_passages=max_passages,
        )
        fused = dedupe_by_docid(fused)

        if fused and fused[0].get("lexical_overlap", 0.0) < min_lexical_overlap_fallback:
            fused = []

        if max_evidence_tokens > 0 and tokenizer is not None and fused:
            fused = pack_passages(fused, tokenizer, max_evidence_tokens)

        fused_refs.append(fused)

    return fused_refs

# ----------------------------- Legacy per-example retrieval (kept for reference) -----------------------------
# (Not used by the new fast path. Kept to minimize surprises.)

def search_with_precomputed_embedding(
    idx: HybridIndex,
    query: str,
    query_embedding: Optional[Any],
    embed_client: Any,
    k: int = 10,
    mode: str = "hybrid",
    rrf_k: int = 60,
    dense_k: int = 50,
    bm25_k: int = 200
) -> List[Dict[str, Any]]:
    """Modified search that uses precomputed embedding to avoid re-embedding."""
    import numpy as np
    import faiss as faiss_local  # safe alias
    from src.scripts.build_medcorp_index import norm_text, default_tokenizer, rrf_fuse
    
    q = norm_text(query)
    dense_list = []
    
    if mode in ("hybrid", "dense"):
        if query_embedding is not None:
            q_emb = query_embedding.astype(np.float32)
            faiss_local.normalize_L2(q_emb.reshape(1, -1))
            sims, ids = idx.faiss_index.search(q_emb.reshape(1, -1), dense_k)
            sims = sims[0]; ids = ids[0]
            dense_list = [(int(i), float(s)) for i, s in zip(ids, sims) if i != -1]
        else:
            q_emb = embed_client.embed_batch([q], is_query=True)[0].astype(np.float32)
            faiss_local.normalize_L2(q_emb.reshape(1, -1))
            sims, ids = idx.faiss_index.search(q_emb.reshape(1, -1), dense_k)
            sims = sims[0]; ids = ids[0]
            dense_list = [(int(i), float(s)) for i, s in zip(ids, sims) if i != -1]

    bm25_list = []
    if mode in ("hybrid", "bm25"):
        if idx.bm25 is None:
            raise RuntimeError("BM25 index not available; rebuild without --no-bm25")
        scores = idx.bm25.get_scores(list(_tok_cached(q)))
        # Fast top-k
        top_idx = np.argpartition(-scores, bm25_k)[:bm25_k]
        top_scores = scores[top_idx]
        order = np.argsort(-top_scores)
        bm25_list = [(int(top_idx[i]), float(top_scores[i])) for i in order]

    if mode == "dense":
        results = [(i, s) for (i, s) in dense_list[:k]]
    elif mode == "bm25":
        results = bm25_list[:k]
    else:
        fused = rrf_fuse([dense_list, bm25_list], rrf_k=rrf_k, topk=k)
        results = fused

    out = []
    for row_idx, score in results:
        rec = idx.df.iloc[row_idx]
        d = {
            "doc_id": rec.get("doc_id"),
            "title":  rec.get("title", ""),
            "text":   rec.get("text", ""),
            "source": rec.get("source", ""),
            "url":    rec.get("url", ""),
            "score":  float(score),
        }
        out.append(d)
    return out

# ----------------------------- Evaluation core -----------------------------

async def evaluate_async(
    index_dir: str,
    eval_path: str,
    save_dir: str,
    ollama_base_url: str,
    embed_model_query: str,
    llm_model: str,
    mode: str,
    k: int,
    dense_k: int,
    bm25_k: int,
    rrf_k: int,
    option_aware: bool,
    use_expansions: bool,
    max_evidence_tokens: int,
    hf_tokenizer: str,
    batch_size: int,
    max_new_tokens: int,
    workers: int,
    limit: Optional[int] = None,
    device: str = "auto",
    # New scoring/filters
    alpha_rrf: float = 0.6,
    beta_overlap: float = 0.4,
    bm25_weight: float = 0.0,
    dense_weight: float = 0.0,
    min_final_threshold: float = 0.35,
    min_lexical_overlap_fallback: float = 0.01,
    max_passages: int = 6,
    # New fast-path knobs
    retrieval_batch: int = 256,
    faiss_gpu: bool = False,
    bm25_scope: str = "base",    # base|all|none
) -> Dict[str, Any]:

    ensure_dir(save_dir)
    retrieved_dir = os.path.join(save_dir, "retrieved_contents")
    ensure_dir(retrieved_dir)
    start = time.time()

    # Load index
    idx = HybridIndex(index_dir=index_dir)
    idx.load()

    # Optional: move FAISS to GPU(s)
    if faiss_gpu and faiss is not None:
        try:
            idx.faiss_index = faiss.index_cpu_to_all_gpus(idx.faiss_index)
            print("FAISS index moved to GPU(s).")
        except Exception as e:
            print(f"FAISS GPU not used ({e}). Falling back to CPU.")

    # Embedding client (query) – MedEmbed or Ollama
    if embed_model_query.strip() == "abhinand/MedEmbed-large-v0.1":
        print(f"Using MedEmbed for query embeddings on device='{device}'")
        embed_client = MedEmbedClient(model_name=embed_model_query, device=device)
    else:
        embed_client = OllamaClient(
            base_url=ollama_base_url,
            model_doc=embed_model_query,
            model_query=embed_model_query,
        )

    # LLM client
    llm = AsyncOllamaClient(llm_model, max_workers=workers, base_url=ollama_base_url)

    # Shared tokenizer (only for token counting / evidence packing)
    tok = AutoTokenizer.from_pretrained(hf_tokenizer, use_fast=True)

    # Data
    data = load_json_or_jsonl(eval_path)
    if limit is not None:
        data = data[:limit]
        print(f"Limited to {len(data)} examples")

    # Batched retrieval + prompt building (fast path)
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    print(f"Processing {len(data)} examples with batched retrieval (batch={retrieval_batch})...")
    for start_i in tqdm(range(0, len(data), retrieval_batch), desc="Retrieving (batched)"):
        batch_examples = data[start_i:start_i + retrieval_batch]

        batch_refs = retrieve_batch_fast(
            batch_examples=batch_examples,
            idx=idx,
            embed_client=embed_client,
            mode=mode,
            k=k,
            dense_k=dense_k,
            bm25_k=bm25_k,
            rrf_k=rrf_k,
            use_expansions=use_expansions,
            option_aware=option_aware,
            max_evidence_tokens=max_evidence_tokens,
            tokenizer=tok,
            alpha_rrf=alpha_rrf,
            beta_overlap=beta_overlap,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            min_final_threshold=min_final_threshold,
            min_lexical_overlap_fallback=min_lexical_overlap_fallback,
            max_passages=max_passages,
            bm25_scope=bm25_scope,
        )

        for ex, refs in zip(batch_examples, batch_refs):
            question = (ex.get("question") or "").strip()
            options  = make_options(ex)
            gold     = gold_letter(ex)
            prompt   = build_prompt(question, options, refs)

            prompts.append(prompt)
            metas.append({
                "example": ex,
                "gold": gold,
                "num_contexts": len(refs),
                "refs": refs,
            })

    preds: List[Dict[str, Any]] = []
    preds_path = os.path.join(save_dir, "predictions.jsonl")
    retrieved_contents_path = os.path.join(retrieved_dir, "retrieved_contents.jsonl")

    with open(preds_path, "w", encoding="utf-8") as pred_file, \
         open(retrieved_contents_path, "w", encoding="utf-8") as retrieved_file:

        for i in tqdm(range(0, len(prompts), batch_size), desc="LLM scoring"):
            batch_prompts = prompts[i:i+batch_size]
            batch_metas   = metas[i:i+batch_size]
            responses = await llm.generate_batch(
                batch_prompts, max_tokens=max_new_tokens, temperature=0.0, top_p=1.0
            )
            batch_preds = []
            for j, (resp, meta) in enumerate(zip(responses, batch_metas)):
                options = make_options(meta["example"])
                allowed_letters = list(options.keys())
                letter = parse_letter(resp, allowed_letters)
                gold   = meta["gold"]
                ex     = meta["example"]
                refs   = meta["refs"]
                is_correct = (letter == gold) if (letter and gold) else None

                pred = {
                    **ex,
                    "prediction": letter,
                    "gold": gold,
                    "is_correct": is_correct,
                    "raw_output": resp,
                    "num_contexts_used": meta["num_contexts"],
                }
                batch_preds.append(pred)
                pred_file.write(json.dumps(pred, ensure_ascii=False) + "\n")
                pred_file.flush()

                retrieved_content = {
                    "question_id": ex.get("id", i*batch_size + j),
                    "question": ex.get("question", ""),
                    "retrieved_passages": [
                        {
                            "doc_id": ref.get("doc_id", ""),
                            "title": ref.get("title", ""),
                            "text": ref.get("text", ""),
                            "source": ref.get("source", ""),
                            "score_rrf": ref.get("rrf_score", ref.get("score", 0.0)),
                            "lexical_overlap": ref.get("lexical_overlap", 0.0),
                            "final_score": ref.get("final_score", ref.get("score", 0.0)),
                            "fusion_votes": ref.get("fusion_votes", 1),
                            "url": ref.get("url", "")
                        } for ref in refs
                    ],
                    "num_passages": len(refs)
                }
                retrieved_file.write(json.dumps(retrieved_content, ensure_ascii=False) + "\n")
                retrieved_file.flush()

            preds.extend(batch_preds)

    # Metrics
    total   = len(preds)
    correct = sum(1 for r in preds if r.get("is_correct") is True)
    acc     = (correct / total) if total else 0.0

    with_ctx   = [r for r in preds if r.get("num_contexts_used", 0) > 0]
    without_ctx= [r for r in preds if r.get("num_contexts_used", 0) == 0]
    acc_with   = (sum(1 for r in with_ctx if r.get("is_correct")) / len(with_ctx)) if with_ctx else 0.0
    acc_without= (sum(1 for r in without_ctx if r.get("is_correct")) / len(without_ctx)) if without_ctx else 0.0

    # Per subject / choice type
    by_subject = defaultdict(lambda: {"n":0,"correct":0})
    by_choice  = defaultdict(lambda: {"n":0,"correct":0})
    for r in preds:
        subj = (r.get("subject_name") or "UNKNOWN").strip()
        ctyp = (r.get("choice_type") or "single").strip()
        by_subject[subj]["n"] += 1
        by_subject[subj]["correct"] += int(bool(r.get("is_correct")))
        by_choice[ctyp]["n"] += 1
        by_choice[ctyp]["correct"] += int(bool(r.get("is_correct")))
    by_subject = {
        k: {"count": v["n"], "accuracy": (v["correct"]/v["n"]) if v["n"] else 0.0}
        for k, v in sorted(by_subject.items(), key=lambda kv: kv[0])
    }
    by_choice = {
        k: {"count": v["n"], "accuracy": (v["correct"]/v["n"]) if v["n"] else 0.0}
        for k, v in sorted(by_choice.items(), key=lambda kv: kv[0])
    }

    elapsed = time.time() - start
    throughput = total / elapsed if elapsed > 0 else 0.0

    print(f"\n=== Performance Summary ===")
    print(f"Total examples: {total}")
    print(f"Processing time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Average time per example: {elapsed/total:.2f}s")
    print(f"===========================")

    # Determine embedding client type and model info
    embedding_client_type = "MedEmbedClient" if embed_model_query.strip() == "abhinand/MedEmbed-large-v0.1" else "OllamaClient"
    embedding_model_actual = embed_model_query
    
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "with_context_count": len(with_ctx),
        "with_context_accuracy": acc_with,
        "without_context_count": len(without_ctx),
        "without_context_accuracy": acc_without,
        "by_subject": by_subject,
        "by_choice_type": by_choice,
        "elapsed_seconds": elapsed,
        "examples_per_second": (total/elapsed) if elapsed > 0 else 0.0,
        "settings": {
            "mode": mode,
            "k": k, "dense_k": dense_k, "bm25_k": bm25_k, "rrf_k": rrf_k,
            "option_aware": option_aware,
            "use_expansions": use_expansions,
            "max_evidence_tokens": max_evidence_tokens,
            "hf_tokenizer": hf_tokenizer,
            "llm_model": llm_model,
            "embed_model_query": embed_model_query,
            "embedding_client_type": embedding_client_type,
            "device": device,
            # New knobs
            "alpha_rrf": alpha_rrf,
            "beta_overlap": beta_overlap,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "min_final_threshold": min_final_threshold,
            "min_lexical_overlap_fallback": min_lexical_overlap_fallback,
            "max_passages": max_passages,
            "retrieval_batch": retrieval_batch,
            "faiss_gpu": faiss_gpu,
            "bm25_scope": bm25_scope,
        },
        "system_info": {
            "index_dir": index_dir,
            "eval_path": eval_path,
            "embedding_model": embedding_model_actual,
            "embedding_backend": embedding_client_type,
            "compute_device": device,
        }
    }
    mets_path = os.path.join(save_dir, "metrics.json")
    with open(mets_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Saved to: {save_dir}")
    print(f"Overall Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"With Context: {acc_with:.4f} (n={len(with_ctx)})")
    print(f"Without Context: {acc_without:.4f} (n={len(without_ctx)})")
    print(f"Elapsed: {elapsed:.2f}s  ({metrics['examples_per_second']:.2f} ex/s)")
    return metrics

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate current hybrid RAG on MedMCQA or MedQA data")
    ap.add_argument("--index-dir", type=str, required=True, help="Path to built hybrid index directory")
    ap.add_argument("--eval-path", type=str, default=os.path.join(PROJECT_ROOT, "data", "medmcqa", "dev_stratified_sample_qe.json"), help="Path to evaluation data (MedMCQA or MedQA format)")
    ap.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "evaluation_results", "eval_hybrid_rag"))
    ap.add_argument("--ollama-base-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--embed-model-query", type=str, default="oscardp96/medcpt-query:latest",
                   help="Query embedding model. Options: oscardp96/medcpt-query:latest, abhinand/MedEmbed-large-v0.1")
    ap.add_argument("--llm-model", type=str, default="thewindmom/llama3-med42-8b")
    ap.add_argument("--medembed", action="store_true",
                   help="Shortcut: use MedEmbed-large-v0.1 for query embeddings")
    ap.add_argument("--model-preset", type=str, choices=["medcpt", "medembed"],
                   help="Use a model preset: 'medcpt' (default) or 'medembed' for embedding model")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for MedEmbed model: auto (GPU if available), cpu, or cuda")

    # Retrieval knobs
    ap.add_argument("--mode", type=str, default="hybrid", choices=["hybrid","dense","bm25"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--dense-k", type=int, default=80)
    ap.add_argument("--bm25-k", type=int, default=400)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--option-aware", action="store_true", help="Use option-aware retrieval for MCQ (slower but more accurate)")
    ap.add_argument("--use-expansions", action="store_true", help="Fuse RRF over question + dataset query_expansions")
    ap.add_argument("--fast-mode", action="store_true", help="Skip option-aware retrieval for faster processing")

    # Evidence packing + tokenizer
    ap.add_argument("--max-evidence-tokens", type=int, default=1200)
    ap.add_argument("--hf-tokenizer", type=str, default="deepseek-ai/DeepSeek-R1")

    # LLM batching
    ap.add_argument("--batch-size", type=int, default=32, help="LLM inference batch size (increase for better GPU utilization)")
    ap.add_argument("--max-new-tokens", type=int, default=2, help="Max tokens to generate (short letter outputs)")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads (increase for GPU utilization)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of examples for testing")

    # Post-fusion scoring/filter knobs
    ap.add_argument("--alpha-rrf", type=float, default=0.6, help="Weight for normalized RRF score")
    ap.add_argument("--beta-overlap", type=float, default=0.4, help="Weight for lexical overlap")
    ap.add_argument("--bm25-weight", type=float, default=0.0, help="Optional weight if bm25 scores are available")
    ap.add_argument("--dense-weight", type=float, default=0.0, help="Optional weight if dense scores are available")
    ap.add_argument("--final-threshold", type=float, default=0.35, help="Minimum final score to keep a passage")
    ap.add_argument("--min-lexical-overlap", type=float, default=0.01, help="If top passage overlap is below this, drop all context")
    ap.add_argument("--max-passages", type=int, default=6, help="Keep at most this many strongest passages per example")

    # NEW fast-path switches
    ap.add_argument("--retrieval-batch", type=int, default=256,
                    help="How many examples to retrieve concurrently (batched dense FAISS).")
    ap.add_argument("--faiss-gpu", action="store_true",
                    help="Use FAISS on GPU (if faiss-gpu is installed).")
    ap.add_argument("--bm25-scope", type=str, choices=["base", "all", "none"], default="base",
                    help="BM25 only for base question, for all queries, or disabled.")

    args = ap.parse_args()

    # --- Normalize / sanitize Ollama base URL (avoid '' -> '/api/embeddings' bug) ---
    def _normalize_base_url(u: Optional[str]) -> str:
        u = (u or "").strip()
        if not u:
            # fallback env then hard default
            u = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        # Accept forms like 'localhost:11434' without scheme
        if not re.match(r'^https?://', u):
            u = 'http://' + u
        # Remove trailing slash(es)
        u = u.rstrip('/')
        return u

    args.ollama_base_url = _normalize_base_url(args.ollama_base_url)

    # Handle model presets and MedEmbed shortcut flag
    if args.model_preset:
        if args.model_preset == "medembed":
            args.embed_model_query = "abhinand/MedEmbed-large-v0.1"
        elif args.model_preset == "medcpt":
            args.embed_model_query = "oscardp96/medcpt-query:latest"

    if args.medembed:
        args.embed_model_query = "abhinand/MedEmbed-large-v0.1"

    # Normalize paths (Windows-safe)
    index_dir = os.path.abspath(args.index_dir)
    eval_path = os.path.abspath(args.eval_path)
    save_dir  = os.path.abspath(args.save_dir)

    asyncio.run(
        evaluate_async(
            index_dir=index_dir,
            eval_path=eval_path,
            save_dir=save_dir,
            ollama_base_url=args.ollama_base_url,
            embed_model_query=args.embed_model_query,
            llm_model=args.llm_model,
            mode=args.mode,
            k=args.k,
            dense_k=args.dense_k,
            bm25_k=args.bm25_k,
            rrf_k=args.rrf_k,
            option_aware=args.option_aware and not args.fast_mode,
            use_expansions=args.use_expansions,
            max_evidence_tokens=args.max_evidence_tokens,
            hf_tokenizer=args.hf_tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            workers=args.workers,
            limit=args.limit,
            device=args.device,
            alpha_rrf=args.alpha_rrf,
            beta_overlap=args.beta_overlap,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            min_final_threshold=args.final_threshold,
            min_lexical_overlap_fallback=args.min_lexical_overlap,
            max_passages=args.max_passages,
            retrieval_batch=args.retrieval_batch,
            faiss_gpu=args.faiss_gpu,
            bm25_scope=args.bm25_scope,
        )
    )

if __name__ == "__main__":
    main()
