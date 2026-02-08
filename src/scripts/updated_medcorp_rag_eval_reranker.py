#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the MedCorp RAG pipeline (Hybrid FAISS+BM25) with support for:
- BGE (and other SentenceTransformers) embeddings
- Optional cross-encoder reranking (e.g., BAAI/bge-reranker-large)
- Batched dense retrieval (single FAISS call per batch)
- Optional FAISS GPU offload (index_cpu_to_all_gpus)
- Fast BM25 top-k via argpartition
- Option-aware MCQ retrieval

Throughput fixes:
- Instantiate the CrossEncoder once and keep it in memory
- Batched reranking across an entire retrieval batch

New in this version:
- --min-ce-score: gate context on top CE score (preferred over lexical gate)
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
    MedEmbedClient,   # keep MedEmbed option
)

# =============== SentenceTransformers client (BGE) ===============
INSTRUCTIONS = {
    "bge_query": "Represent this sentence for searching relevant passages:",
}

class STEmbeddingClient:
    """
    Lightweight SentenceTransformers client with BGE query instruction.
    """
    def __init__(self, model_name: str, device: str = "auto", family: str = "auto", normalize: bool = True):
        self.model_name = model_name
        self.family = family  # "bge" | "auto"
        self.normalize = normalize
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")

    def _prep(self, texts: List[str], is_query: bool) -> List[str]:
        if self.family == "bge" and is_query:
            pfx = INSTRUCTIONS["bge_query"] + " "
            return [pfx + (t or "") for t in texts]
        return texts

    def embed_batch(self, texts: List[str], is_query: bool = False):
        texts = self._prep(texts, is_query)
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        return vecs.astype(np.float32)

def make_query_client(name: str, device: str, ollama_base_url: str):
    """
    Choose the right embedding backend based on name/preset.
    """
    low = (name or "").lower().strip()
    if name.strip() == "abhinand/MedEmbed-large-v0.1":
        return MedEmbedClient(model_name=name, device=device)
    if "bge" in low:  # e.g., BAAI/bge-m3
        return STEmbeddingClient(model_name=name, device=device, family="bge", normalize=True)
    # fallback: Ollama (e.g., MedCPT query encoder served by Ollama)
    return OllamaClient(base_url=ollama_base_url, model_doc=name, model_query=name)

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
    if "options" in ex and isinstance(ex["options"], dict):
        return "medqa"
    elif any(key in ex for key in ["opa", "opb", "opc", "opd"]):
        return "medmcqa"
    else:
        return "medmcqa"

def gold_letter(ex: Dict[str, Any]) -> Optional[str]:
    answer_idx = ex.get("answer_idx")
    if answer_idx and isinstance(answer_idx, str) and answer_idx.upper() in "ABCDE":
        return answer_idx.upper()
    cop = ex.get("cop")
    if isinstance(cop, int) and cop in (1,2,3,4):
        return LETTER_MAP[cop]
    ans = ex.get("answer") or ex.get("gold")
    if isinstance(ans, str) and ans.upper() in "ABCDE":
        return ans.upper()
    return None

def make_options(ex: Dict[str, Any]) -> Dict[str, str]:
    if "options" in ex and isinstance(ex["options"], dict):
        options_dict = ex["options"]
        normalized = {}
        for key in sorted(options_dict.keys()):
            if key.upper() in "ABCDE" and options_dict[key]:
                normalized[key.upper()] = str(options_dict[key]).strip()
        return normalized
    return {
        "A": (ex.get("opa") or "").strip(),
        "B": (ex.get("opb") or "").strip(),
        "C": (ex.get("opc") or "").strip(),
        "D": (ex.get("opd") or "").strip(),
    }

ANSWER_RE = re.compile(r"Answer\s*[:：]\s*([A-E])", re.I)
LETTER_RE  = re.compile(r"\b([A-E])\b")

def parse_letter(text: str, allowed_letters: List[str] = None) -> Optional[str]:
    if allowed_letters is None:
        allowed_letters = ["A", "B", "C", "D", "E"]
    allowed_set = set(l.upper() for l in allowed_letters)
    t = (text or "").strip()
    
    # For CoT responses, prioritize "Answer: X" pattern at the end of the text
    # Look for the pattern in the last few lines of the response
    lines = t.split('\n')
    last_lines = '\n'.join(lines[-5:])  # Check last 5 lines
    m_answer = ANSWER_RE.search(last_lines)
    if m_answer and m_answer.group(1).upper() in allowed_set:
        return m_answer.group(1).upper()
    
    # Also check for "Answer: X" pattern in the entire text (fallback)
    m = ANSWER_RE.search(t)
    if m and m.group(1).upper() in allowed_set:
        return m.group(1).upper()
    
    # For non-CoT responses, check if first character is a valid letter
    if t and t[0].upper() in allowed_set:
        return t[0].upper()
    
    # Look for isolated letters (like "A", "B", etc.) - prefer later occurrences for CoT
    letter_matches = list(LETTER_RE.finditer(t))
    if letter_matches:
        # For CoT, prefer matches closer to the end of the text
        for match in reversed(letter_matches):
            if match.group(1).upper() in allowed_set:
                return match.group(1).upper()
    
    # Look for letter followed by closing parenthesis or period
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

def build_prompt(question: str, options: Dict[str, Any], refs: List[Dict[str, Any]], use_cot: bool = False) -> str:
    opt_str = "\n".join([f"{k}) {v}" for k, v in sorted(options.items()) if v.strip()])
    letters = sorted(options.keys())
    allowed_letters = ", ".join(letters[:-1]) + f" or {letters[-1]}" if len(letters) > 1 else letters[0]

    if use_cot:
        rules_with_refs = (
            "You are a medical expert answering a single-best-answer MCQ.\n"
            "Use the references if they are relevant; if they appear off-topic, ignore them and answer from medical knowledge.\n"
            "Think step by step and explain your reasoning before choosing an option.\n"
            "Output format: First provide your reasoning, then end with 'Answer: X' where X is a single uppercase letter.\n"
            f"Valid answers: {{{' '.join(letters)}}}\n"
            "If unsure, guess.\n"
        )
        rules_without_refs = (
            "You are a medical expert answering a single-best-answer MCQ.\n"
            "Choose exactly one option based on medical knowledge.\n"
            "Think step by step and explain your reasoning before choosing an option.\n"
            "Output format: First provide your reasoning, then end with 'Answer: X' where X is a single uppercase letter.\n"
            f"Valid answers: {{{' '.join(letters)}}}\n"
            "If unsure, guess.\n"
        )
    else:
        rules_with_refs = (
            "You are a medical expert answering a single-best-answer MCQ.\n"
            "Use the references if they are relevant; if they appear off-topic, ignore them and answer from medical knowledge.\n"
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
    scores: Dict[int, float] = {}
    votes: Dict[int, int] = {}
    for lst in rank_lists:
        for rank, (row_idx, _orig) in enumerate(lst):
            scores[row_idx] = scores.get(row_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            votes[row_idx] = votes.get(row_idx, 0) + 1
    fused = [(rid, scores[rid], votes[rid]) for rid in scores]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:topk]

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
    fused = rrf_fuse_with_votes(rank_lists, rrf_k=rrf_k, topk=max(topk, 40))
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

    rrf_norm = _minmax([c["rrf_score"] for c in cands])
    ovlp_vals = [c["lexical_overlap"] for c in cands]
    bm25_norm = _minmax([c.get("bm25", 0.0) for c in cands]) if any("bm25" in c for c in cands) else [0.0]*len(cands
    )
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
    seen = set(); uniq = []
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

@lru_cache(maxsize=200_000)
def _lexical_overlap_cached(q: str, t: str) -> float:
    return _lex_overlap(q, t)

# ----------------------------- Async Ollama LLM -----------------------------

class AsyncOllamaClient:
    def __init__(self, model_name: str, max_workers: int = 4, base_url: Optional[str] = None):
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.base_url = base_url
        # Reuse a single client instance if possible
        try:
            import ollama
            self._client = ollama.Client(host=self.base_url) if self.base_url else ollama
        except Exception:
            self._client = None

    async def generate(self, prompt: str, max_tokens: int = 2, temperature: float = 0.0, top_p: float = 1.0) -> str:
        import ollama  # lazily import (already cached after first)
        client = self._client or (ollama.Client(host=self.base_url) if self.base_url else ollama)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            self.executor,
            lambda: client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens, "stop": ["\n", "\r\n"] if max_tokens <= 5 else None},
            )
        )
        return resp.get("response", "")

    async def generate_batch(self, prompts: List[str], max_tokens: int = 2, temperature: float = 0.0, top_p: float = 1.0) -> List[str]:
        tasks = [self.generate(p, max_tokens=max_tokens, temperature=temperature, top_p=top_p) for p in prompts]
        return await asyncio.gather(*tasks)

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

# ----------------------------- Batch Retrieval + Batched Rerank -----------------------------

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
    bm25_scope: str = "base",
    # batched reranker
    ce: Optional[Any] = None,
    rerank_topk: int = 150,
    rerank_batch_size: int = 128,
    # CE gate
    min_ce_score: Optional[float] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Returns per-example packed refs (list of dicts) in one batched pass, with optional batched reranking.
    """
    # --- Collect queries for all examples in the batch ---
    q_texts: List[str] = []
    q_meta: List[Dict[str, Any]] = []
    questions: List[str] = []   # per example question text
    for ex_idx, ex in enumerate(batch_examples):
        question = (ex.get("question") or "").strip()
        questions.append(question)
        options  = make_options(ex)

        # base
        q_texts.append(question)
        q_meta.append({"ex_idx": ex_idx, "kind": "base"})

        # option-aware
        if option_aware:
            for kopt in sorted(options.keys()):
                ot = options[kopt].strip()
                if ot:
                    q_texts.append(f"{question}\nOption: {ot}")
                    q_meta.append({"ex_idx": ex_idx, "kind": "opt"})

        # expansions
        if use_expansions:
            for qe in sanitize_expansions(ex.get("query_expansions") or []):
                q_texts.append(f"{question} {qe}")
                q_meta.append({"ex_idx": ex_idx, "kind": "exp"})

    # --- Embed and dense search in one go ---
    if mode in ("hybrid", "dense") and q_texts:
        q_embs = embed_client.embed_batch(q_texts, is_query=True)
        q_embs = np.asarray(q_embs, dtype=np.float32)
    else:
        q_embs = None

    dense_lists: List[List[Tuple[int, float]]] = [[] for _ in q_texts]
    if mode in ("hybrid", "dense") and q_embs is not None and len(q_embs) > 0:
        sims, ids = dense_search_batch(idx, q_embs, dense_k)
        for i in range(len(q_texts)):
            dense_lists[i] = [(int(ids[i, j]), float(sims[i, j])) for j in range(dense_k) if ids[i, j] != -1]

    # --- BM25 (optional) ---
    bm25_lists: List[List[Tuple[int, float]]] = [[] for _ in q_texts]
    if mode in ("hybrid", "bm25") and bm25_scope != "none":
        if bm25_scope == "base":
            q_indices = [i for i, m in enumerate(q_meta) if m["kind"] == "base"]
        else:
            q_indices = list(range(len(q_texts)))

        for i in q_indices:
            q = q_texts[i]
            scores = idx.bm25.get_scores(list(_tok_cached(q)))
            top_idx = np.argpartition(-scores, bm25_k)[:bm25_k]
            top_scores = scores[top_idx]
            order = np.argsort(-top_scores)
            bm25_lists[i] = [(int(top_idx[j]), float(top_scores[j])) for j in order]

    # --- Group per example + fuse (RRF + lexical + optional weights) ---
    rank_lists_per_ex: List[List[List[Tuple[int, float]]]] = [[] for _ in batch_examples]
    for qi, meta in enumerate(q_meta):
        if mode == "dense":
            rank_lists_per_ex[meta["ex_idx"]].append(dense_lists[qi])
        elif mode == "bm25":
            rank_lists_per_ex[meta["ex_idx"]].append(bm25_lists[qi])
        else:
            if dense_lists[qi]:
                rank_lists_per_ex[meta["ex_idx"]].append(dense_lists[qi])
            if bm25_lists[qi]:
                rank_lists_per_ex[meta["ex_idx"]].append(bm25_lists[qi])

    # Build candidate pools (pre-rerank)
    pool_for_rerank = max(k, rerank_topk, max_passages * 3)
    cands_per_ex: List[List[Dict[str, Any]]] = []
    for ex_idx, ex in enumerate(batch_examples):
        question = questions[ex_idx]
        rank_lists = rank_lists_per_ex[ex_idx]
        if not rank_lists:
            cands_per_ex.append([])
            continue
        cands = fuse_rescore_filter(
            idx=idx,
            rank_lists=rank_lists,
            question=question,
            topk=pool_for_rerank,
            rrf_k=rrf_k,
            alpha_rrf=alpha_rrf,
            beta_overlap=beta_overlap,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            min_final_threshold=min_final_threshold,
            max_passages=pool_for_rerank,  # don't truncate to final yet
        )
        cands = dedupe_by_docid(cands)
        cands_per_ex.append(cands)

    # --- Batched reranking across the whole retrieval batch ---
    if ce is not None:
        all_pairs: List[Tuple[str, str]] = []
        spans: List[Tuple[int, int, int]] = []  # (ex_idx, start, end)
        cursor = 0
        for ex_idx, cands in enumerate(cands_per_ex):
            cut = min(rerank_topk, len(cands))
            pairs = [(questions[ex_idx], cands[i]["text"]) for i in range(cut)]
            all_pairs.extend(pairs)
            spans.append((ex_idx, cursor, cursor + cut))
            cursor += cut

        if all_pairs:
            try:
                scores = ce.predict(all_pairs, batch_size=rerank_batch_size, show_progress_bar=False)
            except Exception as e:
                print(f"[warn] Reranker predict failed, skipping CE for this batch: {e}")
                scores = None

            if scores is not None:
                # Assign scores back and sort each head slice
                for ex_idx, s, e in spans:
                    seg = scores[s:e]
                    for i in range(e - s):
                        cands_per_ex[ex_idx][i]["ce_score"] = float(seg[i])
                    head = sorted(
                        cands_per_ex[ex_idx][:e - s],
                        key=lambda x: x.get("ce_score", x.get("final_score", 0.0)),
                        reverse=True
                    )
                    cands_per_ex[ex_idx] = head + cands_per_ex[ex_idx][e - s:]

    # --- Finalize: CE gate (if set) -> lexical fallback (only if CE gate not used) -> trim/pack ---
    out_refs: List[List[Dict[str, Any]]] = []
    for ex_idx, cands in enumerate(cands_per_ex):
        # CE-gate: if provided and reranker used, gate by top CE score
        if ce is not None and (min_ce_score is not None) and cands:
            top_ce = cands[0].get("ce_score")
            if (top_ce is not None) and (top_ce < min_ce_score):
                cands = []

        # If we didn't use CE gate, allow lexical fallback gate
        if not (ce is not None and (min_ce_score is not None)):
            if cands and cands[0].get("lexical_overlap", 0.0) < min_lexical_overlap_fallback:
                cands = []

        if max_passages > 0:
            cands = cands[:max_passages]
        if max_evidence_tokens > 0 and tokenizer is not None and cands:
            cands = pack_passages(cands, tokenizer, max_evidence_tokens)
        out_refs.append(cands)

    return out_refs

# ----------------------------- Legacy per-example retrieval (kept) -----------------------------

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
    import numpy as np
    import faiss as faiss_local
    from src.scripts.build_medcorp_index import norm_text, rrf_fuse

    q = norm_text(query)
    dense_list = []
    if mode in ("hybrid", "dense"):
        if query_embedding is not None:
            q_emb = query_embedding.astype(np.float32)
        else:
            q_emb = embed_client.embed_batch([q], is_query=True)[0].astype(np.float32)
        faiss_local.normalize_L2(q_emb.reshape(1, -1))
        sims, ids = idx.faiss_index.search(q_emb.reshape(1, -1), dense_k)
        sims = sims[0]; ids = ids[0]
        dense_list = [(int(i), float(s)) for i, s in zip(ids, sims) if i != -1]

    bm25_list = []
    if mode in ("hybrid", "bm25"):
        scores = idx.bm25.get_scores(list(_tok_cached(q)))
        top_idx = np.argpartition(-scores, bm25_k)[:bm25_k]
        top_scores = scores[top_idx]
        order = np.argsort(-top_scores)
        bm25_list = [(int(top_idx[i]), float(top_scores[i])) for i in order]

    if mode == "dense":
        results = [(i, s) for (i, s) in dense_list[:k]]
    elif mode == "bm25":
        results = bm25_list[:k]
    else:
        results = rrf_fuse([dense_list, bm25_list], rrf_k=rrf_k, topk=k)

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
    # Scoring/filters
    alpha_rrf: float = 0.6,
    beta_overlap: float = 0.4,
    bm25_weight: float = 0.0,
    dense_weight: float = 0.0,
    min_final_threshold: float = 0.35,
    min_lexical_overlap_fallback: float = 0.01,
    max_passages: int = 6,
    # Fast-path knobs
    retrieval_batch: int = 256,
    faiss_gpu: bool = False,
    bm25_scope: str = "base",
    # Reranker controls
    reranker: Optional[str] = None,
    rerank_topk: int = 150,
    rerank_batch_size: int = 128,
    rerank_max_length: int = 384,
    rerank_fp16: bool = False,
    # CE gate
    min_ce_score: Optional[float] = None,
    # Chain of thought
    use_cot: bool = False,
    # Streaming scoring
    stream_mode: bool = False,
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

    # Embedding client (query) – BGE/MedEmbed/Ollama
    embed_client = make_query_client(embed_model_query, device, ollama_base_url)
    print(f"Using embeddings: {embed_model_query}  [{embed_client.__class__.__name__}]")

    # LLM client
    llm = AsyncOllamaClient(llm_model, max_workers=workers, base_url=ollama_base_url)

    # Shared tokenizer (only for token counting / evidence packing)
    tok = AutoTokenizer.from_pretrained(hf_tokenizer, use_fast=True)

    # Construct the CrossEncoder ONCE
    ce = None
    if reranker:
        try:
            from sentence_transformers import CrossEncoder
            import torch
            device_ = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
            ce = CrossEncoder(reranker, device=device_, max_length=rerank_max_length)
            if rerank_fp16 and device_ == "cuda":
                try:
                    ce.model.half()
                except Exception:
                    pass
            print(f"Reranker loaded: {reranker} on {device_} (max_length={rerank_max_length}, fp16={rerank_fp16})")
        except Exception as e:
            print(f"[warn] Failed to load reranker '{reranker}': {e}. Continuing without CE.")
            ce = None

    # Data
    data = load_json_or_jsonl(eval_path)
    if limit is not None:
        data = data[:limit]
        print(f"Limited to {len(data)} examples")

    # Prepare output files early (supports streaming)
    preds: List[Dict[str, Any]] = []
    preds_path = os.path.join(save_dir, "predictions.jsonl")
    retrieved_contents_path = os.path.join(retrieved_dir, "retrieved_contents.jsonl")
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    print(f"Processing {len(data)} examples with batched retrieval (batch={retrieval_batch})... (stream_mode={stream_mode})")

    with open(preds_path, "w", encoding="utf-8") as pred_file, \
         open(retrieved_contents_path, "w", encoding="utf-8") as retrieved_file:
        for start_i in tqdm(range(0, len(data), retrieval_batch), desc="Retrieving (batched)"):
            batch_examples = data[start_i:start_i + retrieval_batch]
            t0 = time.time()
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
                ce=ce,
                rerank_topk=rerank_topk,
                rerank_batch_size=rerank_batch_size,
                min_ce_score=min_ce_score,
            )
            t1 = time.time()

            # Build prompts/metas for this retrieval batch
            local_prompts: List[str] = []
            local_metas: List[Dict[str, Any]] = []
            for ex, refs in zip(batch_examples, batch_refs):
                question = (ex.get("question") or "").strip()
                options  = make_options(ex)
                gold     = gold_letter(ex)
                prompt   = build_prompt(question, options, refs, use_cot=use_cot)
                local_prompts.append(prompt)
                local_metas.append({
                    "example": ex,
                    "gold": gold,
                    "num_contexts": len(refs),
                    "refs": refs,
                })
            print(f"[retrieval batch {start_i//retrieval_batch}] retrieval+prompt={t1-t0:.2f}s examples={len(batch_examples)} prompts={len(local_prompts)}")

            if stream_mode:
                # Score immediately in sub-batches
                for sb in range(0, len(local_prompts), batch_size):
                    sprompts = local_prompts[sb:sb+batch_size]
                    smetas   = local_metas[sb:sb+batch_size]
                    s0 = time.time()
                    # Use higher token limit for CoT
                    actual_max_tokens = 512 if use_cot else max_new_tokens
                    responses = await llm.generate_batch(sprompts, max_tokens=actual_max_tokens, temperature=0.0, top_p=1.0)
                    s1 = time.time()
                    print(f"  [score] sub-batch {sb//batch_size} size={len(sprompts)} time={s1-s0:.2f}s")
                    for j, (resp, meta) in enumerate(zip(responses, smetas)):
                        options = make_options(meta["example"])
                        allowed_letters = list(options.keys())
                        letter = parse_letter(resp, allowed_letters)
                        gold   = meta["gold"]
                        ex     = meta["example"]
                        refs   = meta["refs"]
                        is_correct = (letter == gold) if (letter and gold) else None
                        pred = {**ex, "prediction": letter, "gold": gold, "is_correct": is_correct, "raw_output": resp, "num_contexts_used": meta["num_contexts"]}
                        preds.append(pred)
                        pred_file.write(json.dumps(pred, ensure_ascii=False) + "\n")
                        retrieved_content = {
                            "question_id": ex.get("id", len(preds)-1),
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
                                    "ce_score": ref.get("ce_score", None),
                                    "url": ref.get("url", "")
                                } for ref in refs
                            ],
                            "num_passages": len(refs)
                        }
                        retrieved_file.write(json.dumps(retrieved_content, ensure_ascii=False) + "\n")
                pred_file.flush(); retrieved_file.flush()
            else:
                # Buffer for later single scoring phase
                prompts.extend(local_prompts)
                metas.extend(local_metas)        # Non-streaming scoring phase
        if not stream_mode:
            for i in tqdm(range(0, len(prompts), batch_size), desc="LLM scoring"):
                batch_prompts = prompts[i:i+batch_size]
                batch_metas   = metas[i:i+batch_size]
                s0 = time.time()
                # Use higher token limit for CoT
                actual_max_tokens = 512 if use_cot else max_new_tokens
                responses = await llm.generate_batch(batch_prompts, max_tokens=actual_max_tokens, temperature=0.0, top_p=1.0)
                s1 = time.time()
                print(f"[score] batch {i//batch_size} size={len(batch_prompts)} time={s1-s0:.2f}s")
                for j, (resp, meta) in enumerate(zip(responses, batch_metas)):
                    options = make_options(meta["example"])
                    allowed_letters = list(options.keys())
                    letter = parse_letter(resp, allowed_letters)
                    gold   = meta["gold"]
                    ex     = meta["example"]
                    refs   = meta["refs"]
                    is_correct = (letter == gold) if (letter and gold) else None
                    pred = {**ex, "prediction": letter, "gold": gold, "is_correct": is_correct, "raw_output": resp, "num_contexts_used": meta["num_contexts"]}
                    preds.append(pred)
                    pred_file.write(json.dumps(pred, ensure_ascii=False) + "\n")
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
                                "ce_score": ref.get("ce_score", None),
                                "url": ref.get("url", "")
                            } for ref in refs
                        ],
                        "num_passages": len(refs)
                    }
                    retrieved_file.write(json.dumps(retrieved_content, ensure_ascii=False) + "\n")
                pred_file.flush(); retrieved_file.flush()

    # Metrics
    total   = len(preds)
    correct = sum(1 for r in preds if r.get("is_correct") is True)
    acc     = (correct / total) if total else 0.0

    with_ctx   = [r for r in preds if r.get("num_contexts_used", 0) > 0]
    without_ctx= [r for r in preds if r.get("num_contexts_used", 0) == 0]
    acc_with   = (sum(1 for r in with_ctx if r.get("is_correct")) / len(with_ctx)) if with_ctx else 0.0
    acc_without= (sum(1 for r in without_ctx if r.get("is_correct")) / len(without_ctx)) if without_ctx else 0.0

    by_subject = defaultdict(lambda: {"n":0,"correct":0})
    by_choice  = defaultdict(lambda: {"n":0,"correct":0})
    for r in preds:
        subj = (r.get("subject_name") or "UNKNOWN").strip()
        ctyp = (r.get("choice_type") or "single").strip()
        by_subject[subj]["n"] += 1
        by_subject[subj]["correct"] += int(bool(r.get("is_correct")))
        by_choice[ctyp]["n"] += 1
        by_choice[ctyp]["correct"] += int(bool(r.get("is_correct")))
    by_subject = {k: {"count": v["n"], "accuracy": (v["correct"]/v["n"]) if v["n"] else 0.0}
                  for k, v in sorted(by_subject.items(), key=lambda kv: kv[0])}
    by_choice = {k: {"count": v["n"], "accuracy": (v["correct"]/v["n"]) if v["n"] else 0.0}
                 for k, v in sorted(by_choice.items(), key=lambda kv: kv[0])}

    elapsed = time.time() - start
    throughput = total / elapsed if elapsed > 0 else 0.0

    print(f"\n=== Performance Summary ===")
    print(f"Total examples: {total}")
    print(f"Processing time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Throughput: {throughput:.2f} examples/second")
    print(f"Average time per example: {elapsed/total:.2f}s")
    print(f"===========================")

    # Identify embedding backend for logging
    if embed_model_query.strip() == "abhinand/MedEmbed-large-v0.1":
        embedding_client_type = "MedEmbedClient"
    elif "bge" in embed_model_query.lower():
        embedding_client_type = "STEmbeddingClient"
    else:
        embedding_client_type = "OllamaClient"

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
        "examples_per_second": throughput,
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
            # Scoring knobs
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
            # Reranker knobs
            "reranker": reranker,
            "rerank_topk": rerank_topk,
            "rerank_batch_size": rerank_batch_size,
            "rerank_max_length": rerank_max_length,
            "rerank_fp16": rerank_fp16,
            # CE gate
            "min_ce_score": min_ce_score,
            # Chain of thought
            "use_cot": use_cot,
        },
        "system_info": {
            "index_dir": index_dir,
            "eval_path": eval_path,
            "embedding_model": embed_model_query,
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
    print(f"Elapsed: {elapsed:.2f}s  ({throughput:.2f} ex/s)")
    return metrics

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate hybrid RAG on MedQA/MedMCQA")
    ap.add_argument("--index-dir", type=str, required=True, help="Path to built hybrid index directory")
    ap.add_argument("--eval-path", type=str, default=os.path.join(PROJECT_ROOT, "data", "medmcqa", "dev_stratified_sample_qe.json"), help="Path to evaluation data (MedMCQA or MedQA format)")
    ap.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "evaluation_results", "eval_hybrid_rag"))
    ap.add_argument("--ollama-base-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.104:11434"))

    # Embeddings
    ap.add_argument("--embed-model-query", type=str, default="oscardp96/medcpt-query:latest",
                   help="Query embedding model. Examples: BAAI/bge-m3, abhinand/MedEmbed-large-v0.1, oscardp96/medcpt-query:latest")
    ap.add_argument("--medembed", action="store_true", help="Shortcut: use MedEmbed-large-v0.1 for query embeddings")
    ap.add_argument("--model-preset", type=str, choices=["medcpt", "medembed", "bge"],
                   help="Convenience preset for embedding model")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for ST/MedEmbed models (auto picks CUDA if available)")

    # Retrieval knobs
    ap.add_argument("--mode", type=str, default="hybrid", choices=["hybrid","dense","bm25"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--dense-k", type=int, default=80)
    ap.add_argument("--bm25-k", type=int, default=400)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--option-aware", action="store_true", help="Use option-aware retrieval for MCQ")
    ap.add_argument("--use-expansions", action="store_true", help="Fuse RRF over question + dataset query_expansions")
    ap.add_argument("--fast-mode", action="store_true", help="Skip option-aware retrieval for speed")

    # Evidence packing + tokenizer
    ap.add_argument("--max-evidence-tokens", type=int, default=1200)
    ap.add_argument("--hf-tokenizer", type=str, default="deepseek-ai/DeepSeek-R1")

    # LLM batching
    ap.add_argument("--batch-size", type=int, default=32, help="LLM inference batch size")
    ap.add_argument("--max-new-tokens", type=int, default=2, help="Max tokens to generate (letter outputs)")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads for Ollama calls")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of examples for testing")

    # Scoring/filter knobs
    ap.add_argument("--alpha-rrf", type=float, default=0.6, help="Weight for normalized RRF score")
    ap.add_argument("--beta-overlap", type=float, default=0.4, help="Weight for lexical overlap")
    ap.add_argument("--bm25-weight", type=float, default=0.0, help="Optional weight if bm25 scores are available")
    ap.add_argument("--dense-weight", type=float, default=0.0, help="Optional weight if dense scores are available")
    ap.add_argument("--final-threshold", type=float, default=0.35, help="Minimum final score to keep a passage")
    ap.add_argument("--min-lexical-overlap", type=float, default=0.01, help="If top passage overlap is below this, drop all context")
    ap.add_argument("--max-passages", type=int, default=6, help="Keep at most this many strongest passages per example")

    # Fast-path switches
    ap.add_argument("--retrieval-batch", type=int, default=256, help="How many examples to retrieve concurrently")
    ap.add_argument("--faiss-gpu", action="store_true", help="Use FAISS on GPU (if faiss-gpu is installed)")
    ap.add_argument("--bm25-scope", type=str, choices=["base", "all", "none"], default="base",
                    help="BM25 only for base question, for all queries, or disabled.")

    # Reranker (single instance + batched)
    ap.add_argument("--reranker", type=str, default=None,
                    help="Cross-encoder name, e.g., BAAI/bge-reranker-large")
    ap.add_argument("--rerank-topk", type=int, default=150,
                    help="How many top passages to rerank with the cross-encoder")
    ap.add_argument("--rerank-batch-size", type=int, default=128,
                    help="Batch size for CrossEncoder.predict()")
    ap.add_argument("--rerank-max-length", type=int, default=384,
                    help="Sequence length for the CrossEncoder")
    ap.add_argument("--rerank-fp16", action="store_true",
                    help="Run the CrossEncoder in FP16 on CUDA (faster, less VRAM)")
        # LLM model
    ap.add_argument("--llm-model", type=str, default="thewindmom/llama3-med42-8b", 
                   help="LLM model name for Ollama")

    # CE gate
    ap.add_argument("--min-ce-score", type=float, default=None,
                    help="If set (and CE is enabled), drop all context when top CE score < this threshold")

    # Chain of thought prompting
    ap.add_argument("--use-cot", action="store_true", 
                    help="Enable chain of thought prompting (longer reasoning before answer)")

    # Streaming
    ap.add_argument("--stream-mode", action="store_true", help="Score each retrieval batch immediately")

    args = ap.parse_args()

    # Normalize Ollama base URL
    def _normalize_base_url(u: Optional[str]) -> str:
        u = (u or "").strip()
        if not u:
            u = os.environ.get("OLLAMA_BASE_URL", "http://0.0.0.0:11434")
        if not re.match(r'^https?://', u):
            u = 'http://' + u
        return u.rstrip('/')
    args.ollama_base_url = _normalize_base_url(args.ollama_base_url)

    # Presets
    if args.model_preset:
        if args.model_preset == "medembed":
            args.embed_model_query = "abhinand/MedEmbed-large-v0.1"
        elif args.model_preset == "medcpt":
            args.embed_model_query = "oscardp96/medcpt-query:latest"
        elif args.model_preset == "bge":
            args.embed_model_query = "BAAI/bge-m3"

    if args.medembed:
        args.embed_model_query = "abhinand/MedEmbed-large-v0.1"

    # Paths (Windows-safe)
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
            llm_model=getattr(args, "llm_model", "thewindmom/llama3-med42-8b"),
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
            reranker=args.reranker,
            rerank_topk=args.rerank_topk,
            rerank_batch_size=args.rerank_batch_size,
            rerank_max_length=args.rerank_max_length,
            rerank_fp16=args.rerank_fp16,
            min_ce_score=args.min_ce_score,
            use_cot=args.use_cot,
            stream_mode=args.stream_mode,
        )
    )

if __name__ == "__main__":
    main()
