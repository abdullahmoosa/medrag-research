#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for evaluating the retrieval functionality on medical questions.

This script allows you to:
1. Test a single question against the RAG system
2. Visualize top-k retrieved passages from PubMed
3. Compare vector-only vs. hybrid retrieval results

Usage:
    python src/scripts/test_retriever.py --question "What is the mechanism of action of metformin?"
    
Options:
    --index-dir: Path to the FAISS index
    --kb-dir: Path to the knowledge base
    --embed-model: Embedding model to use
    --top-k: Number of passages to retrieve (default: 5)
    --hybrid: Enable hybrid search (vector + BM25)
"""
import os
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pprint import pprint

# Ensure project root on sys.path when running as a script
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the RAG utilities
from src.rag_utils import (
    build_or_load_index,
    DEFAULT_KB_DIR,
    DEFAULT_INDEX_DIR,
    Passage,
    EmbeddingModel,
    HybridRetriever,
)

DEFAULT_EMBED_MODEL = "oscardp96/medcpt-article"


def highlight_matches(text: str, query: str) -> str:
    """Highlight query terms in the text with ** markers."""
    import re
    highlighted = text
    # Break query into terms
    terms = [t.lower() for t in re.findall(r'\w+', query) if len(t) > 3]
    for term in terms:
        pattern = re.compile(r'\b' + re.escape(term) + r'\w*\b', re.IGNORECASE)
        highlighted = pattern.sub(r'**\g<0>**', highlighted)
    return highlighted


def evaluate_question(
    question: str,
    kb_dir: str = DEFAULT_KB_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    embed_model_id: str = DEFAULT_EMBED_MODEL,
    device: Optional[str] = None,
    top_k: int = 5,
    compare_methods: bool = True,
) -> None:
    """Test retrieval for a single question."""
    print(f"\n{'='*80}\nTesting retrieval for question: \"{question}\"\n{'='*80}\n")
    
    # Load the index and embedder
    index, embedder = build_or_load_index(
        kb_dir=kb_dir,
        index_dir=index_dir,
        embed_model_id=embed_model_id,
        device=device,
    )
    
    # Initialize the hybrid retriever
    hybrid_retriever = HybridRetriever(index, embedder)
    
    # Vector-only retrieval
    print(f"\n{'-'*40}\nVECTOR RETRIEVAL (Top-{top_k})\n{'-'*40}\n")
    q_emb = embedder.embed([question])
    vector_hits = index.search(q_emb, top_k=top_k)
    
    for i, (score, passage) in enumerate(vector_hits, 1):
        print(f"[{i}] Score: {score:.4f} | Source: {passage.meta.get('source', 'unknown')}")
        print(highlight_matches(passage.text, question))
        print(f"{'-'*80}\n")
    
    if compare_methods:
        # Hybrid retrieval (BM25 + Vector)
        print(f"\n{'-'*40}\nHYBRID RETRIEVAL (Top-{top_k})\n{'-'*40}\n")
        hybrid_hits = hybrid_retriever.search(question, top_k=top_k)
        
        for i, (score, passage) in enumerate(hybrid_hits, 1):
            print(f"[{i}] Score: {score:.4f} | Source: {passage.meta.get('source', 'unknown')}")
            print(highlight_matches(passage.text, question))
            print(f"{'-'*80}\n")
        
        # Compare overlap between methods
        vector_passages = {id(p): (s, p) for s, p in vector_hits}
        hybrid_passages = {id(p): (s, p) for s, p in hybrid_hits}
        
        overlap = set(vector_passages.keys()) & set(hybrid_passages.keys())
        print(f"\nOverlap between methods: {len(overlap)}/{top_k} passages")
        
        # Show unique passages from hybrid
        if len(overlap) < top_k:
            print(f"\n{'-'*40}\nUNIQUE PASSAGES FROM HYBRID\n{'-'*40}\n")
            for pid in set(hybrid_passages.keys()) - set(vector_passages.keys()):
                score, passage = hybrid_passages[pid]
                print(f"Score: {score:.4f} | Source: {passage.meta.get('source', 'unknown')}")
                print(highlight_matches(passage.text, question))
                print(f"{'-'*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG retrieval on a single question")
    parser.add_argument("--question", type=str, required=True, 
                        help="Question to test retrieval on")
    parser.add_argument("--kb-dir", type=str, default=DEFAULT_KB_DIR,
                        help="Path to the knowledge base directory")
    parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR,
                        help="Path to the index directory")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL,
                        help="Embedding model to use")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu, cuda)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of passages to retrieve")
    parser.add_argument("--hybrid", action="store_true", default=True,
                        help="Compare vector-only and hybrid retrieval")
    args = parser.parse_args()
    
    evaluate_question(
        question=args.question,
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        embed_model_id=args.embed_model,
        device=args.device,
        top_k=args.top_k,
        compare_methods=args.hybrid,
    )
