#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved RAG system with better context filtering and selection.

Key improvements:
1. Context relevance filtering
2. Adaptive context selection
3. Better prompt engineering
4. Fallback to no-RAG when contexts are poor

Usage:
    python src/scripts/improved_rag_eval.py --hybrid --workers 4 --limit 10
"""
import os
import sys
import json
import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Ensure project root on sys.path when running as a script
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Shared utils
from src.rag_utils import (
    build_or_load_index,
    DEFAULT_KB_DIR,
    DEFAULT_INDEX_DIR,
    Passage,
    EmbeddingModel,
    HybridRetriever,
    load_json_or_jsonl,
    ensure_dir,
)

DEFAULT_EVAL_PATH = os.path.join(PROJECT_ROOT, "data", "medmcqa", "dev_stratified_sample.json")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "evaluation_results", "thewindmom_llama3-med42-8b-improved-rag")
DEFAULT_EMBED_MODEL = "oscardp96/medcpt-article"
DEFAULT_LLM_MODEL = "thewindmom/llama3-med42-8b"

LETTER_RE = re.compile(r"\b([ABCD])\b")
ANSWER_RE = re.compile(r"Answer\s*[:：]\s*([ABCD])", re.I)


# -----------------------------
# Context Quality Assessment
# -----------------------------

def assess_context_relevance(question: str, contexts: List[str], min_score: float = 0.3) -> List[str]:
    """Filter contexts based on relevance to the question."""
    # Simple keyword overlap scoring
    question_words = set(question.lower().split())
    question_words = {w for w in question_words if len(w) > 3}  # Filter short words
    
    relevant_contexts = []
    for context in contexts:
        context_words = set(context.lower().split())
        
        # Calculate overlap score
        overlap = len(question_words.intersection(context_words))
        score = overlap / len(question_words) if question_words else 0
        
        if score >= min_score:
            relevant_contexts.append(context)
    
    return relevant_contexts


def select_best_contexts(question: str, contexts: List[str], max_contexts: int = 3) -> List[str]:
    """Select the most relevant contexts based on keyword overlap and medical terms."""
    
    # Medical keywords that indicate relevance
    medical_keywords = {
        'treatment', 'diagnosis', 'symptoms', 'disease', 'condition', 'therapy',
        'medication', 'drug', 'clinical', 'patient', 'medical', 'syndrome',
        'anatomy', 'physiology', 'pathology', 'surgery', 'procedure'
    }
    
    question_lower = question.lower()
    scored_contexts = []
    
    for context in contexts:
        context_lower = context.lower()
        
        # Score based on question word overlap
        q_words = set(question_lower.split())
        c_words = set(context_lower.split())
        overlap_score = len(q_words.intersection(c_words)) / len(q_words) if q_words else 0
        
        # Bonus for medical terms
        medical_bonus = sum(1 for word in medical_keywords if word in context_lower) * 0.1
        
        # Penalty for very long contexts (research papers)
        length_penalty = max(0, (len(context) - 500) / 1000) * 0.2
        
        final_score = overlap_score + medical_bonus - length_penalty
        scored_contexts.append((final_score, context))
    
    # Sort by score and take top contexts
    scored_contexts.sort(reverse=True, key=lambda x: x[0])
    return [context for score, context in scored_contexts[:max_contexts]]


# -----------------------------
# Async LLM Client (same as before)
# -----------------------------

class AsyncOllamaClient:
    """Asynchronous wrapper for Ollama inference."""
    
    def __init__(self, model_name: str, tokenizer, max_workers: int = 4):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def generate(self, prompt: str, max_tokens: int = 8, temperature: float = 0.0, top_p: float = 1.0):
        """Async generate text from prompt."""
        import ollama
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens
                }
            )
        )
        return response['response']
        
    async def generate_batch(self, prompts: List[str], max_tokens: int = 8, 
                          temperature: float = 0.0, top_p: float = 1.0) -> List[str]:
        """Generate responses for a batch of prompts concurrently."""
        tasks = [
            self.generate(prompt, max_tokens, temperature, top_p)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)


def load_llm(model_id: str = DEFAULT_LLM_MODEL, device: Optional[str] = None, max_workers: int = 4):
    # Check if this is an Ollama model (thewindmom/llama3-med42-8b)
    if "thewindmom/llama3-med42-8b" in model_id:
        try:
            import ollama
            print(f"Using Ollama for LLM generation with model: {model_id}")
            # Dummy tokenizer for compatibility
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            tok.truncation_side = "left"
            
            # Return the async client and tokenizer
            return AsyncOllamaClient(model_id, tok, max_workers=max_workers), tok
            
        except Exception as e:
            print(f"Error using Ollama for LLM: {e}. Falling back to HF model loading.")
    
    # [Rest of HF loading code would be here - same as before]
    raise NotImplementedError("HF model loading not implemented in this snippet")


# -----------------------------
# Improved Prompting
# -----------------------------

def build_improved_prompt(question: str, options: Dict[str, str], contexts: List[str]) -> str:
    """Build improved prompt with better context integration."""
    
    if not contexts:
        # Fallback to no-RAG prompt if no good contexts
        opt_str = "\n".join([f"{k}) {v}" for k, v in options.items()])
        return (
            f"You are a medical expert. Answer this multiple-choice question using your medical knowledge.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{opt_str}\n\n"
            f"Select the correct answer (A, B, C, or D). Respond with only the letter.\n"
            f"Answer: "
        )
    
    # Build prompt with contexts
    ctx_str = "\n\n".join([f"Reference {i+1}: {c[:300]}..." if len(c) > 300 else f"Reference {i+1}: {c}" 
                           for i, c in enumerate(contexts)])
    opt_str = "\n".join([f"{k}) {v}" for k, v in options.items()])
    
    return (
        f"You are a medical expert. Use the provided medical references to help answer this question.\n\n"
        f"Medical References:\n{ctx_str}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_str}\n\n"
        f"Based on the references and your medical knowledge, select the correct answer.\n"
        f"Respond with only the letter (A, B, C, or D).\n"
        f"Answer: "
    )


def parse_letter(text: str) -> Optional[str]:
    # Clean the text first
    text = text.strip()
    
    # Look for "Answer: X" pattern first
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    
    # Look for standalone letter at the beginning
    if len(text) >= 1 and text[0].upper() in 'ABCD':
        return text[0].upper()
    
    # Look for any single letter in the text
    m2 = LETTER_RE.search(text)
    if m2:
        return m2.group(1).upper()
        
    # Last resort: look for patterns like "A)", "B.", etc.
    option_pattern = re.search(r'\b([ABCD])[)\.]', text, re.I)
    if option_pattern:
        return option_pattern.group(1).upper()
    
    return None


# -----------------------------
# MedMCQA helpers (same as before)
# -----------------------------

def gold_letter(ex: Dict[str, Any]) -> Optional[str]:
    cop = ex.get('cop')
    if isinstance(cop, int) and cop in (1, 2, 3, 4):
        return {1: 'A', 2: 'B', 3: 'C', 4: 'D'}[cop]
    ans = ex.get('answer') or ex.get('gold')
    if isinstance(ans, str) and ans.upper() in 'ABCD':
        return ans.upper()
    return None


def make_options(ex: Dict[str, Any]) -> Dict[str, str]:
    return {
        'A': (ex.get('opa') or '').strip(),
        'B': (ex.get('opb') or '').strip(),
        'C': (ex.get('opc') or '').strip(),
        'D': (ex.get('opd') or '').strip(),
    }


# -----------------------------
# Improved Processing
# -----------------------------

async def process_batch_improved(
    batch: List[Dict[str, Any]],
    llm_client,
    tokenizer,
    retriever,
    embedder,
    index,
    top_k: int = 5,
    max_new_tokens: int = 8
) -> List[Dict[str, Any]]:
    """Process a batch with improved context selection and fallback strategy."""
    
    results = []
    
    # Process retrieval in parallel
    async def get_improved_contexts(example):
        q = (example.get('question') or '').strip()
        options = make_options(example)
        
        # Retrieve contexts
        if retriever is not None:
            hits = retriever.search(q, top_k=top_k*2)  # Get more to filter from
            all_contexts = [p.text for _, p in hits]
        else:
            q_emb = embedder.embed([q])
            hits = index.search(q_emb, top_k=top_k*2)
            all_contexts = [p.text for _, p in hits]
        
        # Filter and select best contexts
        relevant_contexts = assess_context_relevance(q, all_contexts, min_score=0.2)
        best_contexts = select_best_contexts(q, relevant_contexts, max_contexts=3)
        
        # Build improved prompt
        prompt = build_improved_prompt(q, options, best_contexts)
        
        return prompt, gold_letter(example), example, len(best_contexts)
    
    # Process all retrievals concurrently
    tasks = [get_improved_contexts(ex) for ex in batch]
    retrieval_results = await asyncio.gather(*tasks)
    
    # Prepare for LLM batch processing
    prompts = []
    golds = []
    examples = []
    context_counts = []
    
    for prompt, gold, example, ctx_count in retrieval_results:
        prompts.append(prompt)
        golds.append(gold)
        examples.append(example)
        context_counts.append(ctx_count)
    
    # Generate all responses concurrently
    responses = await llm_client.generate_batch(
        prompts, 
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0
    )
    
    # Process the results
    for i, (example, response, gold, ctx_count) in enumerate(zip(examples, responses, golds, context_counts)):
        # Clean and parse the response
        cleaned_response = response.strip()
        letter = parse_letter(cleaned_response)
        
        # If no letter found, try to extract from the beginning of response
        if letter is None and cleaned_response:
            first_char = cleaned_response[0].upper()
            if first_char in 'ABCD':
                letter = first_char
        
        is_correct = (letter == gold) if (letter and gold) else None
        
        result = dict(example)
        result.update({
            "prediction": letter,
            "gold": gold,
            "is_correct": is_correct,
            "raw_output": response,
            "num_contexts_used": ctx_count,
        })
        results.append(result)
        
    return results


# -----------------------------
# Main Evaluation Function
# -----------------------------

async def evaluate_improved_rag_async(
    eval_path: str = DEFAULT_EVAL_PATH,
    kb_dir: str = DEFAULT_KB_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    embed_model_id: str = DEFAULT_EMBED_MODEL,
    llm_model_id: str = DEFAULT_LLM_MODEL,
    device: Optional[str] = None,
    top_k: int = 8,  # Retrieve more to filter from
    batch_size: int = 16,
    max_new_tokens: int = 8,
    limit: Optional[int] = None,
    hybrid: bool = True,
    workers: int = 4,
) -> Dict[str, Any]:
    """Improved RAG evaluation with context filtering and adaptive strategies."""
    
    # Create dynamic save directory
    if save_dir == DEFAULT_SAVE_DIR:
        model_name = llm_model_id.replace("/", "_").replace(":", "_")
        retrieval_type = "hybrid" if hybrid else "vector"
        save_dir = os.path.join(PROJECT_ROOT, "evaluation_results", f"{model_name}-improved-rag-{retrieval_type}")
    
    ensure_dir(save_dir)
    start_time = time.time()
    
    # Index and embedder
    index, embedder = build_or_load_index(
        kb_dir=kb_dir,
        index_dir=index_dir,
        embed_model_id=embed_model_id,
        device=device,
        batch_size=batch_size,
    )
    
    # Hybrid retriever
    retriever = HybridRetriever(index, embedder) if hybrid else None

    # LLM with async interface
    llm_client, tok = load_llm(llm_model_id, device=device, max_workers=workers)

    # Load eval data
    data = load_json_or_jsonl(eval_path)
    if limit is not None:
        data = data[:limit]

    preds_out = os.path.join(save_dir, "predictions.jsonl")
    mets_out = os.path.join(save_dir, "metrics.json")

    # Process in batches asynchronously
    all_results = []
    
    # Split data into batches
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    
    # Create tasks for all batches
    tasks = [
        process_batch_improved(
            batch, 
            llm_client, 
            tok, 
            retriever, 
            embedder, 
            index, 
            top_k, 
            max_new_tokens
        ) 
        for batch in batches
    ]
    
    # Process all tasks
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches"):
        results = await task
        all_results.extend(results)
    
    # Compute metrics
    correct = sum(1 for r in all_results if r.get("is_correct"))
    total = len(all_results)
    acc = (correct / total) if total else 0.0
    
    # Additional analysis
    with_context = [r for r in all_results if r.get("num_contexts_used", 0) > 0]
    without_context = [r for r in all_results if r.get("num_contexts_used", 0) == 0]
    
    context_acc = sum(1 for r in with_context if r.get("is_correct")) / len(with_context) if with_context else 0
    no_context_acc = sum(1 for r in without_context if r.get("is_correct")) / len(without_context) if without_context else 0
    
    # Save results
    with open(preds_out, 'w', encoding='utf-8') as fout:
        for result in all_results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    elapsed = time.time() - start_time
    metrics = {
        "total": total, 
        "correct": correct, 
        "accuracy": acc,
        "with_context_count": len(with_context),
        "with_context_accuracy": context_acc,
        "without_context_count": len(without_context), 
        "without_context_accuracy": no_context_acc,
        "elapsed_seconds": elapsed,
        "examples_per_second": total / elapsed if elapsed > 0 else 0
    }
    
    with open(mets_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Saved to: {save_dir}")
    print(f"Overall Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"With Context: {context_acc:.4f} ({len(with_context)} questions)")
    print(f"Without Context: {no_context_acc:.4f} ({len(without_context)} questions)")
    print(f"Elapsed time: {elapsed:.2f}s ({metrics['examples_per_second']:.2f} examples/sec)")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improved RAG eval on MedMCQA")
    parser.add_argument("--kb-dir", type=str, default=DEFAULT_KB_DIR)
    parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--eval-path", type=str, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Run the improved evaluation
    asyncio.run(evaluate_improved_rag_async(
        eval_path=args.eval_path,
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        save_dir=args.save_dir,
        embed_model_id=args.embed_model,
        llm_model_id=args.llm_model,
        device=args.device,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        hybrid=args.hybrid,
        workers=args.workers,
    ))
