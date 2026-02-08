#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Asynchronous RAG evaluation on MedMCQA using PubMed knowledge base.

This script is optimized for parallel processing to maximize throughput:
- Concurrent retrieval operations
- Asynchronous LLM inference
- Batched embedding and generation

Usage:
    python src/scripts/async_rag_eval.py --hybrid --workers 4
"""
import os
import sys
import json
import re
import glob
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
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
)
from src.rag_utils import load_json_or_jsonl, ensure_dir, HybridRetriever

DEFAULT_EVAL_PATH = os.path.join(PROJECT_ROOT, "data", "medmcqa", "dev_stratified_sample.json")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "evaluation_results", "thewindmom_llama3-med42-8b-zero-shot-rag")

DEFAULT_EMBED_MODEL = "oscardp96/medcpt-article"
DEFAULT_LLM_MODEL = "thewindmom/llama3-med42-8b"

LETTER_RE = re.compile(r"\b([ABCD])\b")
ANSWER_RE = re.compile(r"Answer\s*[:：]\s*([ABCD])", re.I)


# -----------------------------
# Async LLM Client
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


# -----------------------------
# LLM generation
# -----------------------------

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
    
    # Normal HF model loading
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model = None
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                load_in_4bit=True,
                device_map="auto",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model.eval()
    
    # Wrap HF model in a class with similar async interface
    class AsyncHFWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            
        async def generate_batch(self, prompts: List[str], max_tokens: int = 8,
                              temperature: float = 0.0, top_p: float = 1.0) -> List[str]:
            """Generate responses for a batch of prompts."""
            # Process in synchronous way but return with async interface
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )
            input_ids = enc.input_ids.to(self.model.device)
            attn = enc.attention_mask.to(self.model.device)
            input_lengths = attn.sum(dim=1)
            
            with torch.inference_mode():
                gen_ids = self.model.generate(
                    input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            results = []
            for j in range(gen_ids.size(0)):
                start = int(input_lengths[j].item())
                out_tokens = gen_ids[j, start:]
                text = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
                results.append(text)
                
            return results
    
    return AsyncHFWrapper(model, tok), tok


SYS_PROMPT = (
    "You are a medical expert answering multiple-choice questions. "
    "Read the provided context carefully and use it to answer the question. "
    "You MUST respond with ONLY the letter (A, B, C, or D) that corresponds to the correct answer. "
    "Do not provide explanations or additional text. "
    "Format your response exactly as: Answer: [LETTER]"
)


def build_prompt(question: str, options: Dict[str, str], contexts: List[str]) -> str:
    ctx = "\n\n".join([f"[Context {i+1}] {c}" for i, c in enumerate(contexts)])
    opt_str = "\n".join([f"{k}) {v}" for k, v in options.items()])
    return (
        f"[INSTRUCTIONS] {SYS_PROMPT} [/INSTRUCTIONS]\n\n"
        f"[MEDICAL CONTEXT]\n{ctx}\n[/MEDICAL CONTEXT]\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[OPTIONS]\n{opt_str}\n[/OPTIONS]\n\n"
        f"Based on the medical context provided above, select the correct answer from the options.\n"
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
# MedMCQA helpers
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


def truncate_to_context(tokenizer, texts: List[str], max_tokens: int) -> List[str]:
    kept: List[str] = []
    used = 0
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if used + len(toks) <= max_tokens:
            kept.append(t)
            used += len(toks)
        else:
            if max_tokens - used > 16:
                sub = tokenizer.decode(toks[: max_tokens - used], skip_special_tokens=True)
                if sub.strip():
                    kept.append(sub)
                    used = max_tokens
            break
    return kept


# -----------------------------
# Async retrieval and processing
# -----------------------------

async def process_batch(
    batch: List[Dict[str, Any]],
    llm_client,
    tokenizer,
    retriever,
    embedder,
    index,
    top_k: int = 5,
    ctx_token_budget: int = 900,
    max_new_tokens: int = 8
) -> List[Dict[str, Any]]:
    """Process a batch of examples concurrently."""
    # First prepare all prompts with retrieved contexts
    prompts = []
    results = []
    
    # Process retrieval in parallel using ThreadPoolExecutor
    async def get_contexts(example):
        q = (example.get('question') or '').strip()
        options = make_options(example)
        
        # Concurrent retrieval
        if retriever is not None:
            hits = retriever.search(q, top_k=top_k)
            contexts = [p.text for _, p in hits]
        else:
            q_emb = embedder.embed([q])
            hits = index.search(q_emb, top_k=top_k)
            contexts = [p.text for _, p in hits]
            
        contexts = truncate_to_context(tokenizer, contexts, max_tokens=ctx_token_budget)
        prompt = build_prompt(q, options, contexts)
        return prompt, gold_letter(example), example
    
    # Process all retrievals concurrently
    tasks = [get_contexts(ex) for ex in batch]
    retrieval_results = await asyncio.gather(*tasks)
    
    # Prepare for LLM batch processing
    prompts = []
    golds = []
    examples = []
    
    for prompt, gold, example in retrieval_results:
        prompts.append(prompt)
        golds.append(gold)
        examples.append(example)
    
    # Generate all responses concurrently
    responses = await llm_client.generate_batch(
        prompts, 
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0
    )
    
    # Process the results
    for i, (example, response, gold) in enumerate(zip(examples, responses, golds)):
        # Clean and parse the response
        cleaned_response = response.strip()
        letter = parse_letter(cleaned_response)
        
        # If no letter found, try to extract from the beginning of response
        if letter is None and cleaned_response:
            # Look for the first character that might be an answer
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
        })
        results.append(result)
        
    return results


# -----------------------------
# Evaluation
# -----------------------------

async def evaluate_rag_async(
    eval_path: str = DEFAULT_EVAL_PATH,
    kb_dir: str = DEFAULT_KB_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    embed_model_id: str = DEFAULT_EMBED_MODEL,
    llm_model_id: str = DEFAULT_LLM_MODEL,
    device: Optional[str] = None,
    top_k: int = 5,
    ctx_token_budget: int = 900,
    batch_size: int = 16,
    max_new_tokens: int = 8,
    limit: Optional[int] = None,
    hybrid: bool = True,
    workers: int = 4,
) -> Dict[str, Any]:
    """Asynchronous RAG evaluation with concurrent retrieval and inference."""
    
    # Create dynamic save directory based on model name if default is used
    if save_dir == DEFAULT_SAVE_DIR:
        # Extract model name and create directory
        model_name = llm_model_id.replace("/", "_").replace(":", "_")
        retrieval_type = "hybrid" if hybrid else "vector"
        save_dir = os.path.join(PROJECT_ROOT, "evaluation_results", f"{model_name}-zero-shot-rag-{retrieval_type}")
    
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
    # Hybrid retriever (BM25 + Vector) or pure vector
    retriever = HybridRetriever(index, embedder) if hybrid else None

    # LLM with async interface
    llm_client, tok = load_llm(llm_model_id, device=device, max_workers=workers)

    # load eval data
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
        process_batch(
            batch, 
            llm_client, 
            tok, 
            retriever, 
            embedder, 
            index, 
            top_k, 
            ctx_token_budget, 
            max_new_tokens
        ) 
        for batch in batches
    ]
    
    # Process all tasks with proper async iteration
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches"):
        results = await task
        all_results.extend(results)
    
    # Compute metrics
    correct = sum(1 for r in all_results if r.get("is_correct"))
    total = len(all_results)
    acc = (correct / total) if total else 0.0
    
    # Save results
    with open(preds_out, 'w', encoding='utf-8') as fout:
        for result in all_results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    elapsed = time.time() - start_time
    metrics = {
        "total": total, 
        "correct": correct, 
        "accuracy": acc,
        "elapsed_seconds": elapsed,
        "examples_per_second": total / elapsed if elapsed > 0 else 0
    }
    
    with open(mets_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Saved to: {save_dir}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Elapsed time: {elapsed:.2f}s ({metrics['examples_per_second']:.2f} examples/sec)")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Async RAG eval on MedMCQA using PubMed knowledge base")
    parser.add_argument("--kb-dir", type=str, default=DEFAULT_KB_DIR)
    parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--eval-path", type=str, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL,
                        help="HF SentenceTransformer model id, e.g., oscardp96/medcpt-article or with :tag")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ctx-token-budget", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid BM25 + Vector retrieval")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for concurrent processing")
    args = parser.parse_args()

    # Run the async evaluation
    asyncio.run(evaluate_rag_async(
        eval_path=args.eval_path,
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        save_dir=args.save_dir,
        embed_model_id=args.embed_model,
        llm_model_id=args.llm_model,
        device=args.device,
        top_k=args.top_k,
        ctx_token_budget=args.ctx_token_budget,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        hybrid=args.hybrid,
        workers=args.workers,
    ))