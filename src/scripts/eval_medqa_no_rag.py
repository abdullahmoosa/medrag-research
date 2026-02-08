#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Asynchronous evaluation on MedQA (USMLE-style) without RAG (no retrieval context).
"""

import os
import sys
import json
import re
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Sequence

from tqdm import tqdm

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.rag_utils import load_json_or_jsonl, ensure_dir

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("medqa_eval")

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_EVAL_PATH = os.path.join(
    PROJECT_ROOT, "data", "medQA USMLE", "questions", "US", "test.jsonl"
)
DEFAULT_LLM_MODEL = "thewindmom/llama3-med42-8b:latest"
DEFAULT_SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation_results",
    "medqa_usmle",
    "no_rag",
    f"{DEFAULT_LLM_MODEL.replace('/', '_').replace(':', '_')}-{'cot' if '--use-cot' in sys.argv else 'zero-shot'}",
)
DEFAULT_OLLAMA_BASE_URL = "http://172.25.208.1:11434"

# -----------------------------------------------------------------------------
# Regex helpers
# -----------------------------------------------------------------------------
ANSWER_PREFIX_RE = re.compile(r"Answer\s*[:：]\s*([A-Z])", re.I)
LETTER_ANY_RE = re.compile(r"\b([A-Z])\b")
OPTION_TOKEN_RE = re.compile(r"\b([A-Z])[)\.]\s")

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
SYS_PROMPT = (
    "You are a medical expert answering multiple-choice USMLE-style questions. "
    "Respond ONLY with the single capital letter that corresponds to the correct option. "
    "No explanation. Format strictly: Answer: [LETTER]"
)

COT_SYS_PROMPT = (
    "You are a medical expert answering multiple-choice USMLE-style questions. "
    "Think through the problem step by step, then provide your final answer. "
    "End your response with 'Answer: [LETTER]' where [LETTER] is the single capital letter of your choice."
)

# =============================================================================
# Async Ollama Client (robust)
# =============================================================================
class AsyncOllamaClient:
    def __init__(
        self,
        model_name: str,
        tokenizer,
        max_workers: int,
        base_url: str,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
    ):
        import ollama

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

        # single long-lived client
        self.client = ollama.Client(host=self.base_url)

        logger.info("Initialized Ollama client at %s", self.base_url)

    def _sync_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        attempt = 0
        backoff = self.initial_backoff
        last_exc = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                res = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                    },
                )
                if isinstance(res, dict) and "response" in res:
                    return res["response"]
                return getattr(res, "response", "")
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Ollama generate failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2

        raise last_exc

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    self.executor,
                    lambda: self._sync_generate(prompt, max_tokens, temperature, top_p),
                )
            except Exception as e:
                logger.error("Final Ollama failure: %s", e)
                return ""

    async def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        if max_tokens == 8 and any(
            "step by step" in p.lower() or "think through" in p.lower() for p in prompts
        ):
            max_tokens = 256

        tasks = [
            self.generate(p, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            for p in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

# =============================================================================
# HF wrapper (unchanged, safe)
# =============================================================================
class AsyncHFWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        if max_tokens == 8 and any(
            "step by step" in p.lower() or "think through" in p.lower() for p in prompts
        ):
            max_tokens = 256

        enc = self.tokenizer(
            list(prompts),
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

        outputs = []
        for i in range(gen_ids.size(0)):
            start = int(input_lengths[i].item())
            outputs.append(
                self.tokenizer.decode(gen_ids[i, start:], skip_special_tokens=True)
            )
        return outputs

# =============================================================================
# LLM loader
# =============================================================================
def load_llm(
    model_id: str,
    device: Optional[str],
    max_workers: int,
    ollama_base_url: str,
):
    raw_id = model_id.strip()
    if raw_id.lower().startswith("m:"):
        raw_id = raw_id.split(":", 1)[1].strip()

    is_ollama = ":" in raw_id

    if is_ollama:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        tok.truncation_side = "left"
        return (
            AsyncOllamaClient(
                raw_id,
                tok,
                max_workers=max_workers,
                base_url=ollama_base_url,
            ),
            tok,
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return AsyncHFWrapper(model, tok), tok

# =============================================================================
# Prompt / parsing
# =============================================================================
def build_prompt_no_rag(question: str, options: Dict[str, str], use_cot: bool) -> str:
    opt_lines = [f"{k}) {options[k]}" for k in sorted(options.keys()) if options[k]]
    allowed = " ".join(sorted(options.keys()))
    sys_prompt = COT_SYS_PROMPT if use_cot else SYS_PROMPT
    return (
        f"[INSTRUCTIONS] {sys_prompt} The valid answer choices are: {allowed}. [/INSTRUCTIONS]\n\n"
        f"[QUESTION]\n{question}\n\n[OPTIONS]\n" +
        "\n".join(opt_lines) +
        "\n[/OPTIONS]\n\nAnswer: "
    )

def extract_gold_letter(ex: Dict[str, Any]) -> Optional[str]:
    letter = ex.get("answer_idx") or ex.get("gold")
    if isinstance(letter, str) and len(letter) == 1:
        return letter.upper()
    return None

def parse_prediction(raw: str, allowed: Sequence[str]) -> Optional[str]:
    if not raw:
        return None
    allowed = set(a.upper() for a in allowed)
    m = ANSWER_PREFIX_RE.search(raw)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    for ch in raw:
        if ch.upper() in allowed:
            return ch.upper()
    return None

# =============================================================================
# Batch processing
# =============================================================================
async def process_batch_no_rag(batch, llm_client, max_new_tokens, use_cot):
    prompts, golds, opt_keys = [], [], []

    for ex in batch:
        opts = {k.upper(): v.strip() for k, v in ex["options"].items()}
        prompts.append(build_prompt_no_rag(ex["question"], opts, use_cot))
        golds.append(extract_gold_letter(ex))
        opt_keys.append(sorted(opts.keys()))

    responses = await llm_client.generate_batch(
        prompts, max_tokens=max_new_tokens, temperature=0.0, top_p=1.0
    )

    out = []
    for ex, resp, gold, keys in zip(batch, responses, golds, opt_keys):
        pred = parse_prediction(resp, keys)
        out.append({
            **ex,
            "prediction": pred,
            "gold": gold,
            "is_correct": pred == gold if pred and gold else None,
            "raw_output": resp,
        })
    return out

# =============================================================================
# Evaluation
# =============================================================================
async def evaluate_medqa_no_rag_async(
    eval_path,
    save_dir,
    llm_model_id,
    device,
    batch_size,
    max_new_tokens,
    limit,
    workers,
    use_cot,
    ollama_base_url,
):
    ensure_dir(save_dir)

    llm_client, _ = load_llm(
        llm_model_id,
        device=device,
        max_workers=workers,
        ollama_base_url=ollama_base_url,
    )

    data = load_json_or_jsonl(eval_path)
    if limit:
        data = data[:limit]

    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    tasks = [process_batch_no_rag(b, llm_client, max_new_tokens, use_cot) for b in batches]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            results.extend(await coro)
        except Exception:
            logger.exception("Batch failed")

    correct = sum(r["is_correct"] for r in results if r["is_correct"] is not None)
    accuracy = correct / len(results)

    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "predictions.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    metrics = {
        "total": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "model": llm_model_id,
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Accuracy: %.4f (%d/%d)", accuracy, correct, len(results))
    return metrics

# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    args = parser.parse_args()

    asyncio.run(
        evaluate_medqa_no_rag_async(
            args.eval_path,
            args.save_dir,
            args.llm_model,
            args.device,
            args.batch_size,
            args.max_new_tokens,
            args.limit,
            args.workers,
            args.use_cot,
            args.ollama_base_url,
        )
    )
