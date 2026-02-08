#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Async-batched evaluation of fine-tuned DeepSeek-8B on MedMCQA dev stratified set (zero-shot)
- Loads the fine-tuned model & adapters (same paths as evaluate_finetuned_deepseek8b.py)
- Evaluates dev_stratified_sample.json (JSON or JSONL)
- Uses safe batching instead of concurrent generate() calls (thread-unsafe)
- Enforces 1 second delay between batches
- Saves predictions & metrics to evaluation_results/finetuned-deepseek-8b-reasoning-zero-shot
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from unsloth import FastLanguageModel
import math
try:
    from transformers.generation.logits_process import LogitsProcessor
except Exception:  # fallback for older transformers
    from transformers.generation.logits_process import LogitsProcessor

# ----------------------- Paths & Constants -----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_ID = os.path.join(BASE_DIR, "../../models/deepseek_8b")
ADAPTERS = os.path.join(BASE_DIR, "../../models/fine_tuned_models/qlora_medmcqa_tb")
INPUT_DEFAULT = os.path.join(BASE_DIR, "../../data/medmcqa/dev_stratified_sample.json")
SAVE_DIR_DEFAULT = os.path.join(BASE_DIR, '../../evaluation_results/finetuned-deepseek-8b-reasoning-zero-shot')

# Match the exact SYS prompt used during finetuning
SYS_PROMPT = (
    "[SYS] You are an expert medical AI assistant. "
    "Provide your chain-of-thought inside <Reasoning> tags and finish "
    "with the correct letter. [/SYS]\n"
)

ANSWER_PATTERNS = [
    r"Answer\s*[:：]\s*([ABCD])\b",
    r"\bFinal\s*Answer\s*[:：-]?\s*([ABCD])\b",
    r"\bPrediction\s*[:：-]?\s*([ABCD])\b",
    r"\bOption\s*[:：-]?\s*([ABCD])\b",
]
REASONING_PATTERN = re.compile(r"<Reasoning>(.*?)</Reasoning>", re.S | re.I)

# ----------------------- Utilities -----------------------

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return []
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass
        items: List[Dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------- Evaluator -----------------------

class AsyncFinetunedDeepseekEvaluator:
    def __init__(
        self,
        model_path: str = MODEL_ID,
        adapters_path: str = ADAPTERS,
        save_dir: str = SAVE_DIR_DEFAULT,
        max_concurrency: int = 2,
        delay_seconds: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.save_dir = os.path.abspath(save_dir)
        ensure_dir(self.save_dir)

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # Optional CUDA perf tweaks
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        # Load base model
        self.model, self.tok = FastLanguageModel.from_pretrained(
            model_path,
            load_in_4bit=True,
            device_map=self.device if self.device.startswith("cuda") else None,
        )
        # Ensure tokenizer behavior is stable for decoder-only models
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"
        # Load LoRA adapters
        self.model.load_adapter(adapters_path)
        # Ensure inference patches are applied
        try:
            FastLanguageModel.for_inference(self.model)
        except Exception:
            pass
        self.model.is_loaded_in_4bit = True
        self.model.eval()

        self.batch_size = max(1, int(max_concurrency))
        self.delay_seconds = max(0.0, float(delay_seconds))
        self._last_request_ts: float = 0.0
        self.max_new_tokens = 192

        # Disallow model from emitting hidden CoT markers seen in some DeepSeek variants
        self.bad_words_ids: List[List[int]] = []
        bad_words = [
            "<think>", "</think>", "<think", "</think",
            "<|begin_of_thought|>", "<|end_of_thought|>",
            "<p", "</p", "<div", "</div", "<span", "</span",
            "<script", "</script", "<br", "</br",
            "http://", "https://", "://"
        ]
        for bw in bad_words:
            ids = self.tok(bw, add_special_tokens=False).input_ids
            if ids:
                self.bad_words_ids.append(ids)

    # ---------- Prompt formatting (zero-shot) ----------
    def format_prompt(self, ex: Dict[str, Any]) -> str:
        q = ex.get('question', '').strip()
        opa = ex.get('opa', '').strip()
        opb = ex.get('opb', '').strip()
        opc = ex.get('opc', '').strip()
        opd = ex.get('opd', '').strip()
        prompt = (
            f"{SYS_PROMPT}{q}\n"
            f"A) {opa}\nB) {opb}\nC) {opc}\nD) {opd}\n\n"
            "### Response:\n"
            "<Reasoning></Reasoning>\n"
            "Answer: "
        )
        return prompt

    # ---------- Extraction ----------
    def extract_letter(self, txt: str) -> Optional[str]:
        # Prefer the last explicit Answer match
        last_match = None
        for pat in ANSWER_PATTERNS:
            for m in re.finditer(pat, txt, re.I):
                last_match = m
        if last_match:
            return last_match.group(1).upper()
        # fallback: take first capital A-D near the end
        tail = txt[-400:]
        m = re.search(r"\b([ABCD])\b", tail)
        return m.group(1).upper() if m else None

    def extract_reasoning(self, txt: str) -> Optional[str]:
        m = REASONING_PATTERN.search(txt)
        if m:
            return m.group(1).strip()
        # Fallback 1: from last <Reasoning> to </Reasoning> or Answer:
        start = txt.rfind("<Reasoning>")
        if start != -1:
            tail = txt[start + len("<Reasoning>"):]
            end_tag_idx = tail.find("</Reasoning>")
            ans_match = re.search(r"\bAnswer\s*[:：]", tail, re.I)
            if end_tag_idx != -1:
                return tail[:end_tag_idx].strip()
            if ans_match:
                return tail[:ans_match.start()].strip()
            return tail.strip()
        # Fallback 2: between last '### Response:' and 'Answer:'
        resp_idx = txt.rfind("### Response:")
        if resp_idx != -1:
            seg = txt[resp_idx + len("### Response:"):]
            ans_match = re.search(r"\bAnswer\s*[:：]", seg, re.I)
            if ans_match:
                return seg[:ans_match.start()].strip()
            return seg.strip()
        return None

    def gold_letter(self, ex: Dict[str, Any]) -> Optional[str]:
        cop = ex.get('cop')
        if isinstance(cop, int) and cop in (1, 2, 3, 4):
            return {1: 'A', 2: 'B', 3: 'C', 4: 'D'}[cop]
        answer = ex.get('answer') or ex.get('gold')
        if isinstance(answer, str) and answer.upper() in 'ABCD':
            return answer.upper()
        return None

    async def _respect_rate_limit(self):
        now = time.monotonic()
        wait = self.delay_seconds - (now - self._last_request_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_ts = time.monotonic()

    # ---------- Batch generate ----------
    def _batch_generate_sync(self, prompts: List[str]) -> List[str]:
        class StopAfterAnswerProcessor(LogitsProcessor):
            def __init__(self, tok, start_lengths: torch.Tensor):
                self.tok = tok
                self.start_lengths = start_lengths
                self.done = [False] * start_lengths.size(0)
                self.ans_re = re.compile(r"Answer\s*:?\s*([ABCD])", re.IGNORECASE)

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                bsz = input_ids.size(0)
                for i in range(bsz):
                    if self.done[i]:
                        scores[i, self.tok.eos_token_id] = 1e9
                        continue
                    start = int(self.start_lengths[i].item())
                    gen = input_ids[i, start:]
                    if gen.numel() == 0:
                        continue
                    text = self.tok.decode(gen, skip_special_tokens=True)
                    if self.ans_re.search(text) or ("</Reasoning>" in text and re.search(r"[ABCD](?![A-Z])", text[-6:])):
                        self.done[i] = True
                        scores[i, self.tok.eos_token_id] = 1e9
                return scores

        with torch.inference_mode():
            enc = self.tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            input_ids = enc.input_ids.to(self.model.device)
            attn_mask = enc.attention_mask.to(self.model.device)
            input_lengths = attn_mask.sum(dim=1)

            logits_processor = [StopAfterAnswerProcessor(self.tok, input_lengths)]

            gen_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
                repetition_penalty=1.05,
                logits_processor=logits_processor,
            )
            if self.bad_words_ids:
                gen_kwargs["bad_words_ids"] = self.bad_words_ids

            outputs = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                **gen_kwargs,
            )
            texts: List[str] = []
            for i in range(outputs.size(0)):
                start_idx = int(input_lengths[i].item())
                gen_tokens = outputs[i, start_idx:]
                text = self.tok.decode(gen_tokens, skip_special_tokens=True)
                texts.append(text)
            return texts

    async def _batch_generate(self, prompts: List[str]) -> List[str]:
        await self._respect_rate_limit()
        return await asyncio.to_thread(self._batch_generate_sync, prompts)

    # ---------- Dataset evaluation ----------
    async def evaluate(
        self,
        data: List[Dict[str, Any]],
        save_dir: Optional[str] = None,
        save_basename: str = 'predictions.jsonl',
    ) -> Tuple[str, Optional[str]]:
        out_dir = os.path.abspath(save_dir or self.save_dir)
        ensure_dir(out_dir)
        pred_path = os.path.join(out_dir, save_basename)
        metrics_path = os.path.join(out_dir, 'metrics.json')

        total = len(data)
        correct = 0
        total_with_gold = 0

        print(f"Starting evaluation: {total} samples | batch_size={self.batch_size} | delay={self.delay_seconds}s", flush=True)
        t0 = time.time()

        with open(pred_path, 'w', encoding='utf-8') as f:
            for start in range(0, total, self.batch_size):
                batch_t0 = time.time()
                batch = data[start:start + self.batch_size]
                prompts = [self.format_prompt(ex) for ex in batch]
                try:
                    gen_texts = await self._batch_generate(prompts)
                except Exception as e:
                    for i, ex in enumerate(batch):
                        out = {**ex, 'prediction': None, 'reasoning': None,
                               'error': f'generate_error: {type(e).__name__}: {e}'}
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    done = min(start + self.batch_size, total)
                    elapsed = time.time() - t0
                    avg = elapsed / max(1, done)
                    eta = avg * max(0, total - done)
                    print(f"Batch {done//self.batch_size}/{math.ceil(total/self.batch_size)} failed: {type(e).__name__} | done {done}/{total} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)
                    continue

                for ex, gen in zip(batch, gen_texts):
                    gen_clean = self._clean_text(gen)
                    reasoning = self.extract_reasoning(gen_clean)
                    pred = self.extract_letter(gen_clean)
                    gold = self.gold_letter(ex)
                    is_correct = (pred == gold) if (pred and gold) else None
                    if is_correct is not None:
                        total_with_gold += 1
                        if is_correct:
                            correct += 1
                    out = {
                        **ex,
                        'prediction': pred,
                        'gold': gold,
                        'is_correct': is_correct,
                        'reasoning': reasoning,
                        'raw_output': gen,
                        'cleaned_output': gen_clean,
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

                done = min(start + self.batch_size, total)
                batch_time = time.time() - batch_t0
                elapsed = time.time() - t0
                avg = elapsed / max(1, done)
                eta = avg * max(0, total - done)
                print(f"Batch {done//self.batch_size}/{math.ceil(total/self.batch_size)} done in {batch_time:.1f}s | completed {done}/{total} | avg/sample {avg:.2f}s | ETA {eta:.1f}s", flush=True)

        elapsed = time.time() - t0
        accuracy = (correct / total_with_gold) if total_with_gold else None
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total': total,
            'evaluated_with_gold': total_with_gold,
            'correct': correct,
            'accuracy': accuracy,
            'max_concurrency': self.batch_size,
            'delay_seconds': self.delay_seconds,
            'model_path': MODEL_ID,
            'adapters_path': ADAPTERS,
            'elapsed_seconds': elapsed,
            'throughput_samples_per_sec': (total / elapsed) if elapsed > 0 else None,
        }
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(summary, mf, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {pred_path}", flush=True)
        print(f"Saved metrics to {metrics_path}", flush=True)
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.4f} ({correct}/{total_with_gold})", flush=True)
        print(f"Total time: {elapsed:.1f}s | Throughput: {summary['throughput_samples_per_sec']:.2f} samples/s", flush=True)
        return pred_path, metrics_path

    def _clean_text(self, txt: str) -> str:
        # Keep only the segment after the last response header if present
        idx = txt.rfind("### Response:")
        if idx != -1:
            txt = txt[idx:]
        # Drop repeated URL-like or corrupt tokens
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S | re.I)
        txt = re.sub(r"https?://\S+", "", txt)
        txt = re.sub(r"[:/]{2,}", "/", txt)
        return txt.strip()


# ----------------------- CLI -----------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Async evaluation of fine-tuned DeepSeek-8B (zero-shot)")
    p.add_argument('--input', type=str, default=INPUT_DEFAULT, help='Path to dev_stratified_sample.json (JSON or JSONL)')
    p.add_argument('--save-dir', type=str, default=SAVE_DIR_DEFAULT, help='Output directory for predictions & metrics')
    p.add_argument('--max-concurrency', type=int, default=2, help='Batch size (items per generate call)')
    p.add_argument('--delay-seconds', type=float, default=1.0, help='Global delay between batches (seconds)')
    p.add_argument('--limit', type=int, default=0, help='Optional limit of samples for quick runs (0 = all)')
    return p.parse_args()


def main():
    args = parse_args()

    data = load_json_or_jsonl(args.input)
    if args.limit and args.limit > 0:
        data = data[:args.limit]
    if not data:
        raise SystemExit(f"No data loaded from: {args.input}")

    evaluator = AsyncFinetunedDeepseekEvaluator(
        model_path=MODEL_ID,
        adapters_path=ADAPTERS,
        save_dir=args.save_dir,
        max_concurrency=args.max_concurrency,
        delay_seconds=args.delay_seconds,
    )

    asyncio.run(evaluator.evaluate(data))


if __name__ == '__main__':
    main()
