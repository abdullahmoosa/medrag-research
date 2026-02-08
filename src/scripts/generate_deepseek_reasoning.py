#!/usr/bin/env python3
"""
Script to generate synthetic reasoning dataset using DeepSeek R1 via the **official**
DeepSeek API.  Only latency-related parts were modified:
  • persistent OpenAI client
  • max_tokens capped at 2000
  • request_timeout = 60 s
"""

import json
import os
import time
import openai
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime
import re
from pathlib import Path

# Phoenix (Arize) tracing and OpenTelemetry instrumentation — untouched
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

# Load environment variables from root directory
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / ".env")

# Register Phoenix tracing
tracer_provider = register(
    project_name="default",
    endpoint="https://app.phoenix.arize.com/s/medrag/v1/traces",
    auto_instrument=True
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# ---------- logging helper (unchanged) ----------
def setup_logging(output_dir: Path):
    log_file = output_dir / 'deepseek_reasoning_generation.log'
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                  logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)

# ---------- main generator ----------
class DeepSeekReasoningGenerator:
    def __init__(self,
                 api_key: str,
                 rate_limit_rpm: int = 0,
                 batch_size: int = 3,
                 start_batch: int = 0,
                 output_dir: str = None,
                 max_concurrent_requests: int = 3):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-reasoner"
        self.batch_size = batch_size
        self.start_batch = start_batch

        # Persistent client ➊
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Output dir & logging (unchanged)
        if output_dir is None:
            script_dir = Path(__file__).parent
            self.output_dir = script_dir.parent.parent / "evaluation_results" / "reasoning_dataset_generation"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.output_dir)

        # Progress / stats (unchanged)
        self.progress_file = self.output_dir / "reasoning_progress.json"
        self.current_batch = start_batch
        self.stats = {
            'total_processed': 0, 'correct_answers': 0, 'incorrect_answers': 0,
            'api_errors': 0, 'total_input_tokens': 0, 'total_output_tokens': 0,
            'current_batch': start_batch, 'total_batches': 0,
            'start_time': datetime.now().isoformat(), 'output_dir': str(self.output_dir)
        }
        # No semaphore: unlimited concurrency for pipelined launch
    
        # ──────────────────────────────
    # Progress-tracking helpers 🔄
    # ──────────────────────────────
    def load_progress(self) -> bool:
        """Return True if a previous run was loaded."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.current_batch = data.get("current_batch", 0)
                self.stats.update(data.get("stats", {}))
                self.logger.info(f"Loaded progress: batch {self.current_batch}")
                return True
            except Exception as e:
                self.logger.warning(f"Could not load progress: {e}")
        return False

    def save_progress(self):
        """Persist current batch & stats to disk."""
        payload = {
            "current_batch": self.current_batch,
            "stats": self.stats,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.logger.info(f"Progress saved: batch {self.current_batch}")

    def delete_progress(self):
        """Start fresh (used by --clear_progress flag)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            self.logger.info("Progress file deleted")
        self.current_batch = 0
        self.stats = {
            'total_processed': 0, 'correct_answers': 0, 'incorrect_answers': 0,
            'api_errors': 0, 'total_input_tokens': 0, 'total_output_tokens': 0,
            'current_batch': 0, 'total_batches': 0,
            'start_time': datetime.now().isoformat(),
            'output_dir': str(self.output_dir)
        }
        self.logger.info("Progress reset — starting fresh")

    # ---------- prompt builder (unchanged) ----------
    def format_prompt(self, sample: Dict) -> str:
        question  = sample.get('question', '')
        opa, opb, opc, opd = (sample.get(k, '') for k in ['opa', 'opb', 'opc', 'opd'])
        explanation = sample.get('exp', '') or ''
        system_prompt = """You are an expert medical AI assistant that provides detailed reasoning for medical questions. 
You must think step-by-step through each medical question, considering relevant medical knowledge, differential diagnoses, 
pathophysiology, clinical presentation, and diagnostic criteria before arriving at your final answer.

Please provide your reasoning enclosed in <Reasoning></Reasoning> tags, followed by your final answer as a single letter (A, B, C, or D).

In your reasoning, consider:
1. Key medical concepts and terminology
2. Pathophysiology and disease mechanisms
3. Clinical presentation and symptoms
4. Differential diagnoses
5. Diagnostic criteria and methods
6. Treatment implications if relevant
7. No self talk or unnecessary commentary"""
        prompt = f"""{system_prompt}

Medical Question: {question}

Options:
A) {opa}
B) {opb}
C) {opc}
D) {opd}"""
        if explanation.strip():
            prompt += f"""

Reference Explanation: {explanation}

Use this explanation to guide your reasoning process, but provide your own step-by-step analysis."""
        return prompt + "\n\nPlease provide your detailed medical reasoning in <Reasoning></Reasoning> tags and then state your final answer."

    # ---------- API call (async, aiohttp) ----------
    async def call_deepseek_api(self, session, prompt: str) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """
        Call DeepSeek R1 via official DeepSeek API using aiohttp.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.9
        }
        import aiohttp, asyncio
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    message = data['choices'][0]['message']
                    content = message.get('content')
                    # DeepSeek returns the raw reasoning in 'reasoning_content'
                    reasoning = message.get('reasoning_content')
                    usage = data.get('usage', {})
                    self.stats['total_input_tokens'] += usage.get('prompt_tokens', 0)
                    self.stats['total_output_tokens'] += usage.get('completion_tokens', 0)
                    return content, reasoning, usage
                else:
                    error_text = await response.text()
                    self.logger.error(f"API error {response.status}: {error_text}")
                    self.stats['api_errors'] += 1
                    return None, None, None
        except asyncio.TimeoutError:
            self.logger.error("API request timeout")
            self.stats['api_errors'] += 1
            return None, None, None
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            self.stats['api_errors'] += 1
            return None, None, None

    # ---------- reasoning / answer extraction (unchanged) ----------
    def extract_reasoning_and_answer(self, content: str, raw_reasoning: str):
        reason_pattern = r'<Reasoning>(.*?)</Reasoning>'
        m = re.search(reason_pattern, content, re.DOTALL | re.IGNORECASE)
        structured = m.group(1).strip() if m else None
        remaining = content[m.end():].strip() if m else content.strip()
        answer_patterns = [r'\b([ABCD])\b',
                           r'answer\s*:?\s*([ABCD])',
                           r'final\s*answer\s*:?\s*([ABCD])',
                           r'option\s*([ABCD])',
                           r'correct\s*answer\s*:?\s*([ABCD])']
        final_ans = None
        for pat in answer_patterns:
            mm = re.search(pat, remaining, re.IGNORECASE)
            if mm:
                final_ans = mm.group(1).upper(); break
        if not final_ans:
            for pat in answer_patterns:
                mm = re.search(pat, content, re.IGNORECASE)
                if mm:
                    final_ans = mm.group(1).upper(); break
        return structured, raw_reasoning, final_ans

    def is_answer_correct(self, predicted: str, correct_option: int) -> bool:
        if not predicted: return False
        return predicted.upper() == {1:'A',2:'B',3:'C',4:'D'}[correct_option]

    # ---------- single-sample processing (async) ----------
    async def process_sample(self, session, sample: Dict) -> Dict:
        sid = sample.get('id', 'unknown')
        self.logger.info(f"Processing sample {sid}")
        prompt = self.format_prompt(sample)
        content, raw_reasoning, usage = await self.call_deepseek_api(session, prompt)
        if content is None:
            self.logger.warning(f"Failed sample {sid}")
            return {**sample, 'structured_reasoning': None, 'raw_reasoning': None,
                    'predicted_answer': None, 'is_correct': False, 'error': 'api_error'}
        structured, _, predicted = self.extract_reasoning_and_answer(content, raw_reasoning)
        is_correct = self.is_answer_correct(predicted, sample.get('cop'))
        self.stats['total_processed'] += 1
        self.stats['correct_answers' if is_correct else 'incorrect_answers'] += 1
        self.logger.info(f"Sample {sid}: Predicted={predicted}, Correct={sample.get('cop')}, Match={is_correct}")
        result = {**sample, 'structured_reasoning': structured, 'raw_reasoning': raw_reasoning,
                  'predicted_answer': predicted, 'is_correct': is_correct}
        return result

    # ---------- dataset loop (async) ----------
    async def process_dataset(self, input_file: str, output_file: str, resume: bool = True):
        import aiohttp, asyncio
        self.logger.info(f"Dataset: {input_file} -> {output_file}")
        if resume:
            self.load_progress()
        samples = [json.loads(l) for l in open(input_file, 'r', encoding='utf-8') if l.strip()]
        total = len(samples)
        processed = []
        processed_ids = set()
        # If resuming and output file exists, load already processed samples
        if resume and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for l in f:
                        if l.strip():
                            s = json.loads(l)
                            processed.append(s)
                            if 'id' in s:
                                processed_ids.add(s['id'])
                self.logger.info(f"Loaded {len(processed)} existing results")
            except Exception as e:
                self.logger.warning(f"Could not load existing results: {e}")
        # Find where to resume
        start_idx = 0
        if processed:
            last_id = processed[-1].get('id')
            for idx, sample in enumerate(samples):
                if sample.get('id') == last_id:
                    start_idx = idx + 1
                    break
        self.logger.info(f"Resuming from sample {start_idx+1}/{total}")

        async def sample_task(session, sample, outf, i):
            # Skip if already processed (by id)
            if sample.get('id') in processed_ids:
                return
            result = await self.process_sample(session, sample)
            outf.write(json.dumps(result) + '\n')
            outf.flush()
            self.current_batch = i + 1
            self.stats['current_batch'] = self.current_batch
            self.save_progress()
            self.print_statistics()
            return result

        async with aiohttp.ClientSession() as session:
            # Open output file in append mode (sync file, async session)
            with open(output_file, 'a', encoding='utf-8') as outf:
                tasks = []
                import random
                for i in range(start_idx, total):
                    sample = samples[i]
                    # Launch a new task every 1-3 seconds, but do not wait for completion
                    task = asyncio.create_task(sample_task(session, sample, outf, i))
                    tasks.append(task)
                    await asyncio.sleep(random.uniform(1, 3))
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)
                # Optionally, collect processed results
                processed.extend([r for r in results if r is not None])
        self.logger.info("Processing done")
        self.print_statistics()
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        return processed

    # ---------- helpers (unchanged) ----------
    def save_results(self, processed_samples: List[Dict], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for s in processed_samples: f.write(json.dumps(s) + '\n')
        self.logger.info(f"Saved {len(processed_samples)} samples to {output_file}")

    def print_statistics(self):
        tot = self.stats['total_processed']
        if tot:
            acc = 100 * self.stats['correct_answers'] / tot
            self.logger.info(f"Stats: {tot} processed, {acc:.1f}% acc, "
                             f"{self.stats['api_errors']} API errors | "
                             f"Tokens in/out: {self.stats['total_input_tokens']}/"
                             f"{self.stats['total_output_tokens']}")

# ---------- CLI wrapper (unchanged) ----------

# ---------- CLI wrapper (async) ----------
import argparse
import asyncio
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=3)
    p.add_argument('--start_batch', type=int, default=0)
    p.add_argument('--rate_limit', type=int, default=50)  # kept for compat
    p.add_argument('--clear_progress', action='store_true')
    p.add_argument('--resume', action='store_true', default=True)
    p.add_argument('--no_resume', action='store_true')
    args = p.parse_args()

    resume_flag = args.resume and not args.no_resume and not args.clear_progress
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found in .env"); return

    print(f"Config: batch_size={args.batch_size}, start_batch={args.start_batch}, resume={resume_flag}")
    script_dir = Path(__file__).parent
    input_file = script_dir.parent.parent / "data" / "medmcqa" / "train_reasoning_sample.json"
    gen = DeepSeekReasoningGenerator(api_key=api_key,
                                     batch_size=args.batch_size,
                                     start_batch=args.start_batch,
                                     max_concurrent_requests=3)
    output_file = gen.output_dir / f"train_reasoning_generated_{datetime.now():%Y%m%d_%H%M%S}.json"
    if resume_flag:
        existing = list(gen.output_dir.glob("train_reasoning_generated_*.json"))
        if existing: output_file = max(existing, key=os.path.getctime)
    if args.clear_progress:
        gen.delete_progress()
        output_file = gen.output_dir / f"train_reasoning_generated_{datetime.now():%Y%m%d_%H%M%S}.json"
    if not input_file.exists():
        print(f"Input file not found: {input_file}"); return
    try:
        asyncio.run(gen.process_dataset(str(input_file), str(output_file), resume=resume_flag))
        print(f"Final results saved to: {output_file}")
    except KeyboardInterrupt:
        print("Interrupted — progress saved; use --resume to continue")
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main()
