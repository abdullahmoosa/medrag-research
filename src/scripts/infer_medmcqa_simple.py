#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedMCQA inference (DeepSeek-8B + LoRA via Unsloth)
- Standard zero-shot CoT (default) with apples-to-apples prompt
- Optional few-shot CoT exemplars pulled from your fine-tune set
- Optional first-step debias (A/B/C/D) and digit masking
- Deterministic 1-step decoding: pick next token among {A,B,C,D}

Usage examples:
  # Baseline (no few-shot, no debias)
  python infer_fs.py

  # Few-shot CoT with 6 exemplars (subject-aware when possible)
  python infer_fs.py --shots 6

  # Few-shot + light debias against "A" bias you saw
  python infer_fs.py --shots 6 --bias-A -0.5 --bias-B 0.5 --bias-C 0.2 --bias-D 0.4
"""

import os, json, random, math
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from unsloth import FastLanguageModel

# ---------- Paths & defaults (match your current setup) ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_ID = os.path.join(BASE_DIR, "../../models/deepseek_8b")
ADAPTERS = os.path.join(BASE_DIR, "../../models/fine_tuned_models/qlora_medmcqa_tb")
SAVE_DIR_DEFAULT = os.path.join(BASE_DIR, "../../evaluation_results/infer_constrained")
DEV_DEFAULT = os.path.join(BASE_DIR, "../../data/medmcqa/dev_stratified_sample.json")
SHOTS_FILE_DEFAULT = os.path.join(BASE_DIR, "../../data/medmcqa/train_reasoning_sample.json")  # your uploaded set

# ---------- Prompt pieces ----------
SYS = (
    "[SYS] You are an expert medical AI assistant. "
    "Provide your chain-of-thought inside <Reasoning> tags and finish "
    "with the correct letter. [/SYS]\n"
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    # Try JSON
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    # JSONL
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line: 
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows

def gold_letter(ex: Dict[str, Any]) -> Optional[str]:
    cop = ex.get('cop')
    if isinstance(cop, int) and cop in (1,2,3,4):
        return {1:'A',2:'B',3:'C',4:'D'}[cop]
    ans = ex.get('answer') or ex.get('gold')
    if isinstance(ans, str) and ans.upper() in "ABCD":
        return ans.upper()
    return None

def get_reasoning(ex: Dict[str, Any], max_chars: int = 900) -> str:
    # Your SFT used 'structured_reasoning'; your sample has 'exp' — support both
    r = ex.get("structured_reasoning") or ex.get("exp") or ""
    r = (r or "").strip().replace("</s>", "")
    if max_chars is not None and len(r) > max_chars:
        r = r[:max_chars]
    return r

def format_exemplar(ex: Dict[str, Any]) -> str:
    # Full few-shot example with reasoning + final answer
    q = (ex.get('question') or '').strip()
    a = (ex.get('opa') or '').strip()
    b = (ex.get('opb') or '').strip()
    c = (ex.get('opc') or '').strip()
    d = (ex.get('opd') or '').strip()
    rsn = get_reasoning(ex)
    letter = gold_letter(ex) or "A"
    return (
        f"{q}\n"
        f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
        "### Response:\n"
        f"<Reasoning>{rsn}</Reasoning>\n"
        f"Answer: {letter}\n\n"
    )

def format_eval_prompt(ex: Dict[str, Any], fewshot_block: str = "") -> str:
    # Evaluation prompt: SYS + optional few-shot block + test question → "Answer: "
    q = (ex.get('question') or '').strip()
    a = (ex.get('opa') or '').strip()
    b = (ex.get('opb') or '').strip()
    c = (ex.get('opc') or '').strip()
    d = (ex.get('opd') or '').strip()
    return (
        f"{SYS}"
        f"{fewshot_block}"
        f"{q}\n"
        f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
        "### Response:\n"
        "<Reasoning></Reasoning>\n"
        "Answer: "
    )

# ---------- Few-shot selection ----------
def index_by_subject(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx = defaultdict(list)
    for r in rows:
        s = r.get("subject_name") or "Unknown"
        idx[s].append(r)
    return idx

def pick_few_shot(
    shots: int,
    shots_pool: List[Dict[str, Any]],
    eval_item: Dict[str, Any],
    rng: random.Random,
    subject_aware: bool = True,
) -> List[Dict[str, Any]]:
    if shots <= 0:
        return []
    # Avoid leakage by excluding same id or exact question text
    eval_id = str(eval_item.get("id") or "")
    eval_q  = (eval_item.get("question") or "").strip()
    pool = [r for r in shots_pool if str(r.get("id") or "") != eval_id and (r.get("question") or "").strip() != eval_q]
    if not pool:
        return []

    if subject_aware:
        subj = eval_item.get("subject_name") or "Unknown"
        subj_pool = [r for r in pool if (r.get("subject_name") or "Unknown") == subj]
        take_from_subj = min(shots, len(subj_pool))
        out = rng.sample(subj_pool, take_from_subj) if take_from_subj > 0 else []
        remaining = shots - len(out)
        if remaining > 0:
            others = [r for r in pool if r not in out]
            out.extend(rng.sample(others, min(remaining, len(others))))
        return out
    else:
        return rng.sample(pool, min(shots, len(pool)))

def build_fewshot_block(exemplars: List[Dict[str, Any]]) -> str:
    # Concatenate exemplars (no SYS inside the block)
    return "".join(format_exemplar(r) for r in exemplars)

# ---------- Token control helpers ----------
def build_token_sets(tok) -> Tuple[Dict[str,int], Dict[str,int], List[int]]:
    surface = ["A","B","C","D","1","2","3","4"]
    ids = {s: tok.encode(s, add_special_tokens=False) for s in surface}
    single = {s: v[0] for s,v in ids.items() if len(v) >= 1}
    letter_ids = {k: single[k] for k in ["A","B","C","D"] if k in single}
    digit_ids  = {k: single[k] for k in ["1","2","3","4"] if k in single}
    allowed    = list(single.values())
    return letter_ids, digit_ids, allowed

def debias_next_logits(next_logits: torch.Tensor,
                       letter_ids: Dict[str,int],
                       bias_map: Optional[Dict[str,float]]) -> None:
    if not bias_map: return
    for k, t_id in letter_ids.items():
        if k in bias_map:
            next_logits[..., t_id] += float(bias_map[k])

def choose_letter(next_logits: torch.Tensor,
                  letter_ids: Dict[str,int],
                  digit_ids: Dict[str,int],
                  allow_digits: bool=False) -> str:
    if next_logits.dim() == 1:
        next_logits = next_logits.unsqueeze(0)
    L_ids = list(letter_ids.values())
    L_sub = next_logits[..., L_ids]
    L_idx = torch.argmax(L_sub, dim=-1).item()
    L_token = L_ids[L_idx]
    L_name  = ["A","B","C","D"][L_idx]
    if not allow_digits:
        return L_name
    D_ids = list(digit_ids.values())
    D_sub = next_logits[..., D_ids]
    D_idx = torch.argmax(D_sub, dim=-1).item()
    D_token = D_ids[D_idx]
    D_name  = ["1","2","3","4"][D_idx]
    if next_logits[0, L_token] >= next_logits[0, D_token]:
        return L_name
    return {"1":"A","2":"B","3":"C","4":"D"}[D_name]

# ---------- Inference ----------
def run_inference(
    input_path: str,
    base_model: str,
    adapters_path: str,
    save_dir: str,
    batch_size: int = 4,
    device: Optional[str] = None,
    limit: Optional[int] = None,
    # few-shot config
    shots: int = 0,
    shots_file: Optional[str] = None,
    subject_aware: bool = False,
    seed: int = 42,
    exemplar_reasoning_chars: int = 600,
    # decoding toggles
    allow_digits: bool = False,
    bias_A: float = 0.0, bias_B: float = 0.0, bias_C: float = 0.0, bias_D: float = 0.0,
):
    ensure_dir(save_dir)
    rng = random.Random(seed)

    # Load eval & exemplar pools
    eval_rows = load_json_or_jsonl(input_path)
    if limit is not None:
        eval_rows = eval_rows[:limit]

    shots_pool = load_json_or_jsonl(shots_file) if (shots > 0 and shots_file) else []
    # pre-truncate reasoning for exemplars to keep prompt compact
    if shots_pool and exemplar_reasoning_chars is not None:
        for r in shots_pool:
            if "structured_reasoning" in r and isinstance(r["structured_reasoning"], str):
                r["structured_reasoning"] = r["structured_reasoning"][:exemplar_reasoning_chars]
            elif "exp" in r and isinstance(r["exp"], str):
                r["exp"] = r["exp"][:exemplar_reasoning_chars]

    # Model & tokenizer
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tok = FastLanguageModel.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map=device if device.startswith("cuda") else None,
    )
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"; tok.truncation_side = "left"
    model.load_adapter(adapters_path)
    try: FastLanguageModel.for_inference(model)
    except Exception: pass
    model.eval()

    letter_ids, digit_ids, _ = build_token_sets(tok)
    bias_map = {"A":bias_A, "B":bias_B, "C":bias_C, "D":bias_D}

    preds_out = os.path.join(save_dir, "predictions.jsonl")
    mets_out  = os.path.join(save_dir, "metrics.json")

    correct = 0
    total   = len(eval_rows)

    with open(preds_out, 'w', encoding='utf-8') as fout, torch.inference_mode():
        i = 0
        while i < total:
            # Build prompts batch with (optional) per-item few-shot blocks
            batch = eval_rows[i:i+batch_size]
            prompts = []
            for ex in batch:
                exs = pick_few_shot(shots, shots_pool, ex, rng, subject_aware=subject_aware)
                fewshot_block = build_fewshot_block(exs) if exs else ""
                prompts.append(format_eval_prompt(ex, fewshot_block=fewshot_block))

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(model.device)

            outputs = model(**enc)
            next_logits = outputs.logits[:, -1, :]  # [B,V]

            for j in range(next_logits.size(0)):
                nl = next_logits[j]
                # small optional debias
                debias_next_logits(nl, letter_ids, bias_map)
                letter = choose_letter(nl, letter_ids, digit_ids, allow_digits=allow_digits)
                gold   = gold_letter(batch[j])
                is_correct = (letter == gold) if (letter and gold) else None
                if is_correct:
                    correct += 1

                rec = dict(batch[j])
                rec.update({
                    "prediction": letter,
                    "gold": gold,
                    "is_correct": is_correct,
                    "raw_output": letter,
                })
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            i += batch_size

    acc = (correct/total) if total else 0.0
    with open(mets_out, 'w', encoding='utf-8') as f:
        json.dump({"total": total, "correct": correct, "accuracy": acc}, f, indent=2)
    print(f"Done. Saved to: {save_dir}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Your original defaults
    parser.add_argument("--input", type=str, default=DEV_DEFAULT)
    parser.add_argument("--base-model", type=str, default=MODEL_ID)
    parser.add_argument("--adapters", type=str, default=ADAPTERS)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1)  # kept for CLI parity (unused)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    # Few-shot controls (OFF by default)
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot exemplars to prepend per question.")
    parser.add_argument("--shots-file", type=str, default=SHOTS_FILE_DEFAULT, help="Path to CoT exemplars (train set).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for few-shot selection.")
    parser.add_argument("--exemplar-reasoning-chars", type=int, default=600, help="Truncate exemplar reasoning to this many chars.")

    # Decoding tweaks (default: comparable baseline)
    parser.add_argument("--allow-digits", action="store_true", help="Allow digits 1-4 as alternatives; otherwise force A-D.")
    parser.add_argument("--bias-A", type=float, default=0.0)
    parser.add_argument("--bias-B", type=float, default=0.0)
    parser.add_argument("--bias-C", type=float, default=0.0)
    parser.add_argument("--bias-D", type=float, default=0.0)

    args = parser.parse_args()
    run_inference(
        input_path=args.input,
        base_model=args.base_model,
        adapters_path=args.adapters,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        shots=args.shots,
        shots_file=args.shots_file,
        seed=args.seed,
        exemplar_reasoning_chars=args.exemplar_reasoning_chars,
        allow_digits=args.allow_digits,
        bias_A=args.bias_A, bias_B=args.bias_B, bias_C=args.bias_C, bias_D=args.bias_D,
    )
