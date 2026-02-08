#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qLoRA finetune of DeepSeek-8B on MedMCQA   –   logs → TensorBoard
"""

import os, re

# ─── MINIMAL FIX ──────────────────────────────────────────────
# Disable Unsloth’s gradient-checkpointing fast-path (not yet
# stable on Windows / Transformers ≤4.54).  **Nothing else changes.**
os.environ["UNSLOTH_DISABLE_GRADIENT_CHECKPOINTING"] = "1"
# ──────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

BASE_DIR = os.path.dirname(__file__)
MODEL_ID = os.path.join(BASE_DIR, "../../models/deepseek_8b")
TRAIN    = os.path.join(BASE_DIR, "../../data/medmcqa/train_reasoning.jsonl")
DEV      = os.path.join(BASE_DIR, "../../data/medmcqa/dev_reasoning.jsonl")
OUT_DIR  = os.path.join(BASE_DIR,
              "../../models/fine_tuned_models/qlora_medmcqa_tb")

# ───────── model ─────────
model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID, load_in_4bit=True, device_map="cuda:0"
)
tok.pad_token = tok.eos_token

import types
def _no_to(self, *args, **kwargs):
    return self                       # already on the correct GPU & dtype
model.to = types.MethodType(_no_to, model)

lora_cfg = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","down_proj","up_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model.add_adapter(lora_cfg)
model.is_loaded_in_4bit = True        # guard flag
model.gradient_checkpointing_enable()
# ───────── data ─────────
ds  = load_dataset("json", data_files={"train":TRAIN, "validation":DEV})


SYS = ("[SYS] You are an expert medical AI assistant. "
       "Provide your chain-of-thought inside <Reasoning> tags and finish "
       "with the correct letter. [/SYS]\n")

def formatting_func(ex):
    if isinstance(ex["question"], list):           # batched call
        out = []
        for q, a, b, c, d, cop, rsn in zip(
                ex["question"], ex["opa"], ex["opb"], ex["opc"],
                ex["opd"], ex["cop"], ex["structured_reasoning"]):
            prompt = f"{SYS}{q.strip()}\n" \
                     f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n" \
                     "### Response:\n" \
                     f"<Reasoning>{rsn.strip()[:1500]}</Reasoning>\n" \
                     f"Answer: {'ABCD'[cop-1]}</s>"
            out.append(prompt)
        return out
    prompt = f"{SYS}{ex['question'].strip()}\n" \
             f"A) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}\n\n" \
             "### Response:\n" \
             f"<Reasoning>{ex['structured_reasoning'].strip()[:1500]}</Reasoning>\n" \
             f"Answer: {'ABCD'[ex['cop']-1]}</s>"
    return [prompt]

def _extract_letter(txt):
    m = re.search(r"Answer\s*:?\s*([ABCD])", txt, re.I)
    return m.group(1).upper() if m else None

def compute_metrics(eval_pred):
    preds, _ = eval_pred
    outs = tok.batch_decode(preds, skip_special_tokens=True)
    gold = ["ABCD"[x["cop"]-1] for x in ds["validation"]]
    acc  = sum(p==g for p, g in zip(map(_extract_letter, outs), gold)) / len(gold)
    return {"accuracy": acc}

args = TrainingArguments(
    output_dir               = OUT_DIR,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8,
    num_train_epochs         = 3,
    save_steps               = 260,
    evaluation_strategy      = "steps",   # "epoch" also works
    eval_steps               = 50,        # evaluate every 50 opt-steps
    logging_steps            = 5,
    report_to                = "tensorboard",
    logging_dir              = "tb_logs_medmcqa_v2",
)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tok,
    train_dataset = ds["train"],
    eval_dataset  = ds["validation"],
    formatting_func = formatting_func,
    compute_metrics = compute_metrics,
    args          = args,
    move_model_to_device = False,      # still required for 8-bit
)

trainer.train()
model.save_pretrained(OUT_DIR)
print("✅ Done – adapters saved to:", OUT_DIR)
