#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned DeepSeek-8B on MedMCQA validation set
"""

import os, re
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

BASE_DIR = os.path.dirname(__file__)
MODEL_ID = os.path.join(BASE_DIR, "../../models/deepseek_8b")
ADAPTERS = os.path.join(BASE_DIR, "../../models/fine_tuned_models/qlora_medmcqa_tb")
DEV      = os.path.join(BASE_DIR, "../../data/medmcqa/dev_reasoning.jsonl")

# Load model with adapters
model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID, load_in_4bit=True, device_map="cuda:0"
)
tok.pad_token = tok.eos_token
model.load_adapter(ADAPTERS)
model.is_loaded_in_4bit = True

# Load validation data
ds = load_dataset("json", data_files={"validation": DEV})
val_ds = ds["validation"]

SYS = ("[SYS] You are an expert medical AI assistant. "
       "Provide your chain-of-thought inside <Reasoning> tags and finish "
       "with the correct letter. [/SYS]\n")

def formatting_func(ex):
    prompt = f"{SYS}{ex['question'].strip()}\n" \
             f"A) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}\n\n" \
             "### Response:\n" \
             f"<Reasoning>{ex['structured_reasoning'].strip()[:1500]}</Reasoning>\n" \
             f"Answer:"
    return prompt

def _extract_letter(txt):
    m = re.search(r"Answer\s*:?\s*([ABCD])", txt, re.I)
    return m.group(1).upper() if m else None

def _extract_reasoning(txt):
    m = re.search(r"<Reasoning>(.*?)</Reasoning>", txt, re.S)
    return m.group(1).strip() if m else None

# Run predictions
import json
results = []
for ex in val_ds:
    prompt = formatting_func(ex)
    input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=32, do_sample=False)
    decoded = tok.decode(output[0], skip_special_tokens=True)
    letter = _extract_letter(decoded)
    reasoning = _extract_reasoning(decoded)
    gold = "ABCD"[ex["cop"]-1]
    results.append({
        "question": ex["question"],
        "gold": gold,
        "prediction": letter,
        "reasoning": reasoning
    })

# Compute accuracy

num_correct = sum(r["prediction"] == r["gold"] for r in results)
acc = num_correct / len(results)
print(f"Validation accuracy (full set): {acc:.3f}")

# Print a few predictions
for i in range(min(5, len(results))):
    print(f"Q: {results[i]['question']}")
    print(f"Gold: {results[i]['gold']}, Pred: {results[i]['prediction']}")
    print(f"Reasoning: {results[i]['reasoning']}")
    print("---")

# Save results to JSON
save_dir = os.path.join(BASE_DIR, '../../evaluation_results/finetuned-deepseek-8b-reasoning-v1')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'validation_predictions.json')
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Results saved to {save_path}")