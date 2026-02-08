"""Run evaluation on MedQA USMLE dev set with on-the-fly normalization.

This script adapts MedQA-style records (question, options dict, answer/answer_idx) to the
internal evaluator expectations used by evaluation.py and batch_evaluation.py.
"""
import json
from batch_evaluation import BatchEvaluator

DATA_PATH = r"e:\nusrat\medrag\data\medQA USMLE\questions\US\dev.jsonl"

# Load original MedQA style JSONL
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        # Normalize: create legacy fields if missing so existing code paths still work
        options = rec.get("options", {})
        # Map first four options (A-D) if present for backward compatibility
        rec.setdefault("opa", options.get("A"))
        rec.setdefault("opb", options.get("B"))
        rec.setdefault("opc", options.get("C"))
        rec.setdefault("opd", options.get("D"))
        # Provide a 'cop' (1-4) if answer_idx is within A-D range (for older metrics expectations)
        ans_letter = rec.get("answer_idx") or rec.get("answer") or rec.get("answer_letter")
        if ans_letter:
            letter = ans_letter.strip().upper()
            letter_to_num = {"A":1, "B":2, "C":3, "D":4}
            if letter not in rec:
                rec["correct_option_letter"] = letter
            if letter in letter_to_num:
                rec.setdefault("cop", letter_to_num[letter])
        records.append(rec)

# Initialize evaluator (use new split label)
evaluator = BatchEvaluator(
    model_name="OussamaELALLAM/MedExpert:latest",
    use_thinking_model=False,
    data_split="medqa_dev"
)

# Do not reset automatically to prevent accidental deletion; comment out if needed
# evaluator.reset_evaluation()

API_URL = "http://localhost:11434/api/generate"

evaluator.evaluate_batch(records, API_URL, batch_size=1000)
