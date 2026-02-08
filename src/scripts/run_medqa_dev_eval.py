"""Evaluate model on MedQA (USMLE) dev split.
Mirrors run_zero_shot_dev.py but points to the MedQA dev.jsonl dataset.
Only changed what is necessary: data file path and data_split label.
"""
import json
from batch_evaluation import BatchEvaluator

# Path to MedQA (USMLE) dev dataset (JSONL: one JSON object per line)
DATA_PATH = r"e:\nusrat\medrag\data\medQA USMLE\questions\US\dev.jsonl"

# Load data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dev_data = [json.loads(line) for line in f]

# Initialize evaluator (keeping same model + config, only split label differs)
evaluator = BatchEvaluator(
    model_name="OussamaELALLAM/MedExpert:latest",
    use_thinking_model=False,  # stay consistent with zero-shot standard evaluation
    data_split="medqa_dev"    # distinguish this dataset in logs/results
)

# Reset any prior evaluation state
evaluator.reset_evaluation()

# Local model API endpoint
API_URL = "http://localhost:11434/api/generate"

# Run batch evaluation (large batch size since we process in one go)
evaluator.evaluate_batch(dev_data, API_URL, batch_size=1000)
