import json
import os
from transformers import AutoTokenizer

# Use DeepSeek tokenizer from HuggingFace
# Model: DeepSeek-R1-Distill-Llama-8B
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Input and output paths
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/medmcqa/train_reasoning_generated.json'))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/medmcqa/train_reasoning_filtered.jsonl'))

MAX_TOTAL_TOKENS = 3000

kept = 0
removed = 0

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        question = item.get('question', '')
        options = item.get('options', '')
        if isinstance(options, list):
            options = ' '.join(str(opt) for opt in options)
        elif not isinstance(options, str):
            options = str(options)
        # Remove if is_correct is False
        if item.get('is_correct', True) is False:
            removed += 1
            continue
        # Check if structured_reasoning is missing, empty, or contains 'api error'
        structured_reasoning = item.get('structured_reasoning', None)
        if (
            not structured_reasoning
            or not isinstance(structured_reasoning, str)
            or not structured_reasoning.strip()
            or 'api error' in structured_reasoning.lower()
        ):
            removed += 1
            continue
        reasoning = structured_reasoning
        total_tokens = sum([
            len(tokenizer.encode(question, add_special_tokens=False)),
            len(tokenizer.encode(options, add_special_tokens=False)),
            len(tokenizer.encode(reasoning, add_special_tokens=False))
        ])
        if total_tokens <= MAX_TOTAL_TOKENS:
            fout.write(json.dumps(item) + '\n')
            kept += 1
        else:
            removed += 1

print(f"Filtered dataset saved to {output_path}")
print(f"Kept: {kept} | Removed: {removed}")
