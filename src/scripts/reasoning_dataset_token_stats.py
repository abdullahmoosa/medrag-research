import json
import os
from transformers import AutoTokenizer

# Use DeepSeek tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

# Path to your JSON file (fix for Windows and workspace structure)
json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/medmcqa/train_reasoning_generated.json'))

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

def main():
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} items from {json_path}")


    question_tokens = []
    structured_reasoning_tokens = []
    raw_reasoning_tokens = []

    # DeepSeek max token limit
    max_tokens = 4096
    over_limit_count = 0
    over_limit_indices = []

    for idx, item in enumerate(data):
        q_tokens = count_tokens(item.get('question', ''))
        sr_tokens = count_tokens(item.get('structured_reasoning', ''))
        rr_tokens = count_tokens(item.get('raw_reasoning', ''))
        question_tokens.append(q_tokens)
        structured_reasoning_tokens.append(sr_tokens)
        raw_reasoning_tokens.append(rr_tokens)
        # Check if any field exceeds max_tokens
        if q_tokens > max_tokens or sr_tokens > max_tokens or rr_tokens > max_tokens:
            over_limit_count += 1
            over_limit_indices.append(idx)

    print("Average tokens per field (DeepSeek tokenizer):")
    print(f"Question: {sum(question_tokens)/len(question_tokens):.2f}")
    print(f"Structured Reasoning: {sum(structured_reasoning_tokens)/len(structured_reasoning_tokens):.2f}")
    print(f"Raw Reasoning: {sum(raw_reasoning_tokens)/len(raw_reasoning_tokens):.2f}")

    print(f"\nNumber of examples exceeding {max_tokens} tokens in any field: {over_limit_count}")
    if over_limit_count > 0:
        print(f"Indices of over-limit examples (first 10): {over_limit_indices[:10]}")

if __name__ == "__main__":
    main()