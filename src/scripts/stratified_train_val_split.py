import json
import os
import random
from sklearn.model_selection import StratifiedShuffleSplit

# Paths
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/medmcqa/train_reasoning_filtered.jsonl'))
train_out = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/medmcqa/train_reasoning.jsonl'))
val_out = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/medmcqa/dev_reasoning.jsonl'))

# Load data
with open(input_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

subject_names = [item['subject_name'] for item in data]

# Stratified split (90% train, 10% val)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
indices = list(range(len(data)))
for train_idx, val_idx in splitter.split(indices, subject_names):
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

# Save splits as JSONL
with open(train_out, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')
with open(val_out, 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

print(f"Train split saved to {train_out} ({len(train_data)} samples)")
print(f"Validation split saved to {val_out} ({len(val_data)} samples)")
