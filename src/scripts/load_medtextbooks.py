from datasets import load_dataset
import pandas as pd

# ❶  Load just a few rows so it’s quick
ds = load_dataset("MedRAG/textbooks", split="train", streaming=True)
rows = [next(iter(ds)) for _ in range(5)]      # take first 5 samples
df = pd.DataFrame(rows)
print(df.head(5))
