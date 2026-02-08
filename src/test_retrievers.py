import pandas as pd
df = pd.read_parquet("../models/medcorp_index/docstore.parquet")
def hits(term):
    m = df["text"].str.contains(term, case=False, na=False)
    return int(m.sum())
for t in ["ethambutol", "optic neuritis", "red-green", "color blindness", "ophthalmolog", "ocular toxicity"]:
    print(t, hits(t))