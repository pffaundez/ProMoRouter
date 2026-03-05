import json
import glob
import pandas as pd

files = glob.glob("data/interaction_logs/grpp_il_v1/shards/*.jsonl")

rows = []

for f in files:
    model = f.split("__")[-1].replace(".jsonl","")
    
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            
            rows.append({
                "model": model,
                "reward": r.get("reward"),
                "cost": r.get("cost"),
                "performance": r.get("performance")
            })

df = pd.DataFrame(rows)

print("\nROWS:", len(df))

print("\nMEAN METRICS\n")
print(df.groupby("model")[["reward","cost","performance"]].mean())

print("\nNULL CHECK\n")
print(df.isnull().sum())