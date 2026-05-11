import json
from pathlib import Path

ROOT = Path("outputs/router_bipartite_qnorm")

rows = []

for d in ROOT.glob("seed_*"):
    log = d / "training_log.json"
    if not log.exists():
        continue

    data = json.loads(log.read_text())
    for ep in data["epochs"]:
        rows.append({
            "seed": d.name,
            "epoch": ep["epoch"],
            "val_P": ep["val_perf"],
            "val_R": ep["val_reward"],
            "val_C": ep["val_cost"],
            "path": d,
        })

# best by P
best = max(rows, key=lambda x: x["val_P"])

print("\nBEST BY VAL PERFORMANCE:\n")
print(best)