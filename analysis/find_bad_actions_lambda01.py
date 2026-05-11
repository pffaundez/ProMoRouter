# analysis/find_bad_actions_lambda01.py

import json
from collections import Counter
from pathlib import Path

PATH = Path("outputs/router_bipartite_qnorm/diagnose_lambda_gap_01_vs_05.json")
data = json.loads(PATH.read_text())

rows = data["rows"]

bad = Counter()
good = Counter()
bad_cost = Counter()

for r in rows:
    m = r["lam01_model"]
    p = r["lam01_prompt"]
    key = (m, p)

    p01 = r["lam01_P"]
    p05 = r["lam05_P"]
    c01 = r["lam01_C"]
    c05 = r["lam05_C"]

    # peor performance que lam=0.5
    if p01 < p05:
        bad[key] += 1
    else:
        good[key] += 1

    # más costo y no mejor performance
    if c01 > c05 and p01 <= p05:
        bad_cost[key] += 1

print("\n=== WORST ACTIONS (λ=0.1 worse than 0.5) ===")
for (m, p), n in bad.most_common(15):
    print(f"{m:16s} | {p:10s} | worse={n} | good={good[(m,p)]}")

print("\n=== COSTLY BUT NOT BETTER ===")
for (m, p), n in bad_cost.most_common(15):
    print(f"{m:16s} | {p:10s} | bad_cost={n}")