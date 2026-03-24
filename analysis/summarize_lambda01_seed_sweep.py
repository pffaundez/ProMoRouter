import json
from pathlib import Path
from statistics import mean, stdev

IN_DIR = Path("outputs/router_bipartite_qnorm_seed_sweep")

files = sorted(IN_DIR.glob("reward_qnorm_lam_01__seed_*.json"))
if not files:
    raise SystemExit("No seed files found.")

rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]

Ps = [r["P"] for r in rows]
Cs = [r["C"] for r in rows]
Rs = [r["R"] for r in rows]

print("Seeds:", [r["seed"] for r in rows])
print(f"P mean={mean(Ps):.4f} std={stdev(Ps) if len(Ps)>1 else 0:.4f}")
print(f"C mean={mean(Cs):.4f} std={stdev(Cs) if len(Cs)>1 else 0:.4f}")
print(f"R mean={mean(Rs):.4f} std={stdev(Rs) if len(Rs)>1 else 0:.4f}")

best = max(rows, key=lambda r: r["R"])
print("\nBest by R:")
print(json.dumps(best, indent=2))

best_p = max(rows, key=lambda r: r["P"])
print("\nBest by P:")
print(json.dumps(best_p, indent=2))