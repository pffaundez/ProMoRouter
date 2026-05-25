#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean

IN_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")
OUT_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only_direct_qnorm.jsonl")

LAMBDAS = [0.1, 0.5, 0.9]
REWARD_KEYS = {
    0.1: "reward_qnorm_lam_01",
    0.5: "reward_qnorm_lam_05",
    0.9: "reward_qnorm_lam_09",
}


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_written = 0
    missing_direct = 0

    with IN_PATH.open() as fin, OUT_PATH.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            n_rows += 1

            direct_edges = [
                e for e in row["action_edges"]
                if e.get("prompt") == "direct"
            ]

            if not direct_edges:
                missing_direct += 1
                continue

            by_model = {}
            for e in direct_edges:
                m = e["model"]
                by_model.setdefault(m, []).append(e)

            candidates = []
            for model, edges in sorted(by_model.items()):
                cand = {
                    "model": model,
                    "prompt_policy": "direct",
                    "n_prompt_variants": 1,
                    "prompt_variants": ["direct"],

                    "avg_performance": safe_mean([e.get("performance") for e in edges]),
                    "avg_cost_norm": safe_mean([e.get("cost_norm_query") for e in edges]),
                    "avg_cost_proxy_money": safe_mean([e.get("cost_proxy_money") for e in edges]),
                    "avg_tokens_total": safe_mean([e.get("tokens_total") for e in edges]),
                }

                for lam, rk in REWARD_KEYS.items():
                    cand[f"avg_{rk}"] = safe_mean([e.get(rk) for e in edges])

                candidates.append(cand)

            oracle_model_only = {}
            for lam, rk in REWARD_KEYS.items():
                reward_field = f"avg_{rk}"
                best = max(
                    candidates,
                    key=lambda c: c[reward_field] if c[reward_field] is not None else float("-inf"),
                )
                oracle_model_only[rk] = {
                    "model": best["model"],
                    "prompt_policy": "direct",
                    "performance": best["avg_performance"],
                    "cost_norm_query": best["avg_cost_norm"],
                    "reward": best[reward_field],
                }

            out_row = {
                "qid": row["qid"],
                "task": row["task"],
                "query_text": row["query_text"],
                "prompt_policy": "direct",
                "num_models": len(candidates),
                "model_candidates": [c["model"] for c in candidates],
                "candidates": candidates,
                "oracle_model_only": oracle_model_only,
            }

            fout.write(json.dumps(out_row) + "\n")
            n_written += 1

    print("Wrote:", OUT_PATH)
    print("input rows:", n_rows)
    print("written rows:", n_written)
    print("missing direct:", missing_direct)


if __name__ == "__main__":
    main()