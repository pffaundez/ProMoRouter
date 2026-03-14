import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_lambdas.jsonl")
LAMBDA_KEYS = ["reward_lam_01", "reward_lam_05", "reward_lam_09"]


def safe_mean(values):
    return sum(values) / len(values) if values else None


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    # Store reward values grouped by (model, prompt) for each lambda setting.
    rewards_by_lambda_pair = {
        lam: defaultdict(list) for lam in LAMBDA_KEYS
    }

    # Also keep average performance and cost for interpretability.
    performance_by_pair = defaultdict(list)
    cost_by_pair = defaultdict(list)

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            model = row.get("model")
            prompt = row.get("prompt")
            pair = (model, prompt)

            perf = row.get("performance", {})
            primary = perf.get("primary") if isinstance(perf, dict) else None

            cost = row.get("cost", {})
            tokens_total = cost.get("tokens_total") if isinstance(cost, dict) else None

            if primary is not None:
                performance_by_pair[pair].append(float(primary))

            if tokens_total is not None:
                cost_by_pair[pair].append(float(tokens_total))

            for lam in LAMBDA_KEYS:
                reward = row.get(lam)
                if reward is not None:
                    rewards_by_lambda_pair[lam][pair].append(float(reward))

    print("==== BEST FIXED PAIR BASELINE ====")

    for lam in LAMBDA_KEYS:
        rows = []
        for pair, rewards in rewards_by_lambda_pair[lam].items():
            avg_reward = safe_mean(rewards)
            avg_perf = safe_mean(performance_by_pair[pair])
            avg_cost = safe_mean(cost_by_pair[pair])

            rows.append(
                {
                    "pair": pair,
                    "avg_reward": avg_reward,
                    "avg_performance": avg_perf,
                    "avg_cost_tokens": avg_cost,
                    "n": len(rewards),
                }
            )

        rows.sort(key=lambda x: x["avg_reward"], reverse=True)

        print(f"\n===== {lam} =====")
        if not rows:
            print("No valid rows found.")
            continue

        best = rows[0]
        model, prompt = best["pair"]

        print("Best fixed pair:")
        print(f"  model: {model}")
        print(f"  prompt: {prompt}")
        print(f"  avg_reward: {best['avg_reward']:.6f}")
        print(f"  avg_performance: {best['avg_performance']:.6f}" if best["avg_performance"] is not None else "  avg_performance: None")
        print(f"  avg_cost_tokens: {best['avg_cost_tokens']:.6f}" if best["avg_cost_tokens"] is not None else "  avg_cost_tokens: None")
        print(f"  n: {best['n']}")

        print("\nTop 10 pairs:")
        for row in rows[:10]:
            model, prompt = row["pair"]
            perf_str = f"{row['avg_performance']:.6f}" if row["avg_performance"] is not None else "None"
            cost_str = f"{row['avg_cost_tokens']:.6f}" if row["avg_cost_tokens"] is not None else "None"
            print(
                f"  ({model}, {prompt}) "
                f"reward={row['avg_reward']:.6f} "
                f"perf={perf_str} "
                f"cost_tokens={cost_str} "
                f"n={row['n']}"
            )


if __name__ == "__main__":
    main()