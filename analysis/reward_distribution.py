import json
import math
from collections import defaultdict

DATA_PATH = "data/interaction_logs/grpp_il_v1/train_clean.jsonl"


def safe_mean(values):
    return sum(values) / len(values) if values else None


def safe_std(values):
    if not values:
        return None
    mu = safe_mean(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    idx = (len(sorted_values) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]

    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def main():
    rewards = []
    rewards_by_model = defaultdict(list)
    rewards_by_task = defaultdict(list)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            reward = row.get("reward")
            if reward is None:
                continue

            model = row.get("model")
            task = row.get("task")

            rewards.append(reward)
            rewards_by_model[model].append(reward)
            rewards_by_task[task].append(reward)

    if not rewards:
        print("No reward values found.")
        return

    rewards_sorted = sorted(rewards)

    print("==== GLOBAL REWARD STATS ====")
    print(f"count = {len(rewards)}")
    print(f"mean  = {safe_mean(rewards):.6f}")
    print(f"std   = {safe_std(rewards):.6f}")
    print(f"min   = {min(rewards):.6f}")
    print(f"p05   = {percentile(rewards_sorted, 0.05):.6f}")
    print(f"p25   = {percentile(rewards_sorted, 0.25):.6f}")
    print(f"p50   = {percentile(rewards_sorted, 0.50):.6f}")
    print(f"p75   = {percentile(rewards_sorted, 0.75):.6f}")
    print(f"p95   = {percentile(rewards_sorted, 0.95):.6f}")
    print(f"max   = {max(rewards):.6f}")

    print("\n==== AVERAGE REWARD BY MODEL ====")
    rows = []
    for model, vals in rewards_by_model.items():
        rows.append((model, safe_mean(vals), safe_std(vals), len(vals)))
    rows.sort(key=lambda x: x[1], reverse=True)

    for model, avg, std, n in rows:
        print(f"{model:20s} mean={avg:.6f} std={std:.6f} n={n}")

    print("\n==== AVERAGE REWARD BY TASK ====")
    rows = []
    for task, vals in rewards_by_task.items():
        rows.append((task, safe_mean(vals), safe_std(vals), len(vals)))
    rows.sort(key=lambda x: x[0])

    for task, avg, std, n in rows:
        print(f"{task:12s} mean={avg:.6f} std={std:.6f} n={n}")


if __name__ == "__main__":
    main()