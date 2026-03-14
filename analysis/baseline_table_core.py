import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_lambdas.jsonl")

LAMBDA_CONFIGS = [
    ("reward_lam_01", 0.1),
    ("reward_lam_05", 0.5),
    ("reward_lam_09", 0.9),
]


def safe_mean(values):
    return sum(values) / len(values) if values else None


def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            perf = row.get("performance", {})
            primary = perf.get("primary") if isinstance(perf, dict) else None
            cost_norm = row.get("cost_norm")

            # Keep only rows that can be evaluated consistently.
            if primary is None or cost_norm is None:
                continue

            rows.append(row)

    return rows


def group_by_qid(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    return grouped


def summarize_selected_rows(rows, reward_key):
    perf_values = []
    cost_values = []
    reward_values = []

    for row in rows:
        perf = row["performance"]["primary"]
        cost = row["cost_norm"]
        reward = row.get(reward_key)

        if perf is None or cost is None or reward is None:
            continue

        perf_values.append(float(perf))
        cost_values.append(float(cost))
        reward_values.append(float(reward))

    return {
        "P": safe_mean(perf_values),
        "C": safe_mean(cost_values),
        "R": safe_mean(reward_values),
        "n": len(reward_values),
    }


def evaluate_fixed_model(rows, reward_key, model_name):
    selected = [r for r in rows if r["model"] == model_name]
    return summarize_selected_rows(selected, reward_key)


def evaluate_fixed_prompt(rows, reward_key, prompt_name):
    selected = [r for r in rows if r["prompt"] == prompt_name]
    return summarize_selected_rows(selected, reward_key)


def evaluate_fixed_pair(rows, reward_key, model_name, prompt_name):
    selected = [
        r for r in rows
        if r["model"] == model_name and r["prompt"] == prompt_name
    ]
    return summarize_selected_rows(selected, reward_key)


def evaluate_oracle(grouped_rows, reward_key):
    selected = []

    for qid, candidates in grouped_rows.items():
        valid = [r for r in candidates if r.get(reward_key) is not None]
        if not valid:
            continue

        best = max(valid, key=lambda r: r[reward_key])
        selected.append(best)

    return summarize_selected_rows(selected, reward_key)


def find_largest_and_smallest_model(rows):
    # Parameter sizes are inferred from model names used in this project.
    size_map = {
        "mistral-7b": 7,
        "qwen2.5-7b": 7,
        "llama3.1-8b": 8,
        "qwen2.5-14b": 14,
        "yi-34b": 34,
        "codellama-34b": 34,
        "mixtral-8x7b": 56,   # approximate total expert count, heuristic for ranking only
        "llama3.1-70b": 70,
        "qwen2.5-72b": 72,
    }

    models = sorted(set(r["model"] for r in rows))
    models = [m for m in models if m in size_map]

    smallest = min(models, key=lambda m: (size_map[m], m))
    largest = max(models, key=lambda m: (size_map[m], m))
    return smallest, largest


def find_best_fixed_model(rows, reward_key):
    rewards_by_model = defaultdict(list)

    for row in rows:
        reward = row.get(reward_key)
        if reward is None:
            continue
        rewards_by_model[row["model"]].append(float(reward))

    best_model = max(rewards_by_model.items(), key=lambda kv: safe_mean(kv[1]))[0]
    return best_model


def find_best_fixed_prompt(rows, reward_key):
    rewards_by_prompt = defaultdict(list)

    for row in rows:
        reward = row.get(reward_key)
        if reward is None:
            continue
        rewards_by_prompt[row["prompt"]].append(float(reward))

    best_prompt = max(rewards_by_prompt.items(), key=lambda kv: safe_mean(kv[1]))[0]
    return best_prompt


def find_best_fixed_pair(rows, reward_key):
    rewards_by_pair = defaultdict(list)

    for row in rows:
        reward = row.get(reward_key)
        if reward is None:
            continue
        pair = (row["model"], row["prompt"])
        rewards_by_pair[pair].append(float(reward))

    best_pair = max(rewards_by_pair.items(), key=lambda kv: safe_mean(kv[1]))[0]
    return best_pair


def format_metric(x):
    return "--" if x is None else f"{x:.3f}"


def main():
    rows = load_rows()
    grouped = group_by_qid(rows)

    smallest_model, largest_model = find_largest_and_smallest_model(rows)

    print("==== CORE BASELINE TABLE ====")
    print(f"Rows used: {len(rows)}")
    print(f"Queries used: {len(grouped)}")
    print(f"Smallest model: {smallest_model}")
    print(f"Largest model: {largest_model}")

    all_results = {}

    for reward_key, lam in LAMBDA_CONFIGS:
        best_model = find_best_fixed_model(rows, reward_key)
        best_prompt = find_best_fixed_prompt(rows, reward_key)
        best_pair_model, best_pair_prompt = find_best_fixed_pair(rows, reward_key)

        results = {
            "Largest LLM": evaluate_fixed_model(rows, reward_key, largest_model),
            "Smallest LLM": evaluate_fixed_model(rows, reward_key, smallest_model),
            "Best Fixed Model": evaluate_fixed_model(rows, reward_key, best_model),
            "Best Fixed Prompt": evaluate_fixed_prompt(rows, reward_key, best_prompt),
            "Best Fixed Pair": evaluate_fixed_pair(rows, reward_key, best_pair_model, best_pair_prompt),
            "Oracle": evaluate_oracle(grouped, reward_key),
        }

        all_results[reward_key] = {
            "lambda": lam,
            "best_model": best_model,
            "best_prompt": best_prompt,
            "best_pair": (best_pair_model, best_pair_prompt),
            "results": results,
        }

    for reward_key, payload in all_results.items():
        print(f"\n===== lambda = {payload['lambda']} ({reward_key}) =====")
        print(f"Best model for this lambda: {payload['best_model']}")
        print(f"Best prompt for this lambda: {payload['best_prompt']}")
        print(f"Best pair for this lambda: {payload['best_pair']}")

        print("\nMethod                 P       C       R       n")
        print("---------------------------------------------------")
        for method, metrics in payload["results"].items():
            print(
                f"{method:20s} "
                f"{format_metric(metrics['P']):>6s}  "
                f"{format_metric(metrics['C']):>6s}  "
                f"{format_metric(metrics['R']):>6s}  "
                f"{metrics['n']:>6d}"
            )

    print("\n==== LATEX ROWS ====")
    for method in ["Largest LLM", "Smallest LLM", "Best Fixed Model", "Best Fixed Prompt", "Best Fixed Pair", "Oracle"]:
        vals = []
        for reward_key, _ in LAMBDA_CONFIGS:
            metrics = all_results[reward_key]["results"][method]
            vals.extend([
                format_metric(metrics["P"]),
                format_metric(metrics["C"]),
                format_metric(metrics["R"]),
            ])
        print(
            f"{method} & "
            + " & ".join(vals)
            + r" \\"
        )


if __name__ == "__main__":
    main()