import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only.jsonl")

LAMBDA_CONFIGS = [
    ("reward_lam_01", 0.1),
    ("reward_lam_05", 0.5),
    ("reward_lam_09", 0.9),
]


def safe_mean(values):
    return sum(values) / len(values) if values else None


def load_queries():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    queries = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def model_size_map():
    # Parameter sizes are used only to define largest/smallest model baselines.
    return {
        "mistral-7b": 7,
        "qwen2.5-7b": 7,
        "llama3.1-8b": 8,
        "qwen2.5-14b": 14,
        "yi-34b": 34,
        "codellama-34b": 34,
        "mixtral-8x7b": 56,   # heuristic aggregate size for ranking only
        "llama3.1-70b": 70,
        "qwen2.5-72b": 72,
    }


def find_smallest_and_largest_models(queries):
    sizes = model_size_map()
    models = set()

    for q in queries:
        for cand in q.get("candidates", []):
            model = cand.get("model")
            if model in sizes:
                models.add(model)

    if not models:
        raise RuntimeError("No known models found in router_model_only dataset.")

    smallest = min(models, key=lambda m: (sizes[m], m))
    largest = max(models, key=lambda m: (sizes[m], m))
    return smallest, largest


def summarize_selected_candidates(candidates, reward_key):
    perf_values = []
    cost_values = []
    reward_values = []

    for cand in candidates:
        perf = cand.get("avg_performance")
        cost = cand.get("avg_cost_norm")
        reward = cand.get(reward_key)

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


def evaluate_fixed_model(queries, reward_key, model_name):
    selected = []

    for q in queries:
        for cand in q.get("candidates", []):
            if cand.get("model") == model_name:
                selected.append(cand)
                break

    return summarize_selected_candidates(selected, reward_key)


def find_best_fixed_model(queries, reward_key):
    rewards_by_model = defaultdict(list)

    for q in queries:
        for cand in q.get("candidates", []):
            model = cand.get("model")
            reward = cand.get(reward_key)
            if model is None or reward is None:
                continue
            rewards_by_model[model].append(float(reward))

    if not rewards_by_model:
        raise RuntimeError(f"No valid rewards found for {reward_key}")

    return max(rewards_by_model.items(), key=lambda kv: safe_mean(kv[1]))[0]


def evaluate_oracle_model_only(queries, reward_key):
    selected = []

    for q in queries:
        oracle = q.get("oracle_model_only", {}).get(reward_key)
        if oracle is None:
            continue

        target_model = oracle.get("model")
        if target_model is None:
            continue

        for cand in q.get("candidates", []):
            if cand.get("model") == target_model:
                selected.append(cand)
                break

    return summarize_selected_candidates(selected, reward_key)


def format_metric(x):
    return "--" if x is None else f"{x:.3f}"


def main():
    queries = load_queries()
    smallest_model, largest_model = find_smallest_and_largest_models(queries)

    print("==== MODEL-ONLY BASELINE TABLE ====")
    print(f"Queries used: {len(queries)}")
    print(f"Smallest model: {smallest_model}")
    print(f"Largest model: {largest_model}")

    all_results = {}

    for reward_key, lam in LAMBDA_CONFIGS:
        best_fixed_model = find_best_fixed_model(queries, reward_key)

        results = {
            "Largest LLM": evaluate_fixed_model(queries, reward_key, largest_model),
            "Smallest LLM": evaluate_fixed_model(queries, reward_key, smallest_model),
            "Best Fixed Model": evaluate_fixed_model(queries, reward_key, best_fixed_model),
            "Oracle Model-Only": evaluate_oracle_model_only(queries, reward_key),
        }

        all_results[reward_key] = {
            "lambda": lam,
            "best_fixed_model": best_fixed_model,
            "results": results,
        }

    for reward_key, payload in all_results.items():
        print(f"\n===== lambda = {payload['lambda']} ({reward_key}) =====")
        print(f"Best fixed model for this lambda: {payload['best_fixed_model']}")
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
    for method in ["Largest LLM", "Smallest LLM", "Best Fixed Model", "Oracle Model-Only"]:
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