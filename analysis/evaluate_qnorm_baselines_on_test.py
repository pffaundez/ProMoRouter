import json
import random
from pathlib import Path
from collections import defaultdict, Counter

MODEL_ONLY_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only_qnorm.jsonl")
BIPARTITE_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")
OUTPUT_PATH = Path("outputs/baselines_qnorm/baselines_qnorm_test_results.json")

LAMBDA_CONFIGS = [
    ("reward_qnorm_lam_01", 0.1),
    ("reward_qnorm_lam_05", 0.5),
    ("reward_qnorm_lam_09", 0.9),
]

SEED = 42


def set_seed(seed: int):
    random.seed(seed)


def safe_mean(values):
    return sum(values) / len(values) if values else None


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def split_qids(rows, train_ratio=0.7, val_ratio=0.15):
    qids = sorted(row["qid"] for row in rows)
    random.shuffle(qids)

    n = len(qids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_qids = set(qids[:n_train])
    val_qids = set(qids[n_train:n_train + n_val])
    test_qids = set(qids[n_train + n_val:])

    return train_qids, val_qids, test_qids


def model_size_map():
    return {
        "mistral-7b": 7,
        "qwen2.5-7b": 7,
        "llama3.1-8b": 8,
        "qwen2.5-14b": 14,
        "yi-34b": 34,
        "codellama-34b": 34,
        "mixtral-8x7b": 56,
        "llama3.1-70b": 70,
        "qwen2.5-72b": 72,
    }


def find_smallest_and_largest_models(model_only_rows):
    sizes = model_size_map()
    models = set()

    for row in model_only_rows:
        for cand in row["candidates"]:
            model = cand["model"]
            if model in sizes:
                models.add(model)

    smallest = min(models, key=lambda m: (sizes[m], m))
    largest = max(models, key=lambda m: (sizes[m], m))
    return smallest, largest


def summarize_selected(selected, reward_key, perf_key, cost_key):
    perf_values = []
    cost_values = []
    reward_values = []

    for row in selected:
        perf = row.get(perf_key)
        cost = row.get(cost_key)
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


def evaluate_fixed_model(model_only_rows, qids, reward_key, model_name):
    selected = []

    for row in model_only_rows:
        if row["qid"] not in qids:
            continue

        match = None
        for cand in row["candidates"]:
            if cand["model"] == model_name:
                match = cand
                break

        if match is not None:
            selected.append(match)

    return summarize_selected(
        selected,
        reward_key=reward_key,
        perf_key="avg_performance",
        cost_key="avg_cost_norm",
    )


def find_best_fixed_model(model_only_rows, train_qids, reward_key):
    rewards_by_model = defaultdict(list)

    for row in model_only_rows:
        if row["qid"] not in train_qids:
            continue

        for cand in row["candidates"]:
            reward = cand.get(reward_key)
            if reward is None:
                continue
            rewards_by_model[cand["model"]].append(float(reward))

    return max(rewards_by_model.items(), key=lambda kv: safe_mean(kv[1]))[0]


def evaluate_oracle_model_only(model_only_rows, qids, reward_key):
    selected = []

    for row in model_only_rows:
        if row["qid"] not in qids:
            continue

        oracle = row.get("oracle_model_only", {}).get(reward_key)
        if oracle is None:
            continue

        target_model = oracle["model"]

        for cand in row["candidates"]:
            if cand["model"] == target_model:
                selected.append(cand)
                break

    return summarize_selected(
        selected,
        reward_key=reward_key,
        perf_key="avg_performance",
        cost_key="avg_cost_norm",
    )


def find_best_fixed_pair(bipartite_rows, train_qids, reward_key):
    rewards_by_pair = defaultdict(list)

    for row in bipartite_rows:
        if row["qid"] not in train_qids:
            continue

        for edge in row["action_edges"]:
            reward = edge.get(reward_key)
            if reward is None:
                continue
            pair = (edge["model"], edge["prompt"])
            rewards_by_pair[pair].append(float(reward))

    return max(rewards_by_pair.items(), key=lambda kv: safe_mean(kv[1]))[0]


def evaluate_fixed_pair(bipartite_rows, qids, reward_key, model_name, prompt_name):
    selected = []

    for row in bipartite_rows:
        if row["qid"] not in qids:
            continue

        match = None
        for edge in row["action_edges"]:
            if edge["model"] == model_name and edge["prompt"] == prompt_name:
                match = edge
                break

        if match is not None:
            selected.append(match)

    return summarize_selected(
        selected,
        reward_key=reward_key,
        perf_key="performance",
        cost_key="cost_norm_query",
    )


def evaluate_oracle_full(bipartite_rows, qids, reward_key):
    selected = []

    for row in bipartite_rows:
        if row["qid"] not in qids:
            continue

        oracle = row.get("oracle_full", {}).get(reward_key)
        if oracle is None:
            continue

        target_model = oracle["model"]
        target_prompt = oracle["prompt"]

        for edge in row["action_edges"]:
            if edge["model"] == target_model and edge["prompt"] == target_prompt:
                selected.append(edge)
                break

    return summarize_selected(
        selected,
        reward_key=reward_key,
        perf_key="performance",
        cost_key="cost_norm_query",
    )


def format_metric(x):
    return "--" if x is None else f"{x:.3f}"


def main():
    set_seed(SEED)

    model_only_rows = load_jsonl(MODEL_ONLY_PATH)
    bipartite_rows = load_jsonl(BIPARTITE_PATH)

    train_qids, val_qids, test_qids = split_qids(model_only_rows)
    smallest_model, largest_model = find_smallest_and_largest_models(model_only_rows)

    test_task_counts = Counter(row["task"] for row in model_only_rows if row["qid"] in test_qids)

    print("==== QNORM BASELINES ON TEST SPLIT (POST-ALPACA) ====")
    print(f"Total queries: {len(model_only_rows)}")
    print(f"Train queries: {len(train_qids)}")
    print(f"Val queries: {len(val_qids)}")
    print(f"Test queries: {len(test_qids)}")
    print(f"Test task counts: {dict(test_task_counts)}")
    print(f"Smallest model: {smallest_model}")
    print(f"Largest model: {largest_model}")

    all_results = {}

    for reward_key, lam in LAMBDA_CONFIGS:
        best_fixed_model = find_best_fixed_model(model_only_rows, train_qids, reward_key)
        best_pair_model, best_pair_prompt = find_best_fixed_pair(bipartite_rows, train_qids, reward_key)

        results = {
            "Largest LLM": evaluate_fixed_model(model_only_rows, test_qids, reward_key, largest_model),
            "Smallest LLM": evaluate_fixed_model(model_only_rows, test_qids, reward_key, smallest_model),
            "Best Fixed Model": evaluate_fixed_model(model_only_rows, test_qids, reward_key, best_fixed_model),
            "Best Fixed Pair": evaluate_fixed_pair(bipartite_rows, test_qids, reward_key, best_pair_model, best_pair_prompt),
            "Oracle Model-Only": evaluate_oracle_model_only(model_only_rows, test_qids, reward_key),
            "Oracle": evaluate_oracle_full(bipartite_rows, test_qids, reward_key),
        }

        all_results[reward_key] = {
            "lambda": lam,
            "best_fixed_model": best_fixed_model,
            "best_fixed_pair": (best_pair_model, best_pair_prompt),
            "results": results,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    for reward_key, payload in all_results.items():
        print(f"\n===== lambda = {payload['lambda']} ({reward_key}) =====")
        print(f"Best fixed model selected on train: {payload['best_fixed_model']}")
        print(f"Best fixed pair selected on train: {payload['best_fixed_pair']}")
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
    for method in [
        "Largest LLM",
        "Smallest LLM",
        "Best Fixed Model",
        "Best Fixed Pair",
        "Oracle Model-Only",
        "Oracle",
    ]:
        vals = []
        for reward_key, _ in LAMBDA_CONFIGS:
            metrics = all_results[reward_key]["results"][method]
            vals.extend([
                format_metric(metrics["P"]),
                format_metric(metrics["C"]),
                format_metric(metrics["R"]),
            ])
        print(f"{method} & " + " & ".join(vals) + r" \\")

    print(f"\nSaved results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()