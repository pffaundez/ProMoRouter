import json
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np

BIPARTITE_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")

LAMBDA_CONFIGS = [
    ("reward_qnorm_lam_01", 0.1),
    ("reward_qnorm_lam_05", 0.5),
    ("reward_qnorm_lam_09", 0.9),
]

OUT_DIR = Path("outputs/analytics_heatmaps_qnorm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE_MAP = {
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


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def split_qids(rows, train_ratio=0.7, val_ratio=0.15, seed=42):
    import random

    random.seed(seed)
    qids = sorted(row["qid"] for row in rows)
    random.shuffle(qids)

    n = len(qids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_qids = set(qids[:n_train])
    val_qids = set(qids[n_train:n_train + n_val])
    test_qids = set(qids[n_train + n_val:])

    return train_qids, val_qids, test_qids


def sort_models_by_size(models):
    return sorted(models, key=lambda m: (MODEL_SIZE_MAP.get(m, 10**9), m))


def get_model_prompt_axes(rows):
    models = set()
    prompts = set()

    for row in rows:
        for edge in row.get("action_edges", []):
            if edge.get("model") is not None:
                models.add(edge["model"])
            if edge.get("prompt") is not None:
                prompts.add(edge["prompt"])

    ordered_models = sort_models_by_size(models)
    ordered_prompts = sorted(prompts)
    return ordered_models, ordered_prompts


def save_heatmap(matrix, row_labels, col_labels, title, out_path, fmt=".3f"):
    fig, ax = plt.subplots(figsize=(1.5 * len(col_labels) + 4, 0.7 * len(row_labels) + 3))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            text = "--" if np.isnan(val) else format(val, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_avg_reward_heatmap(rows, qids, reward_key, models, prompts):
    values = defaultdict(list)

    for row in rows:
        if row["qid"] not in qids:
            continue
        for edge in row.get("action_edges", []):
            reward = edge.get(reward_key)
            if reward is not None:
                values[(edge["model"], edge["prompt"])].append(float(reward))

    matrix = np.full((len(models), len(prompts)), np.nan)

    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            vals = values.get((model, prompt), [])
            if vals:
                matrix[i, j] = sum(vals) / len(vals)

    return matrix


def build_avg_perf_heatmap(rows, qids, models, prompts):
    values = defaultdict(list)

    for row in rows:
        if row["qid"] not in qids:
            continue
        for edge in row.get("action_edges", []):
            perf = edge.get("performance")
            if perf is not None:
                values[(edge["model"], edge["prompt"])].append(float(perf))

    matrix = np.full((len(models), len(prompts)), np.nan)

    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            vals = values.get((model, prompt), [])
            if vals:
                matrix[i, j] = sum(vals) / len(vals)

    return matrix


def build_avg_cost_heatmap(rows, qids, models, prompts):
    values = defaultdict(list)

    for row in rows:
        if row["qid"] not in qids:
            continue
        for edge in row.get("action_edges", []):
            cost = edge.get("cost_norm_query")
            if cost is not None:
                values[(edge["model"], edge["prompt"])].append(float(cost))

    matrix = np.full((len(models), len(prompts)), np.nan)

    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            vals = values.get((model, prompt), [])
            if vals:
                matrix[i, j] = sum(vals) / len(vals)

    return matrix


def build_oracle_choice_heatmap(rows, qids, reward_key, models, prompts):
    counts = Counter()

    for row in rows:
        if row["qid"] not in qids:
            continue
        oracle = row.get("oracle_full", {}).get(reward_key)
        if oracle is None:
            continue
        counts[(oracle["model"], oracle["prompt"])] += 1

    matrix = np.zeros((len(models), len(prompts)), dtype=float)

    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            matrix[i, j] = counts.get((model, prompt), 0)

    return matrix


def main():
    rows = load_jsonl(BIPARTITE_DATA_PATH)
    train_qids, val_qids, test_qids = split_qids(rows)
    models, prompts = get_model_prompt_axes(rows)

    print(f"Loaded queries: {len(rows)}")
    print(f"Train/Val/Test = {len(train_qids)}/{len(val_qids)}/{len(test_qids)}")
    print(f"Ordered models: {models}")
    print(f"Prompts: {prompts}")

    for reward_key, lam in LAMBDA_CONFIGS:
        reward_matrix = build_avg_reward_heatmap(rows, train_qids, reward_key, models, prompts)
        perf_matrix = build_avg_perf_heatmap(rows, train_qids, models, prompts)
        cost_matrix = build_avg_cost_heatmap(rows, train_qids, models, prompts)
        oracle_matrix = build_oracle_choice_heatmap(rows, test_qids, reward_key, models, prompts)

        lam_tag = str(lam).replace(".", "")

        save_heatmap(
            reward_matrix,
            models,
            prompts,
            f"Avg reward by model/prompt (train) | lambda={lam}",
            OUT_DIR / f"heatmap_reward_train_lambda_{lam_tag}.png",
        )

        save_heatmap(
            perf_matrix,
            models,
            prompts,
            "Avg performance by model/prompt (train)",
            OUT_DIR / f"heatmap_perf_train_lambda_{lam_tag}.png",
        )

        save_heatmap(
            cost_matrix,
            models,
            prompts,
            "Avg qnorm cost by model/prompt (train)",
            OUT_DIR / f"heatmap_cost_train_lambda_{lam_tag}.png",
        )

        save_heatmap(
            oracle_matrix,
            models,
            prompts,
            f"Oracle selected pair counts (test) | lambda={lam}",
            OUT_DIR / f"heatmap_oracle_counts_test_lambda_{lam_tag}.png",
            fmt=".0f",
        )

        print(f"[done] lambda={lam}")

    print(f"Saved heatmaps to: {OUT_DIR}")


if __name__ == "__main__":
    main()