import json
from pathlib import Path
from collections import Counter

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


def get_model_task_axes(rows):
    models = set()
    tasks = set()

    for row in rows:
        if row.get("task") is not None:
            tasks.add(row["task"])
        for edge in row.get("action_edges", []):
            if edge.get("model") is not None:
                models.add(edge["model"])

    ordered_models = sort_models_by_size(models)
    ordered_tasks = sorted(tasks)
    return ordered_models, ordered_tasks


def build_oracle_model_task_count_heatmap(rows, qids, reward_key, models, tasks):
    counts = Counter()

    for row in rows:
        if row["qid"] not in qids:
            continue

        task = row.get("task")
        oracle = row.get("oracle_full", {}).get(reward_key)
        if task is None or oracle is None:
            continue

        model = oracle.get("model")
        if model is None:
            continue

        counts[(model, task)] += 1

    matrix = np.zeros((len(models), len(tasks)), dtype=int)

    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            matrix[i, j] = counts.get((model, task), 0)

    return matrix


def save_count_heatmap(matrix, row_labels, col_labels, title, out_path):
    fig, ax = plt.subplots(figsize=(1.4 * len(col_labels) + 4, 0.7 * len(row_labels) + 3))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    rows = load_jsonl(BIPARTITE_DATA_PATH)
    train_qids, val_qids, test_qids = split_qids(rows)
    models, tasks = get_model_task_axes(rows)

    print(f"Loaded queries: {len(rows)}")
    print(f"Train/Val/Test = {len(train_qids)}/{len(val_qids)}/{len(test_qids)}")
    print(f"Ordered models: {models}")
    print(f"Tasks: {tasks}")

    for reward_key, lam in LAMBDA_CONFIGS:
        matrix = build_oracle_model_task_count_heatmap(
            rows=rows,
            qids=test_qids,
            reward_key=reward_key,
            models=models,
            tasks=tasks,
        )

        lam_tag = str(lam).replace(".", "")
        out_path = OUT_DIR / f"heatmap_oracle_model_vs_task_counts_test_lambda_{lam_tag}.png"

        save_count_heatmap(
            matrix=matrix,
            row_labels=models,
            col_labels=tasks,
            title=f"Oracle model vs task counts (test) | lambda={lam}",
            out_path=out_path,
        )

        print(f"[done] lambda={lam} -> {out_path}")

    print(f"Saved heatmaps to: {OUT_DIR}")


if __name__ == "__main__":
    main()