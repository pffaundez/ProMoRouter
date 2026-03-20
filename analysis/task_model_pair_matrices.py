import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

BIPARTITE_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")
QUERY_EMB_PATH = Path("data/router/query_embeddings.pt")
MODEL_DIR = Path("outputs/router_bipartite_qnorm")
OUTPUT_DIR = Path("outputs/router_bipartite_qnorm/task_model_pair_analysis")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42

LAMBDA_CONFIGS = [
    ("reward_qnorm_lam_01", 0.1),
    ("reward_qnorm_lam_05", 0.5),
    ("reward_qnorm_lam_09", 0.9),
]

TASK_ORDER = ["gsm8k", "hotpotqa", "squad"]
PROMPT_ORDER = ["direct", "cot", "decompose", "selfcheck"]
MODEL_ORDER = [
    "mistral-7b",
    "qwen2.5-7b",
    "llama3.1-8b",
    "qwen2.5-14b",
    "yi-34b",
    "codellama-34b",
    "mixtral-8x7b",
    "llama3.1-70b",
    "qwen2.5-72b",
]

TASK_TO_ID = {k: i for i, k in enumerate(TASK_ORDER)}
PROMPT_TO_ID = {k: i for i, k in enumerate(PROMPT_ORDER)}
MODEL_TO_ID = {k: i for i, k in enumerate(MODEL_ORDER)}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BipartiteRouterMLP(nn.Module):
    def __init__(self, query_dim, num_tasks, num_prompts, num_models, hidden_dim=256):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, 32)
        self.prompt_emb = nn.Embedding(num_prompts, 32)
        self.model_emb = nn.Embedding(num_models, 32)

        self.mlp = nn.Sequential(
            nn.Linear(query_dim + 32 + 32 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_emb, task_id, prompt_id, model_id):
        t_emb = self.task_emb(task_id)
        p_emb = self.prompt_emb(prompt_id)
        m_emb = self.model_emb(model_id)
        x = torch.cat([query_emb, t_emb, p_emb, m_emb], dim=-1)
        return self.mlp(x).squeeze(-1)


def load_router_queries():
    rows = []
    with BIPARTITE_DATA_PATH.open("r", encoding="utf-8") as f:
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


def build_test_action_rows(router_queries, query_embs, lambda_key, test_qids):
    action_rows = []

    for row in router_queries:
        qid = row["qid"]
        task = row["task"]

        if qid not in test_qids:
            continue
        if qid not in query_embs:
            continue
        if task not in TASK_TO_ID:
            continue

        q_emb = query_embs[qid]

        for edge in row["action_edges"]:
            prompt = edge.get("prompt")
            model = edge.get("model")
            reward = edge.get(lambda_key)
            perf = edge.get("performance")
            cost = edge.get("cost_norm_query")

            if reward is None or perf is None or cost is None:
                continue
            if prompt not in PROMPT_TO_ID or model not in MODEL_TO_ID:
                continue

            action_rows.append(
                {
                    "qid": qid,
                    "task": task,
                    "prompt": prompt,
                    "model": model,
                    "query_emb": q_emb,
                    "task_id": TASK_TO_ID[task],
                    "prompt_id": PROMPT_TO_ID[prompt],
                    "model_id": MODEL_TO_ID[model],
                    "reward": float(reward),
                    "performance": float(perf),
                    "cost": float(cost),
                }
            )

    return action_rows


def load_checkpoint_model(lambda_key, query_dim):
    ckpt_path = MODEL_DIR / f"router_bipartite_{lambda_key}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = BipartiteRouterMLP(
        query_dim=query_dim,
        num_tasks=len(TASK_TO_ID),
        num_prompts=len(PROMPT_TO_ID),
        num_models=len(MODEL_TO_ID),
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def select_best_action_per_query(model, action_rows):
    by_qid = defaultdict(list)

    # Score one action at a time for simplicity and transparency.
    with torch.no_grad():
        for row in action_rows:
            q = row["query_emb"].unsqueeze(0).to(DEVICE)
            t = torch.tensor([row["task_id"]], dtype=torch.long, device=DEVICE)
            p = torch.tensor([row["prompt_id"]], dtype=torch.long, device=DEVICE)
            m = torch.tensor([row["model_id"]], dtype=torch.long, device=DEVICE)

            pred = model(q, t, p, m).item()

            enriched = dict(row)
            enriched["pred"] = float(pred)
            by_qid[row["qid"]].append(enriched)

    selected = []
    for qid, candidates in by_qid.items():
        best = max(candidates, key=lambda x: x["pred"])
        selected.append(best)

    return selected


def build_task_model_matrix(selected_rows):
    matrix = pd.DataFrame(0, index=TASK_ORDER, columns=MODEL_ORDER, dtype=int)

    for row in selected_rows:
        matrix.loc[row["task"], row["model"]] += 1

    return matrix


def build_task_model_prompt_summary(selected_rows):
    # Nested structure: task -> model -> prompt -> count
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for row in selected_rows:
        summary[row["task"]][row["model"]][row["prompt"]] += 1

    return summary


def make_prompt_label(prompt_counts: dict):
    if not prompt_counts:
        return ""
    ordered = sorted(prompt_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join([f"{p}:{c}" for p, c in ordered])


def save_prompt_detail_csv(summary, output_path: Path):
    records = []

    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            prompt_counts = summary[task][model]
            total = sum(prompt_counts.values())
            records.append(
                {
                    "task": task,
                    "model": model,
                    "total_selected": total,
                    "prompt_breakdown": make_prompt_label(prompt_counts),
                    **{f"prompt_{p}": prompt_counts.get(p, 0) for p in PROMPT_ORDER},
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path):
    plt.figure(figsize=(12, 3.8))
    plt.imshow(matrix.values, aspect="auto")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.colorbar(label="Selected queries")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_console_summary(lambda_key: str, selected_rows, matrix: pd.DataFrame, prompt_summary):
    print(f"\n==== {lambda_key} ====")
    print(f"Selected queries: {len(selected_rows)}")
    print("\nTask x Model counts:")
    print(matrix.to_string())

    print("\nPrompt breakdown inside each non-zero (task, model) cell:")
    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            counts = prompt_summary[task][model]
            total = sum(counts.values())
            if total == 0:
                continue
            label = make_prompt_label(counts)
            print(f"  task={task:8s} model={model:16s} total={total:3d} | {label}")


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    router_queries = load_router_queries()
    query_embs = torch.load(QUERY_EMB_PATH)
    _, _, test_qids = split_qids(router_queries)

    query_dim = next(iter(query_embs.values())).shape[0]

    all_json_summary = {}

    for lambda_key, lam in LAMBDA_CONFIGS:
        model = load_checkpoint_model(lambda_key, query_dim)

        action_rows = build_test_action_rows(
            router_queries=router_queries,
            query_embs=query_embs,
            lambda_key=lambda_key,
            test_qids=test_qids,
        )

        selected_rows = select_best_action_per_query(model, action_rows)
        matrix = build_task_model_matrix(selected_rows)
        prompt_summary = build_task_model_prompt_summary(selected_rows)

        print_console_summary(lambda_key, selected_rows, matrix, prompt_summary)

        matrix_csv_path = OUTPUT_DIR / f"{lambda_key}_task_model_matrix.csv"
        matrix.to_csv(matrix_csv_path)

        prompt_csv_path = OUTPUT_DIR / f"{lambda_key}_task_model_prompt_breakdown.csv"
        save_prompt_detail_csv(prompt_summary, prompt_csv_path)

        heatmap_path = OUTPUT_DIR / f"{lambda_key}_task_model_heatmap.png"
        plot_heatmap(
            matrix=matrix,
            title=f"Task vs Model selections ({lambda_key})",
            output_path=heatmap_path,
        )

        all_json_summary[lambda_key] = {
            "lambda": lam,
            "matrix": matrix.to_dict(),
            "prompt_summary": {
                task: {
                    model: dict(prompt_summary[task][model])
                    for model in MODEL_ORDER
                }
                for task in TASK_ORDER
            },
        }

    summary_json_path = OUTPUT_DIR / "task_model_pair_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_json_summary, f, indent=2)

    print("\nSaved files:")
    for lambda_key, _ in LAMBDA_CONFIGS:
        print(OUTPUT_DIR / f"{lambda_key}_task_model_matrix.csv")
        print(OUTPUT_DIR / f"{lambda_key}_task_model_prompt_breakdown.csv")
        print(OUTPUT_DIR / f"{lambda_key}_task_model_heatmap.png")
    print(summary_json_path)


if __name__ == "__main__":
    main()