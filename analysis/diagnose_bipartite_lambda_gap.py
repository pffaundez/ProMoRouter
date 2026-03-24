import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BIPARTITE_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")
QUERY_EMB_PATH = Path("data/router/query_embeddings.pt")
MODEL_DIR = Path("outputs/router_bipartite_qnorm")

SEED = 42
BATCH_SIZE = 128
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TASK_TO_ID = {
    "gsm8k": 0,
    "hotpotqa": 1,
    "squad": 2,
    "alpaca": 3,
}

PROMPT_TO_ID = {
    "direct": 0,
    "cot": 1,
    "decompose": 2,
    "selfcheck": 3,
}

MODEL_TO_ID = {
    "mistral-7b": 0,
    "qwen2.5-7b": 1,
    "llama3.1-8b": 2,
    "qwen2.5-14b": 3,
    "yi-34b": 4,
    "codellama-34b": 5,
    "mixtral-8x7b": 6,
    "llama3.1-70b": 7,
    "qwen2.5-72b": 8,
}

LAMBDA_KEYS = [
    "reward_qnorm_lam_01",
    "reward_qnorm_lam_05",
    "reward_qnorm_lam_09",
]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class FlatEdgeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "qid": ex["qid"],
            "task": ex["task"],
            "query_emb": ex["query_emb"].float(),
            "task_id": torch.tensor(ex["task_id"], dtype=torch.long),
            "prompt_id": torch.tensor(ex["prompt_id"], dtype=torch.long),
            "model_id": torch.tensor(ex["model_id"], dtype=torch.long),
            "prompt": ex["prompt"],
            "model": ex["model"],
            "performance": torch.tensor(ex["performance"], dtype=torch.float32),
            "cost": torch.tensor(ex["cost"], dtype=torch.float32),
            "reward": torch.tensor(ex["reward"], dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "qid": [x["qid"] for x in batch],
        "task": [x["task"] for x in batch],
        "query_emb": torch.stack([x["query_emb"] for x in batch]),
        "task_id": torch.stack([x["task_id"] for x in batch]),
        "prompt_id": torch.stack([x["prompt_id"] for x in batch]),
        "model_id": torch.stack([x["model_id"] for x in batch]),
        "prompt": [x["prompt"] for x in batch],
        "model": [x["model"] for x in batch],
        "performance": torch.stack([x["performance"] for x in batch]),
        "cost": torch.stack([x["cost"] for x in batch]),
        "reward": torch.stack([x["reward"] for x in batch]),
    }


def build_examples(rows, query_embs, qids, lambda_key):
    examples = []

    for row in rows:
        qid = row["qid"]
        task = row["task"]

        if qid not in qids:
            continue
        if qid not in query_embs:
            continue
        if task not in TASK_TO_ID:
            continue

        q_emb = query_embs[qid]

        for edge in row["action_edges"]:
            reward = edge.get(lambda_key)
            perf = edge.get("performance")
            cost = edge.get("cost_norm_query")
            prompt = edge.get("prompt")
            model = edge.get("model")

            if reward is None or perf is None or cost is None:
                continue
            if prompt not in PROMPT_TO_ID or model not in MODEL_TO_ID:
                continue

            examples.append({
                "qid": qid,
                "task": task,
                "query_emb": q_emb,
                "task_id": TASK_TO_ID[task],
                "prompt_id": PROMPT_TO_ID[prompt],
                "model_id": MODEL_TO_ID[model],
                "prompt": prompt,
                "model": model,
                "performance": float(perf),
                "cost": float(cost),
                "reward": float(reward),
            })

    return examples


def load_model(model_path: Path, query_dim: int):
    model = BipartiteRouterMLP(
        query_dim=query_dim,
        num_tasks=len(TASK_TO_ID),
        num_prompts=len(PROMPT_TO_ID),
        num_models=len(MODEL_TO_ID),
    ).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def select_actions(model, examples):
    ds = FlatEdgeDataset(examples)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    by_qid = defaultdict(list)

    with torch.no_grad():
        for batch in dl:
            pred = model(
                batch["query_emb"].to(DEVICE),
                batch["task_id"].to(DEVICE),
                batch["prompt_id"].to(DEVICE),
                batch["model_id"].to(DEVICE),
            ).cpu()

            for i in range(len(batch["qid"])):
                by_qid[batch["qid"][i]].append({
                    "qid": batch["qid"][i],
                    "task": batch["task"][i],
                    "model": batch["model"][i],
                    "prompt": batch["prompt"][i],
                    "pred": float(pred[i]),
                    "performance": float(batch["performance"][i]),
                    "cost": float(batch["cost"][i]),
                    "reward": float(batch["reward"][i]),
                })

    selected = {}
    for qid, cands in by_qid.items():
        selected[qid] = max(cands, key=lambda x: x["pred"])
    return selected


def summarize_by_task(selected):
    by_task = defaultdict(list)
    for x in selected.values():
        by_task[x["task"]].append(x)

    out = {}
    for task, vals in by_task.items():
        out[task] = {
            "n": len(vals),
            "P": sum(v["performance"] for v in vals) / len(vals),
            "C": sum(v["cost"] for v in vals) / len(vals),
            "R": sum(v["reward"] for v in vals) / len(vals),
            "top_models": Counter(v["model"] for v in vals).most_common(3),
            "top_prompts": Counter(v["prompt"] for v in vals).most_common(3),
        }
    return out


def compare_policies(sel_a, sel_b, name_a="lam01", name_b="lam05"):
    shared_qids = sorted(set(sel_a) & set(sel_b))

    more_cost_not_better = 0
    higher_perf = 0
    lower_perf = 0
    equal_perf = 0

    rows = []
    for qid in shared_qids:
        a = sel_a[qid]
        b = sel_b[qid]

        dp = a["performance"] - b["performance"]
        dc = a["cost"] - b["cost"]
        dr = a["reward"] - b["reward"]

        if a["cost"] > b["cost"] and a["performance"] <= b["performance"]:
            more_cost_not_better += 1

        if dp > 1e-12:
            higher_perf += 1
        elif dp < -1e-12:
            lower_perf += 1
        else:
            equal_perf += 1

        rows.append({
            "qid": qid,
            "task": a["task"],
            f"{name_a}_model": a["model"],
            f"{name_a}_prompt": a["prompt"],
            f"{name_a}_P": a["performance"],
            f"{name_a}_C": a["cost"],
            f"{name_a}_R": a["reward"],
            f"{name_b}_model": b["model"],
            f"{name_b}_prompt": b["prompt"],
            f"{name_b}_P": b["performance"],
            f"{name_b}_C": b["cost"],
            f"{name_b}_R": b["reward"],
            "delta_P": dp,
            "delta_C": dc,
            "delta_R": dr,
        })

    return {
        "shared_qids": len(shared_qids),
        "higher_perf": higher_perf,
        "lower_perf": lower_perf,
        "equal_perf": equal_perf,
        "more_cost_not_better": more_cost_not_better,
        "rows": rows,
    }


def main():
    set_seed(SEED)

    rows = load_jsonl(BIPARTITE_DATA_PATH)
    query_embs = torch.load(QUERY_EMB_PATH)
    _, _, test_qids = split_qids(rows)

    query_dim = next(iter(query_embs.values())).shape[0]

    model_01 = load_model(MODEL_DIR / "router_bipartite_reward_qnorm_lam_01.pt", query_dim)
    model_05 = load_model(MODEL_DIR / "router_bipartite_reward_qnorm_lam_05.pt", query_dim)
    model_09 = load_model(MODEL_DIR / "router_bipartite_reward_qnorm_lam_09.pt", query_dim)

    ex_01 = build_examples(rows, query_embs, test_qids, "reward_qnorm_lam_01")
    ex_05 = build_examples(rows, query_embs, test_qids, "reward_qnorm_lam_05")
    ex_09 = build_examples(rows, query_embs, test_qids, "reward_qnorm_lam_09")

    sel_01 = select_actions(model_01, ex_01)
    sel_05 = select_actions(model_05, ex_05)
    sel_09 = select_actions(model_09, ex_09)

    print("==== BY TASK: lambda 0.1 ====")
    print(json.dumps(summarize_by_task(sel_01), indent=2))

    print("\n==== BY TASK: lambda 0.5 ====")
    print(json.dumps(summarize_by_task(sel_05), indent=2))

    print("\n==== BY TASK: lambda 0.9 ====")
    print(json.dumps(summarize_by_task(sel_09), indent=2))

    cmp_01_05 = compare_policies(sel_01, sel_05, "lam01", "lam05")
    print("\n==== LAMBDA 0.1 vs 0.5 ====")
    print("shared_qids =", cmp_01_05["shared_qids"])
    print("lam01 higher_perf =", cmp_01_05["higher_perf"])
    print("lam01 lower_perf =", cmp_01_05["lower_perf"])
    print("equal_perf =", cmp_01_05["equal_perf"])
    print("lam01 more_cost_not_better =", cmp_01_05["more_cost_not_better"])

    out_path = MODEL_DIR / "diagnose_lambda_gap_01_vs_05.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cmp_01_05, f, indent=2)

    print(f"\nSaved detailed comparison to: {out_path}")


if __name__ == "__main__":
    main()