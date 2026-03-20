import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROUTER_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only.jsonl")
QUERY_EMB_PATH = Path("data/router/query_embeddings.pt")
OUTPUT_DIR = Path("outputs/router_model_only")

LAMBDA_CONFIGS = [
    ("reward_lam_01", 0.1),
    ("reward_lam_05", 0.5),
    ("reward_lam_09", 0.9),
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
HIDDEN_DIM = 256

TASK_TO_ID = {
    "gsm8k": 0,
    "hotpotqa": 1,
    "squad": 2,
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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_router_queries():
    rows = []
    with ROUTER_DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_flat_examples(router_queries, query_embs, lambda_key):
    examples = []

    for row in router_queries:
        qid = row["qid"]
        task = row["task"]

        if qid not in query_embs:
            continue
        if task not in TASK_TO_ID:
            continue

        q_emb = query_embs[qid]

        for cand in row["candidates"]:
            model = cand["model"]
            reward = cand.get(lambda_key)
            perf = cand.get("avg_performance")
            cost = cand.get("avg_cost_norm")

            if reward is None or perf is None or cost is None:
                continue
            if model not in MODEL_TO_ID:
                continue

            examples.append(
                {
                    "qid": qid,
                    "task": task,
                    "model": model,
                    "query_emb": q_emb,
                    "task_id": TASK_TO_ID[task],
                    "model_id": MODEL_TO_ID[model],
                    "reward": float(reward),
                    "performance": float(perf),
                    "cost": float(cost),
                }
            )

    return examples


def split_by_qid(examples, train_ratio=0.7, val_ratio=0.15):
    qids = sorted(set(ex["qid"] for ex in examples))
    random.shuffle(qids)

    n = len(qids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_qids = set(qids[:n_train])
    val_qids = set(qids[n_train:n_train + n_val])
    test_qids = set(qids[n_train + n_val:])

    train = [ex for ex in examples if ex["qid"] in train_qids]
    val = [ex for ex in examples if ex["qid"] in val_qids]
    test = [ex for ex in examples if ex["qid"] in test_qids]

    return train, val, test


class RouterDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "query_emb": ex["query_emb"].float(),
            "task_id": torch.tensor(ex["task_id"], dtype=torch.long),
            "model_id": torch.tensor(ex["model_id"], dtype=torch.long),
            "reward": torch.tensor(ex["reward"], dtype=torch.float32),
            "performance": torch.tensor(ex["performance"], dtype=torch.float32),
            "cost": torch.tensor(ex["cost"], dtype=torch.float32),
            "qid": ex["qid"],
            "model": ex["model"],
        }


class RouterMLP(nn.Module):
    def __init__(self, query_dim, num_tasks, num_models, hidden_dim):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, 32)
        self.model_emb = nn.Embedding(num_models, 32)

        self.mlp = nn.Sequential(
            nn.Linear(query_dim + 32 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_emb, task_id, model_id):
        t_emb = self.task_emb(task_id)
        m_emb = self.model_emb(model_id)
        x = torch.cat([query_emb, t_emb, m_emb], dim=-1)
        return self.mlp(x).squeeze(-1)


def collate_fn(batch):
    return {
        "query_emb": torch.stack([x["query_emb"] for x in batch]),
        "task_id": torch.stack([x["task_id"] for x in batch]),
        "model_id": torch.stack([x["model_id"] for x in batch]),
        "reward": torch.stack([x["reward"] for x in batch]),
        "performance": torch.stack([x["performance"] for x in batch]),
        "cost": torch.stack([x["cost"] for x in batch]),
        "qid": [x["qid"] for x in batch],
        "model": [x["model"] for x in batch],
    }


def evaluate_model_selection(model, dataset):
    model.eval()

    by_qid = defaultdict(list)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in loader:
            query_emb = batch["query_emb"].to(DEVICE)
            task_id = batch["task_id"].to(DEVICE)
            model_id = batch["model_id"].to(DEVICE)

            pred = model(query_emb, task_id, model_id).cpu()

            for i in range(len(batch["qid"])):
                by_qid[batch["qid"][i]].append(
                    {
                        "pred": float(pred[i]),
                        "reward": float(batch["reward"][i]),
                        "performance": float(batch["performance"][i]),
                        "cost": float(batch["cost"][i]),
                        "model": batch["model"][i],
                    }
                )

    selected = []
    for _, candidates in by_qid.items():
        best = max(candidates, key=lambda x: x["pred"])
        selected.append(best)

    avg_reward = sum(x["reward"] for x in selected) / len(selected)
    avg_perf = sum(x["performance"] for x in selected) / len(selected)
    avg_cost = sum(x["cost"] for x in selected) / len(selected)

    return {
        "queries": len(selected),
        "avg_reward": avg_reward,
        "avg_performance": avg_perf,
        "avg_cost": avg_cost,
    }


def train_one_lambda(lambda_key, router_queries, query_embs):
    examples = build_flat_examples(router_queries, query_embs, lambda_key)
    train_ex, val_ex, test_ex = split_by_qid(examples)

    train_ds = RouterDataset(train_ex)
    val_ds = RouterDataset(val_ex)
    test_ds = RouterDataset(test_ex)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    query_dim = examples[0]["query_emb"].shape[0]
    model = RouterMLP(
        query_dim=query_dim,
        num_tasks=len(TASK_TO_ID),
        num_models=len(MODEL_TO_ID),
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_reward = -1e9
    best_state = None

    print(f"\n==== TRAINING {lambda_key} ====")
    print(f"Examples: {len(examples)} | Train: {len(train_ex)} | Val: {len(val_ex)} | Test: {len(test_ex)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            query_emb = batch["query_emb"].to(DEVICE)
            task_id = batch["task_id"].to(DEVICE)
            model_id = batch["model_id"].to(DEVICE)
            reward = batch["reward"].to(DEVICE)

            pred = model(query_emb, task_id, model_id)
            loss = criterion(pred, reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        val_metrics = evaluate_model_selection(model, val_ds)
        avg_loss = total_loss / max(1, len(train_loader))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_loss:.6f} | "
            f"val_reward={val_metrics['avg_reward']:.6f} | "
            f"val_perf={val_metrics['avg_performance']:.6f} | "
            f"val_cost={val_metrics['avg_cost']:.6f}"
        )

        if val_metrics["avg_reward"] > best_val_reward:
            best_val_reward = val_metrics["avg_reward"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed: no best model state found.")

    model.load_state_dict(best_state)
    test_metrics = evaluate_model_selection(model, test_ds)

    out_path = OUTPUT_DIR / f"router_model_only_{lambda_key}.pt"
    torch.save(best_state, out_path)

    result = {
        "lambda_key": lambda_key,
        "queries": test_metrics["queries"],
        "P": test_metrics["avg_performance"],
        "C": test_metrics["avg_cost"],
        "R": test_metrics["avg_reward"],
        "model_path": str(out_path),
    }

    print("\n==== TEST RESULTS ====")
    print(f"queries={result['queries']}")
    print(f"avg_reward={result['R']:.6f}")
    print(f"avg_performance={result['P']:.6f}")
    print(f"avg_cost={result['C']:.6f}")
    print(f"saved_model={out_path}")

    return result


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==== TRAIN ROUTER MODEL-ONLY (ALL LAMBDAS) ====")
    print("Device:", DEVICE)

    router_queries = load_router_queries()
    query_embs = torch.load(QUERY_EMB_PATH)

    all_results = []

    for lambda_key, lam in LAMBDA_CONFIGS:
        set_seed(SEED)
        result = train_one_lambda(lambda_key, router_queries, query_embs)
        result["lambda"] = lam
        all_results.append(result)

    results_path = OUTPUT_DIR / "router_model_only_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n==== OVERLEAF ROW ====")
    vals = []
    for result in all_results:
        vals.extend([
            f"{result['P']:.3f}",
            f"{result['C']:.3f}",
            f"{result['R']:.3f}",
        ])
    print("GraphRouter (model-only) & " + " & ".join(vals) + r" \\")

    print(f"\nSaved results: {results_path}")


if __name__ == "__main__":
    main()