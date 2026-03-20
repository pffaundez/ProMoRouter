import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROUTER_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only_qnorm.jsonl")
QUERY_EMB_PATH = Path("data/router/query_embeddings.pt")
OUTPUT_DIR = Path("outputs/router_model_only_qnorm")

LAMBDA_CONFIGS = [
    ("reward_qnorm_lam_01", 0.1),
    ("reward_qnorm_lam_05", 0.5),
    ("reward_qnorm_lam_09", 0.9),
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 42
BATCH_SIZE = 128
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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    rows = []
    with ROUTER_DATA_PATH.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_examples(rows, query_embs, lambda_key):
    examples = []

    for row in rows:
        qid = row["qid"]
        task = row["task"]

        if qid not in query_embs or task not in TASK_TO_ID:
            continue

        for cand in row["candidates"]:
            reward = cand.get(lambda_key)
            if reward is None:
                continue

            examples.append({
                "qid": qid,
                "query_emb": query_embs[qid],
                "task_id": TASK_TO_ID[task],
                "model_id": MODEL_TO_ID[cand["model"]],
                "reward": float(reward),
                "performance": float(cand["avg_performance"]),
                "cost": float(cand["avg_cost_norm"]),
            })

    return examples


def split_by_qid(examples):
    qids = list(set(e["qid"] for e in examples))
    random.shuffle(qids)

    n = len(qids)
    train_q = set(qids[:int(0.7*n)])
    val_q = set(qids[int(0.7*n):int(0.85*n)])
    test_q = set(qids[int(0.85*n):])

    def filt(qset):
        return [e for e in examples if e["qid"] in qset]

    return filt(train_q), filt(val_q), filt(test_q)


class DatasetWrapper(Dataset):
    def __init__(self, ex):
        self.ex = ex

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        return {
            "query_emb": e["query_emb"].float(),
            "task_id": torch.tensor(e["task_id"]),
            "model_id": torch.tensor(e["model_id"]),
            "reward": torch.tensor(e["reward"]),
            "performance": torch.tensor(e["performance"]),
            "cost": torch.tensor(e["cost"]),
            "qid": e["qid"]
        }


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.task_emb = nn.Embedding(len(TASK_TO_ID), 32)
        self.model_emb = nn.Embedding(len(MODEL_TO_ID), 32)

        self.mlp = nn.Sequential(
            nn.Linear(dim + 64, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, q, t, m):
        x = torch.cat([q, self.task_emb(t), self.model_emb(m)], dim=-1)
        return self.mlp(x).squeeze(-1)


def collate(batch):
    return {
        k: torch.stack([x[k] for x in batch]) if k not in ["qid"] else [x[k] for x in batch]
        for k in batch[0]
    }


def eval_model(model, dataset):
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate)
    model.eval()

    by_qid = defaultdict(list)

    with torch.no_grad():
        for b in loader:
            pred = model(
                b["query_emb"].to(DEVICE),
                b["task_id"].to(DEVICE),
                b["model_id"].to(DEVICE)
            ).cpu()

            for i in range(len(b["qid"])):
                by_qid[b["qid"][i]].append({
                    "pred": float(pred[i]),
                    "reward": float(b["reward"][i]),
                    "performance": float(b["performance"][i]),
                    "cost": float(b["cost"][i]),
                })

    sel = [max(v, key=lambda x: x["pred"]) for v in by_qid.values()]

    return {
        "queries": len(sel),
        "R": sum(x["reward"] for x in sel)/len(sel),
        "P": sum(x["performance"] for x in sel)/len(sel),
        "C": sum(x["cost"] for x in sel)/len(sel),
    }


def train_one(lambda_key, rows, emb):
    ex = build_examples(rows, emb, lambda_key)
    tr, va, te = split_by_qid(ex)

    tr_dl = DataLoader(DatasetWrapper(tr), batch_size=128, shuffle=True, collate_fn=collate)

    model = Model(ex[0]["query_emb"].shape[0]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best = None
    best_val = -1e9

    for ep in range(EPOCHS):
        model.train()
        for b in tr_dl:
            pred = model(
                b["query_emb"].to(DEVICE),
                b["task_id"].to(DEVICE),
                b["model_id"].to(DEVICE)
            )
            loss = loss_fn(pred, b["reward"].to(DEVICE))

            opt.zero_grad()
            loss.backward()
            opt.step()

        val = eval_model(model, DatasetWrapper(va))

        print(f"Epoch {ep+1:02d} | val_R={val['R']:.4f}")

        if val["R"] > best_val:
            best_val = val["R"]
            best = model.state_dict()

    model.load_state_dict(best)
    test = eval_model(model, DatasetWrapper(te))

    print("\nTEST:", test)
    return test


def main():
    set_seed(SEED)
    rows = load_data()
    emb = torch.load(QUERY_EMB_PATH)

    results = []

    for key, lam in LAMBDA_CONFIGS:
        print("\n====", key, "====")
        res = train_one(key, rows, emb)
        results.append(res)

    print("\n==== OVERLEAF ====")
    vals = []
    for r in results:
        vals += [f"{r['P']:.3f}", f"{r['C']:.3f}", f"{r['R']:.3f}"]

    print("GraphRouter (model-only, qnorm) & " + " & ".join(vals) + r" \\")


if __name__ == "__main__":
    main()