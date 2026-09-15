#!/usr/bin/env python3
"""Train/evaluate a flat 36-action MLP baseline with qnorm rewards.

This baseline uses the same closed prompt/model pool and shared query splits as
ProMoRouter, but performs no graph message passing. It scores each prompt-model
pair independently from concatenated query, prompt, and model embeddings.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from router.data_validation import require_complete_action_space
from router.splits import load_or_create_splits

LAMBDA_CONFIGS = (
    ("reward_qnorm_lam_01", 0.1),
    ("reward_qnorm_lam_05", 0.5),
    ("reward_qnorm_lam_09", 0.9),
)

EXPECTED_PROMPTS = ("direct", "cot", "decompose", "selfcheck")
EXPECTED_MODELS = (
    "mistral-7b", "qwen2.5-7b", "llama3.1-8b",
    "qwen2.5-14b", "yi-34b", "codellama-34b",
    "mixtral-8x7b", "llama3.1-70b", "qwen2.5-72b",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_embedding_object(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("embedding_by_id", "embeddings_by_id"):
            if isinstance(obj.get(key), dict):
                return obj[key]
        if isinstance(obj.get("ids"), (list, tuple)) and isinstance(obj.get("embeddings"), torch.Tensor):
            return {name: obj["embeddings"][i].float() for i, name in enumerate(obj["ids"])}
        for key in ("embeddings", "embedding", "tensor", "values"):
            if isinstance(obj.get(key), torch.Tensor):
                return obj[key].float()
        if obj and all(isinstance(value, torch.Tensor) for value in obj.values()):
            return obj
    raise ValueError(f"Unsupported embedding format: {path}")


def vector_for(table, key, order=None):
    if isinstance(table, dict):
        if key not in table:
            raise KeyError(f"Missing embedding for {key}")
        return table[key].float().view(-1)
    if order is None:
        raise ValueError("Tensor embeddings require an explicit identifier order")
    return table[order.index(key)].float().view(-1)


class FlatActionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_examples(row, query_table, prompt_table, model_table,
                  prompt_order, model_order, reward_key, device):
    q = vector_for(query_table, row["qid"]).to(device)
    features, rewards, performances, costs, actions = [], [], [], [], []
    for cand in row["action_edges"]:
        p = cand["prompt"]
        m = cand["model"]
        features.append(torch.cat([
            q,
            vector_for(prompt_table, p, prompt_order).to(device),
            vector_for(model_table, m, model_order).to(device),
        ]))
        rewards.append(float(cand[reward_key]))
        performances.append(float(cand["performance"]))
        costs.append(float(cand["cost_norm_query"]))
        actions.append((p, m))
    return {
        "x": torch.stack(features),
        "reward": torch.tensor(rewards, dtype=torch.float32, device=device),
        "performance": performances,
        "cost": costs,
        "actions": actions,
    }


@torch.no_grad()
def evaluate(model, rows, query_table, prompt_table, model_table,
             prompt_order, model_order, reward_key, device):
    model.eval()
    selected = []
    for row in rows:
        batch = make_examples(row, query_table, prompt_table, model_table,
                              prompt_order, model_order, reward_key, device)
        index = int(torch.argmax(model(batch["x"])).item())
        selected.append((batch["performance"][index],
                         batch["cost"][index],
                         batch["actions"][index]))
    lam = dict(LAMBDA_CONFIGS)[reward_key]
    return {
        "queries": len(selected),
        "P": mean(item[0] for item in selected),
        "C": mean(item[1] for item in selected),
        "R": mean(item[0] - lam * item[1] for item in selected),
        "prompt_counts": dict(Counter(item[2][0] for item in selected)),
        "model_counts": dict(Counter(item[2][1] for item in selected)),
    }


def train_lambda(args, rows, train_qids, val_qids, test_qids,
                 query_table, prompt_table, model_table,
                 prompt_order, model_order, reward_key, device):
    train_rows = [r for r in rows if r["qid"] in train_qids]
    val_rows = [r for r in rows if r["qid"] in val_qids]
    test_rows = [r for r in rows if r["qid"] in test_qids]
    input_dim = (
        vector_for(query_table, rows[0]["qid"]).numel()
        + vector_for(prompt_table, prompt_order[0], prompt_order).numel()
        + vector_for(model_table, model_order[0], model_order).numel()
    )
    model = FlatActionMLP(input_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best_state, best_reward, stale = None, -float("inf"), 0
    lam = dict(LAMBDA_CONFIGS)[reward_key]

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = list(train_rows)
        random.shuffle(shuffled)
        losses = []
        for row in shuffled:
            batch = make_examples(row, query_table, prompt_table, model_table,
                                  prompt_order, model_order, reward_key, device)
            scores = model(batch["x"])
            rewards = batch["reward"]
            mse = F.mse_loss(scores, rewards)
            target = torch.argmax(rewards).view(1)
            ce = F.cross_entropy(scores.view(1, -1), target)
            loss = mse + args.ce_weight * ce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_rows, query_table, prompt_table, model_table,
                       prompt_order, model_order, reward_key, device)
        if val["R"] > best_reward + args.min_delta:
            best_reward = val["R"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if args.verbose:
            print(f"seed={args.seed} {reward_key} epoch={epoch:03d} "
                  f"loss={mean(losses):.6f} val_R={val['R']:.6f}")
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint captured")
    model.load_state_dict(best_state)
    result = evaluate(model, test_rows, query_table, prompt_table, model_table,
                      prompt_order, model_order, reward_key, device)
    result.update({"lambda": lam, "lambda_key": reward_key,
                   "split_manifest": str(args.split_manifest)})
    return model, result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=Path,
                   default=Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl"))
    p.add_argument("--query-emb-path", type=Path, default=Path("data/router/query_embeddings.pt"))
    p.add_argument("--prompt-emb-path", type=Path, default=Path("data/router/prompt_embeddings.pt"))
    p.add_argument("--model-emb-path", type=Path, default=Path("data/router/model_embeddings.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/p1_router_flat_mlp_qnorm"))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ce-weight", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    rows = load_jsonl(args.data_path)
    prompt_order = list(EXPECTED_PROMPTS)
    model_order = list(EXPECTED_MODELS)
    require_complete_action_space(
        rows,
        expected_prompts=EXPECTED_PROMPTS,
        expected_models=EXPECTED_MODELS,
    )
    query_table = load_embedding_object(args.query_emb_path)
    prompt_table = load_embedding_object(args.prompt_emb_path)
    model_table = load_embedding_object(args.model_emb_path)
    seeds = args.seeds or [args.seed]
    all_results = []
    for seed in seeds:
        args.seed = seed
        set_seed(seed)
        args.split_manifest = args.split_manifest or (
            Path("data/router/splits") / f"qnorm_complete_seed{seed}.json"
        )
        qids = [r["qid"] for r in rows]
        train_qids, val_qids, test_qids = load_or_create_splits(
            qids, manifest_path=args.split_manifest, seed=seed,
            train_ratio=0.70, val_ratio=0.15
        )
        for reward_key, _ in LAMBDA_CONFIGS:
            model, result = train_lambda(
                args, rows, set(train_qids), set(val_qids), set(test_qids),
                query_table, prompt_table, model_table,
                prompt_order, model_order, reward_key, device
            )
            model_path = args.output_dir / f"flat_mlp_{reward_key}_seed{seed}.pt"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "reward_key": reward_key, "seed": seed,
                        "config": vars(args)}, model_path)
            result["model_path"] = str(model_path)
            all_results.append(result)
        seed_results = [r for r in all_results if r.get("split_manifest") == str(args.split_manifest)]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / f"flat_mlp_qnorm_results_seed{seed}.json"
        out.write_text(json.dumps(seed_results, indent=2), encoding="utf-8")
        args.split_manifest = None if len(seeds) > 1 else args.split_manifest
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
