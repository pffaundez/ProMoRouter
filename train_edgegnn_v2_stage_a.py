#!/usr/bin/env python3
"""Train one Edge-GNN v2 Stage A arm for one seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from collections import Counter
from pathlib import Path

import torch

from router.data_validation import require_complete_action_space
from router.edgegnn_v2_stage_a import (
    ARMS, FlatSharedObjectiveRouter, MatchedEdgeRouter, build_ego_graph_batch,
    shared_objective, topology_metadata,
)
from router.splits import load_or_create_splits

PROMPTS = ("direct", "cot", "decompose", "selfcheck")
MODELS = ("mistral-7b", "qwen2.5-7b", "llama3.1-8b", "qwen2.5-14b", "yi-34b",
          "codellama-34b", "mixtral-8x7b", "llama3.1-70b", "qwen2.5-72b")
TASKS = ("gsm8k", "hotpotqa", "squad", "alpaca")
LAMBDAS = (("reward_qnorm_lam_01", 0.1), ("reward_qnorm_lam_05", 0.5), ("reward_qnorm_lam_09", 0.9))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--query-emb-path", type=Path, required=True)
    p.add_argument("--task-emb-path", type=Path, required=True)
    p.add_argument("--prompt-emb-path", type=Path, required=True)
    p.add_argument("--model-emb-path", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--deterministic", action="store_true", required=True)
    return p.parse_args()


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_table(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("embedding_by_id", "embeddings_by_id"):
            if isinstance(obj.get(key), dict): return {k: v.float() for k, v in obj[key].items()}
        if isinstance(obj.get("ids"), (list, tuple)) and isinstance(obj.get("embeddings"), torch.Tensor):
            return {name: obj["embeddings"][i].float() for i, name in enumerate(obj["ids"])}
        for key in ("embeddings", "embedding", "tensor", "values"):
            if isinstance(obj.get(key), torch.Tensor): return obj[key].float()
        if obj and all(isinstance(value, torch.Tensor) for value in obj.values()):
            return {k: v.float() for k, v in obj.items()}
    raise ValueError(f"unsupported embedding format: {path}")


def vector(table, key, order=None):
    if isinstance(table, dict): return table[key].float().view(-1)
    if order is None: raise ValueError("tensor table requires identifier order")
    return table[order.index(key)].float().view(-1)


def graph_for(rows, query_table, task_table, prompt_table, model_table, device):
    qids = [row["qid"] for row in rows]
    graph = build_ego_graph_batch(
        qids,
        torch.stack([vector(query_table, row["qid"]) for row in rows]),
        torch.stack([vector(task_table, row["task"], TASKS) for row in rows]),
        torch.stack([vector(prompt_table, item, PROMPTS) for item in PROMPTS]),
        torch.stack([vector(model_table, item, MODELS) for item in MODELS]),
    )
    return graph.to(device)


def labels_for(rows, reward_key, device):
    rewards, perf, cost, actions = [], [], [], []
    canonical = [(p, m) for p in PROMPTS for m in MODELS]
    for row in rows:
        by_pair = {(e["prompt"], e["model"]): e for e in row["action_edges"]}
        rewards.append([float(by_pair[pair][reward_key]) for pair in canonical])
        perf.append([float(by_pair[pair]["performance"]) for pair in canonical])
        cost.append([float(by_pair[pair]["cost_norm_query"]) for pair in canonical])
        actions.append(canonical)
    return (torch.tensor(rewards, device=device), torch.tensor(perf, device=device),
            torch.tensor(cost, device=device), actions)


def batches(rows, size, shuffle):
    rows = list(rows)
    if shuffle: random.shuffle(rows)
    for start in range(0, len(rows), size): yield rows[start:start + size]


def make_model(arm, dims, args, device):
    if arm == "flat_mlp_shared_objective":
        model = FlatSharedObjectiveRouter(dims["query"], dims["prompt"], dims["model"], args.hidden_dim, args.dropout)
    else:
        model = MatchedEdgeRouter(dims, args.hidden_dim, args.dropout, arm == "edgegnn_v2_full_mp")
    return model.to(device)


@torch.no_grad()
def evaluate(model, rows, tables, reward_key, lam, args, device):
    model.eval(); selected = []
    for chunk in batches(rows, args.query_batch_size, False):
        graph = graph_for(chunk, *tables, device)
        _, perf, cost, actions = labels_for(chunk, reward_key, device)
        chosen = model(graph).argmax(dim=-1)
        for i, index in enumerate(chosen.tolist()):
            p, m = actions[i][index]
            selected.append((chunk[i]["qid"], float(perf[i, index]), float(cost[i, index]), p, m, chunk[i]["task"]))
    p_avg = sum(x[1] for x in selected) / len(selected)
    c_avg = sum(x[2] for x in selected) / len(selected)
    return {"queries": len(selected), "P": p_avg, "C": c_avg, "R": p_avg - lam * c_avg,
            "prompt_counts": dict(Counter(x[3] for x in selected)),
            "model_counts": dict(Counter(x[4] for x in selected)),
            "task_counts": dict(Counter(x[5] for x in selected)),
            "selections": [{"qid": x[0], "P": x[1], "C": x[2], "R": x[1] - lam * x[2],
                            "prompt": x[3], "model": x[4], "task": x[5]} for x in selected]}


def main():
    args = parse_args(); seed_all(args.seed); device = torch.device(args.device)
    rows = [json.loads(line) for line in args.data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require_complete_action_space(rows, PROMPTS, MODELS)
    query_table, task_table, prompt_table, model_table = map(load_table, (
        args.query_emb_path, args.task_emb_path, args.prompt_emb_path, args.model_emb_path))
    tables = (query_table, task_table, prompt_table, model_table)
    train_ids, val_ids, test_ids = load_or_create_splits(
        [r["qid"] for r in rows], args.split_manifest, args.seed, 0.70, 0.15)
    subsets = {name: [r for r in rows if r["qid"] in ids]
               for name, ids in (("train", set(train_ids)), ("val", set(val_ids)), ("test", set(test_ids)))}
    sample = rows[0]
    dims = {"query": vector(query_table, sample["qid"]).numel(),
            "task": vector(task_table, sample["task"], TASKS).numel(),
            "prompt": vector(prompt_table, PROMPTS[0], PROMPTS).numel(),
            "model": vector(model_table, MODELS[0], MODELS).numel()}
    fingerprint = hashlib.sha256(args.data_path.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True); results = []
    for reward_key, lam in LAMBDAS:
        seed_all(args.seed); model = make_model(args.arm, dims, args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_reward, best_state, stale = -float("inf"), None, 0
        active_parameter_names = set()
        for _epoch in range(1, args.epochs + 1):
            model.train()
            for chunk in batches(subsets["train"], args.query_batch_size, True):
                graph = graph_for(chunk, *tables, device)
                rewards, _, _, _ = labels_for(chunk, reward_key, device)
                loss = shared_objective(model(graph), rewards)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                active_parameter_names.update(name for name, parameter in model.named_parameters() if parameter.grad is not None)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
            val = evaluate(model, subsets["val"], tables, reward_key, lam, args, device)
            if val["R"] > best_reward:
                best_reward = val["R"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
            else: stale += 1
            if stale >= args.patience: break
        if best_state is None: raise RuntimeError("no checkpoint captured")
        model.load_state_dict(best_state)
        checkpoint = args.output_dir / f"{args.arm}_{reward_key}_seed{args.seed}.pt"
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        torch.save({"state_dict": best_state, "config": config, "lambda_key": reward_key}, checkpoint)
        result = evaluate(model, subsets["test"], tables, reward_key, lam, args, device)
        result.update({
            "arm_id": args.arm, "seed": args.seed, "lambda_key": reward_key, "lambda": lam,
            "dataset_path": str(args.data_path), "dataset_sha256": fingerprint,
            "split_manifest": str(args.split_manifest), "topology": topology_metadata(len(subsets["test"]), 4, 9),
            "processing_layers": 2 if args.arm != "flat_mlp_shared_objective" else 0,
            "message_passing_layers": 2 if args.arm == "edgegnn_v2_full_mp" else 0,
            "objective": {"listwise_alpha": .10, "ce_alpha": .05, "entropy_beta": .005, "temperature": .10},
            "early_stopping": "validation_reward", "deterministic": True,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
            "checkpoint_path": str(checkpoint), "config": config,
            "parameter_counts": {
                "total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "active_with_gradient": sum(p.numel() for name, p in model.named_parameters() if name in active_parameter_names),
                "inactive_trainable": sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and name not in active_parameter_names),
            },
        })
        results.append(result)
    output = args.output_dir / f"stage_a_results_seed{args.seed}.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved_results={output}")


if __name__ == "__main__": main()
