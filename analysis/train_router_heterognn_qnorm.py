"""
GraphRouter++ heterogeneous GNN router with query-normalized rewards.

This script is intended as a drop-in, paper-consistent replacement for
train_router_bipartite_qnorm.py. It implements message passing over a
heterogeneous graph with node types:

    task, query, prompt, model

and scores each candidate action (q, p, m) as:

    s(q,p,m) = MLP([h_q ; h_p ; h_m])

Important leakage rule:
    performance/cost/reward labels are NEVER used as message-passing inputs.
    They are only used as supervised targets and for final evaluation.

Expected input files, matching the current repo:
    data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl
    data/router/query_embeddings.pt

Run:
    python train_router_heterognn_qnorm.py

Optional:
    python train_router_heterognn_qnorm.py --epochs 50 --hidden-dim 256
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BIPARTITE_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")
DEFAULT_QUERY_EMB_PATH = Path("data/router/query_embeddings.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/router_heterognn_qnorm")

LAMBDA_CONFIGS = [
    ("reward_qnorm_lam_01", 0.1),
    ("reward_qnorm_lam_05", 0.5),
    ("reward_qnorm_lam_09", 0.9),
]

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

ID_TO_PROMPT = {v: k for k, v in PROMPT_TO_ID.items()}
ID_TO_MODEL = {v: k for k, v in MODEL_TO_ID.items()}

NODE_TYPES = ("task", "query", "prompt", "model")

# relation name -> (source type, destination type)
RELATIONS = {
    "task_to_query": ("task", "query"),
    "query_to_task": ("query", "task"),
    "prompt_to_query": ("prompt", "query"),
    "query_to_prompt": ("query", "prompt"),
    "model_to_query": ("model", "query"),
    "query_to_model": ("query", "model"),
    "prompt_to_model": ("prompt", "model"),
    "model_to_prompt": ("model", "prompt"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_router_queries(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class ActionExample:
    qid: str
    task: str
    task_id: int
    prompt: str
    prompt_id: int
    model: str
    model_id: int
    reward: float
    performance: float
    cost: float


def build_examples(router_queries: List[dict], query_embs: Dict[str, torch.Tensor], lambda_key: str) -> List[ActionExample]:
    examples: List[ActionExample] = []
    for row in router_queries:
        qid = row["qid"]
        task = row["task"]
        if qid not in query_embs or task not in TASK_TO_ID:
            continue

        for edge in row.get("action_edges", []):
            prompt = edge.get("prompt")
            model = edge.get("model")
            reward = edge.get(lambda_key)
            perf = edge.get("performance")
            cost = edge.get("cost_norm_query")
            if reward is None or perf is None or cost is None:
                continue
            if prompt not in PROMPT_TO_ID or model not in MODEL_TO_ID:
                continue
            examples.append(
                ActionExample(
                    qid=qid,
                    task=task,
                    task_id=TASK_TO_ID[task],
                    prompt=prompt,
                    prompt_id=PROMPT_TO_ID[prompt],
                    model=model,
                    model_id=MODEL_TO_ID[model],
                    reward=float(reward),
                    performance=float(perf),
                    cost=float(cost),
                )
            )
    if not examples:
        raise RuntimeError(f"No examples found for lambda key {lambda_key}")
    return examples


def split_qids(examples: List[ActionExample], train_ratio=0.7, val_ratio=0.15) -> Tuple[set, set, set]:
    qids = sorted({ex.qid for ex in examples})
    random.shuffle(qids)
    n = len(qids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_qids = set(qids[:n_train])
    val_qids = set(qids[n_train : n_train + n_val])
    test_qids = set(qids[n_train + n_val :])
    return train_qids, val_qids, test_qids


def filter_examples(examples: List[ActionExample], qids: set) -> List[ActionExample]:
    return [ex for ex in examples if ex.qid in qids]


@dataclass
class HeteroGraphBatch:
    x_dict: Dict[str, torch.Tensor]
    edge_index_dict: Dict[str, torch.Tensor]
    qid_to_local: Dict[str, int]

    def to(self, device: torch.device) -> "HeteroGraphBatch":
        return HeteroGraphBatch(
            x_dict={k: v.to(device) for k, v in self.x_dict.items()},
            edge_index_dict={k: v.to(device) for k, v in self.edge_index_dict.items()},
            qid_to_local=self.qid_to_local,
        )


def _edge_index(edges: Iterable[Tuple[int, int]]) -> torch.Tensor:
    edges = list(edges)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_graph_for_qids(qids: set, router_queries: List[dict], query_embs: Dict[str, torch.Tensor]) -> HeteroGraphBatch:
    """Builds a structural graph for the selected queries.

    This graph includes only known structural information: task membership,
    candidate prompts, candidate models, and the prompt-model action lattice.
    It does NOT include reward/performance/cost as edge attributes.
    """
    selected_rows = [r for r in router_queries if r["qid"] in qids and r["qid"] in query_embs and r["task"] in TASK_TO_ID]
    selected_rows = sorted(selected_rows, key=lambda r: r["qid"])

    qid_to_local = {row["qid"]: i for i, row in enumerate(selected_rows)}
    if not selected_rows:
        raise RuntimeError("Cannot build graph with zero selected queries.")

    query_x = torch.stack([query_embs[row["qid"]].float() for row in selected_rows], dim=0)
    task_ids = torch.arange(len(TASK_TO_ID), dtype=torch.long)
    prompt_ids = torch.arange(len(PROMPT_TO_ID), dtype=torch.long)
    model_ids = torch.arange(len(MODEL_TO_ID), dtype=torch.long)

    tq, qt = [], []
    pq, qp = [], []
    mq, qm = [], []

    for row in selected_rows:
        q_local = qid_to_local[row["qid"]]
        t_id = TASK_TO_ID[row["task"]]
        tq.append((t_id, q_local))
        qt.append((q_local, t_id))

        # In real inference, all candidate prompts/models are known alternatives.
        # This is structural availability, not an observed reward label.
        for p_id in PROMPT_TO_ID.values():
            pq.append((p_id, q_local))
            qp.append((q_local, p_id))
        for m_id in MODEL_TO_ID.values():
            mq.append((m_id, q_local))
            qm.append((q_local, m_id))

    pm = [(p_id, m_id) for p_id in PROMPT_TO_ID.values() for m_id in MODEL_TO_ID.values()]
    mp = [(m_id, p_id) for p_id in PROMPT_TO_ID.values() for m_id in MODEL_TO_ID.values()]

    return HeteroGraphBatch(
        x_dict={
            "query": query_x,
            "task": task_ids,
            "prompt": prompt_ids,
            "model": model_ids,
        },
        edge_index_dict={
            "task_to_query": _edge_index(tq),
            "query_to_task": _edge_index(qt),
            "prompt_to_query": _edge_index(pq),
            "query_to_prompt": _edge_index(qp),
            "model_to_query": _edge_index(mq),
            "query_to_model": _edge_index(qm),
            "prompt_to_model": _edge_index(pm),
            "model_to_prompt": _edge_index(mp),
        },
        qid_to_local=qid_to_local,
    )


class HeteroSageLayer(nn.Module):
    """Small dependency-free heterogeneous GraphSAGE-style layer."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_lin = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in NODE_TYPES})
        self.rel_lin = nn.ModuleDict({rel: nn.Linear(hidden_dim, hidden_dim, bias=False) for rel in RELATIONS})
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES})
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(iter(h.values())).device
        aggs = {nt: torch.zeros_like(h[nt]) for nt in NODE_TYPES}

        for rel, (src_type, dst_type) in RELATIONS.items():
            edge_index = edge_index_dict[rel]
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            msg = self.rel_lin[rel](h[src_type][src])
            out = torch.zeros_like(h[dst_type])
            out.index_add_(0, dst, msg)

            deg = torch.zeros(h[dst_type].size(0), device=device, dtype=h[dst_type].dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h[dst_type].dtype))
            out = out / deg.clamp_min(1.0).unsqueeze(-1)
            aggs[dst_type] = aggs[dst_type] + out

        new_h = {}
        for nt in NODE_TYPES:
            z = self.self_lin[nt](h[nt]) + aggs[nt]
            z = self.norm[nt](z)
            z = F.relu(z)
            new_h[nt] = self.dropout(z)
        return new_h


class GraphRouterPPHeteroGNN(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_tasks: int,
        num_prompts: int,
        num_models: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.prompt_emb = nn.Embedding(num_prompts, hidden_dim)
        self.model_emb = nn.Embedding(num_models, hidden_dim)
        self.layers = nn.ModuleList([HeteroSageLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, graph: HeteroGraphBatch) -> Dict[str, torch.Tensor]:
        h = {
            "query": self.query_proj(graph.x_dict["query"]),
            "task": self.task_emb(graph.x_dict["task"]),
            "prompt": self.prompt_emb(graph.x_dict["prompt"]),
            "model": self.model_emb(graph.x_dict["model"]),
        }
        for layer in self.layers:
            h = layer(h, graph.edge_index_dict)
        return h

    def score_actions(
        self,
        h: Dict[str, torch.Tensor],
        q_idx: torch.Tensor,
        prompt_id: torch.Tensor,
        model_id: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h["query"][q_idx], h["prompt"][prompt_id], h["model"][model_id]], dim=-1)
        return self.scorer(x).squeeze(-1)


def tensorize_examples(examples: List[ActionExample], graph: HeteroGraphBatch, device: torch.device) -> Dict[str, torch.Tensor]:
    missing = [ex.qid for ex in examples if ex.qid not in graph.qid_to_local]
    if missing:
        raise RuntimeError(f"{len(missing)} examples refer to qids not present in the graph; first={missing[0]}")

    return {
        "q_idx": torch.tensor([graph.qid_to_local[ex.qid] for ex in examples], dtype=torch.long, device=device),
        "prompt_id": torch.tensor([ex.prompt_id for ex in examples], dtype=torch.long, device=device),
        "model_id": torch.tensor([ex.model_id for ex in examples], dtype=torch.long, device=device),
        "reward": torch.tensor([ex.reward for ex in examples], dtype=torch.float32, device=device),
    }


def minibatch_indices(n: int, batch_size: int, device: torch.device, shuffle: bool = True) -> Iterable[torch.Tensor]:
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


@torch.no_grad()
def evaluate_action_selection(
    model: GraphRouterPPHeteroGNN,
    graph: HeteroGraphBatch,
    examples: List[ActionExample],
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    h = model.encode(graph)
    data = tensorize_examples(examples, graph, device)

    preds = []
    for idx in minibatch_indices(len(examples), batch_size, device, shuffle=False):
        pred = model.score_actions(h, data["q_idx"][idx], data["prompt_id"][idx], data["model_id"][idx])
        preds.append(pred.cpu())
    preds = torch.cat(preds).tolist()

    by_qid = defaultdict(list)
    for pred, ex in zip(preds, examples):
        by_qid[ex.qid].append(
            {
                "pred": float(pred),
                "reward": ex.reward,
                "performance": ex.performance,
                "cost": ex.cost,
                "task": ex.task,
                "prompt": ex.prompt,
                "model": ex.model,
            }
        )

    selected = [max(cands, key=lambda x: x["pred"]) for cands in by_qid.values()]
    if not selected:
        raise RuntimeError("Evaluation received zero selected actions.")

    prompt_counts = defaultdict(int)
    model_counts = defaultdict(int)
    task_counts = defaultdict(int)
    for row in selected:
        prompt_counts[row["prompt"]] += 1
        model_counts[row["model"]] += 1
        task_counts[row["task"]] += 1

    return {
        "queries": len(selected),
        "avg_reward": sum(x["reward"] for x in selected) / len(selected),
        "avg_performance": sum(x["performance"] for x in selected) / len(selected),
        "avg_cost": sum(x["cost"] for x in selected) / len(selected),
        "prompt_counts": dict(prompt_counts),
        "model_counts": dict(model_counts),
        "task_counts": dict(task_counts),
    }


def train_one_lambda(
    lambda_key: str,
    lam: float,
    router_queries: List[dict],
    query_embs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    examples = build_examples(router_queries, query_embs, lambda_key)
    train_qids, val_qids, test_qids = split_qids(examples, args.train_ratio, args.val_ratio)

    train_ex = filter_examples(examples, train_qids)
    val_ex = filter_examples(examples, val_qids)
    test_ex = filter_examples(examples, test_qids)

    # Train graph contains only train queries. Val/test graphs include train queries
    # plus the target split queries. This gives eval queries structural context but
    # no reward/performance/cost labels as inputs.
    train_graph = build_graph_for_qids(train_qids, router_queries, query_embs).to(device)
    val_graph = build_graph_for_qids(train_qids | val_qids, router_queries, query_embs).to(device)
    test_graph = build_graph_for_qids(train_qids | val_qids | test_qids, router_queries, query_embs).to(device)

    train_data = tensorize_examples(train_ex, train_graph, device)
    query_dim = next(iter(query_embs.values())).shape[0]

    model = GraphRouterPPHeteroGNN(
        query_dim=query_dim,
        num_tasks=len(TASK_TO_ID),
        num_prompts=len(PROMPT_TO_ID),
        num_models=len(MODEL_TO_ID),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_reward = -1e9
    best_state = None
    stale_epochs = 0

    print(f"\n==== TRAINING HETERO-GNN {lambda_key} / lambda={lam} ====")
    print(f"Actions: total={len(examples)} train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")
    print(f"Queries: train={len(train_qids)} val={len(val_qids)} test={len(test_qids)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        h = model.encode(train_graph)
        total_loss = 0.0
        total_items = 0

        for idx in minibatch_indices(len(train_ex), args.batch_size, device, shuffle=True):
            pred = model.score_actions(
                h,
                train_data["q_idx"][idx],
                train_data["prompt_id"][idx],
                train_data["model_id"][idx],
            )
            loss = F.mse_loss(pred, train_data["reward"][idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * idx.numel()
            total_items += idx.numel()

            # h depends on model parameters. Re-encode after each optimizer step
            # to avoid stale activations across mini-batches.
            h = model.encode(train_graph)

        val_metrics = evaluate_action_selection(model, val_graph, val_ex, device, args.batch_size)
        train_loss = total_loss / max(1, total_items)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_R={val_metrics['avg_reward']:.6f} | "
            f"val_P={val_metrics['avg_performance']:.6f} | val_C={val_metrics['avg_cost']:.6f}"
        )

        if val_metrics["avg_reward"] > best_val_reward:
            best_val_reward = val_metrics["avg_reward"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was captured.")

    model.load_state_dict(best_state)
    test_metrics = evaluate_action_selection(model, test_graph, test_ex, device, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"router_heterognn_{lambda_key}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "lambda_key": lambda_key,
            "lambda": lam,
            "config": vars(args),
            "task_to_id": TASK_TO_ID,
            "prompt_to_id": PROMPT_TO_ID,
            "model_to_id": MODEL_TO_ID,
        },
        model_path,
    )

    result = {
        "lambda_key": lambda_key,
        "lambda": lam,
        "queries": test_metrics["queries"],
        "P": test_metrics["avg_performance"],
        "C": test_metrics["avg_cost"],
        "R": test_metrics["avg_reward"],
        "prompt_counts": test_metrics["prompt_counts"],
        "model_counts": test_metrics["model_counts"],
        "task_counts": test_metrics["task_counts"],
        "model_path": str(model_path),
    }

    print("\n==== TEST RESULTS ====")
    print(f"queries={result['queries']}")
    print(f"P={result['P']:.6f} C={result['C']:.6f} R={result['R']:.6f}")
    print(f"prompt_counts={result['prompt_counts']}")
    print(f"model_counts={result['model_counts']}")
    print(f"saved_model={model_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphRouter++ Hetero-GNN with qnorm rewards.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_BIPARTITE_DATA_PATH)
    parser.add_argument("--query-emb-path", type=Path, default=DEFAULT_QUERY_EMB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print("==== TRAIN GRAPHROUTER++ HETERO-GNN QNORM ====")
    print("Device:", device)
    print("Data:", args.data_path)
    print("Query embeddings:", args.query_emb_path)

    router_queries = load_router_queries(args.data_path)
    query_embs = torch.load(args.query_emb_path, map_location="cpu")

    all_results = []
    for lambda_key, lam in LAMBDA_CONFIGS:
        set_seed(args.seed)
        result = train_one_lambda(lambda_key, lam, router_queries, query_embs, args, device)
        all_results.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "router_heterognn_qnorm_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n==== OVERLEAF ROW ====")
    vals = []
    for result in all_results:
        vals.extend([f"{result['P']:.3f}", f"{result['C']:.3f}", f"{result['R']:.3f}"])
    print("GraphRouter++ Hetero-GNN (qnorm) & " + " & ".join(vals) + r" \\")
    print(f"\nSaved results: {results_path}")


if __name__ == "__main__":
    main()
