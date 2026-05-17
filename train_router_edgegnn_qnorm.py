"""
GraphRouter++ Edge-GNN router with query-normalized rewards.

This is the more GraphRouter-like variant:

  * The graph used for message passing contains only TRAIN-observed routing
    structure, derived from the top-k prompt/model actions per train query.
  * Validation/test queries are connected only to their task node, not to all
    prompt/model candidates. Candidate actions are scored as edge predictions.
  * Reward/performance/cost are never used as node features or edge attributes.
  * The scorer uses both contextualized GNN embeddings and residual/raw projected
    text embeddings to preserve query/task/prompt/model semantics.

Run:
  CUDA_VISIBLE_DEVICES=1 python train_router_edgegnn_qnorm.py

Recommended seeds:
  CUDA_VISIBLE_DEVICES=1 python train_router_edgegnn_qnorm.py --seed 1
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
DEFAULT_TASK_EMB_PATH = Path("data/router/task_embeddings.pt")
DEFAULT_PROMPT_EMB_PATH = Path("data/router/prompt_embeddings.pt")
DEFAULT_MODEL_EMB_PATH = Path("data/router/model_embeddings.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/router_edgegnn_qnorm")

LAMBDA_CONFIGS = [
    ("reward_qnorm_lam_01", 0.1),
    ("reward_qnorm_lam_05", 0.5),
    ("reward_qnorm_lam_09", 0.9),
]

TASK_TO_ID = {"gsm8k": 0, "hotpotqa": 1, "squad": 2, "alpaca": 3}
PROMPT_TO_ID = {"direct": 0, "cot": 1, "decompose": 2, "selfcheck": 3}
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

NODE_TYPES = ("task", "query", "prompt", "model")
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
    return set(qids[:n_train]), set(qids[n_train : n_train + n_val]), set(qids[n_train + n_val :])


def filter_examples(examples: List[ActionExample], qids: set) -> List[ActionExample]:
    return [ex for ex in examples if ex.qid in qids]


def group_by_qid(examples: List[ActionExample]) -> Dict[str, List[ActionExample]]:
    out: Dict[str, List[ActionExample]] = defaultdict(list)
    for ex in examples:
        out[ex.qid].append(ex)
    return dict(out)


def _edge_index(edges: Iterable[Tuple[int, int]]) -> torch.Tensor:
    edges = sorted(set(edges))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


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


def select_observed_train_edges(train_examples: List[ActionExample], top_k: int, threshold_quantile: float | None) -> List[ActionExample]:
    """Select routing edges visible to the GNN from TRAIN labels only.

    The selected examples define observed graph structure. Labels are not passed
    as edge attributes; labels are used only here offline to choose which train
    actions are considered historically successful edges, analogous to observed
    train edges in GraphRouter.
    """
    selected: List[ActionExample] = []
    for _, cands in group_by_qid(train_examples).items():
        cands = sorted(cands, key=lambda ex: ex.reward, reverse=True)
        if threshold_quantile is not None:
            rewards = torch.tensor([ex.reward for ex in cands], dtype=torch.float32)
            threshold = float(torch.quantile(rewards, threshold_quantile).item())
            kept = [ex for ex in cands if ex.reward >= threshold]
        else:
            kept = cands[:top_k]
        if not kept:
            kept = cands[: max(1, top_k)]
        selected.extend(kept[: max(1, top_k)] if threshold_quantile is None else kept)
    return selected


def build_edge_graph(
    query_node_qids: set,
    router_queries: List[dict],
    query_embs: Dict[str, torch.Tensor],
    task_embs: torch.Tensor,
    prompt_embs: torch.Tensor,
    model_embs: torch.Tensor,
    observed_train_edges: List[ActionExample],
    include_full_prompt_model_lattice: bool = False,
) -> HeteroGraphBatch:
    """Build graph for message passing.

    All selected query nodes get task edges. Only train-observed successful
    actions create query-prompt/query-model/prompt-model edges. Thus val/test
    queries are not connected to candidate prompts/models before prediction.
    """
    selected_rows = [
        r for r in router_queries if r["qid"] in query_node_qids and r["qid"] in query_embs and r["task"] in TASK_TO_ID
    ]
    selected_rows = sorted(selected_rows, key=lambda r: r["qid"])
    if not selected_rows:
        raise RuntimeError("Cannot build graph with zero selected queries.")

    qid_to_local = {row["qid"]: i for i, row in enumerate(selected_rows)}
    query_x = torch.stack([query_embs[row["qid"]].float() for row in selected_rows], dim=0)

    tq, qt = [], []
    for row in selected_rows:
        q_local = qid_to_local[row["qid"]]
        t_id = TASK_TO_ID[row["task"]]
        tq.append((t_id, q_local))
        qt.append((q_local, t_id))

    pq, qp, mq, qm, pm, mp = [], [], [], [], [], []
    for ex in observed_train_edges:
        if ex.qid not in qid_to_local:
            continue
        q_local = qid_to_local[ex.qid]
        p_id = ex.prompt_id
        m_id = ex.model_id
        pq.append((p_id, q_local))
        qp.append((q_local, p_id))
        mq.append((m_id, q_local))
        qm.append((q_local, m_id))
        pm.append((p_id, m_id))
        mp.append((m_id, p_id))

    if include_full_prompt_model_lattice:
        for p_id in PROMPT_TO_ID.values():
            for m_id in MODEL_TO_ID.values():
                pm.append((p_id, m_id))
                mp.append((m_id, p_id))

    return HeteroGraphBatch(
        x_dict={
            "query": query_x,
            "task": task_embs.float(),
            "prompt": prompt_embs.float(),
            "model": model_embs.float(),
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
            aggs[dst_type] = aggs[dst_type] + out / deg.clamp_min(1.0).unsqueeze(-1)

        new_h = {}
        for nt in NODE_TYPES:
            z = self.self_lin[nt](h[nt]) + aggs[nt]
            z = self.norm[nt](z)
            z = F.relu(z)
            new_h[nt] = self.dropout(z)
        return new_h


class GraphRouterPPEdgeGNN(nn.Module):
    def __init__(
        self,
        query_dim: int,
        task_dim: int,
        prompt_dim: int,
        model_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.task_proj = nn.Linear(task_dim, hidden_dim)
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        self.model_proj = nn.Linear(model_dim, hidden_dim)
        self.layers = nn.ModuleList([HeteroSageLayer(hidden_dim, dropout) for _ in range(num_layers)])
        # h_q,h_p,h_m + raw_q,raw_p,raw_m + pairwise products
        scorer_in = hidden_dim * 9
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def initial_project(self, graph: HeteroGraphBatch) -> Dict[str, torch.Tensor]:
        return {
            "query": self.query_proj(graph.x_dict["query"]),
            "task": self.task_proj(graph.x_dict["task"]),
            "prompt": self.prompt_proj(graph.x_dict["prompt"]),
            "model": self.model_proj(graph.x_dict["model"]),
        }

    def encode(self, graph: HeteroGraphBatch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x0 = self.initial_project(graph)
        h = {k: v for k, v in x0.items()}
        for layer in self.layers:
            h = layer(h, graph.edge_index_dict)
        return h, x0

    def score_actions(
        self,
        h: Dict[str, torch.Tensor],
        x0: Dict[str, torch.Tensor],
        q_idx: torch.Tensor,
        prompt_id: torch.Tensor,
        model_id: torch.Tensor,
    ) -> torch.Tensor:
        hq, hp, hm = h["query"][q_idx], h["prompt"][prompt_id], h["model"][model_id]
        xq, xp, xm = x0["query"][q_idx], x0["prompt"][prompt_id], x0["model"][model_id]
        feats = torch.cat([hq, hp, hm, xq, xp, xm, hq * hp, hq * hm, hp * hm], dim=-1)
        return self.scorer(feats).squeeze(-1)


def tensorize_qid_group(cands: List[ActionExample], graph: HeteroGraphBatch, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "q_idx": torch.tensor([graph.qid_to_local[ex.qid] for ex in cands], dtype=torch.long, device=device),
        "prompt_id": torch.tensor([ex.prompt_id for ex in cands], dtype=torch.long, device=device),
        "model_id": torch.tensor([ex.model_id for ex in cands], dtype=torch.long, device=device),
        "reward": torch.tensor([ex.reward for ex in cands], dtype=torch.float32, device=device),
        "action_id": torch.tensor([ex.prompt_id * len(MODEL_TO_ID) + ex.model_id for ex in cands], dtype=torch.long, device=device),
    }


def qid_batches(qids: List[str], batch_size: int, shuffle: bool) -> Iterable[List[str]]:
    qids = list(qids)
    if shuffle:
        random.shuffle(qids)
    for i in range(0, len(qids), batch_size):
        yield qids[i : i + batch_size]


def compute_batch_loss(
    model: GraphRouterPPEdgeGNN,
    graph: HeteroGraphBatch,
    grouped: Dict[str, List[ActionExample]],
    batch_qids: List[str],
    device: torch.device,
    listwise_alpha: float,
    ce_alpha: float,
    entropy_beta: float,
    temperature: float,
) -> torch.Tensor:
    h, x0 = model.encode(graph)
    losses = []
    action_probs_accum = torch.zeros(len(PROMPT_TO_ID) * len(MODEL_TO_ID), device=device)
    for qid in batch_qids:
        cands = grouped[qid]
        data = tensorize_qid_group(cands, graph, device)
        scores = model.score_actions(h, x0, data["q_idx"], data["prompt_id"], data["model_id"])
        rewards = data["reward"]
        mse = F.mse_loss(scores, rewards)

        target_dist = F.softmax(rewards / max(temperature, 1e-6), dim=0)
        kl = F.kl_div(F.log_softmax(scores, dim=0), target_dist, reduction="batchmean")
        best_idx = torch.argmax(rewards).view(1)
        ce = F.cross_entropy(scores.view(1, -1), best_idx)
        loss = mse + listwise_alpha * kl + ce_alpha * ce
        losses.append(loss)

        if entropy_beta > 0:
            probs = F.softmax(scores, dim=0)
            action_probs_accum.index_add_(0, data["action_id"], probs.detach() if False else probs)

    total = torch.stack(losses).mean()
    if entropy_beta > 0 and batch_qids:
        dist = action_probs_accum / action_probs_accum.sum().clamp_min(1e-8)
        entropy = -(dist * (dist + 1e-8).log()).sum()
        # maximize entropy over aggregate action usage to discourage early fixed-pair collapse
        total = total - entropy_beta * entropy
    return total


@torch.no_grad()
def evaluate_action_selection(
    model: GraphRouterPPEdgeGNN,
    graph: HeteroGraphBatch,
    examples: List[ActionExample],
    device: torch.device,
) -> dict:
    model.eval()
    h, x0 = model.encode(graph)
    by_qid = group_by_qid(examples)
    selected = []
    for qid, cands in by_qid.items():
        data = tensorize_qid_group(cands, graph, device)
        scores = model.score_actions(h, x0, data["q_idx"], data["prompt_id"], data["model_id"])
        best = int(torch.argmax(scores).item())
        ex = cands[best]
        selected.append(ex)

    prompt_counts, model_counts, task_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    for ex in selected:
        prompt_counts[ex.prompt] += 1
        model_counts[ex.model] += 1
        task_counts[ex.task] += 1

    return {
        "queries": len(selected),
        "avg_reward": sum(ex.reward for ex in selected) / len(selected),
        "avg_performance": sum(ex.performance for ex in selected) / len(selected),
        "avg_cost": sum(ex.cost for ex in selected) / len(selected),
        "prompt_counts": dict(prompt_counts),
        "model_counts": dict(model_counts),
        "task_counts": dict(task_counts),
    }


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        # allow metadata-style files, but prefer plain tensor
        for key in ("embeddings", "tensor"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].float()
    raise TypeError(f"Unsupported embedding file format at {path}")


def train_one_lambda(
    lambda_key: str,
    lam: float,
    router_queries: List[dict],
    query_embs: Dict[str, torch.Tensor],
    task_embs: torch.Tensor,
    prompt_embs: torch.Tensor,
    model_embs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    examples = build_examples(router_queries, query_embs, lambda_key)
    train_qids, val_qids, test_qids = split_qids(examples, args.train_ratio, args.val_ratio)
    train_ex = filter_examples(examples, train_qids)
    val_ex = filter_examples(examples, val_qids)
    test_ex = filter_examples(examples, test_qids)
    observed_edges = select_observed_train_edges(train_ex, top_k=args.edge_top_k, threshold_quantile=args.edge_threshold_quantile)

    train_graph = build_edge_graph(
        train_qids,
        router_queries,
        query_embs,
        task_embs,
        prompt_embs,
        model_embs,
        observed_edges,
        include_full_prompt_model_lattice=args.full_prompt_model_lattice,
    ).to(device)
    val_graph = build_edge_graph(
        train_qids | val_qids,
        router_queries,
        query_embs,
        task_embs,
        prompt_embs,
        model_embs,
        observed_edges,
        include_full_prompt_model_lattice=args.full_prompt_model_lattice,
    ).to(device)
    test_graph = build_edge_graph(
        train_qids | test_qids,
        router_queries,
        query_embs,
        task_embs,
        prompt_embs,
        model_embs,
        observed_edges,
        include_full_prompt_model_lattice=args.full_prompt_model_lattice,
    ).to(device)

    grouped_train = group_by_qid(train_ex)
    train_qid_list = sorted(grouped_train)
    query_dim = next(iter(query_embs.values())).shape[0]
    model = GraphRouterPPEdgeGNN(
        query_dim=query_dim,
        task_dim=task_embs.shape[1],
        prompt_dim=prompt_embs.shape[1],
        model_dim=model_embs.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"\n==== TRAINING EDGE-GNN {lambda_key} / lambda={lam} ====")
    print(f"Actions: total={len(examples)} train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")
    print(f"Queries: train={len(train_qids)} val={len(val_qids)} test={len(test_qids)}")
    print(f"Observed train graph edges from top_k={args.edge_top_k}: {len(observed_edges)} action edges")
    print(
        f"Loss: mse + {args.listwise_alpha}*KL + {args.ce_alpha}*CE - {args.entropy_beta}*entropy | "
        f"temperature={args.temperature}"
    )

    best_val_reward = -1e9
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch_qids in qid_batches(train_qid_list, args.query_batch_size, shuffle=True):
            loss = compute_batch_loss(
                model=model,
                graph=train_graph,
                grouped=grouped_train,
                batch_qids=batch_qids,
                device=device,
                listwise_alpha=args.listwise_alpha,
                ce_alpha=args.ce_alpha,
                entropy_beta=args.entropy_beta,
                temperature=args.temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        val_metrics = evaluate_action_selection(model, val_graph, val_ex, device)
        train_loss = total_loss / max(1, total_batches)
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
        raise RuntimeError("Training failed: no best model state captured.")
    model.load_state_dict(best_state)
    test_metrics = evaluate_action_selection(model, test_graph, test_ex, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"router_edgegnn_{lambda_key}_seed{args.seed}.pt"
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
    p = argparse.ArgumentParser(description="Train GraphRouter++ Edge-GNN with qnorm rewards.")
    p.add_argument("--data-path", type=Path, default=DEFAULT_BIPARTITE_DATA_PATH)
    p.add_argument("--query-emb-path", type=Path, default=DEFAULT_QUERY_EMB_PATH)
    p.add_argument("--task-emb-path", type=Path, default=DEFAULT_TASK_EMB_PATH)
    p.add_argument("--prompt-emb-path", type=Path, default=DEFAULT_PROMPT_EMB_PATH)
    p.add_argument("--model-emb-path", type=Path, default=DEFAULT_MODEL_EMB_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--query-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--edge-top-k", type=int, default=3, help="Top-k train actions per query used as visible graph edges.")
    p.add_argument("--edge-threshold-quantile", type=float, default=None, help="Optional reward quantile threshold for visible edges.")
    p.add_argument("--full-prompt-model-lattice", action="store_true", help="Also connect all prompt-model pairs structurally.")
    p.add_argument("--listwise-alpha", type=float, default=0.10)
    p.add_argument("--ce-alpha", type=float, default=0.05)
    p.add_argument("--entropy-beta", type=float, default=0.005)
    p.add_argument("--temperature", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print("==== TRAIN GRAPHROUTER++ EDGE-GNN QNORM ====")
    print("Device:", device)
    print("Data:", args.data_path)
    print("Query embeddings:", args.query_emb_path)
    print("Task embeddings:", args.task_emb_path)
    print("Prompt embeddings:", args.prompt_emb_path)
    print("Model embeddings:", args.model_emb_path)

    router_queries = load_router_queries(args.data_path)
    query_embs = torch.load(args.query_emb_path, map_location="cpu")
    task_embs = load_tensor(args.task_emb_path)
    prompt_embs = load_tensor(args.prompt_emb_path)
    model_embs = load_tensor(args.model_emb_path)

    all_results = []
    for lambda_key, lam in LAMBDA_CONFIGS:
        set_seed(args.seed)
        result = train_one_lambda(lambda_key, lam, router_queries, query_embs, task_embs, prompt_embs, model_embs, args, device)
        all_results.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"router_edgegnn_qnorm_results_seed{args.seed}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n==== OVERLEAF ROW ====")
    vals = []
    for r in all_results:
        vals.extend([f"{r['P']:.3f}", f"{r['C']:.3f}", f"{r['R']:.3f}"])
    print("GraphRouter++ Edge-GNN (qnorm) & " + " & ".join(vals) + r" \\")
    print(f"\nSaved results: {results_path}")


if __name__ == "__main__":
    main()
