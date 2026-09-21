"""Core models and graph construction for Edge-GNN v2 Stage A.

The graph batch is a disjoint union of per-query ego graphs. Prompt, model and
task nodes are copied for every query, so information cannot cross queries.
Realized outcomes are carried only as labels and never enter graph features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ARMS = ("edgegnn_v2_full_mp", "edgegnn_v2_self_only", "flat_mlp_shared_objective")
NODE_TYPES = ("query", "task", "prompt", "model")
RELATIONS = {
    "query_to_task": ("query", "task"),
    "task_to_query": ("task", "query"),
    "query_to_prompt": ("query", "prompt"),
    "prompt_to_query": ("prompt", "query"),
    "query_to_model": ("query", "model"),
    "model_to_query": ("model", "query"),
}


def _edge_index(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    values = list(edges)
    if not values:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(values, dtype=torch.long).t().contiguous()


def aggregate(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Deterministic sum-by-index without CUDA atomics."""
    selector = F.one_hot(index, num_classes=size).to(values.dtype).transpose(0, 1)
    return selector @ values


@dataclass
class EgoGraphBatch:
    x_dict: Dict[str, torch.Tensor]
    edge_index_dict: Dict[str, torch.Tensor]
    query_ids: tuple[str, ...]
    prompt_count: int
    model_count: int

    def to(self, device: torch.device) -> "EgoGraphBatch":
        return EgoGraphBatch(
            {key: value.to(device) for key, value in self.x_dict.items()},
            {key: value.to(device) for key, value in self.edge_index_dict.items()},
            self.query_ids,
            self.prompt_count,
            self.model_count,
        )

    @property
    def action_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_idx, p_idx, m_idx = [], [], []
        for q in range(len(self.query_ids)):
            for p in range(self.prompt_count):
                for m in range(self.model_count):
                    q_idx.append(q)
                    p_idx.append(q * self.prompt_count + p)
                    m_idx.append(q * self.model_count + m)
        return (torch.tensor(q_idx), torch.tensor(p_idx), torch.tensor(m_idx))


def build_ego_graph_batch(
    query_ids: Sequence[str],
    query_vectors: torch.Tensor,
    task_vectors: torch.Tensor,
    prompt_vectors: torch.Tensor,
    model_vectors: torch.Tensor,
) -> EgoGraphBatch:
    """Build a disjoint ego graph for every query in the batch."""
    n = len(query_ids)
    if n == 0 or query_vectors.size(0) != n or task_vectors.size(0) != n:
        raise ValueError("query IDs, query vectors and task vectors must be non-empty and aligned")
    p_count, m_count = prompt_vectors.size(0), model_vectors.size(0)
    if p_count == 0 or m_count == 0:
        raise ValueError("prompt and model pools must be non-empty")
    qp, pq, qm, mq, qt, tq = [], [], [], [], [], []
    for q in range(n):
        qt.append((q, q)); tq.append((q, q))
        for p in range(p_count):
            local = q * p_count + p
            qp.append((q, local)); pq.append((local, q))
        for m in range(m_count):
            local = q * m_count + m
            qm.append((q, local)); mq.append((local, q))
    return EgoGraphBatch(
        x_dict={
            "query": query_vectors.float(),
            "task": task_vectors.float(),
            "prompt": prompt_vectors.float().repeat(n, 1),
            "model": model_vectors.float().repeat(n, 1),
        },
        edge_index_dict={
            "query_to_task": _edge_index(qt), "task_to_query": _edge_index(tq),
            "query_to_prompt": _edge_index(qp), "prompt_to_query": _edge_index(pq),
            "query_to_model": _edge_index(qm), "model_to_query": _edge_index(mq),
        },
        query_ids=tuple(query_ids), prompt_count=p_count, model_count=m_count,
    )


class HeteroLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_linear = nn.ModuleDict({n: nn.Linear(hidden_dim, hidden_dim) for n in NODE_TYPES})
        self.rel_linear = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim, bias=False) for r in RELATIONS})
        self.norm = nn.ModuleDict({n: nn.LayerNorm(hidden_dim) for n in NODE_TYPES})
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, edges, include_messages: bool):
        incoming = {n: torch.zeros_like(states[n]) for n in NODE_TYPES}
        if include_messages:
            for relation, (source_type, target_type) in RELATIONS.items():
                source, target = edges[relation]
                messages = self.rel_linear[relation](states[source_type][source])
                summed = aggregate(messages, target, states[target_type].size(0))
                degree = aggregate(messages.new_ones((target.numel(), 1)), target, states[target_type].size(0))
                # Degree-normalized mean is computed independently per relation.
                incoming[target_type] = incoming[target_type] + summed / degree.clamp_min(1.0)
        return {
            node_type: self.dropout(F.relu(self.norm[node_type](
                self.self_linear[node_type](states[node_type]) + incoming[node_type]
            )))
            for node_type in NODE_TYPES
        }


class MatchedEdgeRouter(nn.Module):
    """Shared scorer used by the full-MP and self-only matched arms."""
    def __init__(self, dims: Dict[str, int], hidden_dim=256, dropout=0.1, message_passing=True):
        super().__init__()
        self.message_passing = message_passing
        self.projections = nn.ModuleDict({n: nn.Linear(dims[n], hidden_dim) for n in NODE_TYPES})
        self.layers = nn.ModuleList([HeteroLayer(hidden_dim, dropout) for _ in range(2)])
        self.edge_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def compose_edges(self, initial, graph: EgoGraphBatch) -> torch.Tensor:
        """Pre-message-passing candidate representation shared by both arms."""
        _, p_idx, m_idx = (x.to(initial["query"].device) for x in graph.action_indices)
        prompt, model = initial["prompt"][p_idx], initial["model"][m_idx]
        return self.edge_projection(torch.cat((prompt, model, prompt * model), dim=-1))

    def forward(self, graph: EgoGraphBatch) -> torch.Tensor:
        initial = {n: self.projections[n](graph.x_dict[n]) for n in NODE_TYPES}
        states = initial
        for layer in self.layers:
            states = layer(states, graph.edge_index_dict, include_messages=self.message_passing)
        q_idx, p_idx, m_idx = (x.to(states["query"].device) for x in graph.action_indices)
        q, p, m = states["query"][q_idx], states["prompt"][p_idx], states["model"][m_idx]
        edge = self.compose_edges(initial, graph)
        return self.scorer(torch.cat((q, p, m, edge), dim=-1)).view(len(graph.query_ids), -1)


class FlatSharedObjectiveRouter(nn.Module):
    def __init__(self, query_dim, prompt_dim, model_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(query_dim + prompt_dim + model_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: EgoGraphBatch) -> torch.Tensor:
        q_idx, p_idx, m_idx = (x.to(graph.x_dict["query"].device) for x in graph.action_indices)
        features = torch.cat((graph.x_dict["query"][q_idx], graph.x_dict["prompt"][p_idx], graph.x_dict["model"][m_idx]), -1)
        return self.net(features).view(len(graph.query_ids), -1)


def shared_objective(scores, rewards, listwise_alpha=0.10, ce_alpha=0.05,
                     entropy_beta=0.005, temperature=0.10):
    if scores.shape != rewards.shape:
        raise ValueError("scores and rewards must have the same [queries, actions] shape")
    mse = F.mse_loss(scores, rewards)
    targets = F.softmax(rewards / max(temperature, 1e-6), dim=-1)
    kl = F.kl_div(F.log_softmax(scores, dim=-1), targets, reduction="batchmean")
    ce = F.cross_entropy(scores, rewards.argmax(dim=-1))
    usage = F.softmax(scores, dim=-1).mean(dim=0)
    entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
    return mse + listwise_alpha * kl + ce_alpha * ce - entropy_beta * entropy


def topology_metadata(query_count: int, prompt_count: int, model_count: int) -> dict:
    return {
        "topology_id": "independent_complete_query_ego_v1",
        "nodes_per_query": 2 + prompt_count + model_count,
        "directed_edges_per_query": 2 + 2 * prompt_count + 2 * model_count,
        "batch_query_nodes": query_count,
        "cross_query_message_passing": False,
        "reward_selected_topology": False,
    }
