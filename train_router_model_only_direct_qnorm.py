#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train/evaluate GraphRouter-direct under query-normalized cost.

This script implements a model-only router with a fixed `direct` prompting policy.
It reads the prompt-model interaction log, filters action_edges to prompt == "direct",
and trains a model-only scorer over (query, model) candidates.

Expected input:
  data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl

Expected embeddings:
  data/router/query_embeddings.pt
  data/router/model_embeddings.pt

Outputs:
  outputs/router_model_only_direct_qnorm/
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean, stdev
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")
ROUTER_DATA_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only_direct_qnorm.jsonl")

QUERY_EMB_PATH = Path("data/router/query_embeddings.pt")
MODEL_EMB_PATH = Path("data/router/model_embeddings.pt")

OUTPUT_DIR = Path("outputs/router_model_only_direct_qnorm")

LAMBDAS = {
    "reward_qnorm_lam_01": 0.1,
    "reward_qnorm_lam_05": 0.5,
    "reward_qnorm_lam_09": 0.9,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def build_direct_model_only_dataset(
    source_path: Path = SOURCE_DATA_PATH,
    out_path: Path = ROUTER_DATA_PATH,
):
    """
    Build model-only direct dataset.

    One row per query.
    Each candidate corresponds to one model under prompt == "direct".
    """
    source_rows = load_jsonl(source_path)
    out_rows = []

    skipped_no_direct = 0
    model_counter = Counter()

    for qidx, row in enumerate(source_rows):
        qid = row["qid"]
        task = row.get("task")
        query_text = row.get("query_text", "")

        candidates = []
        seen_models = set()

        for e in row.get("action_edges", []):
            if e.get("prompt") != "direct":
                continue

            model = e.get("model")
            if model is None:
                continue

            # Avoid duplicates if any exist.
            if model in seen_models:
                continue
            seen_models.add(model)

            cand = {
                "model": model,
                "prompt": "direct",
                "performance": safe_float(e.get("performance")),
                "cost_norm_query": safe_float(e.get("cost_norm_query")),
                "cost_proxy_money": safe_float(e.get("cost_proxy_money")),
                "tokens_total": safe_float(e.get("tokens_total")),
            }

            for reward_key, lam in LAMBDAS.items():
                if reward_key in e:
                    cand[reward_key] = safe_float(e[reward_key])
                else:
                    cand[reward_key] = cand["performance"] - lam * cand["cost_norm_query"]

            candidates.append(cand)
            model_counter[model] += 1

        if not candidates:
            skipped_no_direct += 1
            continue

        oracle = {}
        for reward_key in LAMBDAS:
            best = max(candidates, key=lambda c: c[reward_key])
            oracle[reward_key] = {
                "model": best["model"],
                "prompt": "direct",
                "performance": best["performance"],
                "cost_norm_query": best["cost_norm_query"],
                "reward": best[reward_key],
            }

        out_rows.append(
            {
                "qid": qid,
                "qidx": qidx,
                "task": task,
                "query_text": query_text,
                "prompt_policy": "direct",
                "num_models": len(candidates),
                "model_candidates": [c["model"] for c in candidates],
                "candidates": candidates,
                "oracle_model_only_direct": oracle,
            }
        )

    write_jsonl(out_path, out_rows)

    print("\n==== BUILT DIRECT MODEL-ONLY DATASET ====")
    print(f"source: {source_path}")
    print(f"saved:  {out_path}")
    print(f"queries: {len(out_rows)}")
    print(f"skipped_no_direct: {skipped_no_direct}")
    print("model coverage:")
    for m, n in model_counter.most_common():
        print(f"  {m:16s} {n}")

    return out_rows


def load_or_build_direct_dataset(args):
    if args.rebuild_direct_dataset or not args.router_data.exists():
        return build_direct_model_only_dataset(args.source_data, args.router_data)
    return load_jsonl(args.router_data)


def load_embeddings(path: Path):
    obj = torch.load(path, map_location="cpu")

    # Case 1: already a tensor
    if isinstance(obj, torch.Tensor):
        return obj

    # Case 2: common saved formats with embeddings inside a dict
    if isinstance(obj, dict):
        for key in ["embeddings", "embedding", "x", "tensor", "values"]:
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]

        # Case 3: dict mapping ids/names -> tensors
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            keys = sorted(obj.keys())
            return torch.stack([obj[k] for k in keys], dim=0)

        # Debug-friendly error
        print(f"\nUnsupported embedding dict format at {path}")
        print(f"Available keys: {list(obj.keys())[:20]}")
        for k, v in list(obj.items())[:5]:
            print(f"  {k}: type={type(v)}")
        raise ValueError(f"Could not extract tensor embeddings from {path}")

    raise ValueError(f"Unsupported embedding format at {path}: type={type(obj)}")

def _to_1d_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().view(-1)
    return torch.tensor(x, dtype=torch.float32).view(-1)


def build_query_embedding_table(rows, raw_query_embeddings):
    """
    Builds a query embedding matrix aligned with the qids in rows.

    Returns:
        query_x: Tensor [num_queries, dim]
        qid_to_idx: dict qid -> row index
    """
    qids = [r["qid"] for r in rows]
    qid_to_idx = {qid: i for i, qid in enumerate(qids)}

    if torch.is_tensor(raw_query_embeddings):
        if raw_query_embeddings.shape[0] != len(rows):
            raise ValueError(
                f"Query embedding tensor has {raw_query_embeddings.shape[0]} rows, "
                f"but dataset has {len(rows)} queries. Need dict-style embeddings or aligned tensor."
            )
        return raw_query_embeddings.float(), qid_to_idx

    if isinstance(raw_query_embeddings, dict):
        missing = [qid for qid in qids if qid not in raw_query_embeddings]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} qids in query embeddings. "
                f"First missing examples: {missing[:5]}"
            )

        query_x = torch.stack([_to_1d_tensor(raw_query_embeddings[qid]) for qid in qids], dim=0)
        return query_x.float(), qid_to_idx

    raise ValueError(f"Unsupported query embeddings type: {type(raw_query_embeddings)}")


def build_model_embedding_table(models, raw_model_embeddings):
    """
    Builds a model embedding matrix aligned with the model list.
    Supports either a tensor already aligned with `models`, or a dict keyed by model name.
    """
    if torch.is_tensor(raw_model_embeddings):
        if raw_model_embeddings.shape[0] != len(models):
            raise ValueError(
                f"Model embedding tensor has {raw_model_embeddings.shape[0]} rows, "
                f"but there are {len(models)} models."
            )
        return raw_model_embeddings.float()

    if isinstance(raw_model_embeddings, dict):
        # Case: {"embeddings": tensor} was already handled in load_embeddings,
        # so here we assume model-name -> embedding.
        missing = [m for m in models if m not in raw_model_embeddings]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} models in model embeddings. "
                f"Missing examples: {missing[:5]}"
            )

        model_x = torch.stack([_to_1d_tensor(raw_model_embeddings[m]) for m in models], dim=0)
        return model_x.float()

    raise ValueError(f"Unsupported model embeddings type: {type(raw_model_embeddings)}")

def build_model_vocab(rows):
    models = sorted({c["model"] for r in rows for c in r["candidates"]})
    return models, {m: i for i, m in enumerate(models)}


def split_rows(rows, seed: int, train_frac=0.70, val_frac=0.15):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    return train, val, test


class DirectModelOnlyRouter(nn.Module):
    def __init__(
        self,
        query_dim: int,
        model_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()

        in_dim = query_dim + model_dim + 4

        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_emb, model_emb, scalar_feats):
        x = torch.cat([query_emb, model_emb, scalar_feats], dim=-1)
        return self.scorer(x).squeeze(-1)


def make_query_batch(
    row,
    query_embeddings,
    model_embeddings,
    model_to_idx,
    reward_key,
    device,
):
    qidx = int(row["qidx"])
    q_emb = query_embeddings[qidx].to(device)

    q_list = []
    m_list = []
    feat_list = []
    y_reward = []
    y_perf = []
    y_cost = []
    model_names = []

    for cand in row["candidates"]:
        model = cand["model"]
        if model not in model_to_idx:
            continue

        midx = model_to_idx[model]
        m_emb = model_embeddings[midx].to(device)

        perf = safe_float(cand["performance"])
        cost = safe_float(cand["cost_norm_query"])
        reward = safe_float(cand[reward_key])
        tokens = safe_float(cand.get("tokens_total", 0.0))
        money = safe_float(cand.get("cost_proxy_money", 0.0))

        # Light scalar features. Cost/perf are available in offline training,
        # but at decision time the router uses learned scores over candidates;
        # these features are candidate metadata from logs.
        scalar = torch.tensor(
            [
                cost,
                np.log1p(tokens),
                np.log1p(money),
                1.0,
            ],
            dtype=torch.float32,
            device=device,
        )

        q_list.append(q_emb)
        m_list.append(m_emb)
        feat_list.append(scalar)
        y_reward.append(reward)
        y_perf.append(perf)
        y_cost.append(cost)
        model_names.append(model)

    if not q_list:
        return None

    return {
        "q": torch.stack(q_list, dim=0),
        "m": torch.stack(m_list, dim=0),
        "feat": torch.stack(feat_list, dim=0),
        "reward": torch.tensor(y_reward, dtype=torch.float32, device=device),
        "perf": torch.tensor(y_perf, dtype=torch.float32, device=device),
        "cost": torch.tensor(y_cost, dtype=torch.float32, device=device),
        "models": model_names,
    }


def train_one(
    args,
    rows_train,
    rows_val,
    query_embeddings,
    model_embeddings,
    model_to_idx,
    reward_key,
    device,
):
    model = DirectModelOnlyRouter(
        query_dim=query_embeddings.shape[1],
        model_dim=model_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_R = -1e9
    bad_epochs = 0

    print(f"Loss: MSE + {args.ce_weight}*CE - {args.entropy_weight}*entropy | temperature={args.temperature}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(rows_train)

        losses = []

        for row in rows_train:
            batch = make_query_batch(
                row,
                query_embeddings,
                model_embeddings,
                model_to_idx,
                reward_key,
                device,
            )
            if batch is None:
                continue

            scores = model(batch["q"], batch["m"], batch["feat"])
            rewards = batch["reward"]

            # Regression target.
            mse = F.mse_loss(scores, rewards)

            # Classification target: best candidate under reward.
            target = torch.argmax(rewards).view(1)
            ce = F.cross_entropy((scores / args.temperature).view(1, -1), target)

            probs = F.softmax(scores / args.temperature, dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()

            loss = mse + args.ce_weight * ce - args.entropy_weight * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(
            model,
            rows_val,
            query_embeddings,
            model_embeddings,
            model_to_idx,
            reward_key,
            device,
        )

        train_loss = mean(losses) if losses else float("nan")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_R={val_metrics['R']:.6f} | "
            f"val_P={val_metrics['P']:.6f} | "
            f"val_C={val_metrics['C']:.6f}"
        )

        if val_metrics["R"] > best_val_R + args.min_delta:
            best_val_R = val_metrics["R"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def evaluate(
    model,
    rows,
    query_embeddings,
    model_embeddings,
    model_to_idx,
    reward_key,
    device,
):
    model.eval()

    Ps, Cs, Rs = [], [], []
    model_counts = Counter()

    for row in rows:
        batch = make_query_batch(
            row,
            query_embeddings,
            model_embeddings,
            model_to_idx,
            reward_key,
            device,
        )
        if batch is None:
            continue

        scores = model(batch["q"], batch["m"], batch["feat"])
        idx = int(torch.argmax(scores).item())

        p = float(batch["perf"][idx].detach().cpu())
        c = float(batch["cost"][idx].detach().cpu())

        lam = LAMBDAS[reward_key]
        r = p - lam * c

        Ps.append(p)
        Cs.append(c)
        Rs.append(r)
        model_counts[batch["models"][idx]] += 1

    return {
        "queries": len(Ps),
        "P": mean(Ps) if Ps else float("nan"),
        "C": mean(Cs) if Cs else float("nan"),
        "R": mean(Rs) if Rs else float("nan"),
        "model_counts": dict(model_counts),
    }


def oracle_direct(rows, reward_key):
    Ps, Cs, Rs = [], [], []
    model_counts = Counter()

    for row in rows:
        best = max(row["candidates"], key=lambda c: c[reward_key])
        p = safe_float(best["performance"])
        c = safe_float(best["cost_norm_query"])
        r = safe_float(best[reward_key])

        Ps.append(p)
        Cs.append(c)
        Rs.append(r)
        model_counts[best["model"]] += 1

    return {
        "queries": len(Ps),
        "P": mean(Ps) if Ps else float("nan"),
        "C": mean(Cs) if Cs else float("nan"),
        "R": mean(Rs) if Rs else float("nan"),
        "model_counts": dict(model_counts),
    }

def unwrap_embedding_tensor(obj, name: str):
    """
    Converts embedding files loaded as either Tensor or dict into a Tensor.

    Some embedding files are saved directly as torch.Tensor.
    Others are saved as dictionaries containing the tensor under keys such as:
    'embeddings', 'query_embeddings', 'model_embeddings', etc.
    """
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        candidate_keys = [
            "embeddings",
            "embedding",
            "query_embeddings",
            "model_embeddings",
            "task_embeddings",
            "prompt_embeddings",
        ]

        for k in candidate_keys:
            if k in obj and torch.is_tensor(obj[k]):
                print(f"Loaded {name} tensor from dict key: {k}")
                return obj[k]

        # Fallback: find the first tensor with at least 2 dimensions
        for k, v in obj.items():
            if torch.is_tensor(v) and v.ndim >= 2:
                print(f"Loaded {name} tensor from dict key: {k}")
                return v

        raise ValueError(
            f"Could not find a tensor inside {name} embedding dict. "
            f"Available keys: {list(obj.keys())}"
        )

    raise TypeError(
        f"Unsupported {name} embedding object type: {type(obj)}"
    )


def run_seed(args, seed: int):
    set_seed(seed)

    rows = load_or_build_direct_dataset(args)

    raw_query_embeddings = load_embeddings(args.query_embeddings)
    raw_model_embeddings = load_embeddings(args.model_embeddings)

    query_embeddings = unwrap_embedding_tensor(raw_query_embeddings, "query")
    model_embeddings = unwrap_embedding_tensor(raw_model_embeddings, "model")

    models, model_to_idx = build_model_vocab(rows)

    def _unwrap_tensor_container(obj, preferred_keys):
        """
        Supports common torch.save formats:
        - Tensor
        - {"embeddings": Tensor}
        - {"query_embeddings": Tensor}
        - {"model_embeddings": Tensor}
        - {"qid": Tensor, ...}
        - {"model_name": Tensor, ...}
        """
        if torch.is_tensor(obj):
            return obj

        if isinstance(obj, dict):
            for k in preferred_keys:
                if k in obj and torch.is_tensor(obj[k]):
                    return obj[k]

            # Some files store a single tensor under an arbitrary key.
            tensor_values = [v for v in obj.values() if torch.is_tensor(v)]
            if len(tensor_values) == 1:
                return tensor_values[0]

        return obj

    def _to_float_embeddings(obj):
        """
        Converts tensors to float. If obj is a dict of embeddings, converts each value.
        """
        if torch.is_tensor(obj):
            return obj.float()

        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if torch.is_tensor(v):
                    out[k] = v.float()
                elif isinstance(v, (list, tuple)):
                    out[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    out[k] = v
            return out

        raise ValueError(f"Unsupported embedding object type: {type(obj)}")

    query_embeddings = _unwrap_tensor_container(
        raw_query_embeddings,
        preferred_keys=[
            "query_embeddings",
            "embeddings",
            "x",
            "tensor",
        ],
    )

    model_embeddings = _unwrap_tensor_container(
        raw_model_embeddings,
        preferred_keys=[
            "model_embeddings",
            "embeddings",
            "x",
            "tensor",
        ],
    )

    query_embeddings = _to_float_embeddings(query_embeddings)
    model_embeddings = _to_float_embeddings(model_embeddings)

    # If model embeddings are stored as a dict keyed by model name, stack them
    # in the exact model order used by this dataset.
    if isinstance(model_embeddings, dict):
        missing = [m for m in models if m not in model_embeddings]
        if missing:
            raise KeyError(
                "Model embeddings are stored as a dict, but these models are missing: "
                + ", ".join(missing)
            )
        model_embeddings = torch.stack(
            [
                model_embeddings[m]
                if torch.is_tensor(model_embeddings[m])
                else torch.tensor(model_embeddings[m], dtype=torch.float32)
                for m in models
            ],
            dim=0,
        ).float()

    if not torch.is_tensor(model_embeddings):
        raise ValueError(
            f"model_embeddings must be a tensor or dict of model->embedding, got {type(model_embeddings)}"
        )

    if len(models) != model_embeddings.shape[0]:
        print("\nWARNING:")
        print(f"  dataset models: {len(models)}")
        print(f"  model embeddings: {model_embeddings.shape[0]}")
        print("  Assuming sorted dataset model order matches embedding order only if dimensions agree.")
        print("  If this warning appears, verify model embedding metadata.")

    rows_train, rows_val, rows_test = split_rows(
        rows,
        seed=seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print("\n==== DIRECT MODEL-ONLY SPLIT ====")
    print(f"seed={seed}")
    print(f"queries: train={len(rows_train)} val={len(rows_val)} test={len(rows_test)}")
    print(f"models={models}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    seed_results = {
        "seed": seed,
        "method": "GraphRouter-direct",
        "prompt_policy": "direct",
        "train_queries": len(rows_train),
        "val_queries": len(rows_val),
        "test_queries": len(rows_test),
        "models": models,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)

    for reward_key in LAMBDAS:
        lam = LAMBDAS[reward_key]
        print(f"\n==== TRAINING GraphRouter-direct {reward_key} / lambda={lam} ====")

        router = train_one(
            args,
            rows_train,
            rows_val,
            query_embeddings,
            model_embeddings,
            model_to_idx,
            reward_key,
            device,
        )

        test_metrics = evaluate(
            router,
            rows_test,
            query_embeddings,
            model_embeddings,
            model_to_idx,
            reward_key,
            device,
        )

        oracle_metrics = oracle_direct(rows_test, reward_key)

        model_path = args.outdir / f"router_model_only_direct_{reward_key}_seed{seed}.pt"
        torch.save(
            {
                "state_dict": router.state_dict(),
                "seed": seed,
                "reward_key": reward_key,
                "lambda": lam,
                "models": models,
                "prompt_policy": "direct",
                "args": vars(args),
            },
            model_path,
        )

        print("\n==== TEST RESULTS ====")
        print(f"queries={test_metrics['queries']}")
        print(
            f"P={test_metrics['P']:.6f} "
            f"C={test_metrics['C']:.6f} "
            f"R={test_metrics['R']:.6f}"
        )
        print(f"model_counts={test_metrics['model_counts']}")
        print(f"oracle_direct={oracle_metrics}")
        print(f"saved_model={model_path}")

        seed_results[reward_key] = test_metrics
        seed_results[f"{reward_key}_oracle_direct"] = oracle_metrics

    out_path = args.outdir / f"router_model_only_direct_qnorm_results_seed{seed}.json"
    out_path.write_text(json.dumps(seed_results, indent=2), encoding="utf-8")
    print(f"\nSaved results: {out_path}")

    print("\n==== OVERLEAF ROW ====")
    print(format_overleaf_row("GraphRouter-direct (qnorm)", seed_results, mean_only=True))

    return seed_results


def summarize_all(results, outdir: Path):
    if not results:
        return

    summary = {
        "method": "GraphRouter-direct",
        "prompt_policy": "direct",
        "seeds": [r["seed"] for r in results],
        "by_lambda": {},
    }

    print("\n" + "=" * 90)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 90)

    for reward_key in LAMBDAS:
        Ps = [r[reward_key]["P"] for r in results]
        Cs = [r[reward_key]["C"] for r in results]
        Rs = [r[reward_key]["R"] for r in results]

        def sd(xs):
            return stdev(xs) if len(xs) > 1 else 0.0

        summary["by_lambda"][reward_key] = {
            "P_mean": mean(Ps),
            "P_std": sd(Ps),
            "C_mean": mean(Cs),
            "C_std": sd(Cs),
            "R_mean": mean(Rs),
            "R_std": sd(Rs),
        }

        print(
            f"{reward_key}: "
            f"P={mean(Ps):.3f}±{sd(Ps):.3f} "
            f"C={mean(Cs):.3f}±{sd(Cs):.3f} "
            f"R={mean(Rs):.3f}±{sd(Rs):.3f}"
        )

    out_path = outdir / "router_model_only_direct_qnorm_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n==== OVERLEAF ROWS ====")
    print(format_overleaf_row("GraphRouter-direct (qnorm)", summary, mean_only=True))
    print(format_overleaf_row("GraphRouter-direct (qnorm)", summary, mean_only=False))
    print(f"\nSaved summary: {out_path}")


def format_overleaf_row(method_name, obj, mean_only=True):
    vals = []

    # Single-seed result.
    if "by_lambda" not in obj:
        for reward_key in ["reward_qnorm_lam_01", "reward_qnorm_lam_05", "reward_qnorm_lam_09"]:
            r = obj[reward_key]
            vals.extend([r["P"], r["C"], r["R"]])
        return (
            method_name
            + " & "
            + " & ".join(f"{v:.3f}" for v in vals)
            + r" \\"
        )

    # Multi-seed summary.
    for reward_key in ["reward_qnorm_lam_01", "reward_qnorm_lam_05", "reward_qnorm_lam_09"]:
        s = obj["by_lambda"][reward_key]
        triples = [
            (s["P_mean"], s["P_std"]),
            (s["C_mean"], s["C_std"]),
            (s["R_mean"], s["R_std"]),
        ]
        for mu, sig in triples:
            if mean_only:
                vals.append(f"{mu:.3f}")
            else:
                vals.append(f"{mu:.3f} $\\pm$ {sig:.3f}")

    return method_name + " & " + " & ".join(vals) + r" \\"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA_PATH)
    parser.add_argument("--router-data", type=Path, default=ROUTER_DATA_PATH)
    parser.add_argument("--query-embeddings", type=Path, default=QUERY_EMB_PATH)
    parser.add_argument("--model-embeddings", type=Path, default=MODEL_EMB_PATH)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)

    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rebuild-direct-dataset", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ce-weight", type=float, default=0.10)
    parser.add_argument("--entropy-weight", type=float, default=0.002)

    return parser.parse_args()


def main():
    args = parse_args()

    print("==== TRAIN GRAPHROUTER-DIRECT MODEL-ONLY QNORM ====")
    print(f"Source data: {args.source_data}")
    print(f"Router data: {args.router_data}")
    print(f"Query embeddings: {args.query_embeddings}")
    print(f"Model embeddings: {args.model_embeddings}")
    print(f"Output dir: {args.outdir}")
    print(f"Seeds: {args.seeds}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        res = run_seed(args, seed)
        results.append(res)

    summarize_all(results, args.outdir)


if __name__ == "__main__":
    main()