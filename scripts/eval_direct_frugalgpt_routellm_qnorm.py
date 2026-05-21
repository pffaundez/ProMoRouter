#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LAMBDAS = {
    "reward_qnorm_lam_01": 0.1,
    "reward_qnorm_lam_05": 0.5,
    "reward_qnorm_lam_09": 0.9,
}


# -----------------------------
# Loading / preprocessing
# -----------------------------

def load_rows(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            direct = {}
            for e in r.get("action_edges", []):
                if e.get("prompt") == "direct":
                    m = e.get("model")
                    if m is not None:
                        direct[m] = e
            if direct:
                r["_direct_by_model"] = direct
                rows.append(r)
    return rows


def load_query_embeddings(path):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        # Common format: {qid: tensor}
        if all(isinstance(k, str) for k in obj.keys()):
            return {k: torch.as_tensor(v).float() for k, v in obj.items()}

        # Possible format: {"qids": [...], "embeddings": tensor}
        if "qids" in obj and "embeddings" in obj:
            qids = obj["qids"]
            embs = obj["embeddings"]
            return {qid: torch.as_tensor(embs[i]).float() for i, qid in enumerate(qids)}

        if "ids" in obj and "embeddings" in obj:
            qids = obj["ids"]
            embs = obj["embeddings"]
            return {qid: torch.as_tensor(embs[i]).float() for i, qid in enumerate(qids)}

    raise ValueError(
        f"Unsupported query embedding format in {path}. "
        "Expected dict[qid] -> embedding or {'qids': ..., 'embeddings': ...}."
    )


def split_rows(rows, seed, train_frac=0.70, val_frac=0.15):
    qids = sorted([r["qid"] for r in rows])
    rng = random.Random(seed)
    rng.shuffle(qids)

    n = len(qids)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_ids = set(qids[:n_train])
    val_ids = set(qids[n_train:n_train + n_val])
    test_ids = set(qids[n_train + n_val:])

    train = [r for r in rows if r["qid"] in train_ids]
    val = [r for r in rows if r["qid"] in val_ids]
    test = [r for r in rows if r["qid"] in test_ids]

    return train, val, test


def available_models(rows, min_coverage=0.99):
    counts = defaultdict(int)
    for r in rows:
        for m in r["_direct_by_model"]:
            counts[m] += 1

    n = len(rows)
    models = [m for m, c in counts.items() if c / n >= min_coverage]
    models = sorted(models)

    # Fallback: use any observed model if strict coverage is impossible.
    if not models:
        models = sorted(counts.keys())

    return models


def safe_mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def select_models(train_rows, min_coverage=0.99):
    models = available_models(train_rows, min_coverage=min_coverage)

    avg_cost = {}
    avg_perf = {}
    coverage = {}

    n = len(train_rows)
    for m in models:
        costs, perfs = [], []
        present = 0
        for r in train_rows:
            e = r["_direct_by_model"].get(m)
            if e is None:
                continue
            present += 1
            costs.append(float(e["cost_norm_query"]))
            perfs.append(float(e["performance"]))
        coverage[m] = present / n
        avg_cost[m] = safe_mean(costs)
        avg_perf[m] = safe_mean(perfs)

    # Remove models with no valid statistics.
    models = [m for m in models if not math.isnan(avg_cost[m]) and not math.isnan(avg_perf[m])]

    weak_model = min(models, key=lambda m: (avg_cost[m], -avg_perf[m]))
    strong_model = max(models, key=lambda m: (avg_perf[m], -avg_cost[m]))
    cost_order = sorted(models, key=lambda m: (avg_cost[m], -avg_perf[m]))

    return models, weak_model, strong_model, cost_order, avg_cost, avg_perf, coverage


# -----------------------------
# Metrics
# -----------------------------

def reward(perf, cost, lam):
    return (1.0 - lam) * perf - lam * cost


def eval_policy(rows, chooser, lam):
    Ps, Cs, Rs = [], [], []
    prompt_counts = defaultdict(int)
    model_counts = defaultdict(int)

    for r in rows:
        m = chooser(r)
        e = r["_direct_by_model"].get(m)

        # Safety fallback: cheapest available direct model for this query.
        if e is None:
            m = min(
                r["_direct_by_model"],
                key=lambda mm: r["_direct_by_model"][mm]["cost_norm_query"],
            )
            e = r["_direct_by_model"][m]

        p = float(e["performance"])
        c = float(e["cost_norm_query"])
        rr = reward(p, c, lam)

        Ps.append(p)
        Cs.append(c)
        Rs.append(rr)
        prompt_counts["direct"] += 1
        model_counts[m] += 1

    return {
        "P": float(np.mean(Ps)),
        "C": float(np.mean(Cs)),
        "R": float(np.mean(Rs)),
        "prompt_counts": dict(prompt_counts),
        "model_counts": dict(model_counts),
    }


# -----------------------------
# Small neural models
# -----------------------------

class PerfPredictor(nn.Module):
    def __init__(self, qdim, n_models, hidden=128, dropout=0.10):
        super().__init__()
        self.model_emb = nn.Embedding(n_models, 32)
        self.net = nn.Sequential(
            nn.Linear(qdim + 32, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, q, mid):
        me = self.model_emb(mid)
        x = torch.cat([q, me], dim=-1)
        return self.net(x).squeeze(-1)


class RouteClassifier(nn.Module):
    def __init__(self, qdim, hidden=128, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(qdim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, q):
        return self.net(q).squeeze(-1)


def make_q_tensor(rows, qemb):
    xs = []
    keep = []
    for r in rows:
        qid = r["qid"]
        if qid in qemb:
            xs.append(qemb[qid])
            keep.append(r)
    if not xs:
        raise ValueError("No rows have query embeddings. Check qid alignment.")
    return torch.stack(xs).float(), keep


def train_perf_predictor(train_rows, val_rows, qemb, models, epochs=200, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_to_id = {m: i for i, m in enumerate(models)}

    Xq, Xm, y = [], [], []
    for r in train_rows:
        qid = r["qid"]
        if qid not in qemb:
            continue
        for m in models:
            e = r["_direct_by_model"].get(m)
            if e is None:
                continue
            Xq.append(qemb[qid])
            Xm.append(model_to_id[m])
            y.append(float(e["performance"]))

    Xq = torch.stack(Xq).float()
    Xm = torch.tensor(Xm, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)

    qdim = Xq.shape[1]
    net = PerfPredictor(qdim, len(models))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = float("inf")
    patience = 20
    bad = 0

    # Build validation examples.
    Vq, Vm, vy = [], [], []
    for r in val_rows:
        qid = r["qid"]
        if qid not in qemb:
            continue
        for m in models:
            e = r["_direct_by_model"].get(m)
            if e is None:
                continue
            Vq.append(qemb[qid])
            Vm.append(model_to_id[m])
            vy.append(float(e["performance"]))

    if Vq:
        Vq = torch.stack(Vq).float()
        Vm = torch.tensor(Vm, dtype=torch.long)
        vy = torch.tensor(vy, dtype=torch.float32)
    else:
        Vq = Vm = vy = None

    batch_size = min(1024, len(y))
    idx_all = np.arange(len(y))

    for _ in range(epochs):
        net.train()
        np.random.shuffle(idx_all)

        for start in range(0, len(idx_all), batch_size):
            idx = torch.tensor(idx_all[start:start + batch_size], dtype=torch.long)
            pred = net(Xq[idx], Xm[idx])
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            if Vq is not None:
                val_loss = F.mse_loss(net(Vq, Vm), vy).item()
            else:
                val_loss = F.mse_loss(net(Xq, Xm), y).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    return net, model_to_id


def train_route_classifier(train_rows, val_rows, qemb, weak_model, strong_model, epochs=200, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    X, y = [], []
    for r in train_rows:
        qid = r["qid"]
        if qid not in qemb:
            continue

        ew = r["_direct_by_model"].get(weak_model)
        es = r["_direct_by_model"].get(strong_model)
        if ew is None or es is None:
            continue

        # Label: strong is better in raw performance.
        label = 1.0 if float(es["performance"]) > float(ew["performance"]) else 0.0
        X.append(qemb[qid])
        y.append(label)

    X = torch.stack(X).float()
    y = torch.tensor(y, dtype=torch.float32)

    qdim = X.shape[1]
    net = RouteClassifier(qdim)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = float("inf")
    patience = 20
    bad = 0

    VX, Vy = [], []
    for r in val_rows:
        qid = r["qid"]
        if qid not in qemb:
            continue
        ew = r["_direct_by_model"].get(weak_model)
        es = r["_direct_by_model"].get(strong_model)
        if ew is None or es is None:
            continue
        VX.append(qemb[qid])
        Vy.append(1.0 if float(es["performance"]) > float(ew["performance"]) else 0.0)

    if VX:
        VX = torch.stack(VX).float()
        Vy = torch.tensor(Vy, dtype=torch.float32)
    else:
        VX = Vy = None

    batch_size = min(256, len(y))
    idx_all = np.arange(len(y))

    for _ in range(epochs):
        net.train()
        np.random.shuffle(idx_all)

        for start in range(0, len(idx_all), batch_size):
            idx = torch.tensor(idx_all[start:start + batch_size], dtype=torch.long)
            logits = net(X[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            if VX is not None:
                val_loss = F.binary_cross_entropy_with_logits(net(VX), Vy).item()
            else:
                val_loss = F.binary_cross_entropy_with_logits(net(X), y).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    return net


# -----------------------------
# Policies
# -----------------------------

def tune_routellm_threshold(val_rows, qemb, clf, weak_model, strong_model, lam):
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t = 0.5
    best_R = -1e9

    for t in thresholds:
        def chooser(r, threshold=t):
            qid = r["qid"]
            if qid not in qemb:
                return weak_model
            with torch.no_grad():
                prob = torch.sigmoid(clf(qemb[qid].unsqueeze(0))).item()
            return strong_model if prob >= threshold else weak_model

        res = eval_policy(val_rows, chooser, lam)
        if res["R"] > best_R:
            best_R = res["R"]
            best_t = float(t)

    return best_t, best_R


def tune_frugalgpt_tau(val_rows, qemb, perf_pred, model_to_id, cost_order, lam):
    # Tau controls how much predicted quality is required before stopping
    # at a cheap model. Lower tau => cheaper. Higher tau => more fallback.
    taus = np.linspace(0.0, 1.0, 101)
    best_tau = 0.5
    best_R = -1e9

    for tau in taus:
        def chooser(r, t=tau):
            qid = r["qid"]
            available = [m for m in cost_order if m in r["_direct_by_model"]]
            if not available:
                return min(
                    r["_direct_by_model"],
                    key=lambda mm: r["_direct_by_model"][mm]["cost_norm_query"],
                )

            if qid not in qemb:
                return available[0]

            q = qemb[qid].unsqueeze(0)
            preds = {}
            with torch.no_grad():
                for m in available:
                    mid = torch.tensor([model_to_id[m]], dtype=torch.long)
                    preds[m] = perf_pred(q, mid).item()

            for m in available:
                if preds[m] >= t:
                    return m

            # Fallback to best predicted available model.
            return max(available, key=lambda mm: preds[mm])

        res = eval_policy(val_rows, chooser, lam)
        if res["R"] > best_R:
            best_R = res["R"]
            best_tau = float(tau)

    return best_tau, best_R


def make_frugalgpt_chooser(qemb, perf_pred, model_to_id, cost_order, tau):
    def chooser(r):
        qid = r["qid"]
        available = [m for m in cost_order if m in r["_direct_by_model"]]
        if not available:
            return min(
                r["_direct_by_model"],
                key=lambda mm: r["_direct_by_model"][mm]["cost_norm_query"],
            )

        if qid not in qemb:
            return available[0]

        q = qemb[qid].unsqueeze(0)
        preds = {}
        with torch.no_grad():
            for m in available:
                mid = torch.tensor([model_to_id[m]], dtype=torch.long)
                preds[m] = perf_pred(q, mid).item()

        for m in available:
            if preds[m] >= tau:
                return m

        return max(available, key=lambda mm: preds[mm])

    return chooser


def make_routellm_chooser(qemb, clf, weak_model, strong_model, threshold):
    def chooser(r):
        qid = r["qid"]
        if qid not in qemb:
            return weak_model
        with torch.no_grad():
            prob = torch.sigmoid(clf(qemb[qid].unsqueeze(0))).item()
        return strong_model if prob >= threshold else weak_model

    return chooser


# -----------------------------
# Seed run
# -----------------------------

def run_seed(args, seed, rows, qemb):
    train_rows, val_rows, test_rows = split_rows(rows, seed)

    models, weak_model, strong_model, cost_order, avg_cost, avg_perf, coverage = select_models(
        train_rows,
        min_coverage=args.min_coverage,
    )

    print("\n" + "=" * 90)
    print(f"seed={seed}")
    print(f"queries: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print("model coverage / avg direct cost / avg direct performance:")
    for m in cost_order:
        print(
            f"  {m:15s} coverage={coverage[m]:.3f} "
            f"avg_cost={avg_cost[m]:.3f} avg_perf={avg_perf[m]:.3f}"
        )
    print(f"RouteLLM weak_model={weak_model} strong_model={strong_model}")
    print(f"FrugalGPT cost_order={cost_order}")

    perf_pred, model_to_id = train_perf_predictor(
        train_rows=train_rows,
        val_rows=val_rows,
        qemb=qemb,
        models=models,
        epochs=args.epochs,
        lr=args.lr,
        seed=seed,
    )

    clf = train_route_classifier(
        train_rows=train_rows,
        val_rows=val_rows,
        qemb=qemb,
        weak_model=weak_model,
        strong_model=strong_model,
        epochs=args.epochs,
        lr=args.lr,
        seed=seed,
    )

    result = {
        "seed": seed,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "weak_model": weak_model,
        "strong_model": strong_model,
        "cost_order": cost_order,
        "model_coverage": coverage,
        "avg_cost": avg_cost,
        "avg_perf": avg_perf,
        "methods": {},
    }

    for key, lam in LAMBDAS.items():
        route_t, route_val_R = tune_routellm_threshold(
            val_rows, qemb, clf, weak_model, strong_model, lam
        )
        route_chooser = make_routellm_chooser(
            qemb, clf, weak_model, strong_model, route_t
        )
        route_res = eval_policy(test_rows, route_chooser, lam)
        route_res["threshold"] = route_t
        route_res["val_R_at_threshold"] = route_val_R

        frugal_tau, frugal_val_R = tune_frugalgpt_tau(
            val_rows, qemb, perf_pred, model_to_id, cost_order, lam
        )
        frugal_chooser = make_frugalgpt_chooser(
            qemb, perf_pred, model_to_id, cost_order, frugal_tau
        )
        frugal_res = eval_policy(test_rows, frugal_chooser, lam)
        frugal_res["tau"] = frugal_tau
        frugal_res["val_R_at_tau"] = frugal_val_R

        result["methods"].setdefault("RouteLLM-direct", {})[key] = route_res
        result["methods"].setdefault("FrugalGPT-direct", {})[key] = frugal_res

        print(f"\n{key} / lambda={lam}")
        print(
            f"  RouteLLM-direct:   "
            f"P={route_res['P']:.3f} C={route_res['C']:.3f} R={route_res['R']:.3f} "
            f"threshold={route_t:.2f} "
            f"models={route_res['model_counts']}"
        )
        print(
            f"  FrugalGPT-direct: "
            f"P={frugal_res['P']:.3f} C={frugal_res['C']:.3f} R={frugal_res['R']:.3f} "
            f"tau={frugal_tau:.2f} "
            f"models={frugal_res['model_counts']}"
        )

    return result


# -----------------------------
# Aggregation / output
# -----------------------------

def summarize(all_results):
    methods = sorted(all_results[0]["methods"].keys())

    summary = {}
    for method in methods:
        summary[method] = {}
        for key in LAMBDAS:
            vals = []
            for r in all_results:
                vals.append(r["methods"][method][key])
            summary[method][key] = {
                "P_mean": float(np.mean([v["P"] for v in vals])),
                "P_std": float(np.std([v["P"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
                "C_mean": float(np.mean([v["C"] for v in vals])),
                "C_std": float(np.std([v["C"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
                "R_mean": float(np.mean([v["R"] for v in vals])),
                "R_std": float(np.std([v["R"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
            }
    return summary


def print_overleaf_rows(summary):
    print("\n" + "=" * 90)
    print("OVERLEAF ROWS, mean only")
    for method, vals in summary.items():
        row = [method]
        for key in ["reward_qnorm_lam_01", "reward_qnorm_lam_05", "reward_qnorm_lam_09"]:
            row.extend([
                f"{vals[key]['P_mean']:.3f}",
                f"{vals[key]['C_mean']:.3f}",
                f"{vals[key]['R_mean']:.3f}",
            ])
        print(" & ".join(row) + r" \\")

    print("\n" + "=" * 90)
    print("OVERLEAF ROWS, mean ± std")
    for method, vals in summary.items():
        row = [method]
        for key in ["reward_qnorm_lam_01", "reward_qnorm_lam_05", "reward_qnorm_lam_09"]:
            row.extend([
                f"{vals[key]['P_mean']:.3f} $\\pm$ {vals[key]['P_std']:.3f}",
                f"{vals[key]['C_mean']:.3f} $\\pm$ {vals[key]['C_std']:.3f}",
                f"{vals[key]['R_mean']:.3f} $\\pm$ {vals[key]['R_std']:.3f}",
            ])
        print(" & ".join(row) + r" \\")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl",
    )
    parser.add_argument(
        "--query_embeddings",
        default="data/router/query_embeddings.pt",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/direct_router_baselines_qnorm",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--min_coverage", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("==== DIRECT ROUTER BASELINES QNORM ====")
    print(f"Data: {args.data}")
    print(f"Query embeddings: {args.query_embeddings}")
    print(f"Seeds: {args.seeds}")
    print(f"Outdir: {outdir}")

    rows = load_rows(args.data)
    qemb = load_query_embeddings(args.query_embeddings)

    print(f"Loaded rows with direct actions: {len(rows)}")
    print(f"Loaded query embeddings: {len(qemb)}")

    all_results = []
    for seed in args.seeds:
        res = run_seed(args, seed, rows, qemb)
        all_results.append(res)

        seed_path = outdir / f"direct_router_baselines_qnorm_results_seed{seed}.json"
        seed_path.write_text(json.dumps(res, indent=2))
        print(f"\nSaved seed results: {seed_path}")

    summary = summarize(all_results)

    summary_path = outdir / "direct_router_baselines_qnorm_summary.json"
    summary_path.write_text(json.dumps({
        "seeds": args.seeds,
        "summary": summary,
        "all_results": all_results,
    }, indent=2))

    print(f"\nSaved summary: {summary_path}")
    print_overleaf_rows(summary)


if __name__ == "__main__":
    main()
