"""Evaluate ProMoRouter v2 and its controls on a shared closed-pool split."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from analysis.audit_rq1_cost import audit
from audit_graphrouter_joint_fixed import paired
from router import graphrouter_joint as joint
from router.data_validation import require_complete_action_space
from router.splits import load_or_create_splits
from run_unseen_6x3 import fit_graph, mlp_epoch, mlp_features, mlp_score
from train_graphrouter_joint_unseen import summarize
import train_graphrouter_unseen as direct
from train_router_flat_mlp_qnorm import FlatActionMLP
from train_router_support_graph import MODELS, PROMPTS, load_table

LAMBDAS = (0.1, 0.5, 0.9)
PATIENCE = 18
MIN_DELTA = 1e-6
DIRECT = PROMPTS.index("direct")
LOG_DIR = Path("interaction_logs/grpp_il_v1")


def rewards(values, lam):
    return values[..., 0] - lam * values[..., 1]


def validation_reward(scores, values, lam):
    choices = scores.flatten(1).argmax(1, keepdim=True)
    return float(rewards(values, lam).flatten(1).gather(1, choices).mean())


def fit_flat(train_x, train_r, validation, epochs, seed, device):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    net = FlatActionMLP(train_x.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0005, weight_decay=0.0001)
    best, stale = -float("inf"), 0
    for epoch in range(1, epochs + 1):
        mlp_epoch(net, opt, train_x, train_r, rng)
        metric = validation(net)
        if metric > best + MIN_DELTA:
            best, best_epoch, stale = metric, epoch, 0
            state = deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    net.load_state_dict(state)
    return net.eval().requires_grad_(False), best_epoch


def cost_bounds(rows):
    costs = [float(edge["cost_proxy_money"]) for row in rows for edge in row["action_edges"]]
    if not costs or any(not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError("Invalid training costs")
    lo, hi = min(costs), max(costs)
    if lo == hi:
        raise ValueError("Degenerate training cost reference")
    return lo, hi


def fit_and_score(fit, val, test, fit_values, val_values, tables, lam, args):
    device = torch.device(args.device)
    direct_tables = tables[0], tables[1], tables[3]
    fit_mask = torch.ones(len(fit), dtype=torch.bool)
    joint_edges = joint.make_visible(fit, PROMPTS, MODELS, fit_values, lam)
    direct_edges = direct.make_visible(fit, MODELS, fit_values[:, DIRECT], lam)

    def score_joint(net, rows):
        return joint.evaluate_scores(net, fit, rows, PROMPTS, MODELS, tables, joint_edges, device)

    def score_direct(net, rows):
        return direct.evaluate_scores(net, fit, rows, MODELS, direct_tables, direct_edges, device)

    graph_epochs = 1 if args.smoke else 1000
    config = dict(lam=lam, seed=args.seed, patience=PATIENCE, min_delta=MIN_DELTA,
                  metric_name="reward")
    torch.manual_seed(args.seed)
    net, graph_epoch = fit_graph(
        joint.graph_inputs(fit, [], PROMPTS, MODELS, tables, joint_edges, device),
        fit_values, fit_mask, lambda net: validation_reward(score_joint(net, val), val_values, lam),
        graph_epochs, device, **config,
    )
    scores = {"joint_gnn": score_joint(net, test)}
    torch.manual_seed(args.seed)
    net, direct_epoch = fit_graph(
        direct.graph_inputs(fit, [], MODELS, direct_tables, direct_edges, device),
        fit_values[:, DIRECT:DIRECT+1], fit_mask,
        lambda net: validation_reward(score_direct(net, val), val_values[:, DIRECT:DIRECT+1], lam),
        graph_epochs, device, **config,
    )
    scores["graphrouter_direct"] = score_direct(net, test)
    val_x = mlp_features(val, MODELS, tables, device)
    net, flat_epoch = fit_flat(
        mlp_features(fit, MODELS, tables, device), rewards(fit_values, lam).to(device),
        lambda net: validation_reward(mlp_score(net, val_x), val_values, lam),
        1 if args.smoke else 80, args.seed, device,
    )
    scores["flat_mlp"] = mlp_score(net, mlp_features(test, MODELS, tables, device))
    return scores, dict(joint_gnn=graph_epoch, graphrouter_direct=direct_epoch, flat_mlp=flat_epoch)


def evaluate(test, truth, val_values, scores, lo, span, lam, seed):
    results = {}
    direct_methods = {"graphrouter_direct", "fixed_model_direct", "oracle_direct",
                      "largest_llm_direct", "smallest_llm_direct"}
    val_scores = rewards(val_values, lam).mean(0)
    fixed = {
        "fixed_pair": (val_scores, int(val_scores.flatten().argmax())),
        "fixed_model_direct": (val_scores[DIRECT], int(val_scores[DIRECT].argmax())),
        "largest_llm_direct": (val_scores[DIRECT], MODELS.index("qwen2.5-72b")),
        "smallest_llm_direct": (val_scores[DIRECT], MODELS.index("mistral-7b")),
    }
    for name, (shape, action) in fixed.items():
        values = torch.zeros((len(test), *shape.shape))
        values.flatten(1)[:, action] = 1
        scores[name] = values
    scores["oracle_joint"] = rewards(truth, lam)
    scores["oracle_direct"] = scores["oracle_joint"][:, DIRECT]
    for name, values in scores.items():
        if not torch.isfinite(values).all():
            raise ValueError(f"Nonfinite scores for {name}")
        if name in direct_methods:
            summary = direct.summarize(test, values, truth[:, DIRECT], MODELS, lam)
        else:
            summary = summarize(test, values, truth, MODELS, lam)
        summary["mean_weighted_token_proxy"] = lo + span * summary["C"]
        results[name] = summary
    comparisons = {name: paired(results["joint_gnn"], summary, seed)
                   for name, summary in results.items() if name != "joint_gnn"}
    return results, comparisons


def run(args):
    data_path = args.data_root / LOG_DIR / "router_bipartite_qnorm_complete.jsonl"
    raw = data_path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    qids = {row["qid"] for row in rows}
    if len(qids) != len(rows):
        raise ValueError("Duplicate query IDs")
    require_complete_action_space(rows, PROMPTS, MODELS)
    split = load_or_create_splits(qids, args.split_manifest, args.seed)
    fit, val, test = ([row for row in rows if row["qid"] in ids] for ids in split)
    cost_audit = None
    if args.raw_log or not args.smoke:
        cost_audit = audit(args.raw_log or args.data_root / LOG_DIR / "train_clean.jsonl", data_path)
    lo, hi = cost_bounds(fit)
    span = hi - lo
    fit_values = joint.action_data(fit, PROMPTS, MODELS, lo, span)
    val_values = joint.action_data(val, PROMPTS, MODELS, lo, span)
    tables = tuple(load_table(args.data_root / "router" / f"{name}_embeddings.pt")
                   for name in ("query", "task", "prompt", "model"))
    # Compute and retain every learned decision before reading test outcomes.
    predictions = {lam: fit_and_score(fit, val, test, fit_values, val_values, tables, lam, args)
                   for lam in LAMBDAS}
    truth = joint.action_data(test, PROMPTS, MODELS, lo, span)
    regimes = {}
    for lam, (scores, best_epochs) in predictions.items():
        candidate_scores = {name: values.tolist() for name, values in scores.items()}
        methods, comparisons = evaluate(test, truth, val_values, scores, lo, span, lam, args.seed)
        regimes[str(lam)] = dict(methods=methods, comparisons=comparisons,
                                 best_epochs=best_epochs, learned_candidate_scores=candidate_scores)
    return dict(protocol="rq1_closed_pool_v2", seed=args.seed, device=args.device, smoke=args.smoke,
                dataset_sha256=hashlib.sha256(raw).hexdigest(),
                split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
                train_qids=sorted(split[0]), validation_qids=sorted(split[1]),
                test_qids=[row["qid"] for row in test], model_order=MODELS, prompt_order=PROMPTS,
                cost_reference=dict(field="cost_proxy_money", min=lo, max=hi, clipped=False),
                cost_audit=cost_audit, patience=PATIENCE, min_delta=MIN_DELTA, regimes=regimes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.split_manifest.is_file():
        parser.error("an existing split manifest and a new output path are required")
    if args.device != "cuda:0" and not args.smoke:
        parser.error("benchmark requires GPU 0")
    if args.device == "cuda:0" and (os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
                                    or not torch.cuda.is_available()):
        parser.error("CUDA_VISIBLE_DEVICES=0 and a working GPU are required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
