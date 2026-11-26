"""Shared 6-to-9 held-out-model comparison on the ProMoRouter benchmark."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F

from audit_graphrouter_joint_fixed import paired, reward
from router import graphrouter_joint as joint
from router.data_validation import require_complete_action_space
from router.graphrouter_official import EncoderDecoderNet
from router.splits import load_or_create_splits
from train_graphrouter_joint_episodic import state_hash
from train_graphrouter_joint_unseen import summarize
import train_graphrouter_unseen as direct
from train_router_flat_mlp_qnorm import FlatActionMLP
from train_router_support_graph import MODELS, PROMPTS, load_table, select_support, vector

FOLDS = {
    "A": ("qwen2.5-14b", "mistral-7b", "llama3.1-70b"),
    "B": ("qwen2.5-7b", "yi-34b", "mixtral-8x7b"),
    "C": ("llama3.1-8b", "codellama-34b", "qwen2.5-72b"),
}
SEED = 1
LAM = 0.9
SUPPORT_SIZE = 8
MLP_UPDATE_EPOCHS = 1


def fit_graph(inputs, values, fit_mask, validate, epochs, device, *,
              lam=LAM, seed=SEED, patience=None, min_delta=0.0,
              metric_name="macro F1"):
    task, query, models, src, dst, attrs = inputs
    edges = torch.tensor([src, dst], dtype=torch.long, device=device)
    attributes = torch.tensor(attrs, dtype=torch.float32, device=device)
    _, labels = joint.best_action_labels(values, lam)
    labels = labels.to(device)
    actions = values.shape[1] * values.shape[2]
    net = EncoderDecoderNet(query.shape[1], models.shape[1], 8, 3).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.0001)
    rng = torch.Generator().manual_seed(seed)
    best_metric, stale = -float("inf"), 0
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(32):
            selected = fit_mask & (torch.rand(len(fit_mask), generator=rng) >= 0.5)
            if not selected.any():
                selected[fit_mask.nonzero()[0]] = True
            predict = selected.repeat_interleave(actions).to(device)
            scores = net(task, query, models, edges, edge_mask=predict,
                         edge_can_see=~predict, edge_weight=attributes)
            (F.binary_cross_entropy(scores.flatten(), labels[predict]) / 32).backward()
        optimizer.step()
        metric = validate(net)
        if metric > best_metric + min_delta:
            best_metric, best_epoch, stale = metric, epoch, 0
            state = deepcopy(net.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch}, validation {metric_name} {metric:.5f}", flush=True)
        if patience is not None and stale >= patience:
            break
    net.load_state_dict(state)
    return net.eval().requires_grad_(False), best_epoch


def mlp_features(rows, pool, tables, device):
    qt, _, pt, mt = tables
    q = torch.stack([vector(qt, row["qid"], ()) for row in rows])
    p = torch.stack([vector(pt, prompt, PROMPTS) for prompt in PROMPTS])
    m = torch.stack([vector(mt, model, MODELS) for model in pool])
    n, a, b = len(rows), len(PROMPTS), len(pool)
    return torch.cat((q[:, None, None].expand(n, a, b, -1),
                      p[None, :, None].expand(n, a, b, -1),
                      m[None, None, :].expand(n, a, b, -1)), dim=-1).to(device)


def mlp_epoch(net, optimizer, x, rewards, rng):
    net.train()
    for i in torch.randperm(len(x), generator=rng).tolist():
        scores = net(x[i]).flatten()
        target = rewards[i].flatten()
        loss = F.mse_loss(scores, target)
        loss = loss + 0.05 * F.cross_entropy(scores.unsqueeze(0), target.argmax().view(1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()


@torch.inference_mode()
def mlp_score(net, x):
    return net.eval()(x).cpu()


def utility(data):
    return data[..., 0] - LAM * data[..., 1]


def fixed_action(rows, actions, lo, span):
    def average(action):
        return sum(reward(row, action, lo, span, LAM) for row in rows) / len(rows)
    ranked = max(actions, key=average)
    return ranked, average(ranked)


def run(args):
    device = torch.device(args.device)
    raw = (args.data_root / "interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    qids = {row["qid"] for row in rows}
    if len(qids) != len(rows):
        raise ValueError("duplicate query IDs")
    require_complete_action_space(rows, PROMPTS, MODELS)
    if sorted(model for group in FOLDS.values() for model in group) != sorted(MODELS):
        raise ValueError("the holdout folds must partition the model pool")
    train_ids, val_ids, test_ids = load_or_create_splits(qids, args.split_manifest, SEED)
    support_ids = select_support(train_ids, SUPPORT_SIZE, SEED)
    expected_support = {
        "alpaca-train-000097", "alpaca-train-000170", "alpaca-train-000198",
        "gsm8k-train-000171", "hotpotqa-train-000170", "squad-train-000066",
        "squad-train-000099", "squad-train-000130",
    }
    if not args.smoke and support_ids != expected_support:
        raise ValueError("support IDs differ from T-039")
    take = lambda ids: [row for row in rows if row["qid"] in ids]
    context, fit, support, validation, test = map(
        take, (train_ids, train_ids - support_ids, support_ids, val_ids, test_ids))
    if not args.smoke and tuple(map(len, (fit, support, validation, test))) != (549, 8, 119, 121):
        raise ValueError("unexpected benchmark split sizes")
    held = FOLDS[args.fold]
    seen = tuple(m for m in MODELS if m not in held)
    lo, hi = joint.cost_reference(fit, seen)
    span = hi - lo
    tables = tuple(load_table(args.data_root / "router" / f"{name}_embeddings.pt")
                   for name in ("query", "task", "prompt", "model"))
    fit_mask = torch.tensor([row["qid"] not in support_ids for row in context])
    known = joint.action_data(context, PROMPTS, seen, lo, span)
    val = joint.action_data(validation, PROMPTS, seen, lo, span)
    direct_index = PROMPTS.index("direct")
    direct_prompts = ("direct",)
    direct_tables = (tables[0], tables[1], tables[3])
    known_joint = joint.make_visible(context, PROMPTS, seen, known, LAM)
    known_direct = direct.make_visible(context, seen, known[:, direct_index], LAM)

    def score_joint(net, targets, pool, visible):
        return joint.evaluate_scores(net, context, targets, PROMPTS, pool, tables, visible, device)

    def score_direct(net, targets, pool, visible):
        return direct.evaluate_scores(net, context, targets, pool, direct_tables, visible, device).unsqueeze(1)

    def fit_method(name, get_inputs, score, values, val_values, visible):
        print(f"Fitting {name}", flush=True)
        torch.manual_seed(SEED)
        begin = perf_counter()
        expected, _ = joint.best_action_labels(val_values, LAM)
        def validate(net):
            scores = score(net, validation, seen, visible).flatten(1)
            return direct.macro_f1(expected, scores.argmax(1), scores.shape[1])
        net, epoch = fit_graph(get_inputs(), values, fit_mask, validate,
                               1 if args.smoke else 1000, device)
        return net, {"fit_seconds": perf_counter() - begin, "best_epoch": epoch}

    joint_net, joint_info = fit_method(
        "joint GNN", lambda: joint.graph_inputs(context, [], PROMPTS, seen, tables, known_joint, device),
        score_joint, known, val, known_joint)
    direct_net, direct_info = fit_method(
        "GraphRouter Direct", lambda: direct.graph_inputs(context, [], seen, direct_tables, known_direct, device),
        score_direct, known[:, direct_index:direct_index+1],
        val[:, direct_index:direct_index+1], known_direct)

    print("Fitting Flat-MLP", flush=True)
    begin = perf_counter()
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    train_x = mlp_features(fit, seen, tables, device)
    train_r = utility(known[fit_mask]).to(device)
    val_x = mlp_features(validation, seen, tables, device)
    val_r = utility(val).flatten(1)
    mlp = FlatActionMLP(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=5e-4, weight_decay=1e-4)
    best_reward, stale = float("-inf"), 0
    for epoch in range(1, 2 if args.smoke else 81):
        mlp_epoch(mlp, optimizer, train_x, train_r, rng)
        choices = mlp_score(mlp, val_x).flatten(1).argmax(1, keepdim=True)
        mean_r = float(val_r.gather(1, choices).mean())
        if mean_r > best_reward + 1e-6:
            best_reward, best_epoch, stale = mean_r, epoch, 0
            state = deepcopy(mlp.state_dict())
        else:
            stale += 1
        if stale >= 18:
            break
    mlp.load_state_dict(state)
    mlp_info = {"fit_seconds": perf_counter() - begin, "best_epoch": best_epoch}

    # Revealing the held-out support is permitted only after initial model selection.
    support_truth = joint.action_data(support, PROMPTS, held, lo, span)
    all_scores, metadata = {}, {}
    for name, net, score, visible, info, prompts in (
        ("joint_gnn", joint_net, score_joint, known_joint, joint_info, PROMPTS),
        ("graphrouter_direct", direct_net, score_direct, known_direct, direct_info, direct_prompts),
    ):
        before = state_hash(net.state_dict())
        begin = perf_counter()
        if name == "joint_gnn":
            extra = joint.make_visible(support, PROMPTS, held, support_truth, LAM)
        else:
            extra = direct.make_visible(support, held, support_truth[:, direct_index], LAM)
        observed = visible + extra
        info.update(incorporation_seconds=perf_counter() - begin, support_actions=len(extra))
        begin = perf_counter()
        all_scores[name] = score(net, test, MODELS, observed)
        info["scoring_seconds"] = perf_counter() - begin
        base = score(net, test, seen, visible)
        indices = [MODELS.index(model) for model in seen]
        if not torch.allclose(base, all_scores[name][:, :, indices], atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"{name} changed known candidate scores")
        if before != state_hash(net.state_dict()):
            raise RuntimeError(f"{name} modified its frozen weights or buffers")
        metadata[name] = info

    # Replay one epoch of known fitting queries plus the eight support queries.
    # The per-query loss on support includes known and new actions together.
    begin = perf_counter()
    support_x = mlp_features(support, MODELS, tables, device)
    support_r = utility(joint.action_data(support, PROMPTS, MODELS, lo, span)).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=5e-4, weight_decay=1e-4)
    for _ in range(MLP_UPDATE_EPOCHS):
        for i in torch.randperm(len(fit) + len(support), generator=rng).tolist():
            x, r, j = (train_x, train_r, i) if i < len(fit) else (
                support_x, support_r, i - len(fit))
            mlp_epoch(mlp, optimizer, x[j:j+1], r[j:j+1], rng)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mlp_info.update(incorporation_seconds=perf_counter() - begin,
                    support_actions=len(support) * len(PROMPTS) * len(held),
                    update_steps=MLP_UPDATE_EPOCHS * (len(fit) + len(support)))
    begin = perf_counter()
    all_scores["flat_mlp_adapted"] = mlp_score(mlp, mlp_features(test, MODELS, tables, device))
    mlp_info["scoring_seconds"] = perf_counter() - begin
    metadata["flat_mlp_adapted"] = mlp_info

    begin = perf_counter()
    fixed_known, _ = fixed_action(validation, [(p, m) for p in PROMPTS for m in seen], lo, span)
    fixed_fit_seconds = perf_counter() - begin
    begin = perf_counter()
    fixed_adapted, _ = fixed_action(support, [fixed_known] + [(p, m) for p in PROMPTS for m in held], lo, span)
    fixed_adaptation_seconds = perf_counter() - begin
    for name, action in (("fixed_known", fixed_known), ("fixed_adapted", fixed_adapted)):
        begin = perf_counter()
        scores = torch.zeros(len(test), len(PROMPTS), len(MODELS))
        scores[:, PROMPTS.index(action[0]), MODELS.index(action[1])] = 1
        all_scores[name] = scores
        metadata[name] = dict(fit_seconds=fixed_fit_seconds, scoring_seconds=perf_counter() - begin,
                              incorporation_seconds=fixed_adaptation_seconds if name == "fixed_adapted" else 0,
                              support_actions=len(support) * len(PROMPTS) * len(held) if name == "fixed_adapted" else 0)

    truth = joint.action_data(test, PROMPTS, MODELS, lo, span)
    methods = {}
    for name, scores in all_scores.items():
        prompts = direct_prompts if name == "graphrouter_direct" else PROMPTS
        observed = truth[:, direct_index:direct_index+1] if prompts == direct_prompts else truth
        if not torch.isfinite(scores).all():
            raise RuntimeError(f"{name} produced nonfinite scores")
        if prompts == direct_prompts:
            result = direct.summarize(test, scores[:, 0], observed[:, 0], MODELS, LAM)
            result["pair_counts"] = {f"direct|{model}": count for model, count in result["model_counts"].items()}
        else:
            result = summarize(test, scores, observed, MODELS, LAM)
        result["held_out_counts"] = {m: result["model_counts"].get(m, 0) for m in held}
        methods[name] = dict(metrics=result, **metadata[name], candidate_scores=scores.tolist())

    primary = methods["joint_gnn"]["metrics"]
    return dict(torch_version=torch.__version__,
                protocol="unseen_6x3_v1", fold=args.fold, smoke=args.smoke, seed=SEED,
                lambda_cost=LAM, device=str(device), upstream_commit=direct.UPSTREAM_SHA,
                dataset_sha256=hashlib.sha256(raw).hexdigest(),
                split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
                train_qids=sorted(train_ids-support_ids), support_qids=sorted(support_ids),
                validation_qids=sorted(val_ids), test_qids=[row["qid"] for row in test],
                held_out_models=held, seen_models=seen, model_order=MODELS, prompt_order=PROMPTS,
                cost_reference=dict(min=lo, max=hi, clipped=False),
                methods=methods, comparisons={name: paired(primary, result["metrics"], SEED)
                                             for name, result in methods.items() if name != "joint_gnn"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", choices=FOLDS, default="A")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not args.split_manifest.is_file():
        parser.error("the existing seed-1 split manifest is required")
    if not args.smoke and args.device != "cuda:0":
        parser.error("benchmark requires GPU 0")
    if args.device == "cuda:0" and (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available()):
        parser.error("CUDA_VISIBLE_DEVICES=0 and a working GPU are required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    for name, result in payload["methods"].items():
        print(name, result["metrics"]["R"], result["metrics"]["held_out_counts"])
    print(args.output)


if __name__ == "__main__":
    main()
