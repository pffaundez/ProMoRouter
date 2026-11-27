"""Evaluate frozen ProMoRouter Lite with support-based score calibration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from audit_graphrouter_joint_fixed import paired
from router import graphrouter_joint as joint
from router.data_validation import require_complete_action_space
from router.splits import load_or_create_splits
from run_rq1_closed_pool import fit_flat, validation_reward
from run_unseen_6x3 import FOLDS, LAM, SEED, SUPPORT_SIZE, mlp_features, mlp_score, utility
from train_graphrouter_joint_episodic import state_hash
from train_graphrouter_joint_unseen import summarize
from train_router_support_graph import MODELS, PROMPTS, load_table, select_support

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
DATASET = Path("interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl")


def fit_lite(fit, validation, pool, tables, lo, span, epochs, device):
    train_values = joint.action_data(fit, PROMPTS, pool, lo, span)
    validation_values = joint.action_data(validation, PROMPTS, pool, lo, span)
    validation_x = mlp_features(validation, pool, tables, device)
    return fit_flat(
        mlp_features(fit, pool, tables, device),
        utility(train_values).to(device),
        lambda net: validation_reward(mlp_score(net, validation_x), validation_values, LAM),
        epochs, SEED, device,
    )


def residuals(net, support, pool, tables, lo, span, device):
    scores = mlp_score(net, mlp_features(support, pool, tables, device))
    observed = utility(joint.action_data(support, PROMPTS, pool, lo, span))
    return (observed - scores).mean(0)


def calibrate(scores, pool, held, residual, weight):
    corrected = scores.clone()
    indices = [pool.index(model) for model in held]
    corrected[..., indices] += weight * residual
    return corrected


def select_weight(fit, validation, support, seen, tables, epochs, device):
    evidence = {}
    for pseudo in seen:
        fit_pool = tuple(model for model in seen if model != pseudo)
        lo, hi = joint.cost_reference(fit, fit_pool)
        span = hi - lo
        net, epoch = fit_lite(fit, validation, fit_pool, tables, lo, span, epochs, device)
        residual = residuals(net, support, (pseudo,), tables, lo, span, device)
        scores = mlp_score(net, mlp_features(validation, seen, tables, device))
        truth = joint.action_data(validation, PROMPTS, seen, lo, span)
        values = {
            str(weight): validation_reward(
                calibrate(scores, seen, (pseudo,), residual, weight), truth, LAM
            )
            for weight in WEIGHTS
        }
        evidence[pseudo] = dict(best_epoch=epoch, validation_rewards=values)
        print(f"Calibration validation: {pseudo}", flush=True)
    average = {str(w): mean(f["validation_rewards"][str(w)] for f in evidence.values())
               for w in WEIGHTS}
    if not all(math.isfinite(value) for value in average.values()):
        raise ValueError("Nonfinite validation reward")
    weight = max(WEIGHTS, key=lambda w: average[str(w)])
    return weight, average, evidence


def run(args):
    raw = (args.data_root / DATASET).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    qids = {row["qid"] for row in rows}
    if len(qids) != len(rows):
        raise ValueError("Duplicate query IDs")
    require_complete_action_space(rows, PROMPTS, MODELS)
    tr, va, te = load_or_create_splits(qids, args.split_manifest, SEED)
    support_ids = select_support(tr, SUPPORT_SIZE, SEED)
    fit, validation, test, support = (
        [row for row in rows if row["qid"] in ids]
        for ids in (tr - support_ids, va, te, support_ids)
    )
    if not args.smoke and tuple(map(len, (fit, validation, test, support))) != (549, 119, 121, 8):
        raise ValueError("Unexpected benchmark split sizes")
    held = FOLDS[args.fold]
    seen = tuple(model for model in MODELS if model not in held)
    lo, hi = joint.cost_reference(fit, seen)
    span = hi - lo
    device = torch.device(args.device)
    tables = tuple(load_table(args.data_root / "router" / f"{key}_embeddings.pt")
                   for key in ("query", "task", "prompt", "model"))
    epochs = 1 if args.smoke else 80

    start = perf_counter()
    weight, average, evidence = select_weight(fit, validation, support, seen, tables, epochs, device)
    tuning_seconds = perf_counter() - start
    start = perf_counter()
    net, epoch = fit_lite(fit, validation, seen, tables, lo, span, epochs, device)
    fit_seconds = perf_counter() - start
    checkpoint = state_hash(net.state_dict())
    start = perf_counter()
    scores_zero = mlp_score(net, mlp_features(test, MODELS, tables, device))
    scoring_seconds = perf_counter() - start
    start = perf_counter()
    correction = residuals(net, support, held, tables, lo, span, device)
    scores_eight = calibrate(scores_zero, MODELS, held, correction, weight)
    incorporation_seconds = perf_counter() - start
    if state_hash(net.state_dict()) != checkpoint:
        raise RuntimeError("Frozen Lite checkpoint changed")
    indices = [MODELS.index(model) for model in seen]
    if not torch.equal(scores_zero[..., indices], scores_eight[..., indices]):
        raise RuntimeError("Calibration changed known-model scores")
    if not torch.isfinite(scores_eight).all() or not torch.isfinite(scores_zero).all():
        raise ValueError("Nonfinite Lite scores")

    truth = joint.action_data(test, PROMPTS, MODELS, lo, span)
    zero = summarize(test, scores_zero, truth, MODELS, LAM)
    eight = summarize(test, scores_eight, truth, MODELS, LAM)
    for metrics in (zero, eight):
        metrics["held_out_counts"] = {m: metrics["model_counts"].get(m, 0) for m in held}
    changes = int((scores_zero.flatten(1).argmax(1) != scores_eight.flatten(1).argmax(1)).sum())
    return dict(
        protocol="rq2_lite_frozen_calibration_v1", fold=args.fold, seed=SEED,
        lambda_cost=LAM, smoke=args.smoke, device=str(device), torch_version=torch.__version__,
        dataset_sha256=hashlib.sha256(raw).hexdigest(),
        split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        train_qids=sorted(tr - support_ids), validation_qids=sorted(va),
        support_qids=sorted(support_ids), test_qids=[row["qid"] for row in test],
        held_out_models=held, seen_models=seen, model_order=MODELS, prompt_order=PROMPTS,
        cost_reference=dict(min=lo, max=hi, clipped=False),
        checkpoint_sha256=checkpoint, best_epoch=epoch, weights_updated=False,
        calibration=dict(weight=weight, validation_by_weight=average,
                         pseudo_holdouts=evidence, mean_residual_by_prompt_and_held_model=correction.tolist()),
        fit_seconds=fit_seconds, tuning_seconds=tuning_seconds,
        incorporation_seconds=incorporation_seconds, scoring_seconds=scoring_seconds,
        zero_support=dict(support_actions=0, metrics=zero, candidate_scores=scores_zero.tolist()),
        eight_support=dict(support_actions=len(support) * len(PROMPTS) * len(held),
                           metrics=eight, candidate_scores=scores_eight.tolist()),
        changed_selections=changes, delta_R=paired(eight, zero, SEED),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.split_manifest.is_file():
        parser.error("An existing split manifest and a new output path are required")
    if not args.smoke and args.device != "cuda:0":
        parser.error("Benchmark requires GPU 0")
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
    print(f"Fold {args.fold}: weight={result['calibration']['weight']}, "
          f"R(8)={result['eight_support']['metrics']['R']:.6f}")
    print(args.output)


if __name__ == "__main__":
    main()
