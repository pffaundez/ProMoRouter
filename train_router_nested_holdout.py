#!/usr/bin/env python3
"""Compare seen-only and inductive epoch selection on one inner trajectory.

Run from the repository root with the original complete qnorm log and its
embeddings. The outer test split is used only after both epochs are selected.
"""

import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from diagnose_behavior_profiles import diagnose
from router.behavior_profiles import PROFILE_MODES, build_profiles, seen_only_row
from router.data_validation import require_complete_action_space
from router.splits import load_or_create_splits
from train_router_flat_mlp_qnorm import (
    EXPECTED_MODELS, EXPECTED_PROMPTS, LAMBDA_CONFIGS, FlatActionMLP,
    evaluate, load_embedding_object, load_jsonl, make_examples, parse_args, set_seed,
)


def configure(parser):
    parser.set_defaults(output_dir=Path("outputs/v2_nested_holdout"))
    parser.add_argument("--held-out-model", required=True, choices=EXPECTED_MODELS)
    parser.add_argument("--inner-holdout", required=True, choices=EXPECTED_MODELS)
    parser.add_argument("--calibration-queries", type=int, default=8)
    parser.add_argument("--profile-mode", choices=PROFILE_MODES, default="per-prompt")
    parser.add_argument("--reward-weight", type=float, choices=[w for _, w in LAMBDA_CONFIGS],
                        default=0.9)


def trajectory(args, rows, tables, profiles, reward_key, device, epochs):
    set_seed(args.seed)
    batches = [
        make_examples(row, *tables, EXPECTED_PROMPTS, EXPECTED_MODELS, reward_key, device, profiles)
        for row in rows
    ]
    router = FlatActionMLP(batches[0]["x"].shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, epochs + 1):
        router.train()
        order = list(range(len(batches)))
        random.shuffle(order)
        for index in order:
            batch = batches[index]
            scores = router(batch["x"])
            rewards = batch["reward"]
            loss = F.mse_loss(scores, rewards) + args.ce_weight * F.cross_entropy(
                scores.view(1, -1), torch.argmax(rewards).view(1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), args.grad_clip)
            optimizer.step()
        yield epoch, router


def without_candidates(row, excluded):
    return {**row, "action_edges": [
        edge for edge in row["action_edges"] if edge["model"] not in excluded
    ]}


def main():
    args = parse_args(configure)
    if args.seeds is not None or args.epochs < 1:
        raise SystemExit("Use one --seed and a positive --epochs.")
    if args.inner_holdout == args.held_out_model:
        raise SystemExit("The inner and outer holdouts must differ.")
    if args.split_manifest is None:
        args.split_manifest = Path("data/router/splits") / f"qnorm_complete_seed{args.seed}.json"
    rows = load_jsonl(args.data_path)
    require_complete_action_space(rows, expected_prompts=EXPECTED_PROMPTS,
                                  expected_models=EXPECTED_MODELS)
    train, val, test = load_or_create_splits(
        (row["qid"] for row in rows), args.split_manifest, args.seed
    )
    if not 0 < args.calibration_queries < len(train):
        raise SystemExit("Calibration size must be positive and leave training queries.")
    support = set(random.Random(args.seed).sample(sorted(train), args.calibration_queries))
    train -= support
    outer = args.held_out_model
    inner = args.inner_holdout
    reward_key = next(key for key, weight in LAMBDA_CONFIGS if weight == args.reward_weight)
    support_rows = [row for row in rows if row["qid"] in support]
    train_rows = [row for row in rows if row["qid"] in train]
    outer_train = [seen_only_row(row, outer, LAMBDA_CONFIGS) for row in train_rows]
    inner_train = [seen_only_row(row, inner, LAMBDA_CONFIGS) for row in outer_train]
    val_with_inner = [seen_only_row(row, outer, LAMBDA_CONFIGS) for row in rows if row["qid"] in val]
    # Both policies are measured against the same eight-model validation cost scale.
    val_seen = [without_candidates(row, {inner}) for row in val_with_inner]
    tables = tuple(load_embedding_object(path) for path in (
        args.query_emb_path, args.prompt_emb_path, args.model_emb_path
    ))
    inner_profiles = build_profiles(
        [without_candidates(row, {outer}) for row in support_rows],
        tuple(model for model in EXPECTED_MODELS if model != outer),
        EXPECTED_PROMPTS, args.profile_mode,
    )
    device = torch.device(args.device)
    selected = {}
    history = []
    for epoch, router in trajectory(
        args, inner_train, tables, inner_profiles, reward_key, device, args.epochs
    ):
        record = {"epoch": epoch}
        for criterion, candidates in (("seen_only", val_seen), ("inner_enabled", val_with_inner)):
            reward = evaluate(
                router, candidates, *tables, EXPECTED_PROMPTS, EXPECTED_MODELS,
                reward_key, device, inner_profiles
            )["R"]
            record[criterion] = reward
            if criterion not in selected or reward > selected[criterion]["validation_reward"]:
                selected[criterion] = {"epoch": epoch, "validation_reward": reward}
        history.append(record)
        if args.verbose:
            print(json.dumps(record), flush=True)

    full_profiles = build_profiles(support_rows, EXPECTED_MODELS, EXPECTED_PROMPTS, args.profile_mode)
    test_rows = [row for row in rows if row["qid"] in test]
    target_epochs = {decision["epoch"] for decision in selected.values()}
    outcomes = {}
    for epoch, router in trajectory(
        args, outer_train, tables, inner_profiles, reward_key, device, max(target_epochs)
    ):
        if epoch in target_epochs:
            outcomes[epoch] = diagnose({
                "config": vars(args),
                "reward_key": reward_key,
                "state_dict": router.state_dict(),
                "profile_mode": args.profile_mode,
                "profiles": full_profiles,
            }, test_rows, tables)

    results = {
        "outer_holdout": outer,
        "inner_holdout": inner,
        "profile_mode": args.profile_mode,
        "seed": args.seed,
        "lambda": args.reward_weight,
        "support_qids": sorted(support),
        "split_manifest": str(args.split_manifest),
        "inner_curves": history,
        "criteria": {name: {**decision, "outer_metrics": outcomes[decision["epoch"]]}
                     for name, decision in selected.items()},
    }
    output = args.output_dir / outer / inner / args.profile_mode
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{reward_key}_k{len(support)}_seed{args.seed}_epochs{args.epochs}.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results_path": str(path), **results["criteria"]}, indent=2))


if __name__ == "__main__":
    main()
