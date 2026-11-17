#!/usr/bin/env python3
"""Run matched global vs. per-prompt profiles on a held-out original model."""

import json
import random
from pathlib import Path

import torch

from router.behavior_profiles import PROFILE_FEATURES, PROFILE_MODES, build_profiles, seen_only_row
from router.data_validation import require_complete_action_space
from router.splits import load_or_create_splits
from train_router_flat_mlp_qnorm import (
    EXPECTED_MODELS,
    EXPECTED_PROMPTS,
    LAMBDA_CONFIGS,
    load_embedding_object,
    load_jsonl,
    parse_args,
    set_seed,
    train_lambda,
)


def configure(parser):
    parser.set_defaults(output_dir=Path("outputs/v2_behavior_profiles"))
    parser.add_argument("--held-out-model", required=True, choices=EXPECTED_MODELS)
    parser.add_argument("--calibration-queries", type=int, default=8)


def main():
    args = parse_args(configure)
    if args.seeds is not None:
        raise SystemExit("Run one --seed per invocation for this pilot.")
    rows = load_jsonl(args.data_path)
    require_complete_action_space(rows, expected_prompts=EXPECTED_PROMPTS,
                                  expected_models=EXPECTED_MODELS)
    if args.split_manifest is None:
        args.split_manifest = Path("data/router/splits") / f"qnorm_complete_seed{args.seed}.json"
    train, validation, test = load_or_create_splits(
        (row["qid"] for row in rows), args.split_manifest, args.seed
    )
    if not 0 < args.calibration_queries < len(train):
        raise SystemExit("Calibration size must be positive and leave training queries.")
    support = set(random.Random(args.seed).sample(sorted(train), args.calibration_queries))
    train -= support
    support_rows = [row for row in rows if row["qid"] in support]
    experiment_rows = [
        seen_only_row(row, args.held_out_model, LAMBDA_CONFIGS)
        if row["qid"] in train or row["qid"] in validation else row
        for row in rows if row["qid"] not in support
    ]
    query_table, prompt_table, model_table = (
        load_embedding_object(path)
        for path in (args.query_emb_path, args.prompt_emb_path, args.model_emb_path)
    )
    device = torch.device(args.device)
    output = args.output_dir / args.held_out_model / f"k{args.calibration_queries}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in PROFILE_MODES:
        profiles = build_profiles(support_rows, EXPECTED_MODELS, EXPECTED_PROMPTS, mode)
        for reward_key, _ in LAMBDA_CONFIGS:
            set_seed(args.seed)
            model, result = train_lambda(
                args, experiment_rows, train, validation, test,
                query_table, prompt_table, model_table, EXPECTED_PROMPTS,
                EXPECTED_MODELS, reward_key, device, profiles
            )
            checkpoint = output / f"{mode}_{reward_key}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "config": vars(args),
                "reward_key": reward_key,
                "profile_mode": mode,
                "profile_features": PROFILE_FEATURES,
                "profiles": profiles,
                "support_qids": sorted(support),
            }, checkpoint)
            result.update({
                "profile_mode": mode,
                "model_path": str(checkpoint),
                "held_out_selections": result["model_counts"].get(args.held_out_model, 0),
            })
            results.append(result)

    payload = {
        "held_out_model": args.held_out_model,
        "seed": args.seed,
        "support_qids": sorted(support),
        "held_out_calibration_calls": len(support) * len(EXPECTED_PROMPTS),
        "fit_cost_normalization": "query min-max over seen actions",
        "test_cost_normalization": "original query min-max over the full pool",
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
