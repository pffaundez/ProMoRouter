#!/usr/bin/env python3
"""Compare frozen router decisions with and without a held-out model.

Pass one or more run directories, each containing the six generated checkpoints.
Use only trusted checkpoints from your experiments.
"""

import argparse
import json
from pathlib import Path
from statistics import mean


import torch

from router.behavior_profiles import PROFILE_MODES
from train_router_flat_mlp_qnorm import (
    EXPECTED_MODELS,
    EXPECTED_PROMPTS,
    LAMBDA_CONFIGS,
    FlatActionMLP,
    load_embedding_object,
    load_jsonl,
    make_examples,
)

TOLERANCE = 1e-8


def compare_actions(scores, rewards, actions, held_out):
    held = [i for i, (_, model) in enumerate(actions) if model == held_out]
    seen = [i for i, (_, model) in enumerate(actions) if model != held_out]
    if not held or not seen:
        raise ValueError("Both held-out and seen actions are required.")
    selected = max(range(len(actions)), key=scores.__getitem__)
    selected_seen = max(seen, key=scores.__getitem__)
    best_held = max(rewards[i] for i in held)
    best_seen = max(rewards[i] for i in seen)
    oracle_gain = best_held - best_seen
    held_selected = actions[selected][1] == held_out
    return rewards[selected], rewards[selected_seen], oracle_gain, held_selected


@torch.inference_mode()
def diagnose(checkpoint, rows, tables):
    config = checkpoint["config"]
    held_out = config["held_out_model"]
    reward_key = checkpoint["reward_key"]
    weight = dict(LAMBDA_CONFIGS)[reward_key]
    state = checkpoint["state_dict"]
    router = FlatActionMLP(
        state["net.0.weight"].shape[1], config["hidden_dim"], config["dropout"]
    )
    router.load_state_dict(state)
    router.eval()

    paired = []
    for row in rows:
        batch = make_examples(
            row, *tables, EXPECTED_PROMPTS, EXPECTED_MODELS,
            reward_key, torch.device("cpu"), checkpoint["profiles"],
        )
        rewards = [p - weight * c for p, c in zip(batch["performance"], batch["cost"])]
        paired.append(compare_actions(router(batch["x"]).tolist(), rewards, batch["actions"], held_out))

    wins = sum(gap > TOLERANCE for _, _, gap, _ in paired)
    recovered = sum(gap > TOLERANCE and chosen for _, _, gap, chosen in paired)
    return {
        "held_out_model": held_out,
        "profile_mode": checkpoint["profile_mode"],
        "lambda": weight,
        "queries": len(paired),
        "reward_full_pool": mean(full for full, _, _, _ in paired),
        "reward_seen_only": mean(seen for _, seen, _, _ in paired),
        "incorporation_gain": mean(full - seen for full, seen, _, _ in paired),
        "oracle_gain": mean(max(0, gap) for _, _, gap, _ in paired),
        "oracle_wins": wins,
        "recovered": recovered,
        "opportunity_recall": recovered / wins if wins else None,
        "held_out_selections": sum(chosen for _, _, _, chosen in paired),
    }


def diagnose_run(run):
    results = []
    for mode in PROFILE_MODES:
        for reward_key, _ in LAMBDA_CONFIGS:
            path = run / f"{mode}_{reward_key}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
            test_ids = set(manifest["test_qids"])
            if test_ids.intersection(checkpoint["support_qids"]):
                raise ValueError(f"Calibration overlaps test in {path}")
            rows = [row for row in load_jsonl(config["data_path"]) if row["qid"] in test_ids]
            if not test_ids or len(rows) != len(test_ids) or {row["qid"] for row in rows} != test_ids:
                raise ValueError(f"Test data does not match the split in {path}")
            tables = [
                load_embedding_object(config[key])
                for key in ("query_emb_path", "prompt_emb_path", "model_emb_path")
            ]
            result = diagnose(checkpoint, rows, tables)
            result["checkpoint"] = str(path)
            result["seed"] = config["seed"]
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Directories containing checkpoints.")
    parser.add_argument("--json-out", type=Path, help="Optionally save the per-checkpoint metrics.")
    args = parser.parse_args()
    results = [result for run in args.runs for result in diagnose_run(run)]
    for result in results:
        print(
            f"{result['held_out_model']} seed={result['seed']} {result['profile_mode']} "
            f"lambda={result['lambda']:.1f} gain={result['incorporation_gain']:+.4f} "
            f"oracle_gain={result['oracle_gain']:.4f} "
            f"recovered={result['recovered']}/{result['oracle_wins']} "
            f"selected={result['held_out_selections']}/{result['queries']}"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
