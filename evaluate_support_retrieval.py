#!/usr/bin/env python3
"""Compare uniform support averages and fixed cosine retrieval without fitting.

Reuse trusted v2 checkpoints only to recover paths and the exact calibration IDs.
Support outcome means are compared under one common prediction/readout protocol.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from collections import Counter


def support_weights(similarities, temperature, method):
    if method == "uniform":
        return [1.0 / len(similarities)] * len(similarities)
    if method != "cosine":
        raise ValueError("Expected uniform or cosine weighting.")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be positive and finite.")
    maximum = max(similarities)
    weights = [math.exp((value - maximum) / temperature) for value in similarities]
    return [weight / sum(weights) for weight in weights]


def summarize(records):
    wins = sum(record["oracle_win"] for record in records)
    recovered = sum(record["oracle_win"] and record["held_selected"] for record in records)
    selected = [record for record in records if record["held_selected"]]
    deltas = [record["R"] - record["seen_R"] for record in selected]
    return {
        "beneficial_selections": sum(delta > 1e-8 for delta in deltas),
        "harmful_selections": sum(delta < -1e-8 for delta in deltas),
        "tied_selections": sum(abs(delta) <= 1e-8 for delta in deltas),
        "held_out_exact_optimal_pair": sum(
            record["oracle_win"] and record["held_selected"]
            and record["optimal_pair"] for record in records),
        "positive_gain_contribution": sum(max(0, delta) for delta in deltas) / len(records),
        "negative_gain_contribution": sum(min(0, delta) for delta in deltas) / len(records),
        "prompt_counts": dict(Counter(record["prompt"] for record in records)),
        "held_out_prompt_counts": dict(Counter(record["prompt"] for record in selected)),
        "queries": len(records),
        "P": mean(record["P"] for record in records),
        "C": mean(record["C"] for record in records),
        "R": mean(record["R"] for record in records),
        "reward_seen_only": mean(record["seen_R"] for record in records),
        "incorporation_gain": mean(record["R"] - record["seen_R"] for record in records),
        "oracle_gain": mean(record["oracle_gain"] for record in records),
        "oracle_wins": wins,
        "recovered": recovered,
        "opportunity_recall": recovered / wins if wins else None,
        "held_out_selections": sum(record["held_selected"] for record in records),
    }


def main():
    import torch
    import torch.nn.functional as F
    from router.data_validation import require_complete_action_space
    from train_router_flat_mlp_qnorm import (
        EXPECTED_MODELS, EXPECTED_PROMPTS, LAMBDA_CONFIGS,
        load_embedding_object, load_jsonl, vector_for,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="A trusted v2 checkpoint from the target fold.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Fixed softmax temperature, chosen before reading test results.")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    support_weights([0.0], args.temperature, "cosine")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    held_out = config["held_out_model"]
    rows = load_jsonl(config["data_path"])
    require_complete_action_space(rows, expected_prompts=EXPECTED_PROMPTS,
                                  expected_models=EXPECTED_MODELS)
    by_id = {row["qid"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate query IDs.")
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    support_ids = sorted(checkpoint["support_qids"])
    test_ids = sorted(manifest["test_qids"])
    if not support_ids or not test_ids or set(support_ids) & set(test_ids):
        raise ValueError("Support and test must be non-empty and disjoint.")
    if not set(support_ids) <= set(manifest["train_qids"]):
        raise ValueError("Support queries must belong to the training query split.")
    actions = [(p, m) for p in EXPECTED_PROMPTS for m in EXPECTED_MODELS]

    def outcomes(qid):
        indexed = {(edge["prompt"], edge["model"]): edge for edge in by_id[qid]["action_edges"]}
        values = [(float(indexed[action]["performance"]),
                   float(indexed[action]["cost_norm_query"])) for action in actions]
        if any(not math.isfinite(value) for pair in values for value in pair):
            raise ValueError(f"Non-finite support/test outcomes for {qid}")
        return values

    # These are observed support results, available within the calibration budget.
    support_outcomes = [outcomes(qid) for qid in support_ids]
    embeddings = load_embedding_object(config["query_emb_path"])

    def normalized(qid):
        vector = vector_for(embeddings, qid).float().cpu()
        if not torch.isfinite(vector).all() or vector.norm().item() == 0:
            raise ValueError(f"Invalid query embedding for {qid}")
        return F.normalize(vector, dim=0)

    support_vectors = torch.stack([normalized(qid) for qid in support_ids])
    held_indices = [i for i, (_, model) in enumerate(actions) if model == held_out]
    seen_indices = [i for i, (_, model) in enumerate(actions) if model != held_out]
    if not held_indices or not seen_indices:
        raise ValueError("Both seen and held-out candidates are required.")

    # Post-hoc summaries never feed the selection procedure.
    def pool_summary(query_ids, penalty):
        observed_rows = [outcomes(qid) for qid in query_ids]
        action_means = [
            mean(values[i][0] - penalty * values[i][1] for values in observed_rows)
            for i in range(len(actions))
        ]
        best_held = max(held_indices, key=action_means.__getitem__)
        best_seen = max(seen_indices, key=action_means.__getitem__)
        return {
            "queries": len(query_ids),
            "best_held_fixed_pair": actions[best_held],
            "best_held_fixed_reward": action_means[best_held],
            "best_seen_fixed_pair": actions[best_seen],
            "best_seen_fixed_reward": action_means[best_seen],
            "held_prompt_means": {
                actions[i][0]: {
                    "P": mean(values[i][0] for values in observed_rows),
                    "C": mean(values[i][1] for values in observed_rows),
                    "R": action_means[i],
                } for i in held_indices
            },
        }

    calibration_diagnostic = {
        str(weight): {
            "support": pool_summary(support_ids, weight),
            "test_posthoc": pool_summary(test_ids, weight),
        } for _, weight in LAMBDA_CONFIGS
    }
    results = []
    for method in ("uniform", "cosine"):
        all_records = {weight: [] for _, weight in LAMBDA_CONFIGS}
        for qid in test_ids:
            similarities = (support_vectors @ normalized(qid)).tolist()
            weights = support_weights(similarities, args.temperature, method)
            predicted = [
                tuple(sum(weights[j] * support_outcomes[j][i][field]
                          for j in range(len(support_ids))) for field in (0, 1))
                for i in range(len(actions))
            ]
            # Choose before accessing the target query's outcomes.
            selections = {}
            for _, penalty in LAMBDA_CONFIGS:
                scores = [quality - penalty * cost for quality, cost in predicted]
                selections[penalty] = (
                    max(range(len(actions)), key=scores.__getitem__),
                    max(seen_indices, key=scores.__getitem__),
                )
            observed = outcomes(qid)
            for _, penalty in LAMBDA_CONFIGS:
                selected, seen_selected = selections[penalty]
                rewards = [quality - penalty * cost for quality, cost in observed]
                delta = max(rewards[i] for i in held_indices) - max(rewards[i] for i in seen_indices)
                quality, cost = observed[selected]
                all_records[penalty].append({
                    "qid": qid, "prompt": actions[selected][0], "model": actions[selected][1],
                    "P": quality, "C": cost, "R": rewards[selected],
                    "seen_R": rewards[seen_selected],
                    "oracle_win": delta > 1e-8, "oracle_gain": max(0.0, delta),
                    "held_selected": selected in held_indices,
                    "optimal_pair": rewards[selected] >= max(rewards) - 1e-8,
                    "seen_prompt": actions[seen_selected][0],
                    "seen_model": actions[seen_selected][1],
                })
        for penalty, records in all_records.items():
            results.append({"method": method, "lambda": penalty,
                            **summarize(records), "per_query": records})

    output = {
        "held_out_model": held_out, "seed": config["seed"],
        "support_qids": support_ids, "held_out_calibration_calls": len(support_ids) * len(EXPECTED_PROMPTS),
        "temperature": args.temperature,
        "support_cost": "original full-pool query-normalized observed support cost",
        "information_budget": "same support for all candidates; no other training outcomes used",
        "calibration_diagnostic": calibration_diagnostic,
        "metric_notes": {
            "recovered": "Legacy model-only hit count; does not require the optimal prompt.",
            "beneficial_selections": "Held-out selection beats the seen-only policy on that query.",
            "test_posthoc": "Diagnostic means only; never used for fitting or selection.",
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"{result['method']:7s} lambda={result['lambda']:.1f} "
            f"R={result['R']:.4f} gain={result['incorporation_gain']:+.4f} "
            f"oracle_gain={result['oracle_gain']:.4f} "
            f"model_hits={result['recovered']}/{result['oracle_wins']} "
            f"optimal_pairs={result['held_out_exact_optimal_pair']}/{result['oracle_wins']} "
            f"beneficial={result['beneficial_selections']} harmful={result['harmful_selections']} "
            f"tied={result['tied_selections']} "
            f"selected={result['held_out_selections']}/{result['queries']}"
        )

    for penalty, diagnostic in calibration_diagnostic.items():
        support = diagnostic["support"]
        test = diagnostic["test_posthoc"]
        print(
            f"calibration lambda={penalty} "
            f"support held={support['best_held_fixed_reward']:.4f} "
            f"seen={support['best_seen_fixed_reward']:.4f}; "
            f"test held={test['best_held_fixed_reward']:.4f} "
            f"seen={test['best_seen_fixed_reward']:.4f}"
        )
    print("Prompt counts and per-prompt calibration/test means are saved in the JSON.")


if __name__ == "__main__":
    main()
