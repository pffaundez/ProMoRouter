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
    return {
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
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"{result['method']:7s} lambda={result['lambda']:.1f} "
            f"R={result['R']:.4f} gain={result['incorporation_gain']:+.4f} "
            f"oracle_gain={result['oracle_gain']:.4f} "
            f"recovered={result['recovered']}/{result['oracle_wins']} "
            f"selected={result['held_out_selections']}/{result['queries']}"
        )


if __name__ == "__main__":
    main()
