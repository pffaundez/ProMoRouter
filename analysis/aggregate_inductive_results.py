#!/usr/bin/env python3
"""Aggregate the standalone inductive evaluation artifact.

The script deliberately keeps these summaries separate from the P0 outputs.
It reports realized token counts as the cost proxy and also derives a
query-normalized token cost for the protocol's P/C/R summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


LAMBDAS = (0.1, 0.5, 0.9)
REQUIRED_NUMERIC_FIELDS = (
    "performance",
    "input_tokens",
    "output_tokens",
    "tokens_total",
    "cost_proxy_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/inductive_full_corrected.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/inductive_candidates.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/inductive_aggregates.json"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_and_annotate(rows: list[dict], manifest: dict) -> dict:
    seen_models = set(manifest["training_pool"]["models"])
    seen_prompts = set(manifest["training_pool"]["prompts"])
    unseen_models = {item["id"] for item in manifest["unseen_models"]}
    unseen_prompts = {item["id"] for item in manifest["unseen_prompts"]}
    expected_models = seen_models | unseen_models
    expected_prompts = seen_prompts | unseen_prompts

    keys = []
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_NUMERIC_FIELDS if not finite_number(row.get(field))]
        if missing:
            raise SystemExit(f"row {index} has missing/non-finite numeric fields: {missing}")
        if not row.get("response"):
            raise SystemExit(f"row {index} has an empty response")
        if int(row["tokens_total"]) != int(row["input_tokens"]) + int(row["output_tokens"]):
            raise SystemExit(f"row {index} violates tokens_total = input_tokens + output_tokens")
        if int(row["cost_proxy_tokens"]) != int(row["tokens_total"]):
            raise SystemExit(f"row {index} violates cost_proxy_tokens = tokens_total")
        keys.append((row["qid"], row["model"], row["prompt"]))

    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate (qid, model, prompt) keys")
    if len(rows) != 8712:
        raise SystemExit(f"expected 8712 rows, found {len(rows)}")

    qids = {row["qid"] for row in rows}
    models = {row["model"] for row in rows}
    prompts = {row["prompt"] for row in rows}
    if len(qids) != 121 or models != expected_models or prompts != expected_prompts:
        raise SystemExit(
            "unexpected coverage: "
            f"qids={len(qids)}, models={sorted(models)}, prompts={sorted(prompts)}"
        )

    expected_actions = {(model, prompt) for model in expected_models for prompt in expected_prompts}
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_qid[row["qid"]].append(row)
    for qid, qrows in by_qid.items():
        actions = {(row["model"], row["prompt"]) for row in qrows}
        if actions != expected_actions:
            raise SystemExit(f"qid {qid} does not contain the exact 72-action space")
        costs = [float(row["cost_proxy_tokens"]) for row in qrows]
        low, high = min(costs), max(costs)
        for row in qrows:
            row["model_visibility"] = "seen" if row["model"] in seen_models else "unseen"
            row["prompt_visibility"] = "seen" if row["prompt"] in seen_prompts else "unseen"
            row["condition"] = f'{row["model_visibility"]}_model__{row["prompt_visibility"]}_prompt'
            row["pair_training_status"] = (
                "observed_pair" if row["model"] in seen_models and row["prompt"] in seen_prompts
                else "unobserved_pair"
            )
            row["cost_norm_query"] = 0.0 if high == low else (
                float(row["cost_proxy_tokens"]) - low
            ) / (high - low)

    return {
        "qids": len(qids),
        "models": len(models),
        "prompts": len(prompts),
        "actions_per_query": len(expected_actions),
        "unique_action_keys": len(set(keys)),
        "seen_models": sorted(seen_models),
        "unseen_models": sorted(unseen_models),
        "seen_prompts": sorted(seen_prompts),
        "unseen_prompts": sorted(unseen_prompts),
    }


def summarize(rows: list[dict]) -> dict:
    metrics = {
        "rows": len(rows),
        "queries": len({row["qid"] for row in rows}),
        "mean_performance": statistics.fmean(float(row["performance"]) for row in rows),
        "mean_input_tokens": statistics.fmean(float(row["input_tokens"]) for row in rows),
        "mean_output_tokens": statistics.fmean(float(row["output_tokens"]) for row in rows),
        "mean_tokens_total": statistics.fmean(float(row["tokens_total"]) for row in rows),
        "median_tokens_total": statistics.median(float(row["tokens_total"]) for row in rows),
        "sum_tokens_total": sum(int(row["tokens_total"]) for row in rows),
        "mean_cost_proxy_tokens": statistics.fmean(
            float(row["cost_proxy_tokens"]) for row in rows
        ),
        "mean_cost_norm_query": statistics.fmean(float(row["cost_norm_query"]) for row in rows),
    }
    for lam in LAMBDAS:
        key = str(lam).replace(".", "_")
        metrics[f"mean_reward_lambda_{key}"] = statistics.fmean(
            float(row["performance"]) - lam * float(row["cost_norm_query"])
            for row in rows
        )
    return metrics


def group_summaries(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key in sorted(groups):
        labels = dict(zip(fields, key))
        output.append({**labels, **summarize(groups[key])})
    return output


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    coverage = validate_and_annotate(rows, manifest)

    result = {
        "protocol": "standalone_inductive_transformers",
        "source": {
            "path": str(args.input),
            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        },
        "cost_definition": {
            "raw_proxy": "cost_proxy_tokens = tokens_total = input_tokens + output_tokens",
            "normalized_proxy": "per-query min-max normalization across all 72 actions",
            "reward": "performance - lambda * cost_norm_query",
            "lambdas": list(LAMBDAS),
        },
        "coverage": coverage,
        "verified_scope": {
            "observed_training_pairs": 36,
            "unobserved_pairs_from_unseen_nodes": 36,
            "separately_masked_seen_seen_pairs": 0,
            "masked_pair_manifest_available": False,
            "note": (
                "No manifest declares seen-model/seen-prompt pairs masked during training; "
                "masked-pair generalization cannot be aggregated from this artifact alone."
            ),
        },
        "overall": summarize(rows),
        "by_model_visibility": group_summaries(rows, ("model_visibility",)),
        "by_prompt_visibility": group_summaries(rows, ("prompt_visibility",)),
        "by_novelty_condition": group_summaries(rows, ("condition",)),
        "by_pair_training_status": group_summaries(rows, ("pair_training_status",)),
        "by_task_and_novelty_condition": group_summaries(rows, ("task", "condition")),
        "by_model": group_summaries(rows, ("model", "model_visibility")),
        "by_prompt": group_summaries(rows, ("prompt", "prompt_visibility")),
        "by_prompt_model_pair": group_summaries(
            rows,
            ("prompt", "model", "prompt_visibility", "model_visibility", "pair_training_status"),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(
        f"rows={len(rows)} qids={coverage['qids']} "
        f"actions_per_query={coverage['actions_per_query']} output={args.output}"
    )


if __name__ == "__main__":
    main()
