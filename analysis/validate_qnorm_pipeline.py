#!/usr/bin/env python3
"""Validate reward algebra and action coverage in the qnorm router dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LAMBDAS = {
    "reward_qnorm_lam_01": 0.1,
    "reward_qnorm_lam_05": 0.5,
    "reward_qnorm_lam_09": 0.9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl"),
    )
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--expected-prompts", type=int, default=4)
    parser.add_argument("--expected-models", type=int, default=9)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report incomplete action spaces without failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    errors: list[str] = []
    query_count = 0
    edge_count = 0
    task_counts: Counter[str] = Counter()

    with args.data.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_count += 1
            qid = str(row.get("qid"))
            task_counts[str(row.get("task"))] += 1
            edges = row.get("action_edges", [])
            edge_count += len(edges)

            prompts = {edge.get("prompt") for edge in edges}
            models = {edge.get("model") for edge in edges}
            pairs = [(edge.get("prompt"), edge.get("model")) for edge in edges]
            expected_edges = args.expected_prompts * args.expected_models

            if len(prompts) != args.expected_prompts:
                errors.append(f"{qid}: prompts={len(prompts)}, expected={args.expected_prompts}")
            if len(models) != args.expected_models:
                errors.append(f"{qid}: models={len(models)}, expected={args.expected_models}")
            if len(pairs) != len(set(pairs)):
                errors.append(f"{qid}: duplicate prompt--model actions")
            if len(edges) != expected_edges:
                errors.append(f"{qid}: actions={len(edges)}, expected={expected_edges}")

            for edge_idx, edge in enumerate(edges):
                perf = edge.get("performance")
                cost = edge.get("cost_norm_query")
                if perf is None or cost is None:
                    errors.append(f"{qid}/edge{edge_idx}: missing performance or cost")
                    continue
                perf = float(perf)
                cost = float(cost)
                if not 0.0 <= cost <= 1.0:
                    errors.append(f"{qid}/edge{edge_idx}: cost {cost} outside [0,1]")

                for key, lam in LAMBDAS.items():
                    actual = edge.get(key)
                    if actual is None:
                        errors.append(f"{qid}/edge{edge_idx}: missing {key}")
                        continue
                    expected = perf - lam * cost
                    if abs(float(actual) - expected) > args.atol:
                        errors.append(
                            f"{qid}/edge{edge_idx}/{key}: "
                            f"stored={float(actual):.12g}, expected={expected:.12g}"
                        )

    coverage_errors = [
        error for error in errors
        if "prompts=" in error or "models=" in error or "actions=" in error
        or "duplicate" in error
    ]
    algebra_errors = [error for error in errors if error not in coverage_errors]

    print("==== QNORM PIPELINE VALIDATION ====")
    print(f"data: {args.data}")
    print(f"queries: {query_count}")
    print(f"action_edges: {edge_count}")
    print(f"tasks: {dict(task_counts)}")
    print(f"reward/data errors: {len(algebra_errors)}")
    print(f"coverage errors: {len(coverage_errors)}")

    fatal = algebra_errors + ([] if args.allow_incomplete else coverage_errors)
    if fatal:
        for error in fatal[:25]:
            print(f"ERROR: {error}")
        if len(fatal) > 25:
            print(f"... {len(fatal) - 25} additional errors")
        raise SystemExit(1)

    if coverage_errors:
        for error in coverage_errors[:25]:
            print(f"WARNING: {error}")
    print("Validation passed.")


if __name__ == "__main__":
    main()
