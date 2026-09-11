#!/usr/bin/env python3
"""Validate exact population and candidate alignment across repaired router inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("data/interaction_logs/grpp_il_v1")
DEFAULT_BIPARTITE = DEFAULT_ROOT / "router_bipartite_qnorm_complete.jsonl"
DEFAULT_MODEL_ONLY = DEFAULT_ROOT / "router_model_only_qnorm_complete.jsonl"
DEFAULT_DIRECT = DEFAULT_ROOT / "router_model_only_direct_qnorm_complete.jsonl"

EXPECTED_PROMPTS = ("direct", "cot", "decompose", "selfcheck")
EXPECTED_MODELS = (
    "mistral-7b",
    "qwen2.5-7b",
    "llama3.1-8b",
    "qwen2.5-14b",
    "yi-34b",
    "codellama-34b",
    "mixtral-8x7b",
    "llama3.1-70b",
    "qwen2.5-72b",
)
LAMBDAS = {
    "reward_qnorm_lam_01": 0.1,
    "reward_qnorm_lam_05": 0.5,
    "reward_qnorm_lam_09": 0.9,
}


def load_unique_rows(path: Path) -> tuple[list[dict], dict[str, dict]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    by_qid = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("qid")
            if qid is None or not str(qid):
                raise ValueError(f"{path}:{line_no}: missing qid")
            qid = str(qid)
            if qid in by_qid:
                raise ValueError(f"{path}:{line_no}: duplicate qid {qid!r}")
            rows.append(row)
            by_qid[qid] = row
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows, by_qid


def reward_errors(
    item: dict,
    *,
    performance_key: str,
    cost_key: str,
    prefix: str,
    atol: float,
) -> list[str]:
    errors = []
    performance = item.get(performance_key)
    cost = item.get(cost_key)
    if performance is None or cost is None:
        return [f"{prefix}: missing {performance_key} or {cost_key}"]
    performance = float(performance)
    cost = float(cost)
    if not 0.0 <= cost <= 1.0:
        errors.append(f"{prefix}: {cost_key}={cost} outside [0,1]")
    for key, lam in LAMBDAS.items():
        actual = item.get(key)
        if actual is None:
            errors.append(f"{prefix}: missing {key}")
            continue
        expected = performance - lam * cost
        if abs(float(actual) - expected) > atol:
            errors.append(
                f"{prefix}/{key}: stored={float(actual):.12g}, "
                f"expected={expected:.12g}"
            )
    return errors


def validate_bipartite(rows: list[dict], atol: float) -> list[str]:
    expected_pairs = {
        (prompt, model)
        for prompt in EXPECTED_PROMPTS
        for model in EXPECTED_MODELS
    }
    errors = []
    for row in rows:
        qid = str(row["qid"])
        edges = row.get("action_edges", [])
        pairs = [(edge.get("prompt"), edge.get("model")) for edge in edges]
        if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
            errors.append(
                f"bipartite/{qid}: actions={len(pairs)}, "
                f"unique_pairs={len(set(pairs))}"
            )
            continue
        for edge in edges:
            errors.extend(
                reward_errors(
                    edge,
                    performance_key="performance",
                    cost_key="cost_norm_query",
                    prefix=f"bipartite/{qid}/{edge['prompt']}/{edge['model']}",
                    atol=atol,
                )
            )
    return errors


def validate_model_only(
    rows: list[dict],
    *,
    direct: bool,
    atol: float,
) -> list[str]:
    expected_models = set(EXPECTED_MODELS)
    label = "direct" if direct else "model-only"
    errors = []
    for row in rows:
        qid = str(row["qid"])
        candidates = row.get("candidates", [])
        models = [candidate.get("model") for candidate in candidates]
        if len(models) != len(expected_models) or set(models) != expected_models:
            errors.append(
                f"{label}/{qid}: candidates={len(models)}, "
                f"unique_models={len(set(models))}"
            )
            continue
        if row.get("num_models") != len(expected_models):
            errors.append(f"{label}/{qid}: num_models={row.get('num_models')}")
        if direct and row.get("prompt_policy") != "direct":
            errors.append(f"direct/{qid}: prompt_policy is not 'direct'")

        for candidate in candidates:
            model = candidate.get("model")
            if direct and candidate.get("prompt") != "direct":
                errors.append(f"direct/{qid}/{model}: prompt is not 'direct'")
            errors.extend(
                reward_errors(
                    candidate,
                    performance_key=(
                        "performance" if direct else "avg_performance"
                    ),
                    cost_key=(
                        "cost_norm_query" if direct else "avg_cost_norm"
                    ),
                    prefix=f"{label}/{qid}/{model}",
                    atol=atol,
                )
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bipartite", type=Path, default=DEFAULT_BIPARTITE)
    parser.add_argument("--model-only", type=Path, default=DEFAULT_MODEL_ONLY)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--expected-queries", type=int, default=797)
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bipartite_rows, bipartite = load_unique_rows(args.bipartite)
    model_only_rows, model_only = load_unique_rows(args.model_only)
    direct_rows, direct = load_unique_rows(args.direct)

    errors = []
    qid_sets = {
        "bipartite": set(bipartite),
        "model-only": set(model_only),
        "direct": set(direct),
    }
    reference = qid_sets["bipartite"]
    for name, qids in qid_sets.items():
        if qids != reference:
            errors.append(
                f"{name}: qid mismatch "
                f"(missing={sorted(reference - qids)[:10]}, "
                f"extra={sorted(qids - reference)[:10]})"
            )
        if args.expected_queries > 0 and len(qids) != args.expected_queries:
            errors.append(
                f"{name}: queries={len(qids)}, expected={args.expected_queries}"
            )

    for qid in sorted(reference & set(model_only) & set(direct)):
        tasks = {
            str(bipartite[qid].get("task")),
            str(model_only[qid].get("task")),
            str(direct[qid].get("task")),
        }
        if len(tasks) != 1:
            errors.append(f"{qid}: task mismatch={sorted(tasks)}")

    errors.extend(validate_bipartite(bipartite_rows, args.atol))
    errors.extend(validate_model_only(model_only_rows, direct=False, atol=args.atol))
    errors.extend(validate_model_only(direct_rows, direct=True, atol=args.atol))

    print("==== ALIGNED ROUTER INPUT VALIDATION ====")
    print(f"bipartite: {args.bipartite} ({len(bipartite_rows)} queries)")
    print(f"model-only: {args.model_only} ({len(model_only_rows)} queries)")
    print(f"direct: {args.direct} ({len(direct_rows)} queries)")
    print(f"validation errors: {len(errors)}")

    if errors:
        for error in errors[:30]:
            print(f"ERROR: {error}")
        if len(errors) > 30:
            print(f"... {len(errors) - 30} additional errors")
        raise SystemExit(1)
    print("Validation passed: populations, candidates, tasks, and rewards align.")


if __name__ == "__main__":
    main()
