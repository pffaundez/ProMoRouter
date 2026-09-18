#!/usr/bin/env python3
"""Validate and aggregate the canonical five-seed P0 closed-pool outputs.

This script reads only the repaired closed-pool output directories. It does not
read inductive artifacts and never rewrites seed-level results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


LAMBDAS = {
    "reward_qnorm_lam_01": 0.1,
    "reward_qnorm_lam_05": 0.5,
    "reward_qnorm_lam_09": 0.9,
}
SEEDS = (1, 2, 3, 4, 5)
EXPECTED_QUERIES = 121
REWARD_TOLERANCE = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate validated five-seed P0 closed-pool results."
    )
    parser.add_argument(
        "--edge-dir",
        type=Path,
        default=Path("outputs/p0_router_edgegnn_qnorm"),
    )
    parser.add_argument(
        "--direct-dir",
        type=Path,
        default=Path("outputs/p0_router_model_only_direct_qnorm"),
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path("outputs/p0_baselines_qnorm"),
    )
    parser.add_argument(
        "--flat-dir",
        type=Path,
        default=Path("outputs/p1_router_flat_mlp_qnorm"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis/p0_closed_pool_aggregates.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("analysis/p0_closed_pool_table.csv"),
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("analysis/p0_closed_pool_table.tex"),
    )
    return parser.parse_args()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing required result file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} is not numeric: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"{label} is not finite: {value!r}")
    return value


def validate_metrics(metrics: dict, *, label: str, lam: float) -> dict:
    if not isinstance(metrics, dict):
        raise SystemExit(f"{label} is not a metrics object")
    p = finite_float(metrics.get("P"), label=f"{label}.P")
    c = finite_float(metrics.get("C"), label=f"{label}.C")
    r = finite_float(metrics.get("R"), label=f"{label}.R")
    count = metrics.get("queries", metrics.get("n"))
    if count != EXPECTED_QUERIES:
        raise SystemExit(
            f"{label} has {count!r} test queries; expected {EXPECTED_QUERIES}"
        )
    expected_r = p - lam * c
    if abs(r - expected_r) > REWARD_TOLERANCE:
        raise SystemExit(
            f"{label} violates R=P-lambda*C: stored={r}, expected={expected_r}"
        )
    for field in ("prompt_counts", "model_counts", "task_counts"):
        if field in metrics:
            counts = metrics[field]
            if not isinstance(counts, dict) or sum(counts.values()) != EXPECTED_QUERIES:
                raise SystemExit(f"{label}.{field} does not sum to {EXPECTED_QUERIES}")
    return {"P": p, "C": c, "R": r, "n": int(count)}


def expected_file(directory: Path, template: str, seed: int) -> Path:
    path = directory / template.format(seed=seed)
    if not path.is_file():
        raise SystemExit(f"missing seed {seed} result: {path}")
    return path


def validate_manifest(value, *, seed: int, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label} has no split_manifest")
    expected_name = f"qnorm_complete_seed{seed}.json"
    if Path(value).name != expected_name:
        raise SystemExit(
            f"{label} uses split manifest {value!r}; expected basename {expected_name!r}"
        )
    return value


def append_record(records, *, method, family, seed, key, metrics, source, extra=None):
    lam = LAMBDAS[key]
    normalized = validate_metrics(
        metrics, label=f"{source}:{method}:{key}", lam=lam
    )
    records.append(
        {
            "method": method,
            "family": family,
            "seed": seed,
            "lambda_key": key,
            "lambda": lam,
            **normalized,
            "source": str(source),
            "extra": extra or {},
        }
    )


def load_edge(args, records, sources, manifests):
    for seed in SEEDS:
        path = expected_file(
            args.edge_dir, "router_edgegnn_qnorm_results_seed{seed}.json", seed
        )
        payload = read_json(path)
        if not isinstance(payload, list) or len(payload) != len(LAMBDAS):
            raise SystemExit(f"{path} must contain exactly three lambda results")
        by_key = {row.get("lambda_key"): row for row in payload}
        if set(by_key) != set(LAMBDAS):
            raise SystemExit(f"{path} has unexpected lambda keys: {sorted(by_key)}")
        sources.append({"path": str(path), "sha256": sha256(path)})
        for key, lam in LAMBDAS.items():
            row = by_key[key]
            if finite_float(row.get("lambda"), label=f"{path}:{key}.lambda") != lam:
                raise SystemExit(f"{path}:{key} has the wrong lambda")
            manifest = validate_manifest(
                row.get("split_manifest"), seed=seed, label=f"{path}:{key}"
            )
            manifests[(seed, "Edge-GNN", key)] = Path(manifest).name
            append_record(
                records,
                method="Edge-GNN",
                family="learned_joint",
                seed=seed,
                key=key,
                metrics=row,
                source=path,
            )


def load_flat(args, records, sources, manifests):
    for seed in SEEDS:
        path = expected_file(
            args.flat_dir, "flat_mlp_qnorm_results_seed{seed}.json", seed
        )
        payload = read_json(path)
        if not isinstance(payload, list) or len(payload) != len(LAMBDAS):
            raise SystemExit(f"{path} must contain exactly three lambda results")
        by_key = {row.get("lambda_key"): row for row in payload}
        if set(by_key) != set(LAMBDAS):
            raise SystemExit(f"{path} has unexpected lambda keys: {sorted(by_key)}")
        sources.append({"path": str(path), "sha256": sha256(path)})
        for key, lam in LAMBDAS.items():
            row = by_key[key]
            if finite_float(row.get("lambda"), label=f"{path}:{key}.lambda") != lam:
                raise SystemExit(f"{path}:{key} has the wrong lambda")
            manifest = validate_manifest(
                row.get("split_manifest"), seed=seed, label=f"{path}:{key}"
            )
            manifests[(seed, "Flat-MLP", key)] = Path(manifest).name
            append_record(
                records,
                method="Flat-MLP",
                family="learned_no_graph",
                seed=seed,
                key=key,
                metrics=row,
                source=path,
            )


def load_direct(args, records, sources, manifests):
    for seed in SEEDS:
        path = expected_file(
            args.direct_dir,
            "router_model_only_direct_qnorm_results_seed{seed}.json",
            seed,
        )
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise SystemExit(f"{path} must contain an object")
        if payload.get("seed") != seed:
            raise SystemExit(f"{path} stores seed={payload.get('seed')!r}, expected {seed}")
        if payload.get("test_queries") != EXPECTED_QUERIES:
            raise SystemExit(f"{path} has unexpected test_queries")
        if payload.get("prompt_policy") != "direct":
            raise SystemExit(f"{path} does not declare prompt_policy='direct'")
        manifest = validate_manifest(
            payload.get("split_manifest"), seed=seed, label=str(path)
        )
        sources.append({"path": str(path), "sha256": sha256(path)})
        for key in LAMBDAS:
            if key not in payload:
                raise SystemExit(f"{path} is missing {key}")
            manifests[(seed, "GraphRouter-direct", key)] = Path(manifest).name
            append_record(
                records,
                method="GraphRouter-direct",
                family="learned_model_only",
                seed=seed,
                key=key,
                metrics=payload[key],
                source=path,
            )


def load_baselines(args, records, sources):
    methods_seen = None
    for seed in SEEDS:
        path = expected_file(
            args.baselines_dir, "baselines_qnorm_test_results_seed{seed}.json", seed
        )
        payload = read_json(path)
        if not isinstance(payload, dict) or set(payload) != set(LAMBDAS):
            raise SystemExit(f"{path} must contain exactly the three lambda keys")
        sources.append({"path": str(path), "sha256": sha256(path)})
        for key, lam in LAMBDAS.items():
            block = payload[key]
            if finite_float(block.get("lambda"), label=f"{path}:{key}.lambda") != lam:
                raise SystemExit(f"{path}:{key} has the wrong lambda")
            results = block.get("results")
            if not isinstance(results, dict) or not results:
                raise SystemExit(f"{path}:{key}.results is empty or invalid")
            current_methods = set(results)
            if methods_seen is None:
                methods_seen = current_methods
            elif current_methods != methods_seen:
                raise SystemExit(f"{path}:{key} baseline methods differ across files")
            for method, metrics in sorted(results.items()):
                extra = {}
                if method == "Best Fixed Model":
                    extra["selected_model"] = block.get("best_fixed_model")
                elif method == "Best Fixed Pair":
                    extra["selected_pair"] = block.get("best_fixed_pair")
                append_record(
                    records,
                    method=method,
                    family="static_or_oracle",
                    seed=seed,
                    key=key,
                    metrics=metrics,
                    source=path,
                    extra=extra,
                )


def validate_cross_method_coverage(records, manifests):
    counts = Counter((row["method"], row["lambda_key"]) for row in records)
    bad = {key: count for key, count in counts.items() if count != len(SEEDS)}
    if bad:
        raise SystemExit(f"method/lambda groups without five seeds: {bad}")
    learned = ("Edge-GNN", "GraphRouter-direct", "Flat-MLP")
    for seed in SEEDS:
        expected = f"qnorm_complete_seed{seed}.json"
        for method in learned:
            for key in LAMBDAS:
                value = manifests.get((seed, method, key))
                if value != expected:
                    raise SystemExit(
                        f"manifest mismatch for seed={seed}, method={method}, {key}: {value}"
                    )


def aggregate(records):
    groups = {}
    for row in records:
        groups.setdefault((row["method"], row["family"], row["lambda_key"]), []).append(row)
    output = []
    for (method, family, key), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != list(SEEDS):
            raise SystemExit(f"{method}/{key} does not contain seeds 1--5 exactly once")
        item = {
            "method": method,
            "family": family,
            "lambda_key": key,
            "lambda": LAMBDAS[key],
            "seeds": list(SEEDS),
        }
        for metric in ("P", "C", "R"):
            values = [row[metric] for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values)
            item[f"{metric}_by_seed"] = values
        selections = [row["extra"] for row in rows if row["extra"]]
        if selections:
            item["selection_metadata_by_seed"] = selections
        output.append(item)
    return output


def write_csv(path: Path, aggregates):
    fields = [
        "method", "family", "lambda_key", "lambda", "seeds",
        "P_mean", "P_std", "C_mean", "C_std", "R_mean", "R_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in aggregates:
            writer.writerow({field: row[field] for field in fields})


def latex_escape(value: str) -> str:
    return re.sub(r"([&_#%])", r"\\\1", value)


def write_tex(path: Path, aggregates):
    by_method = {}
    for row in aggregates:
        by_method.setdefault(row["method"], {})[row["lambda_key"]] = row
    lines = []
    for method in sorted(by_method):
        values = [latex_escape(method)]
        for key in LAMBDAS:
            row = by_method[method][key]
            for metric in ("P", "C", "R"):
                values.append(
                    f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_std']:.3f}"
                )
        lines.append(" & ".join(values) + " \\\\")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = []
    sources = []
    manifests = {}
    load_edge(args, records, sources, manifests)
    load_direct(args, records, sources, manifests)
    load_baselines(args, records, sources)
    load_flat(args, records, sources, manifests)
    validate_cross_method_coverage(records, manifests)
    aggregates = aggregate(records)

    result = {
        "protocol": "p0_closed_pool_five_seed_qnorm",
        "seeds": list(SEEDS),
        "lambdas": LAMBDAS,
        "expected_test_queries_per_seed": EXPECTED_QUERIES,
        "reward_definition": "R = P - lambda * C",
        "standard_deviation": "sample standard deviation across five seeds",
        "separation": "P0 only; no inductive artifacts are read or aggregated",
        "validation": {
            "source_files": len(sources),
            "normalized_seed_rows": len(records),
            "learned_method_manifests_checked": len(manifests),
            "reward_tolerance": REWARD_TOLERANCE,
            "static_baseline_manifest_limitation": (
                "Baseline JSON files do not serialize split_manifest; their 121-query "
                "coverage is checked, but manifest identity must be established by the "
                "generation command or a separate provenance record."
            ),
        },
        "sources": sorted(sources, key=lambda item: item["path"]),
        "seed_results": records,
        "aggregates": aggregates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, aggregates)
    write_tex(args.output_tex, aggregates)
    print(
        f"validated_sources={len(sources)} normalized_rows={len(records)} "
        f"aggregate_rows={len(aggregates)}"
    )
    print(f"json={args.output_json}")
    print(f"csv={args.output_csv}")
    print(f"tex={args.output_tex}")


if __name__ == "__main__":
    main()
