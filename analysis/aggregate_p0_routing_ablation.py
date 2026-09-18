#!/usr/bin/env python3
"""Validate and aggregate the repaired P0 routing-configuration ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


LAMBDAS = {
    "reward_qnorm_lam_01": 0.1,
    "reward_qnorm_lam_05": 0.5,
    "reward_qnorm_lam_09": 0.9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/p0_routing_ablation.json")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("analysis/p0_routing_ablation_aggregates.json"),
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=Path("analysis/p0_routing_ablation_table.csv"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"{label} is not finite")
    return value


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    common = manifest["common"]
    seeds = args.seeds or common["seeds"]
    if len(seeds) != len(set(seeds)) or any(seed not in common["seeds"] for seed in seeds):
        raise SystemExit("invalid or duplicate seeds")

    sources = []
    rows = []
    for config in manifest["configurations"]:
        for seed in seeds:
            path = (
                Path(common["output_root"])
                / config["id"]
                / f"router_edgegnn_qnorm_results_seed{seed}.json"
            )
            if not path.is_file():
                raise SystemExit(f"missing required result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or len(payload) != 3:
                raise SystemExit(f"{path} must contain three lambda results")
            by_key = {item.get("lambda_key"): item for item in payload}
            if set(by_key) != set(LAMBDAS):
                raise SystemExit(f"{path} has unexpected lambda keys")
            sources.append({"path": str(path), "sha256": sha256(path)})
            for key, lam in LAMBDAS.items():
                item = by_key[key]
                if item.get("queries") != 121:
                    raise SystemExit(f"{path}:{key} must contain 121 test queries")
                if Path(item.get("split_manifest", "")).name != f"qnorm_complete_seed{seed}.json":
                    raise SystemExit(f"{path}:{key} has the wrong split manifest")
                p = finite(item.get("P"), f"{path}:{key}.P")
                c = finite(item.get("C"), f"{path}:{key}.C")
                r = finite(item.get("R"), f"{path}:{key}.R")
                if abs(r - (p - lam * c)) > 1e-8:
                    raise SystemExit(f"{path}:{key} violates reward algebra")
                for field in ("prompt_counts", "model_counts", "task_counts"):
                    if field in item and sum(item[field].values()) != 121:
                        raise SystemExit(f"{path}:{key}.{field} does not sum to 121")
                rows.append({
                    "config_id": config["id"], "seed": seed,
                    "lambda_key": key, "lambda": lam,
                    "P": p, "C": c, "R": r,
                })

    aggregates = []
    for config in manifest["configurations"]:
        for key, lam in LAMBDAS.items():
            group = [
                row for row in rows
                if row["config_id"] == config["id"] and row["lambda_key"] == key
            ]
            if sorted(row["seed"] for row in group) != sorted(seeds):
                raise SystemExit(f"incomplete group: {config['id']} {key}")
            out = {
                "config_id": config["id"],
                "edge_top_k": config["edge_top_k"],
                "full_prompt_model_lattice": config["full_prompt_model_lattice"],
                "lambda_key": key, "lambda": lam, "seeds": sorted(seeds),
            }
            for metric in ("P", "C", "R"):
                values = [row[metric] for row in sorted(group, key=lambda x: x["seed"])]
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                out[f"{metric}_by_seed"] = values
            aggregates.append(out)

    result = {
        "protocol": manifest["protocol"],
        "seeds": sorted(seeds),
        "source_files": len(sources),
        "normalized_rows": len(rows),
        "sources": sources,
        "seed_results": rows,
        "aggregates": aggregates,
        "non_claims": manifest["non_claims"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = [
        "config_id", "edge_top_k", "full_prompt_model_lattice",
        "lambda_key", "lambda", "seeds",
        "P_mean", "P_std", "C_mean", "C_std", "R_mean", "R_std",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in aggregates:
            writer.writerow({field: row[field] for field in fields})
    print(
        f"validated_sources={len(sources)} normalized_rows={len(rows)} "
        f"aggregate_rows={len(aggregates)}"
    )
    print(f"json={args.output_json}")
    print(f"csv={args.output_csv}")


if __name__ == "__main__":
    main()
