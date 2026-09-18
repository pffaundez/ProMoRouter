#!/usr/bin/env python3
"""Run the repaired P0 routing-configuration ablation reproducibly."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/p0_routing_ablation.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--config-ids", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing data/interaction_logs and data/router. "
            "Use this when experiment artifacts live outside the repository."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def result_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"router_edgegnn_qnorm_results_seed{seed}.json"


def artifact_path(value: str, root: Path | None) -> str:
    path = Path(value)
    if root is not None and not path.is_absolute():
        path = root / path
    return str(path)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    common = manifest["common"]
    configurations = manifest["configurations"]
    known_ids = {item["id"] for item in configurations}
    selected_ids = set(args.config_ids or known_ids)
    unknown = selected_ids - known_ids
    if unknown:
        raise SystemExit(f"unknown configuration ids: {sorted(unknown)}")
    seeds = args.seeds or common["seeds"]
    if not seeds or any(seed not in common["seeds"] for seed in seeds):
        raise SystemExit(f"seeds must be selected from {common['seeds']}")

    commands = []
    for config in configurations:
        if config["id"] not in selected_ids:
            continue
        output_dir = Path(common["output_root"]) / config["id"]
        for seed in seeds:
            output = result_path(output_dir, seed)
            if output.exists() and not args.overwrite:
                print(f"skip existing: {output}")
                continue
            command = [
                sys.executable,
                common["trainer"],
                "--data-path", artifact_path(common["data_path"], args.artifact_root),
                "--query-emb-path", artifact_path(common["query_emb_path"], args.artifact_root),
                "--task-emb-path", artifact_path(common["task_emb_path"], args.artifact_root),
                "--prompt-emb-path", artifact_path(common["prompt_emb_path"], args.artifact_root),
                "--model-emb-path", artifact_path(common["model_emb_path"], args.artifact_root),
                "--split-manifest", artifact_path(
                    common["split_manifest_template"].format(seed=seed),
                    args.artifact_root,
                ),
                "--output-dir", str(output_dir),
                "--seed", str(seed),
                "--edge-top-k", str(config["edge_top_k"]),
            ]
            if config["full_prompt_model_lattice"]:
                command.append("--full-prompt-model-lattice")
            if args.device:
                command.extend(["--device", args.device])
            commands.append(command)

    if not commands:
        print("nothing to run")
        return
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    print(f"commands={'planned' if args.dry_run else 'completed'} count={len(commands)}")


if __name__ == "__main__":
    main()
