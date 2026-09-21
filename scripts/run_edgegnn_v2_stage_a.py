#!/usr/bin/env python3
"""Run or dry-run the frozen Edge-GNN v2 Stage A matrix."""
import argparse, json, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, default=Path("configs/edgegnn_v2_stage_a.json"))
    p.add_argument("--artifact-root", type=Path); p.add_argument("--output-root", type=Path, default=Path("outputs/edgegnn_v2_stage_a"))
    p.add_argument("--seeds", type=int, nargs="+"); p.add_argument("--arms", nargs="+"); p.add_argument("--device")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--overwrite", action="store_true"); args = p.parse_args()
    manifest = json.loads(args.manifest.read_text()); known_arms = [a["id"] for a in manifest["arms"]]
    arms, seeds = args.arms or known_arms, args.seeds or manifest["data"]["seeds"]
    if set(arms) - set(known_arms): raise SystemExit("unknown arm")
    if set(seeds) - set(manifest["data"]["seeds"]): raise SystemExit("invalid seed")
    def artifact(value):
        path = Path(value); return str(args.artifact_root / path if args.artifact_root and not path.is_absolute() else path)
    commands = []
    for arm in arms:
        for seed in seeds:
            output_dir = args.output_root / arm
            result = output_dir / f"stage_a_results_seed{seed}.json"
            if result.exists() and not args.overwrite:
                print(f"skip existing: {result}"); continue
            command = [sys.executable, "train_edgegnn_v2_stage_a.py", "--arm", arm,
                "--data-path", artifact(manifest["data"]["path"]),
                "--query-emb-path", artifact(manifest["embeddings"]["query"]),
                "--task-emb-path", artifact(manifest["embeddings"]["task"]),
                "--prompt-emb-path", artifact(manifest["embeddings"]["prompt"]),
                "--model-emb-path", artifact(manifest["embeddings"]["model"]),
                "--split-manifest", artifact(manifest["data"]["split_manifest_template"].format(seed=seed)),
                "--output-dir", str(output_dir), "--seed", str(seed), "--deterministic"]
            if args.device: command += ["--device", args.device]
            commands.append(command)
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run: subprocess.run(command, check=True)
    print(f"commands={'planned' if args.dry_run else 'completed'} count={len(commands)}")

if __name__ == "__main__": main()
