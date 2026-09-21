#!/usr/bin/env python3
"""Strict validation and aggregation for Edge-GNN v2 Stage A."""
import argparse, csv, hashlib, json, math, random, statistics
from pathlib import Path

LAMBDAS = {"reward_qnorm_lam_01": .1, "reward_qnorm_lam_05": .5, "reward_qnorm_lam_09": .9}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): raise SystemExit(f"invalid {label}")
    return float(value)

def bootstrap_mean_ci(values, seed, repetitions=10000):
    rng = random.Random(seed); count = len(values)
    samples = sorted(statistics.fmean(values[rng.randrange(count)] for _ in range(count)) for _ in range(repetitions))
    return [samples[int(.025 * repetitions)], samples[int(.975 * repetitions)]]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, default=Path("configs/edgegnn_v2_stage_a.json"))
    p.add_argument("--input-root", type=Path, default=Path("outputs/edgegnn_v2_stage_a")); p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-json", type=Path, default=Path("analysis/edgegnn_v2_stage_a_aggregates.json"))
    p.add_argument("--output-csv", type=Path, default=Path("analysis/edgegnn_v2_stage_a_table.csv")); args = p.parse_args()
    manifest = json.loads(args.manifest.read_text()); arms = [a["id"] for a in manifest["arms"]]; seeds = args.seeds or manifest["data"]["seeds"]
    if len(seeds) != len(set(seeds)) or set(seeds) - set(manifest["data"]["seeds"]): raise SystemExit("invalid or duplicate seeds")
    if args.dry_run:
        print(f"expected_sources={len(arms) * len(seeds)} expected_rows={len(arms) * len(seeds) * len(LAMBDAS)} expected_aggregates={len(arms) * len(LAMBDAS)}")
        for arm in arms:
            for seed in seeds: print(args.input_root / arm / f"stage_a_results_seed{seed}.json")
        return
    rows, sources, keys = [], [], set()
    for arm in arms:
        for seed in seeds:
            path = args.input_root / arm / f"stage_a_results_seed{seed}.json"
            if not path.is_file(): raise SystemExit(f"missing required result: {path}")
            payload = json.loads(path.read_text()); sources.append({"path": str(path), "sha256": sha(path)})
            if not isinstance(payload, list) or len(payload) != 3: raise SystemExit(f"{path}: expected three results")
            for item in payload:
                key = (item.get("arm_id"), item.get("seed"), item.get("lambda_key"))
                if key in keys or key[0] != arm or key[1] != seed or key[2] not in LAMBDAS: raise SystemExit(f"invalid/duplicate key: {key}")
                keys.add(key); lam = LAMBDAS[key[2]]
                if item.get("queries") != 121 or Path(item.get("split_manifest", "")).name != f"qnorm_complete_seed{seed}.json": raise SystemExit(f"{path}: split/query mismatch")
                if not item.get("deterministic") or item.get("topology", {}).get("cross_query_message_passing") is not False: raise SystemExit(f"{path}: protocol mismatch")
                pval, cval, rval = (number(item.get(x), f"{path}:{x}") for x in ("P", "C", "R"))
                if abs(rval - (pval - lam * cval)) > 1e-8: raise SystemExit(f"{path}: reward algebra")
                for field in ("prompt_counts", "model_counts", "task_counts"):
                    if sum(item.get(field, {}).values()) != 121: raise SystemExit(f"{path}: {field}")
                counts = item.get("parameter_counts", {})
                total, active, inactive = (counts.get(k) for k in ("total_trainable", "active_with_gradient", "inactive_trainable"))
                if not all(isinstance(v, int) and v >= 0 for v in (total, active, inactive)) or total != active + inactive:
                    raise SystemExit(f"{path}: invalid parameter counts")
                selections = item.get("selections", [])
                if len(selections) != 121 or len({entry.get("qid") for entry in selections}) != 121:
                    raise SystemExit(f"{path}: invalid per-query selections")
                selection_r = {}
                for entry in selections:
                    query_r = number(entry.get("R"), f"{path}:selection.R")
                    if abs(query_r - (number(entry.get("P"), "selection.P") - lam * number(entry.get("C"), "selection.C"))) > 1e-8:
                        raise SystemExit(f"{path}: per-query reward algebra")
                    selection_r[entry["qid"]] = query_r
                rows.append({"arm_id": arm, "seed": seed, "lambda_key": key[2], "lambda": lam,
                             "P": pval, "C": cval, "R": rval, "parameter_counts": counts,
                             "_selection_r": selection_r})
    aggregates = []
    for arm in arms:
        for lambda_key, lam in LAMBDAS.items():
            group = sorted((r for r in rows if r["arm_id"] == arm and r["lambda_key"] == lambda_key), key=lambda r:r["seed"])
            if [r["seed"] for r in group] != sorted(seeds): raise SystemExit("incomplete aggregate group")
            out = {"arm_id": arm, "lambda_key": lambda_key, "lambda": lam, "seeds": sorted(seeds)}
            for metric in ("P", "C", "R"):
                values = [r[metric] for r in group]; out[f"{metric}_mean"] = statistics.fmean(values); out[f"{metric}_std"] = statistics.stdev(values) if len(values)>1 else 0.; out[f"{metric}_by_seed"] = values
            aggregates.append(out)
    paired = []
    for seed in seeds:
        for lambda_key, lam in LAMBDAS.items():
            full = next(r for r in rows if r["arm_id"] == "edgegnn_v2_full_mp" and r["seed"] == seed and r["lambda_key"] == lambda_key)
            control = next(r for r in rows if r["arm_id"] == "edgegnn_v2_self_only" and r["seed"] == seed and r["lambda_key"] == lambda_key)
            if set(full["_selection_r"]) != set(control["_selection_r"]): raise SystemExit("paired query IDs differ")
            deltas = [full["_selection_r"][qid] - control["_selection_r"][qid] for qid in sorted(full["_selection_r"])]
            paired.append({"seed": seed, "lambda_key": lambda_key, "lambda": lam,
                           "comparison": "edgegnn_v2_full_mp-minus-edgegnn_v2_self_only",
                           "queries": len(deltas), "R_mean_difference": statistics.fmean(deltas),
                           "R_bootstrap_95_ci": bootstrap_mean_ci(deltas, 20260918 + seed * 10 + int(lam * 10))})
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    result = {"protocol": manifest["protocol"], "source_files": len(sources), "normalized_rows": len(rows), "sources": sources, "seed_results": public_rows, "aggregates": aggregates, "paired_query_comparisons": paired}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(result, indent=2)+"\n")
    fields = ["arm_id","lambda_key","lambda","seeds","P_mean","P_std","C_mean","C_std","R_mean","R_std"]
    with args.output_csv.open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows({k:r[k] for k in fields} for r in aggregates)
    print(f"validated_sources={len(sources)} normalized_rows={len(rows)} aggregate_rows={len(aggregates)}")

if __name__ == "__main__": main()
