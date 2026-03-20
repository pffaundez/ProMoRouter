import argparse
import json
from pathlib import Path
from collections import defaultdict

DEFAULT_INPUT_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_qnorm_lambdas.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only_qnorm.jsonl")

LAMBDA_KEYS = [
    "reward_qnorm_lam_01",
    "reward_qnorm_lam_05",
    "reward_qnorm_lam_09",
]


def safe_mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Group all interaction rows by (qid, model).
    grouped = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            perf = row.get("performance", {})
            primary = perf.get("primary") if isinstance(perf, dict) else None
            cost_norm = row.get("cost_norm_query")

            # Keep only rows that are valid for routing evaluation.
            if primary is None or cost_norm is None:
                continue

            qid = row.get("qid")
            model = row.get("model")
            if qid is None or model is None:
                continue

            grouped[(qid, model)].append(row)

    # Regroup by qid so that each output row corresponds to one query.
    by_qid = defaultdict(list)

    for (qid, model), rows in grouped.items():
        first = rows[0]

        entry = {
            "qid": qid,
            "task": first.get("task"),
            "model": model,
            "query_text": first.get("user_text"),
            "n_prompt_variants": len(rows),
            "avg_performance": safe_mean(
                [float(r["performance"]["primary"]) for r in rows if r["performance"]["primary"] is not None]
            ),
            "avg_cost_norm": safe_mean(
                [float(r["cost_norm_query"]) for r in rows if r.get("cost_norm_query") is not None]
            ),
            "avg_cost_proxy_money": safe_mean(
                [float(r["cost_proxy_money"]) for r in rows if r.get("cost_proxy_money") is not None]
            ),
            "avg_tokens_total": safe_mean(
                [
                    float(r["cost"]["tokens_total"])
                    for r in rows
                    if isinstance(r.get("cost"), dict) and r["cost"].get("tokens_total") is not None
                ]
            ),
            "reward_qnorm_lam_01": safe_mean(
                [float(r["reward_qnorm_lam_01"]) for r in rows if r.get("reward_qnorm_lam_01") is not None]
            ),
            "reward_qnorm_lam_05": safe_mean(
                [float(r["reward_qnorm_lam_05"]) for r in rows if r.get("reward_qnorm_lam_05") is not None]
            ),
            "reward_qnorm_lam_09": safe_mean(
                [float(r["reward_qnorm_lam_09"]) for r in rows if r.get("reward_qnorm_lam_09") is not None]
            ),
            # Keep prompt-level details for traceability/debugging.
            "prompt_variants": [
                {
                    "prompt": r.get("prompt"),
                    "query_text": r.get("user_text"),
                    "performance": r["performance"]["primary"],
                    "cost_norm_query": r.get("cost_norm_query"),
                    "cost_proxy_money": r.get("cost_proxy_money"),
                    "tokens_total": r["cost"].get("tokens_total") if isinstance(r.get("cost"), dict) else None,
                    "reward_qnorm_lam_01": r.get("reward_qnorm_lam_01"),
                    "reward_qnorm_lam_05": r.get("reward_qnorm_lam_05"),
                    "reward_qnorm_lam_09": r.get("reward_qnorm_lam_09"),
                }
                for r in rows
            ],
        }

        by_qid[qid].append(entry)

    # Build one JSON line per query with all model candidates.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for qid, candidates in by_qid.items():
            task = candidates[0]["task"] if candidates else None

            # Compute model-only oracle labels for each lambda.
            oracle_model_only = {}
            for lam in LAMBDA_KEYS:
                valid = [c for c in candidates if c.get(lam) is not None]
                if not valid:
                    oracle_model_only[lam] = None
                else:
                    best = max(valid, key=lambda c: c[lam])
                    oracle_model_only[lam] = {
                        "model": best["model"],
                        "reward": best[lam],
                    }

            out_row = {
                "qid": qid,
                "task": task,
                "num_models": len(candidates),
                "candidates": candidates,
                "oracle_model_only": oracle_model_only,
            }

            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print("==== MODEL-ONLY ROUTER DATASET BUILT ====")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Queries: {len(by_qid)}")

    # Print average number of candidates for sanity check.
    avg_candidates = safe_mean([len(v) for v in by_qid.values()])
    print(
        f"Average model candidates per query: {avg_candidates:.3f}"
        if avg_candidates is not None
        else "Average model candidates per query: None"
    )


if __name__ == "__main__":
    main()