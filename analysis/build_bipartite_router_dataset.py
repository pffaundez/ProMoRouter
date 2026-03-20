import argparse
import json
from pathlib import Path
from collections import defaultdict

DEFAULT_INPUT_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_qnorm_lambdas.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl")

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

    # Group valid interaction rows by query.
    by_qid = defaultdict(list)

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
            if qid is None:
                continue

            by_qid[qid].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_queries = 0
    total_edges = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for qid, rows in by_qid.items():
            first = rows[0]

            task = first.get("task")
            query_text = first.get("user_text")

            prompt_candidates = sorted(set(r["prompt"] for r in rows if r.get("prompt") is not None))
            model_candidates = sorted(set(r["model"] for r in rows if r.get("model") is not None))

            action_edges = []

            for r in rows:
                edge = {
                    "prompt": r.get("prompt"),
                    "model": r.get("model"),
                    "performance": float(r["performance"]["primary"]) if r["performance"]["primary"] is not None else None,
                    "cost_norm_query": float(r["cost_norm_query"]) if r.get("cost_norm_query") is not None else None,
                    "cost_proxy_money": float(r["cost_proxy_money"]) if r.get("cost_proxy_money") is not None else None,
                    "tokens_total": float(r["cost"]["tokens_total"]) if isinstance(r.get("cost"), dict) and r["cost"].get("tokens_total") is not None else None,
                    "reward_qnorm_lam_01": float(r["reward_qnorm_lam_01"]) if r.get("reward_qnorm_lam_01") is not None else None,
                    "reward_qnorm_lam_05": float(r["reward_qnorm_lam_05"]) if r.get("reward_qnorm_lam_05") is not None else None,
                    "reward_qnorm_lam_09": float(r["reward_qnorm_lam_09"]) if r.get("reward_qnorm_lam_09") is not None else None,
                }
                action_edges.append(edge)

            # Oracle over full action space (prompt, model).
            oracle_full = {}
            for lam in LAMBDA_KEYS:
                valid = [e for e in action_edges if e.get(lam) is not None]
                if not valid:
                    oracle_full[lam] = None
                else:
                    best = max(valid, key=lambda e: e[lam])
                    oracle_full[lam] = {
                        "prompt": best["prompt"],
                        "model": best["model"],
                        "reward": best[lam],
                    }

            # Oracle prompt-only after averaging across models.
            oracle_prompt_only = {}
            for lam in LAMBDA_KEYS:
                rewards_by_prompt = defaultdict(list)
                for e in action_edges:
                    reward = e.get(lam)
                    prompt = e.get("prompt")
                    if reward is None or prompt is None:
                        continue
                    rewards_by_prompt[prompt].append(float(reward))

                if not rewards_by_prompt:
                    oracle_prompt_only[lam] = None
                else:
                    best_prompt = max(
                        rewards_by_prompt.items(),
                        key=lambda kv: safe_mean(kv[1])
                    )[0]
                    oracle_prompt_only[lam] = {
                        "prompt": best_prompt,
                        "reward": safe_mean(rewards_by_prompt[best_prompt]),
                    }

            # Oracle model-only after averaging across prompts.
            oracle_model_only = {}
            for lam in LAMBDA_KEYS:
                rewards_by_model = defaultdict(list)
                for e in action_edges:
                    reward = e.get(lam)
                    model = e.get("model")
                    if reward is None or model is None:
                        continue
                    rewards_by_model[model].append(float(reward))

                if not rewards_by_model:
                    oracle_model_only[lam] = None
                else:
                    best_model = max(
                        rewards_by_model.items(),
                        key=lambda kv: safe_mean(kv[1])
                    )[0]
                    oracle_model_only[lam] = {
                        "model": best_model,
                        "reward": safe_mean(rewards_by_model[best_model]),
                    }

            out_row = {
                "qid": qid,
                "task": task,
                "query_text": query_text,
                "num_prompts": len(prompt_candidates),
                "num_models": len(model_candidates),
                "num_edges": len(action_edges),
                "prompt_candidates": prompt_candidates,
                "model_candidates": model_candidates,
                "action_edges": action_edges,
                "oracle_full": oracle_full,
                "oracle_prompt_only": oracle_prompt_only,
                "oracle_model_only": oracle_model_only,
            }

            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            total_queries += 1
            total_edges += len(action_edges)

    print("==== BIPARTITE ROUTER DATASET BUILT ====")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Queries: {total_queries}")
    print(f"Total action edges: {total_edges}")
    print(
        f"Average edges per query: {total_edges / total_queries:.3f}"
        if total_queries > 0
        else "Average edges per query: None"
    )


if __name__ == "__main__":
    main()