import json
import math
from pathlib import Path
from collections import defaultdict

INPUT_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean.jsonl")
OUTPUT_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_qnorm_lambdas.jsonl")

LAMBDAS = {
    "reward_qnorm_lam_01": 0.1,
    "reward_qnorm_lam_05": 0.5,
    "reward_qnorm_lam_09": 0.9,
}

# Cost proxy derived from your local llm_candidates configuration.
# These values are used as model-aware token prices.
MODEL_PRICES = {
    "mistral-7b": {"input_price": 0.2, "output_price": 0.2},
    "qwen2.5-7b": {"input_price": 0.3, "output_price": 0.3},
    "llama3.1-8b": {"input_price": 0.5, "output_price": 0.5},
    "qwen2.5-14b": {"input_price": 0.143, "output_price": 0.43},
    "yi-34b": {"input_price": 0.8, "output_price": 0.8},
    "mixtral-8x7b": {"input_price": 0.6, "output_price": 0.6},
    "codellama-34b": {"input_price": 0.8, "output_price": 0.8},
    "llama3.1-70b": {"input_price": 0.88, "output_price": 0.88},
    "qwen2.5-72b": {"input_price": 1.2, "output_price": 1.2},
}


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_primary_performance(row):
    perf = row.get("performance", {})
    if not isinstance(perf, dict):
        return None
    val = perf.get("primary")
    return None if val is None else float(val)


def get_token_cost_proxy(row):
    model = row.get("model")
    cost = row.get("cost", {})

    if model not in MODEL_PRICES:
        return None
    if not isinstance(cost, dict):
        return None

    tokens_in = cost.get("tokens_in")
    tokens_out = cost.get("tokens_out")

    if tokens_in is None or tokens_out is None:
        return None

    prices = MODEL_PRICES[model]

    # Model-aware token cost proxy.
    proxy = (
        float(tokens_in) * float(prices["input_price"])
        + float(tokens_out) * float(prices["output_price"])
    )
    return proxy


def min_max_normalize(values):
    vmin = min(values)
    vmax = max(values)

    if math.isclose(vmin, vmax):
        return [0.0 for _ in values], vmin, vmax

    norm = [(v - vmin) / (vmax - vmin) for v in values]
    return norm, vmin, vmax


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    rows = load_rows(INPUT_PATH)

    # Group valid rows by query id.
    rows_by_qid = defaultdict(list)
    for idx, row in enumerate(rows):
        perf = get_primary_performance(row)
        cost_proxy = get_token_cost_proxy(row)
        qid = row.get("qid")

        if qid is None:
            continue
        if perf is None or cost_proxy is None:
            continue

        rows_by_qid[qid].append((idx, row, perf, cost_proxy))

    # Precompute query-level normalized costs.
    qnorm_cost_by_row_idx = {}
    query_stats = {}

    for qid, items in rows_by_qid.items():
        cost_values = [item[3] for item in items]
        norm_values, cmin, cmax = min_max_normalize(cost_values)

        for j, (idx, row, perf, cost_proxy) in enumerate(items):
            qnorm_cost_by_row_idx[idx] = {
                "cost_proxy_money": cost_proxy,
                "cost_norm_query": norm_values[j],
            }

        query_stats[qid] = {
            "num_actions": len(items),
            "cost_proxy_min": cmin,
            "cost_proxy_max": cmax,
        }

    written = 0
    skipped = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            perf = get_primary_performance(row)

            row["cost_proxy_money"] = None
            row["cost_norm_query"] = None
            for reward_name in LAMBDAS:
                row[reward_name] = None

            if perf is None or idx not in qnorm_cost_by_row_idx:
                skipped += 1
            else:
                cost_proxy_money = qnorm_cost_by_row_idx[idx]["cost_proxy_money"]
                cost_norm_query = qnorm_cost_by_row_idx[idx]["cost_norm_query"]

                row["cost_proxy_money"] = cost_proxy_money
                row["cost_norm_query"] = cost_norm_query

                for reward_name, lam in LAMBDAS.items():
                    row[reward_name] = perf - lam * cost_norm_query

                written += 1

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("==== QUERY-NORMALIZED LAMBDA REWARDS ADDED ====")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows with query-normalized rewards: {written}")
    print(f"Rows skipped: {skipped}")
    print(f"Queries with valid query-level normalization: {len(query_stats)}")
    print("Added fields: cost_proxy_money, cost_norm_query, reward_qnorm_lam_01, reward_qnorm_lam_05, reward_qnorm_lam_09")


if __name__ == "__main__":
    main()