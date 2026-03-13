import json
import math
from pathlib import Path

INPUT_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean.jsonl")
OUTPUT_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_lambdas.jsonl")

LAMBDAS = {
    "reward_lam_01": 0.1,
    "reward_lam_05": 0.5,
    "reward_lam_09": 0.9,
}


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_tokens_total(row):
    cost = row.get("cost", {})
    if not isinstance(cost, dict):
        return None
    value = cost.get("tokens_total")
    if value is None:
        return None
    return float(value)


def get_primary_performance(row):
    perf = row.get("performance", {})
    if not isinstance(perf, dict):
        return None
    value = perf.get("primary")
    if value is None:
        return None
    return float(value)


def min_max_normalize(values):
    vmin = min(values)
    vmax = max(values)

    # Avoid division by zero if all costs are identical.
    if math.isclose(vmin, vmax):
        return {i: 0.0 for i in range(len(values))}, vmin, vmax

    norm = {
        i: (v - vmin) / (vmax - vmin)
        for i, v in enumerate(values)
    }
    return norm, vmin, vmax


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    rows = load_rows(INPUT_PATH)

    token_values = []
    valid_indices = []

    for i, row in enumerate(rows):
        perf = get_primary_performance(row)
        tokens = get_tokens_total(row)

        # Compute lambda rewards only for rows with valid performance and cost.
        if perf is None or tokens is None:
            continue

        valid_indices.append(i)
        token_values.append(tokens)

    if not token_values:
        raise RuntimeError("No rows with valid performance.primary and cost.tokens_total were found.")

    norm_map_local, cmin, cmax = min_max_normalize(token_values)

    # Map normalized costs back to original row indices.
    norm_cost_by_row_idx = {}
    for local_idx, row_idx in enumerate(valid_indices):
        norm_cost_by_row_idx[row_idx] = norm_map_local[local_idx]

    written = 0
    skipped = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            perf = get_primary_performance(row)
            tokens = get_tokens_total(row)

            row["cost_norm"] = None
            for reward_name in LAMBDAS:
                row[reward_name] = None

            if perf is None or tokens is None or i not in norm_cost_by_row_idx:
                skipped += 1
            else:
                cost_norm = norm_cost_by_row_idx[i]
                row["cost_norm"] = cost_norm

                for reward_name, lam in LAMBDAS.items():
                    row[reward_name] = perf - lam * cost_norm

                written += 1

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("==== LAMBDA REWARDS ADDED ====")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows with lambda rewards: {written}")
    print(f"Rows skipped: {skipped}")
    print(f"Cost min (tokens_total): {cmin}")
    print(f"Cost max (tokens_total): {cmax}")
    print("Added fields: cost_norm, reward_lam_01, reward_lam_05, reward_lam_09")


if __name__ == "__main__":
    main()