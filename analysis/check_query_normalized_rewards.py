import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_qnorm_lambdas.jsonl")


def safe_mean(values):
    return sum(values) / len(values) if values else None


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    reward_values = defaultdict(list)
    cost_ranges_by_qid = defaultdict(list)

    rows_by_qid = defaultdict(list)

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row.get("qid")
            if qid is None:
                continue
            rows_by_qid[qid].append(row)

            for key in ["reward_qnorm_lam_01", "reward_qnorm_lam_05", "reward_qnorm_lam_09"]:
                val = row.get(key)
                if val is not None:
                    reward_values[key].append(float(val))

    valid_qids = 0
    avg_cost_range = []

    for qid, rows in rows_by_qid.items():
        vals = [r.get("cost_norm_query") for r in rows if r.get("cost_norm_query") is not None]
        if not vals:
            continue
        valid_qids += 1
        avg_cost_range.append(max(vals) - min(vals))

    print("==== QUERY-NORMALIZED REWARD SUMMARY ====")
    print(f"Valid qids: {valid_qids}")
    print(f"Average within-query cost_norm range: {safe_mean(avg_cost_range):.6f}" if avg_cost_range else "Average within-query cost_norm range: None")

    for key in ["reward_qnorm_lam_01", "reward_qnorm_lam_05", "reward_qnorm_lam_09"]:
        vals = reward_values[key]
        if not vals:
            print(f"{key}: no values")
            continue
        print(
            f"{key}: "
            f"n={len(vals)} "
            f"mean={safe_mean(vals):.6f} "
            f"min={min(vals):.6f} "
            f"max={max(vals):.6f}"
        )


if __name__ == "__main__":
    main()