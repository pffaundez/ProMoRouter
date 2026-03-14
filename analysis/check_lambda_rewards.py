import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_lambdas.jsonl")


def safe_mean(values):
    return sum(values) / len(values) if values else None


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    values = defaultdict(list)

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for key in ["reward_lam_01", "reward_lam_05", "reward_lam_09"]:
                val = row.get(key)
                if val is not None:
                    values[key].append(float(val))

    print("==== LAMBDA REWARD SUMMARY ====")
    for key in ["reward_lam_01", "reward_lam_05", "reward_lam_09"]:
        vals = values[key]
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