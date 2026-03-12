import json
import glob
import os
from collections import Counter

SHARD_GLOB = "data/interaction_logs/grpp_il_v1/shards/train__*.jsonl"
OUT_PATH = "data/interaction_logs/grpp_il_v1/train_clean.jsonl"

# Keep only successful rows for training.
# If you want to be stricter later, you can also require performance.primary != None.
REQUIRE_PRIMARY = False


def main():
    files = sorted(glob.glob(SHARD_GLOB))
    if not files:
        raise FileNotFoundError(f"No shard files found for pattern: {SHARD_GLOB}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    total_rows = 0
    kept_rows = 0
    dropped_failed = 0
    dropped_no_primary = 0

    kept_by_model = Counter()
    kept_by_task = Counter()
    kept_by_prompt = Counter()

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for path in files:
            with open(path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    row = json.loads(line)
                    total_rows += 1

                    if row.get("failed", False):
                        dropped_failed += 1
                        continue

                    primary = None
                    perf = row.get("performance")
                    if isinstance(perf, dict):
                        primary = perf.get("primary")

                    if REQUIRE_PRIMARY and primary is None:
                        dropped_no_primary += 1
                        continue

                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept_rows += 1

                    kept_by_model[row.get("model")] += 1
                    kept_by_task[row.get("task")] += 1
                    kept_by_prompt[row.get("prompt")] += 1

    print("==== TRAINING DATASET BUILT ====")
    print(f"Input rows: {total_rows}")
    print(f"Kept rows: {kept_rows}")
    print(f"Dropped failed rows: {dropped_failed}")
    print(f"Dropped rows without primary metric: {dropped_no_primary}")
    print(f"Output: {OUT_PATH}")

    print("\nKept rows by model:")
    for k, v in sorted(kept_by_model.items()):
        print(f"  {k}: {v}")

    print("\nKept rows by task:")
    for k, v in sorted(kept_by_task.items()):
        print(f"  {k}: {v}")

    print("\nKept rows by prompt:")
    for k, v in sorted(kept_by_prompt.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()