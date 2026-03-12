import json
import glob
from collections import Counter

files = glob.glob("data/interaction_logs/grpp_il_v1/shards/train__*.jsonl")

if not files:
    print("No shard files found.")
    exit()

total = 0
failed = 0

model_fail = Counter()
task_fail = Counter()
prompt_fail = Counter()

model_rows = Counter()
task_rows = Counter()
prompt_rows = Counter()

for f in files:
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)

            total += 1

            model = row.get("model")
            task = row.get("task")
            prompt = row.get("prompt")

            model_rows[model] += 1
            task_rows[task] += 1
            prompt_rows[prompt] += 1

            if row.get("failed"):
                failed += 1
                model_fail[model] += 1
                task_fail[task] += 1
                prompt_fail[prompt] += 1


print("\n===== DATASET SUMMARY =====")
print("Total rows:", total)
print("Failed rows:", failed)

if total > 0:
    print("Fail rate:", round(failed / total, 6))

print("\n===== ROWS BY MODEL =====")
for k, v in sorted(model_rows.items()):
    print(f"{k:20s} {v}")

print("\n===== FAILS BY MODEL =====")
for k, v in sorted(model_fail.items()):
    print(f"{k:20s} {v}")

print("\n===== ROWS BY TASK =====")
for k, v in sorted(task_rows.items()):
    print(f"{k:20s} {v}")

print("\n===== FAILS BY TASK =====")
for k, v in sorted(task_fail.items()):
    print(f"{k:20s} {v}")

print("\n===== ROWS BY PROMPT =====")
for k, v in sorted(prompt_rows.items()):
    print(f"{k:20s} {v}")

print("\n===== FAILS BY PROMPT =====")
for k, v in sorted(prompt_fail.items()):
    print(f"{k:20s} {v}")

print("\nDone.")