import json
import glob
import pandas as pd

SHARD_PATH = "data/interaction_logs/grpp_il_v1/shards/*.jsonl"

rows = []

for file in glob.glob(SHARD_PATH):
    with open(file) as f:
        for line in f:
            r = json.loads(line)

            perf = r.get("performance") or {}
            cost = r.get("cost") or {}

            rows.append({
                "model": r.get("model"),
                "task": r.get("task"),
                "prompt": r.get("prompt"),

                "failed": r.get("failed", False),

                "reward": r.get("reward"),

                "primary_perf": perf.get("primary"),
                "em": perf.get("em"),
                "f1": perf.get("f1"),
                "acc": perf.get("acc"),

                "tokens_in": cost.get("tokens_in"),
                "tokens_out": cost.get("tokens_out"),
                "tokens_total": cost.get("tokens_total"),
                "latency": cost.get("latency_s"),
            })


df = pd.DataFrame(rows)

print("\nTOTAL ROWS:", len(df))
print("FAILED:", df["failed"].sum())

print("\n==============================")
print("AVERAGE METRICS BY MODEL")
print("==============================")

model_summary = (
    df[df["failed"] == False]
    .groupby("model")
    .agg(
        n=("model", "count"),
        reward_avg=("reward", "mean"),
        perf_avg=("primary_perf", "mean"),
        tokens_avg=("tokens_total", "mean"),
        latency_avg=("latency", "mean"),
    )
    .sort_values("reward_avg", ascending=False)
)

print(model_summary)


print("\n==============================")
print("AVERAGE METRICS BY TASK")
print("==============================")

task_summary = (
    df[df["failed"] == False]
    .groupby("task")
    .agg(
        n=("task", "count"),
        reward_avg=("reward", "mean"),
        perf_avg=("primary_perf", "mean"),
        tokens_avg=("tokens_total", "mean"),
    )
)

print(task_summary)


print("\n==============================")
print("CHECK FOR NULL PERFORMANCE")
print("==============================")

null_perf = df[df["primary_perf"].isnull()]

print("Rows with null performance:", len(null_perf))

if len(null_perf) > 0:
    print(null_perf[["model","task","prompt"]].head())