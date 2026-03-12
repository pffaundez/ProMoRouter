import json
from collections import defaultdict
from pathlib import Path
import statistics

DATA_DIR = Path("data/interaction_logs/grpp_il_v1/shards")

# group rows by query id
queries = defaultdict(list)

for file in DATA_DIR.glob("train__*.jsonl"):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row.get("failed"):
                continue

            qid = row["qid"]
            reward = row.get("reward")

            if reward is None:
                continue

            queries[qid].append(reward)

oracle_rewards = []
for qid, rewards in queries.items():
    oracle_rewards.append(max(rewards))

print("===== ORACLE ROUTING =====")
print("queries:", len(oracle_rewards))
print("oracle mean reward:", statistics.mean(oracle_rewards))
print("oracle std:", statistics.stdev(oracle_rewards))
print("oracle max:", max(oracle_rewards))
print("oracle min:", min(oracle_rewards))