import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/interaction_logs/grpp_il_v1/train_clean_lambdas.jsonl")

LAMBDA_KEYS = [
    "reward_lam_01",
    "reward_lam_05",
    "reward_lam_09",
]


def safe_mean(values):
    return sum(values) / len(values) if values else None


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    # Store rewards per query.
    rewards_by_lambda_qid = {
        lam: defaultdict(list) for lam in LAMBDA_KEYS
    }

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            qid = row.get("qid")
            if qid is None:
                continue

            for lam in LAMBDA_KEYS:
                reward = row.get(lam)
                if reward is not None:
                    rewards_by_lambda_qid[lam][qid].append(float(reward))

    print("==== ORACLE ROUTING UPPER BOUND ====")

    for lam in LAMBDA_KEYS:

        oracle_rewards = []

        for qid, rewards in rewards_by_lambda_qid[lam].items():

            # Oracle selects the best action for the query.
            oracle_reward = max(rewards)
            oracle_rewards.append(oracle_reward)

        print(f"\n===== {lam} =====")

        if not oracle_rewards:
            print("No valid rewards.")
            continue

        print(f"queries: {len(oracle_rewards)}")
        print(f"oracle_avg_reward: {safe_mean(oracle_rewards):.6f}")
        print(f"oracle_min: {min(oracle_rewards):.6f}")
        print(f"oracle_max: {max(oracle_rewards):.6f}")


if __name__ == "__main__":
    main()