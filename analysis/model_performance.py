import json
from collections import defaultdict

DATA_PATH = "data/interaction_logs/grpp_il_v1/train_clean.jsonl"


def safe_mean(values):
    return sum(values) / len(values) if values else None


def main():
    scores_by_model = defaultdict(list)
    scores_by_task_model = defaultdict(list)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            perf = row.get("performance", {})
            primary = perf.get("primary") if isinstance(perf, dict) else None

            # Skip rows without numeric primary score.
            if primary is None:
                continue

            model = row.get("model")
            task = row.get("task")

            scores_by_model[model].append(primary)
            scores_by_task_model[(task, model)].append(primary)

    print("==== AVERAGE PRIMARY SCORE BY MODEL ====")
    ranked_models = []
    for model, vals in scores_by_model.items():
        avg = safe_mean(vals)
        ranked_models.append((model, avg, len(vals)))

    ranked_models.sort(key=lambda x: x[1], reverse=True)

    for model, avg, n in ranked_models:
        print(f"{model:20s} avg={avg:.4f} n={n}")

    print("\n==== AVERAGE PRIMARY SCORE BY (TASK, MODEL) ====")
    rows = []
    for (task, model), vals in scores_by_task_model.items():
        avg = safe_mean(vals)
        rows.append((task, model, avg, len(vals)))

    rows.sort(key=lambda x: (x[0], -x[2], x[1]))

    current_task = None
    for task, model, avg, n in rows:
        if task != current_task:
            print(f"\n[{task}]")
            current_task = task
        print(f"  {model:20s} avg={avg:.4f} n={n}")


if __name__ == "__main__":
    main()