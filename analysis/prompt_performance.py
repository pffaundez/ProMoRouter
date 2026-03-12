import json
from collections import defaultdict

DATA_PATH = "data/interaction_logs/grpp_il_v1/train_clean.jsonl"


def safe_mean(values):
    return sum(values) / len(values) if values else None


def main():
    scores_by_prompt = defaultdict(list)
    scores_by_task_prompt = defaultdict(list)
    scores_by_model_prompt = defaultdict(list)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            perf = row.get("performance", {})
            primary = perf.get("primary") if isinstance(perf, dict) else None

            # Skip rows without numeric primary score.
            if primary is None:
                continue

            prompt = row.get("prompt")
            task = row.get("task")
            model = row.get("model")

            scores_by_prompt[prompt].append(primary)
            scores_by_task_prompt[(task, prompt)].append(primary)
            scores_by_model_prompt[(model, prompt)].append(primary)

    print("==== AVERAGE PRIMARY SCORE BY PROMPT ====")
    ranked_prompts = []
    for prompt, vals in scores_by_prompt.items():
        avg = safe_mean(vals)
        ranked_prompts.append((prompt, avg, len(vals)))

    ranked_prompts.sort(key=lambda x: x[1], reverse=True)

    for prompt, avg, n in ranked_prompts:
        print(f"{prompt:12s} avg={avg:.4f} n={n}")

    print("\n==== AVERAGE PRIMARY SCORE BY (TASK, PROMPT) ====")
    rows = []
    for (task, prompt), vals in scores_by_task_prompt.items():
        avg = safe_mean(vals)
        rows.append((task, prompt, avg, len(vals)))

    rows.sort(key=lambda x: (x[0], -x[2], x[1]))

    current_task = None
    for task, prompt, avg, n in rows:
        if task != current_task:
            print(f"\n[{task}]")
            current_task = task
        print(f"  {prompt:12s} avg={avg:.4f} n={n}")

    print("\n==== AVERAGE PRIMARY SCORE BY (MODEL, PROMPT) ====")
    rows = []
    for (model, prompt), vals in scores_by_model_prompt.items():
        avg = safe_mean(vals)
        rows.append((model, prompt, avg, len(vals)))

    rows.sort(key=lambda x: (x[0], -x[2], x[1]))

    current_model = None
    for model, prompt, avg, n in rows:
        if model != current_model:
            print(f"\n[{model}]")
            current_model = model
        print(f"  {prompt:12s} avg={avg:.4f} n={n}")


if __name__ == "__main__":
    main()