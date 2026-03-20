import json
from pathlib import Path
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt

INPUT_PATH = Path("outputs/router_bipartite_qnorm/router_bipartite_qnorm_results.json")
OUTPUT_DIR = Path("outputs/router_bipartite_qnorm/behavior_analysis")

PROMPT_ORDER = ["direct", "cot", "decompose", "selfcheck"]
MODEL_ORDER = [
    "mistral-7b",
    "qwen2.5-7b",
    "llama3.1-8b",
    "qwen2.5-14b",
    "yi-34b",
    "codellama-34b",
    "mixtral-8x7b",
    "llama3.1-70b",
    "qwen2.5-72b",
]


def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_counts(counts: dict, ordered_keys: list[str]) -> OrderedDict:
    total = sum(counts.get(k, 0) for k in ordered_keys)
    out = OrderedDict()

    for k in ordered_keys:
        value = counts.get(k, 0)
        out[k] = (value / total) if total > 0 else 0.0

    return out


def concentration_stats(distribution: dict[str, float]) -> dict[str, float]:
    values = list(distribution.values())
    max_share = max(values) if values else 0.0
    hhi = sum(v * v for v in values)
    return {
        "max_share": max_share,
        "hhi": hhi,
    }


def print_distribution_table(title: str, lam_label: str, raw_counts: dict, distribution: dict):
    print(f"\n[{title}] lambda={lam_label}")
    print(f"{'item':20s} {'count':>8s} {'share':>10s}")
    print("-" * 42)
    for key, share in distribution.items():
        print(f"{key:20s} {raw_counts.get(key, 0):8d} {share:10.3f}")


def plot_grouped_bars(
    data_by_lambda: dict[str, OrderedDict],
    keys: list[str],
    title: str,
    output_path: Path,
):
    lambdas = list(data_by_lambda.keys())
    x = list(range(len(keys)))
    width = 0.22

    plt.figure(figsize=(10, 4.5))

    for i, lam in enumerate(lambdas):
        vals = [data_by_lambda[lam][k] for k in keys]
        offsets = [xi + (i - 1) * width for xi in x]
        plt.bar(offsets, vals, width=width, label=lam)

    plt.xticks(x, keys, rotation=30, ha="right")
    plt.ylabel("Selection share")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_summary(results):
    prompt_dist_by_lambda = {}
    model_dist_by_lambda = {}

    for row in results:
        lam_key = row["lambda_key"]

        prompt_counts = row.get("prompt_counts", {})
        model_counts = row.get("model_counts", {})

        prompt_dist_by_lambda[lam_key] = normalize_counts(prompt_counts, PROMPT_ORDER)
        model_dist_by_lambda[lam_key] = normalize_counts(model_counts, MODEL_ORDER)

    return prompt_dist_by_lambda, model_dist_by_lambda


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results(INPUT_PATH)
    prompt_dist_by_lambda, model_dist_by_lambda = build_summary(results)

    print("==== ROUTING BEHAVIOR ANALYSIS ====")
    print(f"Input: {INPUT_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Print prompt distributions.
    for row in results:
        lam_key = row["lambda_key"]
        print_distribution_table(
            title="Prompt selection",
            lam_label=lam_key,
            raw_counts=row.get("prompt_counts", {}),
            distribution=prompt_dist_by_lambda[lam_key],
        )

        stats = concentration_stats(prompt_dist_by_lambda[lam_key])
        print(
            f"Prompt concentration | max_share={stats['max_share']:.3f} | "
            f"HHI={stats['hhi']:.3f}"
        )

    # Print model distributions.
    for row in results:
        lam_key = row["lambda_key"]
        print_distribution_table(
            title="Model selection",
            lam_label=lam_key,
            raw_counts=row.get("model_counts", {}),
            distribution=model_dist_by_lambda[lam_key],
        )

        stats = concentration_stats(model_dist_by_lambda[lam_key])
        print(
            f"Model concentration | max_share={stats['max_share']:.3f} | "
            f"HHI={stats['hhi']:.3f}"
        )

    # Save prompt plot.
    plot_grouped_bars(
        data_by_lambda=prompt_dist_by_lambda,
        keys=PROMPT_ORDER,
        title="Prompt usage across cost regimes",
        output_path=OUTPUT_DIR / "prompt_usage_by_lambda.png",
    )

    # Save model plot.
    plot_grouped_bars(
        data_by_lambda=model_dist_by_lambda,
        keys=MODEL_ORDER,
        title="Model usage across cost regimes",
        output_path=OUTPUT_DIR / "model_usage_by_lambda.png",
    )

    # Save machine-readable summary.
    summary = {
        "prompt_distribution": {
            lam: dict(dist) for lam, dist in prompt_dist_by_lambda.items()
        },
        "model_distribution": {
            lam: dict(dist) for lam, dist in model_dist_by_lambda.items()
        },
        "prompt_concentration": {
            lam: concentration_stats(dist) for lam, dist in prompt_dist_by_lambda.items()
        },
        "model_concentration": {
            lam: concentration_stats(dist) for lam, dist in model_dist_by_lambda.items()
        },
    }

    with (OUTPUT_DIR / "routing_behavior_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved files:")
    print(OUTPUT_DIR / "prompt_usage_by_lambda.png")
    print(OUTPUT_DIR / "model_usage_by_lambda.png")
    print(OUTPUT_DIR / "routing_behavior_summary.json")


if __name__ == "__main__":
    main()