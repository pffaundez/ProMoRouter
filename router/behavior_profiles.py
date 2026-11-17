"""Support-only profiles and cost labels for held-out-model experiments."""

import math
from statistics import mean


PROFILE_MODES = ("global", "per-prompt")
PROFILE_FEATURES = ("mean_quality", "log1p_mean_tokens", "log1p_queries", "available")


def finite_value(edge, key):
    value = edge.get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Missing or non-finite {key} for {edge['model']} / {edge['prompt']}")
    return float(value)


def build_profiles(support_rows, model_order, prompt_order, mode):
    """Return equal-width features and repeat the global profile across prompts."""
    if mode not in PROFILE_MODES:
        raise ValueError(f"Unknown profile mode: {mode}")
    buckets = {(m, p): [] for m in model_order for p in prompt_order}
    for row in support_rows:
        for edge in row["action_edges"]:
            key = (edge["model"], edge["prompt"])
            performance = finite_value(edge, "performance")
            tokens = finite_value(edge, "tokens_total")
            if tokens < 0:
                raise ValueError(f"Negative token count for {key} on {row['qid']}")
            buckets[key].append((performance, tokens))

    profiles = {}
    for model in model_order:
        pooled = [item for p in prompt_order for item in buckets[(model, p)]]
        for prompt in prompt_order:
            values = pooled if mode == "global" else buckets[(model, prompt)]
            if not values:
                profiles[(model, prompt)] = (0.0, 0.0, 0.0, 0.0)
                continue
            count = len(values) / len(prompt_order) if mode == "global" else len(values)
            profiles[(model, prompt)] = (
                mean(quality for quality, _ in values),
                math.log1p(mean(tokens for _, tokens in values)),
                math.log1p(count),
                1.0,
            )
    return profiles


def seen_only_row(row, held_out_model, lambda_configs):
    """Rebuild labels so the held-out action costs cannot affect fitting."""
    edges = [edge.copy() for edge in row["action_edges"] if edge["model"] != held_out_model]
    costs = [finite_value(edge, "cost_proxy_money") for edge in edges]
    low, high = min(costs), max(costs)
    for edge, cost in zip(edges, costs):
        normalized = (cost - low) / (high - low) if not math.isclose(high, low) else 0.0
        edge["cost_norm_query"] = normalized
        quality = finite_value(edge, "performance")
        for reward_key, weight in lambda_configs:
            edge[reward_key] = quality - weight * normalized
    return {**row, "action_edges": edges}
