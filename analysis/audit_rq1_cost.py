"""Audit routing costs against the original per-action token log."""
import argparse
import json
import math
from pathlib import Path

from add_query_normalized_lambda_rewards import MODEL_PRICES, get_token_cost_proxy


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit(raw_path, routing_path):
    expected = {}
    for row in rows(raw_path):
        performance = row.get("performance") or {}
        if performance.get("primary") is None:
            continue
        key = (row["qid"], row["prompt"], row["model"])
        if key in expected:
            raise ValueError(f"Duplicate source action: {key}")
        expected[key] = row

    checked = set()
    maximum_error = 0.0
    for row in rows(routing_path):
        for edge in row["action_edges"]:
            model = edge["model"]
            key = (row["qid"], edge["prompt"], model)
            if key in checked or key not in expected or model not in MODEL_PRICES:
                raise ValueError(f"Missing, duplicate or unpriced action: {key}")
            source = expected[key]
            tokens = [float(source["cost"][name]) for name in ("tokens_in", "tokens_out")]
            if not all(math.isfinite(t) and t >= 0 and t.is_integer() for t in tokens):
                raise ValueError(f"Invalid source token counts: {key}")
            calculated = get_token_cost_proxy(source)
            stored = float(edge["cost_proxy_money"])
            if not math.isclose(stored, calculated, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"Cost mismatch for {key}: stored {stored}, expected {calculated}")
            if edge.get("tokens_total") is not None and float(edge["tokens_total"]) != sum(tokens):
                raise ValueError(f"Token total mismatch for {key}")
            maximum_error = max(maximum_error, abs(stored - calculated))
            checked.add(key)
    if not checked:
        raise ValueError("No routing actions found")
    return {
        "actions_checked": len(checked),
        "maximum_absolute_proxy_error": maximum_error,
        "unit": "tokens weighted by the historical price table",
        "scope": "recorded action totals; requires upstream call logs to verify call coverage",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.routing), indent=2))
