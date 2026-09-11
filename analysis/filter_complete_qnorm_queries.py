#!/usr/bin/env python3
"""Write a new qnorm dataset containing only complete prompt--model action spaces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data_validation import find_action_space_issues


DEFAULT_INPUT = Path(
    "data/interaction_logs/grpp_il_v1/router_bipartite_qnorm.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl"
)
EXPECTED_PROMPTS = ("direct", "cot", "decompose", "selfcheck")
EXPECTED_MODELS = (
    "mistral-7b",
    "qwen2.5-7b",
    "llama3.1-8b",
    "qwen2.5-14b",
    "yi-34b",
    "codellama-34b",
    "mixtral-8x7b",
    "llama3.1-70b",
    "qwen2.5-72b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must differ; the source is immutable.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite to replace it."
        )

    rows = load_jsonl(args.input)
    issues = find_action_space_issues(
        rows,
        expected_prompts=EXPECTED_PROMPTS,
        expected_models=EXPECTED_MODELS,
    )
    rejected_qids = {issue.qid for issue in issues}
    kept = [row for row in rows if str(row.get("qid")) not in rejected_qids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_counts = Counter(str(row.get("task")) for row in kept)
    print("==== COMPLETE ACTION-SPACE FILTER ====")
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"input queries: {len(rows)}")
    print(f"kept queries: {len(kept)}")
    print(f"removed queries: {len(issues)}")
    print(f"kept action edges: {sum(len(row['action_edges']) for row in kept)}")
    print(f"task counts: {dict(task_counts)}")
    for issue in issues:
        print(f"REMOVED: {issue.summary()}")


if __name__ == "__main__":
    main()
