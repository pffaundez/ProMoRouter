#!/usr/bin/env python3
"""Merge targeted inductive repair rows and validate the final JSONL artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row):
    return row["qid"], row["model"], row["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--repair", type=Path, nargs=2, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = read_rows(args.full)
    repairs = [row for path in args.repair for row in read_rows(path)]
    repair_map = {key(row): row for row in repairs}
    if len(repair_map) != len(repairs):
        raise SystemExit("duplicate repair keys")
    if len(repairs) != 2:
        raise SystemExit(f"expected exactly 2 repair rows, found {len(repairs)}")

    original_keys = [key(row) for row in rows]
    if len(set(original_keys)) != len(original_keys):
        raise SystemExit("duplicate keys in full artifact")
    missing = set(repair_map) - set(original_keys)
    if missing:
        raise SystemExit(f"repair keys absent from full artifact: {sorted(missing)}")

    replaced = 0
    merged = []
    for row in rows:
        k = key(row)
        if k in repair_map:
            row = repair_map[k]
            replaced += 1
        merged.append(row)

    if replaced != len(repairs):
        raise SystemExit(f"replaced {replaced} rows, expected {len(repairs)}")
    if len(merged) != 8712:
        raise SystemExit(f"expected 8712 rows, found {len(merged)}")
    if len({row["qid"] for row in merged}) != 121:
        raise SystemExit("unexpected query count")
    if len({row["model"] for row in merged}) != 12:
        raise SystemExit("unexpected model count")
    if len({row["prompt"] for row in merged}) != 6:
        raise SystemExit("unexpected prompt count")
    if any(not row.get("response") for row in merged):
        raise SystemExit("empty response remains")
    if any(row.get("performance") is None for row in merged):
        raise SystemExit("missing performance remains")

    expected = {
        ("gsm8k-train-000072", "llama3.1-8b", "cot"): 1.0,
        ("alpaca-train-000102", "llama3.1-70b", "step_back"): 0.5,
    }
    for k, value in expected.items():
        row = next(row for row in merged if key(row) == k)
        if row["performance"] != value:
            raise SystemExit(f"unexpected performance for {k}: {row['performance']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    print(f"merged={len(merged)} replaced={replaced} output={args.output}")


if __name__ == "__main__":
    main()
