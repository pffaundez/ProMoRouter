import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT_PATH = Path(
    "data/interaction_logs/grpp_il_v1/train_clean_qnorm_lambdas.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/interaction_logs/grpp_il_v1/router_model_only_qnorm.jsonl"
)

LAMBDA_KEYS = (
    "reward_qnorm_lam_01",
    "reward_qnorm_lam_05",
    "reward_qnorm_lam_09",
)
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


def safe_mean(values):
    return sum(values) / len(values) if values else None


def load_qid_allowlist(path: Path | None) -> set[str] | None:
    """Load the exact query population from a grouped bipartite JSONL file."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"QID source not found: {path}")

    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("qid")
            if qid is None or not str(qid):
                raise ValueError(f"{path}:{line_no}: missing qid")
            qid = str(qid)
            if qid in qids:
                raise ValueError(f"{path}:{line_no}: duplicate qid {qid!r}")
            qids.add(qid)

    if not qids:
        raise ValueError(f"QID source is empty: {path}")
    return qids


def build_rows(input_path: Path, allowed_qids: set[str] | None) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    grouped = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            perf = row.get("performance", {})
            primary = perf.get("primary") if isinstance(perf, dict) else None
            cost_norm = row.get("cost_norm_query")
            qid = row.get("qid")
            model = row.get("model")

            if primary is None or cost_norm is None or qid is None or model is None:
                continue
            qid = str(qid)
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            grouped[(qid, str(model))].append(row)

    by_qid = defaultdict(list)
    for (qid, model), rows in grouped.items():
        first = rows[0]
        entry = {
            "qid": qid,
            "task": first.get("task"),
            "model": model,
            "query_text": first.get("user_text"),
            "n_prompt_variants": len(rows),
            "avg_performance": safe_mean(
                [float(r["performance"]["primary"]) for r in rows]
            ),
            "avg_cost_norm": safe_mean(
                [float(r["cost_norm_query"]) for r in rows]
            ),
            "avg_cost_proxy_money": safe_mean(
                [
                    float(r["cost_proxy_money"])
                    for r in rows
                    if r.get("cost_proxy_money") is not None
                ]
            ),
            "avg_tokens_total": safe_mean(
                [
                    float(r["cost"]["tokens_total"])
                    for r in rows
                    if isinstance(r.get("cost"), dict)
                    and r["cost"].get("tokens_total") is not None
                ]
            ),
            **{
                key: safe_mean(
                    [float(r[key]) for r in rows if r.get(key) is not None]
                )
                for key in LAMBDA_KEYS
            },
            "prompt_variants": [
                {
                    "prompt": r.get("prompt"),
                    "query_text": r.get("user_text"),
                    "performance": r["performance"]["primary"],
                    "cost_norm_query": r.get("cost_norm_query"),
                    "cost_proxy_money": r.get("cost_proxy_money"),
                    "tokens_total": (
                        r["cost"].get("tokens_total")
                        if isinstance(r.get("cost"), dict)
                        else None
                    ),
                    **{key: r.get(key) for key in LAMBDA_KEYS},
                }
                for r in rows
            ],
        }
        by_qid[qid].append(entry)

    output_rows = []
    for qid in sorted(by_qid):
        candidates = sorted(by_qid[qid], key=lambda item: item["model"])
        oracle_model_only = {}
        for key in LAMBDA_KEYS:
            valid = [candidate for candidate in candidates if candidate.get(key) is not None]
            if not valid:
                oracle_model_only[key] = None
                continue
            best = max(valid, key=lambda candidate: candidate[key])
            oracle_model_only[key] = {"model": best["model"], "reward": best[key]}

        output_rows.append(
            {
                "qid": qid,
                "task": candidates[0]["task"] if candidates else None,
                "num_models": len(candidates),
                "candidates": candidates,
                "oracle_model_only": oracle_model_only,
            }
        )
    return output_rows


def validate_rows(rows: list[dict], allowed_qids: set[str] | None) -> None:
    if not rows:
        raise ValueError("Model-only dataset is empty")

    output_qids = [str(row.get("qid")) for row in rows]
    if len(output_qids) != len(set(output_qids)):
        raise ValueError("Output contains duplicate qids")

    output_qid_set = set(output_qids)
    if allowed_qids is not None and output_qid_set != allowed_qids:
        missing = sorted(allowed_qids - output_qid_set)
        extra = sorted(output_qid_set - allowed_qids)
        raise ValueError(
            "Output qids do not match --qid-source "
            f"(missing={missing[:10]}, extra={extra[:10]})"
        )

    expected_models = set(EXPECTED_MODELS)
    errors = []
    for row in rows:
        qid = str(row["qid"])
        candidates = row.get("candidates", [])
        models = [candidate.get("model") for candidate in candidates]
        if len(models) != len(expected_models) or set(models) != expected_models:
            errors.append(
                f"{qid}: models={len(models)}, unique={len(set(models))}, "
                f"missing={sorted(expected_models - set(models))}, "
                f"unexpected={sorted(set(models) - expected_models)}"
            )
            continue
        for candidate in candidates:
            missing_rewards = [key for key in LAMBDA_KEYS if candidate.get(key) is None]
            if missing_rewards:
                errors.append(
                    f"{qid}/{candidate.get('model')}: missing rewards={missing_rewards}"
                )

    if errors:
        details = "\n".join(f"  - {error}" for error in errors[:20])
        suffix = f"\n  ... {len(errors) - 20} additional errors" if len(errors) > 20 else ""
        raise ValueError(f"Invalid model-only dataset:\n{details}{suffix}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--qid-source",
        type=Path,
        default=None,
        help="Grouped bipartite JSONL defining the exact qid population.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_qids = load_qid_allowlist(args.qid_source)
    rows = build_rows(args.input, allowed_qids)
    validate_rows(rows, allowed_qids)
    write_jsonl(args.output, rows)

    print("==== MODEL-ONLY ROUTER DATASET BUILT ====")
    print(f"Input: {args.input}")
    print(f"QID source: {args.qid_source}")
    print(f"Output: {args.output}")
    print(f"Queries: {len(rows)}")
    print(f"Models per query: {len(EXPECTED_MODELS)}")
    print("Population validation passed.")


if __name__ == "__main__":
    main()
