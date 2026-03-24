#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import requests
from datasets import load_dataset


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_last_number(text: str) -> Optional[str]:
    if text is None:
        return None
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_gold_number(answer_text: str) -> Optional[str]:
    if answer_text is None:
        return None

    if "####" in answer_text:
        tail = answer_text.split("####")[-1].strip()
        num = extract_last_number(tail)
        if num is not None:
            return num

    return extract_last_number(answer_text)


def numeric_em(pred: str, gold: str) -> float:
    p = extract_last_number(pred)
    g = extract_gold_number(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if p == g else 0.0


def clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def build_monolithic_messages(question: str) -> List[Dict[str, str]]:
    user = (
        "Solve the math word problem carefully. "
        "Return only the final numeric answer.\n\n"
        f"Problem:\n{question}\n\n"
        "Final numeric answer:"
    )
    return [{"role": "user", "content": user}]


def build_decomposition(question: str) -> List[str]:
    q = question.strip()
    return [
        f"Identify the key quantities and operations needed to solve: {q}",
        f"Compute the intermediate steps needed to solve: {q}",
        f"Using those steps, what is the final numeric answer to: {q}",
    ]


def build_subquery_messages(question: str, subquery: str) -> List[Dict[str, str]]:
    user = (
        "Answer the sub-question briefly and clearly.\n\n"
        f"Original problem:\n{question}\n\n"
        f"Sub-question:\n{subquery}\n\n"
        "Sub-answer:"
    )
    return [{"role": "user", "content": user}]


def build_aggregator_messages(question: str, subqueries: List[str], subanswers: List[str]) -> List[Dict[str, str]]:
    joined = []
    for i, (sq, sa) in enumerate(zip(subqueries, subanswers), start=1):
        joined.append(f"{i}. {sq}\nAnswer: {clip_text(sa, 180)}")

    user = (
        "Use the sub-answers to solve the original math problem. "
        "Return only the final numeric answer.\n\n"
        f"Original problem:\n{question}\n\n"
        f"{chr(10).join(joined)}\n\n"
        "Final numeric answer:"
    )
    return [{"role": "user", "content": user}]


def chat_completion(
    endpoint: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    timeout_s: int,
    temperature: float = 0.0,
) -> Tuple[str, dict]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    r = requests.post(url, json=payload, timeout=timeout_s)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    return text, data


def usage_to_counts(raw_response: dict) -> Dict[str, int]:
    usage = raw_response.get("usage", {}) or {}
    tin = int(usage.get("prompt_tokens", 0) or 0)
    tout = int(usage.get("completion_tokens", 0) or 0)
    return {
        "tokens_in": tin,
        "tokens_out": tout,
        "tokens_total": tin + tout,
    }


def run_one(
    example: dict,
    endpoint: str,
    model: str,
    mono_max_tokens: int,
    sub_max_tokens: int,
    agg_max_tokens: int,
    timeout_s: int,
    max_question_chars: int,
) -> dict:
    question = clip_text(str(example["question"]).strip(), max_question_chars)
    gold = str(example["answer"]).strip()

    row = {
        "question": question,
        "gold": gold,
        "gold_number": extract_gold_number(gold),
        "status": "ok",
        "error_msg": None,
        "monolithic": None,
        "decomposed": None,
    }

    try:
        mono_messages = build_monolithic_messages(question)
        mono_pred, mono_raw = chat_completion(
            endpoint=endpoint,
            model=model,
            messages=mono_messages,
            max_tokens=mono_max_tokens,
            timeout_s=timeout_s,
            temperature=0.0,
        )
        mono_counts = usage_to_counts(mono_raw)

        row["monolithic"] = {
            "prediction": mono_pred,
            "pred_number": extract_last_number(mono_pred),
            "em": numeric_em(mono_pred, gold),
            **mono_counts,
            "raw_response": mono_raw,
        }

        subqueries = build_decomposition(question)
        subanswers = []
        sub_raw = []
        sub_token_in = 0
        sub_token_out = 0

        for sq in subqueries:
            sq_messages = build_subquery_messages(question, sq)
            sa, sa_raw = chat_completion(
                endpoint=endpoint,
                model=model,
                messages=sq_messages,
                max_tokens=sub_max_tokens,
                timeout_s=timeout_s,
                temperature=0.0,
            )
            subanswers.append(sa)
            sub_raw.append(sa_raw)

            cnt = usage_to_counts(sa_raw)
            sub_token_in += cnt["tokens_in"]
            sub_token_out += cnt["tokens_out"]

        agg_messages = build_aggregator_messages(question, subqueries, subanswers)
        final_pred, final_raw = chat_completion(
            endpoint=endpoint,
            model=model,
            messages=agg_messages,
            max_tokens=agg_max_tokens,
            timeout_s=timeout_s,
            temperature=0.0,
        )
        agg_counts = usage_to_counts(final_raw)

        row["decomposed"] = {
            "subqueries": subqueries,
            "subanswers": subanswers,
            "sub_raw_responses": sub_raw,
            "prediction": final_pred,
            "pred_number": extract_last_number(final_pred),
            "em": numeric_em(final_pred, gold),
            "tokens_in": sub_token_in + agg_counts["tokens_in"],
            "tokens_out": sub_token_out + agg_counts["tokens_out"],
            "tokens_total": (sub_token_in + agg_counts["tokens_in"]) + (sub_token_out + agg_counts["tokens_out"]),
            "raw_response": final_raw,
        }

    except Exception as e:
        row["status"] = "failed"
        row["error_msg"] = str(e)

    return row


def summarize(rows: List[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("monolithic") and r.get("decomposed")]
    failed = [r for r in rows if r.get("status") != "ok"]

    if not ok:
        return {
            "n_ok": 0,
            "n_failed": len(failed),
            "monolithic_em": None,
            "decomposed_em": None,
            "delta_em": None,
            "monolithic_tokens": None,
            "decomposed_tokens": None,
            "delta_tokens": None,
            "tokens_ratio": None,
            "improved": 0,
            "tied": 0,
            "worsened": 0,
        }

    mono_em = mean(r["monolithic"]["em"] for r in ok)
    decomp_em = mean(r["decomposed"]["em"] for r in ok)

    mono_tokens = mean(r["monolithic"]["tokens_total"] for r in ok)
    decomp_tokens = mean(r["decomposed"]["tokens_total"] for r in ok)

    improved = 0
    tied = 0
    worsened = 0
    for r in ok:
        dm = r["decomposed"]["em"] - r["monolithic"]["em"]
        if dm > 0:
            improved += 1
        elif dm < 0:
            worsened += 1
        else:
            tied += 1

    return {
        "n_ok": len(ok),
        "n_failed": len(failed),
        "monolithic_em": mono_em,
        "decomposed_em": decomp_em,
        "delta_em": decomp_em - mono_em,
        "monolithic_tokens": mono_tokens,
        "decomposed_tokens": decomp_tokens,
        "delta_tokens": decomp_tokens - mono_tokens,
        "tokens_ratio": (decomp_tokens / mono_tokens) if mono_tokens > 0 else None,
        "improved": improved,
        "tied": tied,
        "worsened": worsened,
    }


def parse_run_specs(run_specs: List[str]) -> List[Dict[str, str]]:
    parsed = []
    for item in run_specs:
        parts = item.split("|")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --run '{item}'. Expected format: <tag>|<endpoint>|<model>"
            )
        tag, endpoint, model = parts
        parsed.append({
            "tag": tag.strip(),
            "endpoint": endpoint.strip(),
            "model": model.strip(),
        })
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help='Format: "<tag>|<endpoint>|<model>"',
    )
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--mono_max_tokens", type=int, default=24)
    ap.add_argument("--sub_max_tokens", type=int, default=24)
    ap.add_argument("--agg_max_tokens", type=int, default=16)
    ap.add_argument("--max_question_chars", type=int, default=500)
    ap.add_argument("--out_prefix", type=str, required=True)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    runs = parse_run_specs(args.run)

    ds = load_dataset("gsm8k", "main")
    split = ds[args.split]

    indices = list(range(len(split)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    indices = indices[: args.n]

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for run_cfg in runs:
        tag = run_cfg["tag"]
        endpoint = run_cfg["endpoint"]
        model = run_cfg["model"]

        rows = []
        out_jsonl = out_prefix.parent / f"{out_prefix.name}__{tag}.jsonl"

        print(f"\n==== RUNNING {tag} ====")
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model}")
        print(f"Output: {out_jsonl}")

        with out_jsonl.open("w", encoding="utf-8") as f:
            for i, idx in enumerate(indices, start=1):
                ex = split[idx]
                row = run_one(
                    example=ex,
                    endpoint=endpoint,
                    model=model,
                    mono_max_tokens=args.mono_max_tokens,
                    sub_max_tokens=args.sub_max_tokens,
                    agg_max_tokens=args.agg_max_tokens,
                    timeout_s=args.timeout_s,
                    max_question_chars=args.max_question_chars,
                )
                row["tag"] = tag
                row["endpoint"] = endpoint
                row["model"] = model

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)

                if i % args.log_every == 0:
                    print(f"[{i}/{len(indices)}] processed for {tag}")

        summary = summarize(rows)
        summary.update({
            "tag": tag,
            "endpoint": endpoint,
            "model": model,
            "n_requested": len(indices),
            "split": args.split,
            "seed": args.seed,
            "jsonl_path": str(out_jsonl),
        })
        all_summaries.append(summary)

        print("\n==== GSM8K SUMMARY ====")
        print(f"Tag: {tag}")
        print(f"N successful: {summary['n_ok']}")
        print(f"N failed: {summary['n_failed']}")
        if summary["n_ok"] > 0:
            print(f"Monolithic EM: {summary['monolithic_em']:.4f}")
            print(f"Decomposed EM: {summary['decomposed_em']:.4f}")
            print(f"Delta EM: {summary['delta_em']:.4f}")
            print(f"Monolithic avg tokens: {summary['monolithic_tokens']:.2f}")
            print(f"Decomposed avg tokens: {summary['decomposed_tokens']:.2f}")
            print(f"Delta avg tokens: {summary['delta_tokens']:.2f}")
            print(f"Token ratio: {summary['tokens_ratio']:.2f}x")
            print(f"Improved: {summary['improved']}")
            print(f"Tied: {summary['tied']}")
            print(f"Worsened: {summary['worsened']}")
            print(
                f"Overleaf row:\n"
                f"GSM8K (RQ1, {tag}) & {summary['monolithic_em']:.3f} & "
                f"{summary['decomposed_em']:.3f} & {summary['delta_em']:.3f} & "
                f"{summary['monolithic_tokens']:.1f} & {summary['decomposed_tokens']:.1f} & "
                f"{summary['tokens_ratio']:.2f}x \\\\"
            )
        else:
            print("No successful rows.")

    out_summary = out_prefix.parent / f"{out_prefix.name}__summary.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved summary: {out_summary}")


if __name__ == "__main__":
    main()