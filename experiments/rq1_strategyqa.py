#!/usr/bin/env python3
# experiments/rq1_strategyqa.py

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import load_dataset
from tqdm import tqdm


# ----------------------------
# Helpers: yes/no normalization
# ----------------------------
YES = {"yes", "y", "true"}
NO = {"no", "n", "false"}

def normalize_yesno(text: str) -> Optional[str]:
    """
    Map a free-form answer to {yes,no} if possible.
    Returns "yes", "no", or None.
    """
    if text is None:
        return None
    t = text.strip().lower()

    # common wrappers
    t = re.sub(r"^final answer\s*:\s*", "", t).strip()
    t = re.sub(r"^answer\s*:\s*", "", t).strip()

    # take first token/word if present
    first = re.split(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}]+", t)[0].strip()
    if first in YES:
        return "yes"
    if first in NO:
        return "no"

    # sometimes: "Yes, because ..."
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"

    return None

def accuracy(pred_label: Optional[str], gold_label: str) -> int:
    if pred_label is None:
        return 0
    return int(pred_label == gold_label)


# ----------------------------
# OpenAI-compatible call (vLLM)
# ----------------------------
def chat_completion(
    endpoint: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout)
    dt = time.time() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage")
    return text, dt, usage


# ----------------------------
# Decomposition parsing
# ----------------------------
def parse_json_list(text: str) -> List[str]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass

    lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if 3 <= len(ln) <= 220]
    return lines[:4]


# ----------------------------
# Prompts
# ----------------------------
def monolithic_messages(q: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Answer with only 'yes' or 'no'."},
        {"role": "user", "content": f"Question: {q}\nAnswer:"},
    ]

def decompose_messages(q: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You decompose complex questions into minimal sub-questions."},
        {"role": "user", "content":
            "Decompose the following question into 2-4 atomic sub-questions.\n"
            "Return ONLY a JSON array of strings.\n"
            f"Question: {q}"
        },
    ]

def subq_messages(sq: str) -> List[Dict[str, str]]:
    # keep sub-answers concise; they can be short facts used in aggregation
    return [
        {"role": "system", "content": "Answer concisely. If yes/no, answer yes/no; otherwise a short fact."},
        {"role": "user", "content": f"Sub-question: {sq}\nAnswer:"},
    ]

def aggregate_messages(q: str, subqs: List[str], answers: List[str]) -> List[Dict[str, str]]:
    pairs = "\n".join([f"{i+1}) {sq} -> {a}" for i, (sq, a) in enumerate(zip(subqs, answers))])
    return [
        {"role": "system", "content": "You must output ONLY 'yes' or 'no'."},
        {"role": "user", "content":
            f"Original question: {q}\n\n"
            f"Sub-questions and answers:\n{pairs}\n\n"
            "Based on the above, output ONLY 'yes' or 'no'."
        },
    ]


# ----------------------------
# Progress stats
# ----------------------------
@dataclass
class RunningStats:
    n_done: int = 0
    acc_mono: int = 0
    acc_decomp: int = 0

    n_fail_mono: int = 0
    n_fail_decomp: int = 0
    n_fail_sub: int = 0
    n_fail_agg: int = 0

    sum_latency_mono: float = 0.0
    sum_latency_decomp: float = 0.0
    sum_latency_sub: float = 0.0
    sum_latency_agg: float = 0.0
    sum_k: int = 0

    def update(self, row: Dict[str, Any]) -> None:
        self.n_done += 1
        self.acc_mono += int(row.get("acc_mono", 0))
        self.acc_decomp += int(row.get("acc_decomp", 0))

        self.sum_latency_mono += float(row.get("latency_mono_s", 0.0))
        self.sum_latency_decomp += float(row.get("latency_decomp_s", 0.0))
        self.sum_latency_agg += float(row.get("latency_agg_s", 0.0))
        self.sum_k += len(row.get("subqueries") or [])
        self.sum_latency_sub += sum(float(x) for x in (row.get("latency_sub_s") or []))

        if row.get("mono_error"): self.n_fail_mono += 1
        if row.get("decomp_error"): self.n_fail_decomp += 1
        if (row.get("sub_error_count", 0) or 0) > 0: self.n_fail_sub += 1
        if row.get("agg_error"): self.n_fail_agg += 1

    def snapshot(self) -> Dict[str, Any]:
        n = max(self.n_done, 1)
        return {
            "n_done": self.n_done,
            "acc_mono": self.acc_mono / n,
            "acc_decomp": self.acc_decomp / n,
            "avg_k": self.sum_k / n,
            "avg_latency_mono_s": self.sum_latency_mono / n,
            "avg_latency_decomp_s": self.sum_latency_decomp / n,
            "avg_latency_sub_s": self.sum_latency_sub / n,  # total sub latency per sample
            "avg_latency_agg_s": self.sum_latency_agg / n,
            "fail_rate_mono": self.n_fail_mono / n,
            "fail_rate_decomp": self.n_fail_decomp / n,
            "fail_rate_sub": self.n_fail_sub / n,
            "fail_rate_agg": self.n_fail_agg / n,
        }


def load_processed_ids(out_path: str) -> set:
    processed = set()
    if not os.path.exists(out_path):
        return processed
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    processed.add(str(obj["id"]))
            except Exception:
                continue
    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:18000/v1")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/rq1_strategyqa_pilot.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log_every", type=int, default=20)

    ap.add_argument("--max_tokens_mono", type=int, default=16)
    ap.add_argument("--max_tokens_sub", type=int, default=64)
    ap.add_argument("--max_tokens_agg", type=int, default=8)
    ap.add_argument("--max_subqueries", type=int, default=3)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load dataset (StrategyQA)
    ds = load_dataset("tasksource/strategy-qa", split=f"{args.split}[:{args.n}]").shuffle(seed=args.seed)

    processed_ids = load_processed_ids(args.out) if args.resume else set()
    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"

    stats = RunningStats()

    with open(args.out, mode, encoding="utf-8") as f:
        pbar = tqdm(total=len(ds), desc="RQ1 StrategyQA")

        for i, ex in enumerate(ds):
            ex_id = str(i)
            if args.resume and ex_id in processed_ids:
                pbar.update(1)
                continue

            q = ex["question"]
            gold = normalize_yesno(str(ex["answer"]))
            if gold is None:
                gold = "no"


            row: Dict[str, Any] = {
                "id": ex_id,
                "question": q,
                "gold": gold,
                "endpoint": args.endpoint,
                "model": args.model,
                "budgets": {
                    "max_tokens_mono": args.max_tokens_mono,
                    "max_tokens_sub": args.max_tokens_sub,
                    "max_tokens_agg": args.max_tokens_agg,
                    "max_subqueries": args.max_subqueries,
                },
            }

            # Monolithic
            try:
                pred_mono_raw, dt_mono, _u = chat_completion(
                    args.endpoint, args.model, monolithic_messages(q),
                    max_tokens=args.max_tokens_mono,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                pred_mono = normalize_yesno(pred_mono_raw)
                row["pred_mono_raw"] = pred_mono_raw
                row["pred_mono"] = pred_mono
                row["latency_mono_s"] = dt_mono
                row["mono_error"] = None
            except Exception as e:
                row["pred_mono_raw"] = ""
                row["pred_mono"] = None
                row["latency_mono_s"] = 0.0
                row["mono_error"] = str(e)

            row["acc_mono"] = accuracy(row["pred_mono"], gold)

            # Decompose
            subqs: List[str] = []
            try:
                dec_raw, dt_dec, _u = chat_completion(
                    args.endpoint, args.model, decompose_messages(q),
                    max_tokens=256,
                    temperature=0.0,
                    timeout=args.timeout,
                )
                subqs = parse_json_list(dec_raw)[: args.max_subqueries]
                row["decomposition_raw"] = dec_raw
                row["latency_decomp_s"] = dt_dec
                row["decomp_error"] = None
            except Exception as e:
                row["decomposition_raw"] = ""
                row["latency_decomp_s"] = 0.0
                row["decomp_error"] = str(e)
                subqs = []

            row["subqueries"] = subqs

            # Answer subqueries
            subanswers: List[str] = []
            sub_lat: List[float] = []
            sub_err = 0
            for sq in subqs:
                try:
                    a, dt_a, _u = chat_completion(
                        args.endpoint, args.model, subq_messages(sq),
                        max_tokens=args.max_tokens_sub,
                        temperature=args.temperature,
                        timeout=args.timeout,
                    )
                except Exception:
                    a, dt_a = "", 0.0
                    sub_err += 1
                subanswers.append(a)
                sub_lat.append(dt_a)

            row["subanswers"] = subanswers
            row["latency_sub_s"] = sub_lat
            row["sub_error_count"] = sub_err

            # Aggregate
            try:
                pred_dec_raw, dt_agg, _u = chat_completion(
                    args.endpoint, args.model, aggregate_messages(q, subqs, subanswers),
                    max_tokens=args.max_tokens_agg,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                pred_dec = normalize_yesno(pred_dec_raw)
                row["pred_decomp_raw"] = pred_dec_raw
                row["pred_decomp"] = pred_dec
                row["latency_agg_s"] = dt_agg
                row["agg_error"] = None
            except Exception as e:
                row["pred_decomp_raw"] = ""
                row["pred_decomp"] = None
                row["latency_agg_s"] = 0.0
                row["agg_error"] = str(e)

            row["acc_decomp"] = accuracy(row["pred_decomp"], gold)

            # write + stats
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            stats.update(row)
            pbar.update(1)

            if stats.n_done % args.log_every == 0:
                snap = stats.snapshot()
                pbar.set_postfix({
                    "Acc_m": f"{snap['acc_mono']:.2f}",
                    "Acc_d": f"{snap['acc_decomp']:.2f}",
                    "k": f"{snap['avg_k']:.1f}",
                    "fail": f"{snap['fail_rate_decomp']:.2f}",
                })
                print("\n[progress]", json.dumps(snap, indent=2))

        pbar.close()

    final = stats.snapshot()
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(final, sf, indent=2)

    print("\n==== RQ1 Final Summary (StrategyQA) ====")
    print(json.dumps(final, indent=2))
    print(f"Saved: {args.out}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
