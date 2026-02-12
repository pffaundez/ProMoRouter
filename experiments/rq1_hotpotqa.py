#!/usr/bin/env python3
# experiments/rq1_hotpotqa.py

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
# Text utils + metrics
# ----------------------------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)  # keep alnum + spaces
    return s

def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    counts = {}
    for t in pred_toks:
        counts[t] = counts.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if counts.get(t, 0) > 0:
            num_same += 1
            counts[t] -= 1
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


# ----------------------------
# OpenAI-compatible call
# ----------------------------
def chat_completion(
    endpoint: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    """
    Returns: (text, latency_seconds, usage_dict_or_none)
    """
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
    usage = data.get("usage")  # vLLM may or may not provide this depending on version/config
    return text, dt, usage


# ----------------------------
# Prompts
# ----------------------------
def monolithic_messages(q: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. Answer the question concisely."},
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
    return [
        {"role": "system", "content": "You are a helpful assistant. Answer the question concisely."},
        {"role": "user", "content": f"Sub-question: {sq}\nAnswer:"},
    ]

def aggregate_messages(q: str, subqs: List[str], answers: List[str]) -> List[Dict[str, str]]:
    pairs = "\n".join([f"{i+1}) {sq} -> {a}" for i, (sq, a) in enumerate(zip(subqs, answers))])
    return [
        {"role": "system", "content": "You combine sub-answers into the final answer."},
        {"role": "user", "content":
            f"Original question: {q}\n\n"
            f"Sub-questions and answers:\n{pairs}\n\n"
            "Return ONLY the final answer."
        },
    ]


# ----------------------------
# Decomposition parsing
# ----------------------------
def parse_json_list(text: str) -> List[str]:
    """
    Expect a JSON array of strings. If the model wraps it with text, try to recover.
    """
    text = text.strip()
    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    # recover first [...] block
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass

    # fallback: bullet/lines
    lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if 3 <= len(ln) <= 220]
    return lines[:4]


# ----------------------------
# Progress stats
# ----------------------------
@dataclass
class RunningStats:
    n_done: int = 0
    n_fail_mono: int = 0
    n_fail_decomp: int = 0
    n_fail_sub: int = 0
    n_fail_agg: int = 0

    sum_em_mono: int = 0
    sum_em_decomp: int = 0
    sum_f1_mono: float = 0.0
    sum_f1_decomp: float = 0.0

    sum_latency_mono: float = 0.0
    sum_latency_decomp: float = 0.0
    sum_latency_sub: float = 0.0
    sum_latency_agg: float = 0.0

    sum_k: int = 0

    def update(self, row: Dict[str, Any]) -> None:
        self.n_done += 1
        self.sum_em_mono += int(row.get("em_mono", 0))
        self.sum_em_decomp += int(row.get("em_decomp", 0))
        self.sum_f1_mono += float(row.get("f1_mono", 0.0))
        self.sum_f1_decomp += float(row.get("f1_decomp", 0.0))

        self.sum_latency_mono += float(row.get("latency_mono_s", 0.0))
        self.sum_latency_decomp += float(row.get("latency_decomp_s", 0.0))
        self.sum_latency_agg += float(row.get("latency_agg_s", 0.0))

        sub_lat = row.get("latency_sub_s") or []
        self.sum_latency_sub += sum(float(x) for x in sub_lat)

        k = len(row.get("subqueries") or [])
        self.sum_k += k

        if row.get("mono_error"):
            self.n_fail_mono += 1
        if row.get("decomp_error"):
            self.n_fail_decomp += 1
        if row.get("sub_error_count", 0) > 0:
            self.n_fail_sub += 1
        if row.get("agg_error"):
            self.n_fail_agg += 1

    def snapshot(self) -> Dict[str, Any]:
        n = max(self.n_done, 1)
        return {
            "n_done": self.n_done,
            "em_mono": self.sum_em_mono / n,
            "f1_mono": self.sum_f1_mono / n,
            "em_decomp": self.sum_em_decomp / n,
            "f1_decomp": self.sum_f1_decomp / n,
            "avg_k": self.sum_k / n,
            "avg_latency_mono_s": self.sum_latency_mono / n,
            "avg_latency_decomp_s": self.sum_latency_decomp / n,
            "avg_latency_sub_s": self.sum_latency_sub / n,  # per sample total sub latency
            "avg_latency_agg_s": self.sum_latency_agg / n,
            "fail_rate_mono": self.n_fail_mono / n,
            "fail_rate_decomp": self.n_fail_decomp / n,
            "fail_rate_sub": self.n_fail_sub / n,
            "fail_rate_agg": self.n_fail_agg / n,
        }


# ----------------------------
# Resume helpers
# ----------------------------
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
    ap.add_argument("--n", type=int, default=20, help="number of samples from split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/rq1_smoke.jsonl")
    ap.add_argument("--resume", action="store_true", help="resume if out file exists")
    ap.add_argument("--log_every", type=int, default=10, help="print progress every N samples")

    # budgets
    ap.add_argument("--max_tokens_mono", type=int, default=256)
    ap.add_argument("--max_tokens_sub", type=int, default=96)
    ap.add_argument("--max_tokens_agg", type=int, default=64)
    ap.add_argument("--max_subqueries", type=int, default=3)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load dataset
    ds = load_dataset("hotpot_qa", "distractor", split=f"{args.split}[:{args.n}]")
    ds = ds.shuffle(seed=args.seed)

    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(args.out)

    stats = RunningStats()

    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        pbar = tqdm(total=len(ds), desc="RQ1 HotpotQA")

        for i, ex in enumerate(ds):
            ex_id = str(i)  # stable per run; OK for quick RQ1
            if args.resume and ex_id in processed_ids:
                pbar.update(1)
                continue

            q = ex["question"]
            gold = ex["answer"]

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

            # ---- Monolithic
            try:
                pred_mono, dt_mono, usage_mono = chat_completion(
                    args.endpoint, args.model, monolithic_messages(q),
                    max_tokens=args.max_tokens_mono,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                row["pred_mono"] = pred_mono
                row["latency_mono_s"] = dt_mono
                row["usage_mono"] = usage_mono
                row["mono_error"] = None
            except Exception as e:
                row["pred_mono"] = ""
                row["latency_mono_s"] = 0.0
                row["usage_mono"] = None
                row["mono_error"] = str(e)

            row["em_mono"] = exact_match(row["pred_mono"], gold)
            row["f1_mono"] = f1_score(row["pred_mono"], gold)

            # ---- Decompose
            subqs: List[str] = []
            try:
                decomp_text, dt_dec, usage_dec = chat_completion(
                    args.endpoint, args.model, decompose_messages(q),
                    max_tokens=256,
                    temperature=0.0,  # keep deterministic for parsing
                    timeout=args.timeout,
                )
                subqs = parse_json_list(decomp_text)[: args.max_subqueries]
                row["decomposition_raw"] = decomp_text
                row["latency_decomp_s"] = dt_dec
                row["usage_decomp"] = usage_dec
                row["decomp_error"] = None
            except Exception as e:
                row["decomposition_raw"] = ""
                row["latency_decomp_s"] = 0.0
                row["usage_decomp"] = None
                row["decomp_error"] = str(e)
                subqs = []

            row["subqueries"] = subqs

            # ---- Answer subqueries
            subanswers: List[str] = []
            sub_lat: List[float] = []
            sub_err = 0
            for sq in subqs:
                try:
                    a, dt_a, _usage_a = chat_completion(
                        args.endpoint, args.model, subq_messages(sq),
                        max_tokens=args.max_tokens_sub,
                        temperature=args.temperature,
                        timeout=args.timeout,
                    )
                except Exception as e:
                    a, dt_a = "", 0.0
                    sub_err += 1
                subanswers.append(a)
                sub_lat.append(dt_a)

            row["subanswers"] = subanswers
            row["latency_sub_s"] = sub_lat
            row["sub_error_count"] = sub_err

            # ---- Aggregate
            try:
                pred_decomp, dt_agg, usage_agg = chat_completion(
                    args.endpoint, args.model, aggregate_messages(q, subqs, subanswers),
                    max_tokens=args.max_tokens_agg,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                row["pred_decomp"] = pred_decomp
                row["latency_agg_s"] = dt_agg
                row["usage_agg"] = usage_agg
                row["agg_error"] = None
            except Exception as e:
                row["pred_decomp"] = ""
                row["latency_agg_s"] = 0.0
                row["usage_agg"] = None
                row["agg_error"] = str(e)

            row["em_decomp"] = exact_match(row["pred_decomp"], gold)
            row["f1_decomp"] = f1_score(row["pred_decomp"], gold)

            # write + update stats
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            stats.update(row)
            pbar.update(1)

            # live progress
            if stats.n_done % args.log_every == 0:
                snap = stats.snapshot()
                pbar.set_postfix({
                    "EM_m": f"{snap['em_mono']:.2f}",
                    "EM_d": f"{snap['em_decomp']:.2f}",
                    "F1_m": f"{snap['f1_mono']:.2f}",
                    "F1_d": f"{snap['f1_decomp']:.2f}",
                    "k": f"{snap['avg_k']:.1f}",
                    "fail": f"{(snap['fail_rate_decomp']):.2f}",
                })
                print("\n[progress]", json.dumps(snap, indent=2))

        pbar.close()

    # final summary
    final = stats.snapshot()
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(final, sf, indent=2)

    print("\n==== RQ1 Final Summary ====")
    print(json.dumps(final, indent=2))
    print(f"Saved: {args.out}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
