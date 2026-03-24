#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from datasets import load_dataset


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def f1_token(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    counts = {}
    for w in p:
        counts[w] = counts.get(w, 0) + 1

    common = 0
    for w in g:
        if counts.get(w, 0) > 0:
            common += 1
            counts[w] -= 1

    if common == 0:
        return 0.0

    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def clip_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def safe_get_context(example: dict) -> str:
    ctx = example.get("context", "")
    if isinstance(ctx, str):
        return ctx

    if isinstance(ctx, dict):
        titles = ctx.get("title", []) or []
        sentences = ctx.get("sentences", []) or []
        parts = []
        for title, sent_list in zip(titles, sentences):
            if isinstance(sent_list, list):
                sent_text = " ".join(str(x) for x in sent_list)
            else:
                sent_text = str(sent_list)
            parts.append(f"{title}: {sent_text}")
        return "\n\n".join(parts)

    return str(ctx)


def build_short_context(example: dict, max_docs: int = 2, max_sents_per_doc: int = 2, max_chars: int = 900) -> str:
    ctx = example.get("context", {})
    if not isinstance(ctx, dict):
        return clip_text(safe_get_context(example), max_chars)

    titles = ctx.get("title", []) or []
    sentences = ctx.get("sentences", []) or []

    parts = []
    for title, sent_list in list(zip(titles, sentences))[:max_docs]:
        if not isinstance(sent_list, list):
            sent_list = [str(sent_list)]
        short_sents = [str(s).strip() for s in sent_list[:max_sents_per_doc] if str(s).strip()]
        if short_sents:
            parts.append(f"{title}: {' '.join(short_sents)}")

    return clip_text("\n\n".join(parts), max_chars)


def build_monolithic_messages(question: str, context: str) -> List[Dict[str, str]]:
    user = (
        "Answer the question using the context. "
        "Return only a short final answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    return [{"role": "user", "content": user}]


def build_decomposition(question: str) -> List[str]:
    q = question.strip()
    return [
        f"What key intermediate fact is needed to answer: {q}",
        f"Using that intermediate fact, what is the final answer to: {q}",
    ]


def build_subquery_messages(question: str, subquery: str, context: str) -> List[Dict[str, str]]:
    user = (
        "Answer the sub-question briefly using the context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"Sub-question:\n{subquery}\n\n"
        "Sub-answer:"
    )
    return [{"role": "user", "content": user}]


def build_aggregator_messages(question: str, subqueries: List[str], subanswers: List[str]) -> List[Dict[str, str]]:
    joined = []
    for i, (sq, sa) in enumerate(zip(subqueries, subanswers), start=1):
        joined.append(f"{i}. {sq}\nAnswer: {clip_text(sa, 120)}")

    user = (
        "Use the sub-answers to answer the original question. "
        "Return only a short final answer.\n\n"
        f"Original question:\n{question}\n\n"
        f"{chr(10).join(joined)}\n\n"
        "Final answer:"
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


def run_one(
    example: dict,
    endpoint: str,
    model: str,
    mono_max_tokens: int,
    sub_max_tokens: int,
    agg_max_tokens: int,
    timeout_s: int,
    max_docs: int,
    max_sents_per_doc: int,
    max_context_chars: int,
) -> dict:
    question = clip_text(str(example["question"]).strip(), 240)
    gold = str(example["answer"]).strip()
    context = build_short_context(
        example,
        max_docs=max_docs,
        max_sents_per_doc=max_sents_per_doc,
        max_chars=max_context_chars,
    )

    row = {
        "qid": example.get("id", None),
        "task": "hotpotqa",
        "question": question,
        "gold": gold,
        "context": context,
        "status": "ok",
        "error_msg": None,
        "monolithic": None,
        "decomposed": None,
    }

    try:
        mono_messages = build_monolithic_messages(question, context)
        mono_pred, mono_raw = chat_completion(
            endpoint=endpoint,
            model=model,
            messages=mono_messages,
            max_tokens=mono_max_tokens,
            timeout_s=timeout_s,
            temperature=0.0,
        )

        row["monolithic"] = {
            "prediction": mono_pred,
            "em": exact_match(mono_pred, gold),
            "f1": f1_token(mono_pred, gold),
            "raw_response": mono_raw,
        }

        subqueries = build_decomposition(question)
        subanswers = []
        sub_raw = []

        for sq in subqueries:
            sq_messages = build_subquery_messages(question, sq, context)
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

        agg_messages = build_aggregator_messages(question, subqueries, subanswers)
        final_pred, final_raw = chat_completion(
            endpoint=endpoint,
            model=model,
            messages=agg_messages,
            max_tokens=agg_max_tokens,
            timeout_s=timeout_s,
            temperature=0.0,
        )

        row["decomposed"] = {
            "subqueries": subqueries,
            "subanswers": subanswers,
            "sub_raw_responses": sub_raw,
            "prediction": final_pred,
            "em": exact_match(final_pred, gold),
            "f1": f1_token(final_pred, gold),
            "raw_response": final_raw,
        }

    except Exception as e:
        row["status"] = "failed"
        row["error_msg"] = str(e)

    return row


def summarize(rows: List[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("monolithic") and r.get("decomposed")]
    if not ok_rows:
        return {
            "n": 0,
            "monolithic_em": None,
            "decomposed_em": None,
            "delta_em": None,
            "monolithic_f1": None,
            "decomposed_f1": None,
            "delta_f1": None,
        }

    mono_em = sum(r["monolithic"]["em"] for r in ok_rows) / len(ok_rows)
    decomp_em = sum(r["decomposed"]["em"] for r in ok_rows) / len(ok_rows)
    mono_f1 = sum(r["monolithic"]["f1"] for r in ok_rows) / len(ok_rows)
    decomp_f1 = sum(r["decomposed"]["f1"] for r in ok_rows) / len(ok_rows)

    return {
        "n": len(ok_rows),
        "monolithic_em": mono_em,
        "decomposed_em": decomp_em,
        "delta_em": decomp_em - mono_em,
        "monolithic_f1": mono_f1,
        "decomposed_f1": decomp_f1,
        "delta_f1": decomp_f1 - mono_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--mono_max_tokens", type=int, default=32)
    ap.add_argument("--sub_max_tokens", type=int, default=24)
    ap.add_argument("--agg_max_tokens", type=int, default=24)
    ap.add_argument("--max_docs", type=int, default=2)
    ap.add_argument("--max_sents_per_doc", type=int, default=2)
    ap.add_argument("--max_context_chars", type=int, default=900)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    split = ds[args.split]

    indices = list(range(len(split)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    indices = indices[: args.n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, idx in enumerate(indices, start=1):
            ex = split[idx]
            row = run_one(
                example=ex,
                endpoint=args.endpoint,
                model=args.model,
                mono_max_tokens=args.mono_max_tokens,
                sub_max_tokens=args.sub_max_tokens,
                agg_max_tokens=args.agg_max_tokens,
                timeout_s=args.timeout_s,
                max_docs=args.max_docs,
                max_sents_per_doc=args.max_sents_per_doc,
                max_context_chars=args.max_context_chars,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

            if i % args.log_every == 0:
                print(f"[{i}/{len(indices)}] processed")

    summary = summarize(rows)

    print("\n==== RQ1 HOTPOTQA COMPARE ====")
    print(f"Model: {args.model}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Split: {args.split}")
    print(f"N requested: {args.n}")
    print(f"N successful: {summary['n']}")
    print(f"Output: {out_path}")

    if summary["n"] > 0:
        print(f"Monolithic EM: {summary['monolithic_em']:.4f}")
        print(f"Decomposed EM: {summary['decomposed_em']:.4f}")
        print(f"Delta EM: {summary['delta_em']:.4f}")
        print(f"Monolithic F1: {summary['monolithic_f1']:.4f}")
        print(f"Decomposed F1: {summary['decomposed_f1']:.4f}")
        print(f"Delta F1: {summary['delta_f1']:.4f}")
    else:
        print("No successful rows.")


if __name__ == "__main__":
    main()