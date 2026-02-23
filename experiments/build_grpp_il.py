#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build GRPP Interaction Log (GRPP-IL) JSONL dataset from:
- configs/rq2_dataset_builder_smoke.yaml
- configs/prompt_templates.yaml
- configs/llm_candidates.json

Each JSONL row is one (task, qid, query/subquery, prompt, model) execution attempt.

Works with OpenAI-compatible endpoints (vLLM, Ollama OpenAI server, etc.):
POST {endpoint}/chat/completions
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from datasets import load_dataset


# -----------------------------
# Utils: time, io
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Normalization + metrics (QA)
# -----------------------------

_ARTICLES = {"a", "an", "the"}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def squad_f1_em(pred: str, gold_answers: List[str]) -> Tuple[float, float]:
    # take max over all golds (SQuAD convention)
    f1s = [f1_score(pred, g) for g in gold_answers] if gold_answers else [0.0]
    ems = [exact_match(pred, g) for g in gold_answers] if gold_answers else [0.0]
    return max(f1s), max(ems)

# -----------------------------
# ROUGE-L (simple)
# -----------------------------

def lcs_length(a: List[str], b: List[str]) -> int:
    # DP O(n*m) - fine for smoke sizes
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        row = dp[i]
        prev = dp[i-1]
        for j in range(1, m+1):
            if ai == b[j-1]:
                row[j] = prev[j-1] + 1
            else:
                row[j] = max(prev[j], row[j-1])
    return dp[n][m]

def rouge_l(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = lcs_length(pred_toks, gold_toks)
    prec = lcs / len(pred_toks)
    rec = lcs / len(gold_toks)
    if prec + rec == 0:
        return 0.0
    beta = (prec / rec) if rec > 0 else 0.0
    # Standard ROUGE-L F-measure often uses beta=1.2; we keep beta=1 for simplicity in smoke.
    return (2 * prec * rec) / (prec + rec)


# -----------------------------
# Parsing answers (task-specific)
# -----------------------------

FINAL_PAT = re.compile(r"(?:^|\n)\s*Final\s*:\s*(.*)$", re.IGNORECASE)

def extract_after_final(response: str) -> str:
    # take last occurrence of "Final:" line if present
    matches = list(FINAL_PAT.finditer(response))
    if matches:
        ans = matches[-1].group(1).strip()
        if ans:
            return ans
    return response.strip()

NUM_PAT = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_gsm8k_gold(gold: str) -> str:
    # GSM8K gold has "#### <answer>"
    if "####" in gold:
        return gold.split("####")[-1].strip()
    # fallback: last number
    nums = NUM_PAT.findall(gold)
    return nums[-1] if nums else gold.strip()

def extract_final_number(text: str) -> str:
    # try parse "Final:" block first, then last number in it
    t = extract_after_final(text)
    nums = NUM_PAT.findall(t.replace(",", ""))
    return nums[-1] if nums else t.strip()


# -----------------------------
# Prompt rendering
# -----------------------------

def make_context_block(context: Optional[str]) -> str:
    c = (context or "").strip()
    return f"Context:\n{c}\n\n" if c else ""

def render_prompt(template: str, subquery: str, context: Optional[str]) -> str:
    context_block = make_context_block(context)
    return template.format(subquery=subquery, context_block=context_block).strip() + "\n"


# -----------------------------
# OpenAI-compatible client
# -----------------------------

@dataclass
class ModelCandidate:
    key: str
    service: str
    model: str
    hf_id: str
    input_price: float
    output_price: float
    api_endpoint: str

def load_candidates(path: str) -> Dict[str, ModelCandidate]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, ModelCandidate] = {}
    for k, v in raw.items():
        out[k] = ModelCandidate(
            key=k,
            service=v.get("service", "Local"),
            model=v.get("model", k),
            hf_id=v.get("hf_id", k),
            input_price=float(v.get("input_price", 0.0)),
            output_price=float(v.get("output_price", 0.0)),
            api_endpoint=v.get("api_endpoint", ""),
        )
    return out

def get_served_model_ids(endpoint: str, timeout_s: int = 10) -> List[str]:
    try:
        r = requests.get(endpoint.rstrip("/") + "/models", timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        # OpenAI-style: {"data":[{"id":"..."}]}
        ids = []
        for item in data.get("data", []):
            mid = item.get("id")
            if mid:
                ids.append(mid)
        return ids
    except Exception:
        return []

def resolve_model_name_for_endpoint(candidate: ModelCandidate, endpoint: str) -> str:
    """
    Tries to pick a model name that the endpoint will accept.
    Strategy:
    - If /v1/models is available, choose an id matching hf_id or model; else fallback:
      - If endpoint equals candidate.api_endpoint -> use candidate.model (Ollama style)
      - Else -> use candidate.hf_id (vLLM style)
    """
    served = get_served_model_ids(endpoint.rstrip("/"), timeout_s=10)
    if served:
        # Prefer exact match against hf_id then model then candidate key.
        for want in (candidate.hf_id, candidate.model, candidate.key):
            if want in served:
                return want
        # If only one served model, use it (common for vLLM). Smoke will still run.
        if len(served) == 1:
            return served[0]
        # Fallback to hf_id
        return candidate.hf_id

    # No models endpoint: heuristic
    if candidate.api_endpoint and endpoint.rstrip("/") == candidate.api_endpoint.rstrip("/"):
        return candidate.model
    return candidate.hf_id

def chat_completion(
    endpoint: str,
    model_name: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> Tuple[str, Dict[str, Any]]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    # OpenAI-style: choices[0].message.content
    content = j["choices"][0]["message"]["content"]
    return content, j


# -----------------------------
# Dataset loading helpers
# -----------------------------

def make_qid(task: str, idx: int, row: Dict[str, Any]) -> str:
    # Prefer dataset-provided id if present
    for key in ("id", "_id", "qid", "question_id"):
        if key in row and row[key] is not None:
            return f"{task}_{row[key]}"
    return f"{task}_{idx:06d}"

def get_task_query(task_cfg: Dict[str, Any], row: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    # returns (query, context_opt)
    q = str(row.get(task_cfg["query_field"], "")).strip()

    # optional fields (alpaca)
    if "input_field" in task_cfg:
        inp = str(row.get(task_cfg["input_field"], "")).strip()
        if inp:
            q = f"{q}\n\nInput:\n{inp}"

    context = None
    if "context_field" in task_cfg and task_cfg["context_field"]:
        context = str(row.get(task_cfg["context_field"], "")).strip()

    return q, context

def get_gold(task: str, task_cfg: Dict[str, Any], row: Dict[str, Any]) -> Any:
    gf = task_cfg.get("gold_field")
    return row.get(gf) if gf else None


# -----------------------------
# Performance computation
# -----------------------------

def compute_performance(task: str, task_cfg: Dict[str, Any], parsed_answer: str, gold: Any) -> Tuple[Optional[float], str]:
    pm = task_cfg.get("primary_metric")
    metric_name = task_cfg.get("metric_name") or (pm.upper() if pm else "NA")

    if pm is None:
        return None, metric_name

    if task == "hotpotqa":
        # gold is a string answer
        gold_str = str(gold).strip()
        if pm.lower() == "f1":
            return float(f1_score(parsed_answer, gold_str)), metric_name
        if pm.lower() == "em":
            return float(exact_match(parsed_answer, gold_str)), metric_name
        # default f1
        return float(f1_score(parsed_answer, gold_str)), metric_name

    if task == "gsm8k" or task == "cs8k" or task == "cs8mk":
        gold_str = extract_gsm8k_gold(str(gold))
        pred_num = extract_final_number(parsed_answer)
        return (1.0 if normalize_text(pred_num) == normalize_text(gold_str) else 0.0), metric_name

    if task == "squad":
        # gold is dict: {"text":[...], "answer_start":[...]}
        gold_texts = []
        if isinstance(gold, dict):
            gold_texts = [str(x) for x in (gold.get("text") or [])]
        elif isinstance(gold, list):
            gold_texts = [str(x) for x in gold]
        else:
            gold_texts = [str(gold)]
        f1, em = squad_f1_em(parsed_answer, gold_texts)
        if pm.lower() == "em":
            return float(em), metric_name
        return float(f1), metric_name

    if task == "multinews":
        gold_str = str(gold).strip()
        # rougeL
        return float(rouge_l(parsed_answer, gold_str)), metric_name

    # fallback
    return None, metric_name


# -----------------------------
# Cost computation
# -----------------------------

def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cand: ModelCandidate,
) -> float:
    # prices are $ per 1M tokens
    return (input_tokens / 1e6) * cand.input_price + (output_tokens / 1e6) * cand.output_price


# -----------------------------
# Main build loop
# -----------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompt_templates(path: str) -> Dict[str, str]:
    y = load_yaml(path)
    templates = y.get("templates", {})
    if not isinstance(templates, dict) or not templates:
        raise ValueError(f"prompt_templates.yaml missing 'templates' dict: {path}")
    # ensure strings
    out: Dict[str, str] = {}
    for k, v in templates.items():
        out[k] = str(v)
    return out

def build_actions(models: List[str], prompts: List[str], max_units: int, seed: int) -> List[Tuple[str, str]]:
    # cartesian, then cap deterministically with seed
    all_actions = [(p, m) for p in prompts for m in models]
    if max_units is None or max_units <= 0 or max_units >= len(all_actions):
        return all_actions
    rng = random.Random(seed)
    rng.shuffle(all_actions)
    return all_actions[:max_units]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq2_dataset_builder_smoke.yaml", help="Path to builder YAML")
    ap.add_argument("--prompt_templates", default="configs/prompt_templates.yaml", help="Path to prompt templates YAML")
    ap.add_argument("--candidates", default="configs/llm_candidates.json", help="Path to llm_candidates.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    prompt_templates = load_prompt_templates(args.prompt_templates)
    candidates = load_candidates(args.candidates)

    out_jsonl = cfg["out_jsonl"]
    seed = int(cfg.get("seed", 0))
    random.seed(seed)

    split = cfg.get("split", "train")
    n_per_task = int(cfg.get("n_per_task", 10))
    max_units_per_query = int(cfg.get("max_units_per_query", 1))

    endpoint = cfg["endpoint"]
    timeout_s = int(cfg.get("timeout_s", 120))
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 256))

    lambda_cost = float(cfg.get("lambda_cost", 0.0))  # not required; logged only if you want
    cost_proxy = str(cfg.get("cost_proxy", "tokens_total"))

    model_keys: List[str] = list(cfg["pools"]["models"])
    prompt_ids: List[str] = list(cfg["pools"]["prompts"])

    # validate prompts exist
    for pid in prompt_ids:
        if pid not in prompt_templates:
            raise ValueError(f"Prompt id '{pid}' not found in {args.prompt_templates}")

    # validate model keys exist
    for mk in model_keys:
        if mk not in candidates:
            raise ValueError(f"Model key '{mk}' not found in {args.candidates}")

    tasks_cfg: Dict[str, Any] = cfg["tasks"]
    tasks_order = list(tasks_cfg.keys())

    actions = build_actions(model_keys, prompt_ids, max_units_per_query, seed)

    print(f"[build_grpp_il] out={out_jsonl} split={split} seed={seed}")
    print(f"[build_grpp_il] endpoint={endpoint} temp={temperature} max_tokens={max_tokens}")
    print(f"[build_grpp_il] tasks={tasks_order}")
    print(f"[build_grpp_il] prompts={prompt_ids}")
    print(f"[build_grpp_il] models={model_keys}")
    print(f"[build_grpp_il] actions_per_qid={len(actions)} (cap={max_units_per_query})")

    rows_to_write: List[Dict[str, Any]] = []

    for task_name in tasks_order:
        tcfg = tasks_cfg[task_name]
        hf_path = tcfg["hf_path"]
        hf_name = tcfg.get("hf_name", None)
        # Load HF dataset split if exists; else fallback to "train"
        try:
            ds = load_dataset(hf_path, hf_name, split=split)
        except Exception:
            ds = load_dataset(hf_path, hf_name, split="train")

        # sample first n_per_task deterministically
        indices = list(range(min(n_per_task, len(ds))))
        # (for real run you might want random sampling; smoke keeps deterministic)
        for local_idx, idx in enumerate(indices):
            ex = ds[int(idx)]
            qid = make_qid(task_name, int(idx), ex)

            query, context = get_task_query(tcfg, ex)

            # monolithic (as per spec): subquery=query
            subquery = query
            is_decomposed = False
            subquery_id = 0
            num_subqueries = 1

            gold = get_gold(task_name, tcfg, ex)

            for prompt_id, model_key in actions:
                cand = candidates[model_key]
                model_backend = cand.service

                # Resolve model name accepted by the endpoint
                model_name = resolve_model_name_for_endpoint(cand, endpoint)

                prompt_text = render_prompt(
                    template=prompt_templates[prompt_id],
                    subquery=subquery,
                    context=context,
                )

                status = "ok"
                error_msg = ""
                response_text = ""
                parsed_answer = ""
                performance = None
                metric_name = tcfg.get("metric_name") or (tcfg.get("primary_metric", "NA") or "NA")

                input_tokens = 0
                output_tokens = 0
                latency_s = None
                cost = None

                t0 = time.time()
                try:
                    response_text, raw = chat_completion(
                        endpoint=endpoint,
                        model_name=model_name,
                        prompt_text=prompt_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                    )
                    latency_s = round(time.time() - t0, 4)

                    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
                    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    output_tokens = int(usage.get("completion_tokens", 0) or 0)

                    # fallback token proxy if usage missing
                    if input_tokens == 0 and output_tokens == 0:
                        # crude fallback (smoke only)
                        input_tokens = max(1, len(prompt_text.split()) // 0.75)  # approx
                        output_tokens = max(1, len(response_text.split()) // 0.75)

                    # Parsed answer: prompt-agnostic default = after "Final:" if present
                    parsed_answer = extract_after_final(response_text)

                    # Task-specific tweaks
                    if task_name in ("gsm8k", "cs8k", "cs8mk"):
                        parsed_answer = extract_final_number(response_text)
                    elif task_name == "squad":
                        parsed_answer = extract_after_final(response_text)
                    elif task_name == "multinews":
                        parsed_answer = extract_after_final(response_text)

                    performance, metric_name = compute_performance(
                        task=task_name,
                        task_cfg=tcfg,
                        parsed_answer=parsed_answer,
                        gold=gold,
                    )

                    cost = compute_cost_usd(input_tokens, output_tokens, cand)

                except requests.exceptions.Timeout as e:
                    latency_s = round(time.time() - t0, 4)
                    status = "timeout"
                    error_msg = str(e)
                except Exception as e:
                    latency_s = round(time.time() - t0, 4)
                    status = "error"
                    error_msg = repr(e)

                row = {
                    "task": task_name,
                    "qid": qid,
                    "query": query,
                    "subquery": subquery,
                    "is_decomposed": is_decomposed,
                    "subquery_id": subquery_id,
                    "num_subqueries": num_subqueries,
                    "prompt": prompt_id,
                    "prompt_text": prompt_text,
                    "model": model_key,               # IMPORTANT: key from llm_candidates.json
                    "model_backend": model_backend,   # e.g., Local
                    "response": response_text,
                    "parsed_answer": parsed_answer,
                    "performance": performance,
                    "metric_name": metric_name,
                    "cost": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_s": latency_s,
                    "timestamp": utc_now_iso(),
                    "seed": seed,
                    "status": status,
                    "error_msg": error_msg,
                    # Helpful extras for debugging (optional)
                    "endpoint": endpoint,
                    "served_model_name": model_name,
                    "lambda_cost": lambda_cost,
                    "cost_proxy": cost_proxy,
                    "tokens_total": int(input_tokens + output_tokens),
                }

                rows_to_write.append(row)

        # write per task chunk (safer for long runs)
        if rows_to_write:
            write_jsonl(out_jsonl, rows_to_write)
            print(f"[build_grpp_il] wrote {len(rows_to_write)} rows (task={task_name}) -> {out_jsonl}")
            rows_to_write = []

    print("[build_grpp_il] done.")


if __name__ == "__main__":
    main()
