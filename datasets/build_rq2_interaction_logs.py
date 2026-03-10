import os
import json
import time
import random
import re
from typing import Dict, Any, List, Optional, Tuple

import yaml
import requests
from datasets import load_dataset


# -------- Prompt strategies (IDs must match YAML pools.prompts) --------
PROMPTS = {
    "direct": {
        "system": "You are a helpful assistant. Answer concisely and accurately.",
        "user": "{q}",
    },
    "cot": {
        "system": "You are a helpful assistant. Think step by step internally, then provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):",
    },
    "decompose": {
        "system": "You are a helpful assistant. Decompose the problem internally, then provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):",
    },
    "selfcheck": {
        "system": "You are a helpful assistant. Answer, self-check briefly, and provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):",
    },
}


# ----------------- Normalization / metrics -----------------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def f1_token(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    common: Dict[str, int] = {}
    for w in p:
        common[w] = common.get(w, 0) + 1

    num_same = 0
    for w in g:
        if common.get(w, 0) > 0:
            num_same += 1
            common[w] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def acc_numeric(pred: str, gold: str) -> float:
    """Simple numeric extraction for GSM8K-like answers."""

    def extract_num(x: str) -> Optional[str]:
        x = (x or "").strip().replace(",", "")
        m = re.findall(r"-?\d+(?:\.\d+)?", x)
        return m[-1] if m else None

    pn = extract_num(pred)
    gn = extract_num(gold)
    return 1.0 if (pn is not None and gn is not None and pn == gn) else 0.0


# ----------------- OpenAI-compatible client (vLLM / OpenAI-like) -----------------
def chat_completion(
    endpoint: str,
    served_model_name: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> Tuple[str, float, int, int]:
    url = endpoint.rstrip("/") + "/chat/completions"

    payload = {
        "model": served_model_name,
        "messages": [
            {"role": "user", "content": f"{system}\n\n{user}"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()

    r = requests.post(url, json=payload, timeout=timeout_s)
    latency = time.time() - t0

    if not r.ok:
        raise RuntimeError(
            f"HTTP {r.status_code} for {url}. Response body: {r.text}"
        )

    data = r.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    tin = int(usage.get("prompt_tokens", 0))
    tout = int(usage.get("completion_tokens", 0))
    return text, latency, tin, tout


def text_completion(
    endpoint: str,
    served_model_name: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> Tuple[str, float, int, int]:
    url = endpoint.rstrip("/") + "/completions"
    payload = {
        "model": served_model_name,
        "prompt": prompt_text,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()

    r = requests.post(url, json=payload, timeout=timeout_s)
    latency = time.time() - t0

    if not r.ok:
        raise RuntimeError(
            f"HTTP {r.status_code} for {url}. Response body: {r.text}"
        )

    data = r.json()

    text = data["choices"][0]["text"]
    usage = data.get("usage", {}) or {}
    tin = int(usage.get("prompt_tokens", 0))
    tout = int(usage.get("completion_tokens", 0))
    return text, latency, tin, tout


def generate_response(
    endpoint: str,
    api_mode: str,
    served_model_name: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> Tuple[str, float, int, int]:
    api_mode = (api_mode or "chat").strip().lower()

    if api_mode == "chat":
        return chat_completion(
            endpoint=endpoint,
            served_model_name=served_model_name,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    if api_mode == "completion":
        prompt_text = f"{system}\n\n{user}"
        return text_completion(
            endpoint=endpoint,
            served_model_name=served_model_name,
            prompt_text=prompt_text,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    raise ValueError(f"Unsupported api_mode: {api_mode}")


# ----------------- Reward -----------------
def compute_reward(primary: Optional[float], tokens_total: int, lam: float) -> Optional[float]:
    if primary is None:
        return None
    return float(primary) - lam * float(tokens_total)


# ----------------- Task → query rendering -----------------
def make_query(task_cfg: Dict[str, Any], ex: Dict[str, Any]) -> Tuple[str, Any]:
    """returns (query_text, gold_raw)"""
    q = ex.get(task_cfg["query_field"], "")

    if task_cfg.get("context_field"):
        ctx = ex.get(task_cfg["context_field"], "")
        q = f"Context:\n{ctx}\n\nQuestion:\n{q}"

    if task_cfg.get("input_field"):
        inp = ex.get(task_cfg["input_field"], "")
        if inp:
            q = f"Instruction:\n{q}\n\nInput:\n{inp}"

    gold = ex.get(task_cfg.get("gold_field", ""), None)
    return q, gold


def extract_gold(task_name: str, gold_raw: Any) -> str:
    if gold_raw is None:
        return ""
    if task_name == "squad":
        texts = gold_raw.get("text", []) if isinstance(gold_raw, dict) else []
        return texts[0] if texts else ""
    return str(gold_raw)


# ----------------- Scoring -----------------
def score(task_name: str, primary_metric: Optional[str], pred: str, gold: str) -> Dict[str, Any]:
    perf = {
        "primary": None,
        "metric": primary_metric,
        "em": None,
        "f1": None,
        "acc": None,
    }

    if task_name in ("hotpotqa", "squad"):
        perf["em"] = em(pred, gold)
        perf["f1"] = f1_token(pred, gold)
        perf["primary"] = perf["em"] if primary_metric == "em" else perf["f1"]

    elif task_name == "gsm8k":
        perf["acc"] = acc_numeric(pred, gold)
        perf["primary"] = perf["acc"]

    else:
        perf["primary"] = None

    return perf


# ----------------- Split fallback -----------------
def load_with_fallback(hf_path: str, hf_name: Optional[str], preferred_split: str):
    splits_to_try = []

    for s in [preferred_split, "train", "validation", "test"]:
        if s not in splits_to_try:
            splits_to_try.append(s)

    last_error = None

    for split_try in splits_to_try:
        try:
            ds = load_dataset(
                path=hf_path,
                name=hf_name,
                split=split_try,
            )
            return ds, split_try
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not load dataset={hf_path} name={hf_name}. "
        f"Tried splits={splits_to_try}. Last error: {last_error}"
    )


# ----------------- Prompt list normalization -----------------
def normalize_prompt_ids(prompts_cfg: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(prompts_cfg, list):
        raise ValueError(f"Unsupported prompts config type: {type(prompts_cfg)}")
    for p in prompts_cfg:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            pid = p.get("id")
            if not pid:
                raise ValueError(f"Prompt dict missing 'id': {p}")
            out.append(pid)
        else:
            raise ValueError(f"Unsupported prompt entry: {p} ({type(p)})")
    return out


# ----------------- Config deep-merge (base + shard override) -----------------
def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (u or {}).items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


# ----------------- Action selection helper (optional cap) -----------------
def build_actions(models: List[str], prompt_ids: List[str], cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    combos: List[Tuple[str, str]] = [(p, m) for p in prompt_ids for m in models]

    max_units = cfg.get("max_units_per_query", None)
    actions_cfg = cfg.get("actions", {}) or {}
    mode = actions_cfg.get("mode", "cartesian_then_cap")

    if max_units is None:
        return combos

    try:
        max_units = int(max_units)
    except Exception:
        return combos

    if mode == "cartesian_then_cap" and max_units > 0 and len(combos) > max_units:
        seed = int(cfg.get("seed", 0))
        rng = random.Random(seed + 1337)
        combos_copy = combos[:]
        rng.shuffle(combos_copy)
        return combos_copy[:max_units]

    return combos


# ----------------- Main -----------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to shard/override YAML config")
    ap.add_argument(
        "--base_config",
        default="configs/rq2_dataset_builder_full.yaml",
        help="Base YAML (full builder) to deep-merge with shard config",
    )
    args = ap.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(args.config, "r", encoding="utf-8") as f:
        override_cfg = yaml.safe_load(f) or {}

    cfg: Dict[str, Any] = deep_update(base_cfg, override_cfg)

    out_path = cfg["out_jsonl"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seed = int(cfg.get("seed", 0))
    random.seed(seed)

    endpoint = cfg["endpoint"]
    api_mode = cfg.get("api_mode", "chat")
    served_model_name = cfg.get("served_model_name", None)

    timeout_s = int(cfg.get("timeout_s", 120))
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 256))
    lam = float(cfg.get("reward", {}).get("lambda_cost", 1e-5))

    models: List[str] = cfg["pools"]["models"]
    prompt_ids: List[str] = normalize_prompt_ids(cfg["pools"]["prompts"])
    tasks_cfg: Dict[str, Any] = dict(cfg["tasks"])

    if served_model_name is None:
        if len(models) == 1:
            served_model_name = models[0]
        else:
            raise ValueError(
                "served_model_name is required when multiple logical models are listed in pools.models"
            )

    tasks_cfg.pop("multinews", None)

    missing = [p for p in prompt_ids if p not in PROMPTS]
    if missing:
        raise KeyError(f"Prompt ids not found in PROMPTS: {missing}. Available: {list(PROMPTS.keys())}")

    n_per_task = int(cfg.get("n_per_task", 10))
    global_split_pref = cfg.get("split", "train")

    actions = build_actions(models=models, prompt_ids=prompt_ids, cfg=cfg)

    rows: List[Dict[str, Any]] = []
    per_task_counts: Dict[str, int] = {}
    per_task_failed: Dict[str, int] = {}

    for task_name, tcfg in tasks_cfg.items():
        split_pref = tcfg.get("split", global_split_pref)
        hf_path = tcfg["hf_path"]
        hf_name = tcfg.get("hf_name", None)

        ds, used_split = load_with_fallback(hf_path, hf_name, split_pref)
        ds = ds.shuffle(seed=seed).select(range(min(n_per_task, len(ds))))

        for i, ex in enumerate(ds):
            qid = f"{task_name}-{used_split}-{i:06d}"
            query_text, gold_raw = make_query(tcfg, ex)
            gold = extract_gold(task_name, gold_raw)

            for (p, m) in actions:
                sys = PROMPTS[p]["system"]
                usr = PROMPTS[p]["user"].format(q=query_text)

                rec: Dict[str, Any] = {
                    "task": task_name,
                    "qid": qid,
                    "prompt": p,
                    "model": m,  # canonical model id for dataset/cost logic
                    "served_model_name": served_model_name,  # actual vLLM-exposed name
                    "api_mode": api_mode,
                    "system_text": sys,
                    "user_text": usr,
                    "request_preview": f"{sys}\n\n{usr}",
                    "failed": False,
                    "response": "",
                    "performance": None,
                    "cost": None,
                    "reward": None,
                    "error": None,
                }

                try:
                    text, lat, tin, tout = generate_response(
                        endpoint=endpoint,
                        api_mode=api_mode,
                        served_model_name=served_model_name,
                        system=sys,
                        user=usr,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                    )
                    tokens_total = tin + tout
                    perf = score(task_name, tcfg.get("primary_metric", None), text, gold)

                    rec["response"] = text
                    rec["performance"] = perf
                    rec["cost"] = {
                        "tokens_in": tin,
                        "tokens_out": tout,
                        "tokens_total": tokens_total,
                        "latency_s": lat,
                    }
                    rec["reward"] = compute_reward(perf["primary"], tokens_total, lam)

                except Exception as e:
                    rec["failed"] = True
                    rec["error"] = str(e)
                    rec["performance"] = {
                        "primary": None,
                        "metric": tcfg.get("primary_metric", None),
                        "em": None,
                        "f1": None,
                        "acc": None,
                    }
                    rec["cost"] = {
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "tokens_total": 0,
                        "latency_s": 0.0,
                    }
                    rec["reward"] = None

                rows.append(rec)
                per_task_counts[task_name] = per_task_counts.get(task_name, 0) + 1
                if rec["failed"]:
                    per_task_failed[task_name] = per_task_failed.get(task_name, 0) + 1

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_failed = sum(1 for r in rows if r.get("failed"))

    print("==== DONE ====")
    print(f"Base: {args.base_config}")
    print(f"Shard: {args.config}")
    print(f"Out: {out_path}")
    print(f"Rows: {len(rows)}")
    print(f"Failed: {total_failed}")
    print("Per-task rows:", per_task_counts)
    print("Per-task failed:", per_task_failed)


if __name__ == "__main__":
    main()