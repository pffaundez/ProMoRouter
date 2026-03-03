import os, json, time, random, re
from typing import Dict, Any, List, Optional, Tuple

import yaml
import requests
from datasets import load_dataset

# -------- Prompt strategies --------
PROMPTS = {
    "direct": {
        "system": "You are a helpful assistant. Answer concisely and accurately.",
        "user": "{q}"
    },
    "cot": {
        "system": "You are a helpful assistant. Think step by step internally, then provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):"
    },
    "decompose": {
        "system": "You are a helpful assistant. Decompose the problem internally, then provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):"
    },
    "selfcheck": {
        "system": "You are a helpful assistant. Answer, self-check briefly, and provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):"
    },
}

# ----------------- Normalization -----------------
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
    common = {}
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
    def extract_num(x: str) -> Optional[str]:
        m = re.findall(r"-?\d+(?:\.\d+)?", (x or "").replace(",", ""))
        return m[-1] if m else None
    return 1.0 if extract_num(pred) == extract_num(gold) else 0.0

# ----------------- vLLM client -----------------
def chat(endpoint, model, system, user, temperature, max_tokens, timeout_s):
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout_s)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    tin = int(usage.get("prompt_tokens", 0))
    tout = int(usage.get("completion_tokens", 0))
    return text, latency, tin, tout

# ----------------- Reward -----------------
def reward(primary, tokens_total, lam):
    if primary is None:
        return None
    return float(primary) - lam * float(tokens_total)

# ----------------- Query building -----------------
def make_query(task_cfg, ex):
    query = ex.get(task_cfg["query_field"], "")

    if task_cfg.get("context_field"):
        ctx = ex.get(task_cfg["context_field"], "")
        query = f"Context:\n{ctx}\n\nQuestion:\n{query}"

    if task_cfg.get("input_field"):
        inp = ex.get(task_cfg["input_field"], "")
        if inp:
            query = f"Instruction:\n{query}\n\nInput:\n{inp}"

    gold = ex.get(task_cfg.get("gold_field", ""), None)
    return query, gold

def extract_gold(task_name, gold_raw):
    if gold_raw is None:
        return ""
    if task_name == "squad":
        texts = gold_raw.get("text", []) if isinstance(gold_raw, dict) else []
        return texts[0] if texts else ""
    return str(gold_raw)

# ----------------- Scoring -----------------
def score(task_name, primary_metric, pred, gold):
    perf = {
        "primary": None,
        "metric": primary_metric,
        "em": None,
        "f1": None,
        "acc": None
    }

    if task_name in ("hotpotqa", "squad"):
        perf["em"] = em(pred, gold)
        perf["f1"] = f1_token(pred, gold)
        perf["primary"] = perf["f1"]

    elif task_name == "gsm8k":
        perf["acc"] = acc_numeric(pred, gold)
        perf["primary"] = perf["acc"]

    return perf

# ----------------- Split fallback -----------------
def load_with_fallback(hf_path, hf_name, preferred_split):
    for split_try in [preferred_split, "validation", "test", "train"]:
        try:
            return load_dataset(hf_path, hf_name, split=split_try)
        except Exception:
            continue
    raise RuntimeError(f"Could not load dataset {hf_path}")

# ----------------- Main -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    out_path = cfg["out_jsonl"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    endpoint = cfg["endpoint"]
    timeout_s = int(cfg.get("timeout_s", 120))
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 256))

    lam = cfg.get("reward", {}).get("lambda_cost", 1e-5)

    models = cfg["pools"]["models"]
    prompts = cfg["pools"]["prompts"]
    tasks_cfg = cfg["tasks"]

    rows = []

    for task_name, tcfg in tasks_cfg.items():
        n = int(cfg.get("n_per_task", 10))
        split_pref = tcfg.get("split", cfg.get("split", "train"))

        ds = load_with_fallback(
            tcfg["hf_path"],
            tcfg.get("hf_name"),
            split_pref
        )

        ds = ds.shuffle(seed=cfg.get("seed", 0)).select(range(min(n, len(ds))))

        for i, ex in enumerate(ds):
            qid = f"{task_name}-{i:06d}"
            query_text, gold_raw = make_query(tcfg, ex)
            gold = extract_gold(task_name, gold_raw)

            for p in prompts:
                for m in models:

                    sys = PROMPTS[p]["system"]
                    usr = PROMPTS[p]["user"].format(q=query_text)

                    rec = {
                        "task": task_name,
                        "qid": qid,
                        "prompt": p,
                        "model": m,
                        "failed": False,
                    }

                    try:
                        text, lat, tin, tout = chat(
                            endpoint, m, sys, usr,
                            temperature, max_tokens, timeout_s
                        )

                        tokens_total = tin + tout
                        perf = score(task_name, tcfg["primary_metric"], text, gold)

                        rec.update({
                            "response": text,
                            "performance": perf,
                            "cost": tokens_total,
                            "reward": reward(perf["primary"], tokens_total, lam)
                        })

                    except Exception as e:
                        rec["failed"] = True
                        rec["error"] = str(e)

                    rows.append(rec)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("==== DONE ====")
    print(f"Rows: {len(rows)}")
    print(f"Failed: {sum(r['failed'] for r in rows)}")

if __name__ == "__main__":
    main()