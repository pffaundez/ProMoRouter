import os, json, time, random, re
from typing import Dict, Any, List, Optional, Tuple

import yaml
import requests
from datasets import load_dataset

# -------- Prompt strategies (IDs must match YAML pools.prompts) --------
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
        "system": "You are a helpful assistant. Decompose the problem into minimal steps internally, then provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):"
    },
    "selfcheck": {
        "system": "You are a helpful assistant. Answer, then self-check briefly and correct if needed. Provide the final answer only.",
        "user": "{q}\n\nFinal answer (one line):"
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
    # simple numeric extraction for GSM8K-like answers
    def extract_num(x: str) -> Optional[str]:
        x = (x or "").strip()
        m = re.findall(r"-?\d+(?:\.\d+)?", x.replace(",", ""))
        return m[-1] if m else None
    pn = extract_num(pred)
    gn = extract_num(gold)
    return 1.0 if (pn is not None and gn is not None and pn == gn) else 0.0

def rougeL_proxy(pred: str, gold: str) -> float:
    # lightweight ROUGE-L proxy (LCS / len(gold_tokens))
    # For paper-grade results, replace with `evaluate.load("rouge")`.
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p or not g:
        return 0.0
    # LCS DP
    dp = [[0]*(len(g)+1) for _ in range(len(p)+1)]
    for i in range(1, len(p)+1):
        for j in range(1, len(g)+1):
            if p[i-1] == g[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    return float(lcs) / float(len(g))

# ----------------- OpenAI-compatible client (vLLM) -----------------
def chat(endpoint: str, model: str, system: str, user: str,
         temperature: float, max_tokens: int, timeout_s: int) -> Tuple[str, float, int, int]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
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
def reward(primary: Optional[float], tokens_total: int, lam: float) -> Optional[float]:
    if primary is None:
        return None
    return float(primary) - lam * float(tokens_total)

# ----------------- Task → prompt rendering & scoring -----------------
def make_query(task_cfg: Dict[str, Any], ex: Dict[str, Any]) -> Tuple[str, Any]:
    # returns (query_text, gold)
    qf = task_cfg["query_field"]
    query = ex.get(qf, "")

    # Special cases
    if task_cfg.get("context_field"):
        ctx = ex.get(task_cfg["context_field"], "")
        query = f"Context:\n{ctx}\n\nQuestion:\n{query}"

    # Alpaca has instruction + optional input
    if "input_field" in task_cfg:
        inp = ex.get(task_cfg["input_field"], "")
        if inp:
            query = f"Instruction:\n{query}\n\nInput:\n{inp}"

    gold = ex.get(task_cfg.get("gold_field", ""), None)
    return query, gold

def extract_gold(task_name: str, task_cfg: Dict[str, Any], gold_raw: Any) -> str:
    if gold_raw is None:
        return ""
    if task_name == "squad":
        # SQuAD: answers = {text: [...], answer_start: [...]}
        texts = gold_raw.get("text", []) if isinstance(gold_raw, dict) else []
        return texts[0] if texts else ""
    # others are strings already
    return str(gold_raw)

def score(task_name: str, primary_metric: str, pred: str, gold: str) -> Dict[str, Any]:
    # returns performance object with primary + other metrics if applicable
    perf = {"primary": None, "metric": primary_metric, "em": None, "f1": None, "acc": None, "rougeL": None, "judge_score": None, "pass_fail": None}

    if task_name in ("hotpotqa", "squad"):
        perf["em"] = em(pred, gold)
        perf["f1"] = f1_token(pred, gold)
        perf["primary"] = perf["f1"] if primary_metric == "f1" else perf["em"]

    elif task_name == "cs8mk":
        perf["acc"] = acc_numeric(pred, gold)
        perf["primary"] = perf["acc"]

    elif task_name == "multinews":
        perf["rougeL"] = rougeL_proxy(pred, gold)
        perf["primary"] = perf["rougeL"]

    elif task_name == "alpaca":
        # Not defined yet without LLM-judge; we log output but leave None
        perf["judge_score"] = None
        perf["primary"] = None

    elif task_name == "humaneval":
        # Not defined yet without code execution harness; leave None
        perf["pass_fail"] = None
        perf["primary"] = None

    else:
        # unknown task
        perf["primary"] = None

    return perf

# ----------------- Main builder -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    out_path = cfg["out_jsonl"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    random.seed(cfg.get("seed", 0))

    endpoint = cfg["endpoint"]
    timeout_s = int(cfg.get("timeout_s", 120))
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 256))
    lam = float(cfg.get("lambda_cost", 1e-5))

    models = cfg["pools"]["models"]
    prompts = cfg["pools"]["prompts"]
    tasks_cfg: Dict[str, Any] = cfg["tasks"]

    # Build a balanced list of examples across tasks
    rows: List[Dict[str, Any]] = []

    for task_name, tcfg in tasks_cfg.items():
        n = int(cfg.get("n_per_task", 10))
        split = cfg.get("split", "train")

        hf_path = tcfg["hf_path"]
        hf_name = tcfg.get("hf_name", None)

        ds = load_dataset(hf_path, hf_name, split=split)
        ds = ds.shuffle(seed=cfg.get("seed", 0)).select(range(min(n, len(ds))))

        for i, ex in enumerate(ds):
            qid = f"{task_name}-{split}-{i:06d}"
            query_text, gold_raw = make_query(tcfg, ex)
            gold = extract_gold(task_name, tcfg, gold_raw)

            # monolithic unit for now
            unit_id = f"{qid}::mono"
            subquery = query_text

            for p in prompts:
                for m in models:
                    sys = PROMPTS[p]["system"]
                    usr = PROMPTS[p]["user"].format(q=subquery)

                    rec = {
                        "task": task_name,
                        "qid": qid,
                        "query": query_text,
                        "subquery": subquery,
                        "unit_id": unit_id,
                        "is_decomposed": False,
                        "prompt": p,
                        "model": m,
                        "response": "",
                        "performance": None,
                        "cost": None,
                        "reward": None,
                        "failed": False,
                    }

                    try:
                        text, lat, tin, tout = chat(endpoint, m, sys, usr, temperature, max_tokens, timeout_s)
                        rec["response"] = text

                        tokens_total = tin + tout
                        rec["cost"] = {
                            "tokens_in": tin,
                            "tokens_out": tout,
                            "tokens_total": tokens_total,
                            "latency_s": lat,
                            "proxy": cfg.get("cost_proxy", "tokens_total"),
                        }

                        perf = score(task_name, tcfg["primary_metric"], text, gold)
                        rec["performance"] = perf
                        rec["reward"] = reward(perf["primary"], tokens_total, lam)

                    except Exception as e:
                        rec["failed"] = True
                        rec["error"] = str(e)
                        rec["performance"] = {"primary": None, "metric": tcfg["primary_metric"]}
                        rec["cost"] = {"tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "latency_s": 0.0, "proxy": cfg.get("cost_proxy", "tokens_total")}
                        rec["reward"] = None

                    rows.append(rec)

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick summary
    n = len(rows)
    fail = sum(1 for r in rows if r.get("failed"))
    with_reward = [r for r in rows if (r.get("reward") is not None)]
    avg_reward = sum(r["reward"] for r in with_reward) / max(1, len(with_reward))

    print("\n==== RQ2 Builder Done ====")
    print(json.dumps({
        "out": out_path,
        "rows": n,
        "failed": fail,
        "reward_rows": len(with_reward),
        "avg_reward_over_reward_rows": avg_reward,
        "note": "alpaca/humaneval primary metrics are null unless you add judge/test harness",
    }, indent=2))

if __name__ == "__main__":
    main()
