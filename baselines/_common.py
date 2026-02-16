# baselines/_common.py
import argparse, json, time, re, os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import requests

# -----------------------------
# Prompt strategies (IDs + templates)
# -----------------------------
PROMPTS: Dict[str, Dict[str, str]] = {
    "direct": {
        "system": "You are a helpful assistant. Answer concisely and accurately.",
        "user": "{q}"
    },
    "cot": {
        "system": "You are a helpful assistant. Reason step by step internally, then give the final answer.",
        "user": "{q}\n\nProvide the final answer on a single line."
    },
    "decompose": {
        "system": "You are a helpful assistant. Break the problem into minimal steps, solve them, then answer.",
        "user": "{q}\n\nReturn the final answer on a single line."
    },
    "selfcheck": {
        "system": "You are a helpful assistant. Answer, then briefly self-check and correct if needed.",
        "user": "{q}\n\nFinal answer on a single line."
    },
}

# -----------------------------
# Simple accuracy helpers
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def accuracy(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

# -----------------------------
# Reward
# -----------------------------
def compute_reward(quality: float, cost_tokens: int, lam: float) -> float:
    # Cost proxy: total tokens
    return float(quality) - lam * float(cost_tokens)

# -----------------------------
# OpenAI-compatible chat client (vLLM endpoint)
# -----------------------------
@dataclass
class ChatResult:
    text: str
    latency_s: float
    tokens_in: int
    tokens_out: int

def chat_completion(endpoint: str, model: str, system: str, user: str,
                    temperature: float = 0.0, max_tokens: int = 256,
                    timeout: int = 120) -> ChatResult:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})  # vLLM often returns this
    tokens_in = int(usage.get("prompt_tokens", 0))
    tokens_out = int(usage.get("completion_tokens", 0))
    return ChatResult(text=text, latency_s=latency, tokens_in=tokens_in, tokens_out=tokens_out)

# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def print_summary(name: str, rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    if n == 0:
        print(f"[{name}] No rows.")
        return
    avg_q = sum(r.get("quality", 0.0) for r in rows) / n
    avg_cost = sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in rows) / n
    avg_lat = sum(r.get("latency_s", 0.0) for r in rows) / n
    avg_r = sum(r.get("reward", 0.0) for r in rows) / n
    fail = sum(1 for r in rows if r.get("failed", False)) / n
    print(f"\n==== {name} Summary ====")
    print(json.dumps({
        "n": n,
        "avg_quality": avg_q,
        "avg_cost_tokens": avg_cost,
        "avg_latency_s": avg_lat,
        "avg_reward": avg_r,
        "fail_rate": fail,
    }, indent=2))
