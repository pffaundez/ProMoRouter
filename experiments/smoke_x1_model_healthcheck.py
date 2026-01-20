# X1 - CANDIDATE POOL HEALTH CHECK

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

@dataclass
class ModelCandidate:
    name: str
    service: str
    model: str
    api_endpoint: str


def _is_localhost(url: str) -> bool:
    return ("localhost" in url) or ("127.0.0.1" in url)


def _parse_api_keys_env() -> Dict[str, Any]:
    """
    Supports:
      - dict JSON: {"NVIDIA":"k1,k2", "OpenAI":["k1","k2"], "Ollama":""}
      - list JSON: ["k1","k2"]
      - comma-separated: "k1,k2"
      - single key: "k1"
    """
    raw = os.environ.get("API_KEYS", "").strip()
    if not raw:
        return {}

    # Try JSON
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Comma-separated or single
    if "," in raw:
        return {"__default__": [k.strip() for k in raw.split(",") if k.strip()]}
    return {"__default__": [raw]}


def _pick_key_for_service(keys_obj: Dict[str, Any], service: str) -> Optional[str]:
    """
    Returns one API key for the given service, if available.
    - If keys_obj is dict with service name: picks first key.
    - If keys_obj has "__default__": picks first.
    - If value is "" (empty) that's allowed for localhost endpoints.
    """
    if not keys_obj:
        return None

    if isinstance(keys_obj, dict):
        if service in keys_obj:
            v = keys_obj[service]
        elif "__default__" in keys_obj:
            v = keys_obj["__default__"]
        else:
            return None

        if isinstance(v, list):
            return v[0] if v else None
        if isinstance(v, str):
            # Might be comma-separated
            if "," in v:
                parts = [p.strip() for p in v.split(",") if p.strip()]
                return parts[0] if parts else None
            return v  # may be ""
        return None

    # If user provided something else odd, ignore
    return None


def load_candidates(path: str) -> list[ModelCandidate]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: list[ModelCandidate] = []
    for name, cfg in data.items():
        api_endpoint = cfg.get("api_endpoint")
        model = cfg.get("model")
        service = cfg.get("service", "__default__")

        if not api_endpoint or not model:
            raise ValueError(f"Candidate '{name}' missing api_endpoint or model")

        out.append(ModelCandidate(name=name, service=service, model=model, api_endpoint=api_endpoint))

    return out


def _normalize_chat_url(api_endpoint: str) -> str:
    """
    LLMRouter suggests using /v1 endpoints (OpenAI-compatible).
    We accept:
      - .../v1  -> add /chat/completions
      - .../v1/ -> add chat/completions
      - ...     -> if already ends with /chat/completions, keep
    """
    ep = api_endpoint.rstrip("/")
    if ep.endswith("/chat/completions"):
        return ep
    if ep.endswith("/v1"):
        return ep + "/chat/completions"
    # Sometimes endpoints are already full, sometimes only base host
    # We'll assume OpenAI-compatible if it already contains /v1 somewhere
    if "/v1" in ep:
        # e.g. http://host:8000/v1 -> handled above; if not exact, append
        return ep + "/chat/completions"
    # Last resort: try /v1/chat/completions
    return ep + "/v1/chat/completions"


def call_chat_completion(
    candidate: ModelCandidate,
    api_key: Optional[str],
    prompt: str,
    timeout_s: float,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> Tuple[bool, float, str, Optional[Dict[str, Any]]]:
    url = _normalize_chat_url(candidate.api_endpoint)

    headers = {"Content-Type": "application/json"}
    # For localhost endpoints, allow empty/no key
    if api_key is not None and (api_key != "" or not _is_localhost(candidate.api_endpoint)):
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": candidate.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        latency = time.time() - t0
        if r.status_code != 200:
            return False, latency, f"HTTP {r.status_code}: {r.text[:200]}", None

        data = r.json()
        # OpenAI format: choices[0].message.content
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = str(data)[:200]

        return True, latency, content.strip(), data
    except Exception as e:
        latency = time.time() - t0
        return False, latency, f"EXC: {type(e).__name__}: {e}", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="configs/llm_candidates.json")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-models", type=int, default=0, help="0 = all")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    keys_obj = _parse_api_keys_env()
    candidates = load_candidates(args.candidates)
    if args.max_models and args.max_models > 0:
        candidates = candidates[: args.max_models]

    tests = [
        ("hello", "Hello."),
        ("math", "What is 2+2? Reply with just the number."),
    ]

    print(f"Loaded {len(candidates)} candidates from {args.candidates}")
    print("Running X-1 healthcheck...\n")

    ok_count = 0
    for cand in candidates:
        api_key = _pick_key_for_service(keys_obj, cand.service)

        print(f"=== {cand.name} ===")
        print(f" service: {cand.service}")
        print(f" model:   {cand.model}")
        print(f" endpoint:{cand.api_endpoint}")

        all_ok = True
        for test_name, prompt in tests:
            ok, latency, content, raw = call_chat_completion(
                cand, api_key, prompt, timeout_s=args.timeout
            )
            status = "OK" if ok else "FAIL"
            snippet = content.replace("\n", " ")[:120]
            print(f"  [{status}] {test_name:>5}  {latency:6.2f}s  -> {snippet}")

            if args.verbose and raw is not None:
                # show usage if present
                usage = raw.get("usage")
                if usage:
                    print(f"         usage: {usage}")

            all_ok = all_ok and ok

        if all_ok:
            ok_count += 1
        print()

    print(f"Summary: {ok_count}/{len(candidates)} models passed both checks.")
    if ok_count != len(candidates):
        print("Tip: failing models usually mean wrong api_endpoint (/v1), missing API_KEYS, or model name mismatch.")


if __name__ == "__main__":
    main()
