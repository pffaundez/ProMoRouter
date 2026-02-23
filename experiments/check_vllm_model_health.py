#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dry-run fast health check for a pool of models with vLLM OpenAI server.

For each model candidate (from llm_candidates.json):
- start vLLM api_server on a dedicated port
- wait until /v1/models is responsive
- run a tiny /v1/chat/completions request
- terminate server
- record status + timings

Usage:
  python experiments/check_vllm_model_health.py \
    --candidates configs/llm_candidates.json \
    --model_keys mistral-7b qwen2.5-7b llama3.1-8b \
    --base_port 18001 \
    --gpus 0,1,2,3 \
    --tensor_parallel 1 \
    --start_timeout_s 120 \
    --max_model_len 1024 \
    --gpu_mem_util 0.85 \
    --max_tokens 8 \
    --out_json runs/model_health.json

If --model_keys is omitted, it checks all models in llm_candidates.json in file order.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class CheckResult:
    model_key: str
    hf_id: str
    port: int
    gpu: str
    tensor_parallel: int
    status: str              # ok / start_timeout / request_failed / server_failed
    error_msg: str
    served_models: List[str]
    startup_s: Optional[float]
    completion_s: Optional[float]
    response_preview: str


def load_candidates(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wait_http_ok(url: str, timeout_s: int, poll_s: float = 1.0) -> Tuple[bool, float, Optional[dict]]:
    t0 = time.time()
    last_exc = None
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True, time.time() - t0, r.json()
        except Exception as e:
            last_exc = e

        if time.time() - t0 >= timeout_s:
            return False, time.time() - t0, None

        time.sleep(poll_s)


def try_chat_completion(endpoint: str, model_name: str, max_tokens: int) -> Tuple[bool, float, str, str]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
        txt = j["choices"][0]["message"]["content"]
        return True, time.time() - t0, txt, ""
    except Exception as e:
        return False, time.time() - t0, "", repr(e)


def terminate_process_tree(proc: subprocess.Popen, grace_s: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        proc.kill()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="configs/llm_candidates.json")
    ap.add_argument("--model_keys", nargs="*", default=None, help="If omitted, checks all keys in candidates file.")
    ap.add_argument("--base_port", type=int, default=18001)
    ap.add_argument("--gpus", default="0", help="Comma-separated GPU ids to cycle through, e.g. '0,1,2,3'")
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--start_timeout_s", type=int, default=120)
    ap.add_argument("--max_model_len", type=int, default=1024)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--out_json", default="runs/model_health.json")
    ap.add_argument("--keep_logs", action="store_true", help="Keep per-model vLLM logs under runs/model_health_logs/")
    args = ap.parse_args()

    cands = load_candidates(args.candidates)
    keys = args.model_keys if args.model_keys else list(cands.keys())

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        print("ERROR: --gpus parsed empty.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    logs_dir = "runs/model_health_logs"
    if args.keep_logs:
        os.makedirs(logs_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    print(f"[check] candidates={args.candidates}")
    print(f"[check] models={keys}")
    print(f"[check] base_port={args.base_port} gpus={gpus} tp={args.tensor_parallel}")
    print("")

    for i, mk in enumerate(keys):
        if mk not in cands:
            print(f"[WARN] model_key not found in candidates: {mk}")
            continue

        hf_id = cands[mk].get("hf_id", mk)
        port = args.base_port + i
        gpu = gpus[i % len(gpus)]
        endpoint = f"http://localhost:{port}/v1"

        log_path = None
        stdout = subprocess.DEVNULL
        stderr = subprocess.STDOUT
        if args.keep_logs:
            log_path = os.path.join(logs_dir, f"{mk.replace('/', '_')}_p{port}.log")
            f = open(log_path, "w", encoding="utf-8")
            stdout = f
            stderr = subprocess.STDOUT

        print(f"[{i+1}/{len(keys)}] {mk} -> hf_id={hf_id} gpu={gpu} port={port}")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", hf_id,
            "--port", str(port),
            "--gpu-memory-utilization", str(args.gpu_mem_util),
            "--max-model-len", str(args.max_model_len),
            "--tensor-parallel-size", str(args.tensor_parallel),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        proc = None
        startup_s = None
        completion_s = None
        served_models: List[str] = []
        response_preview = ""
        status = "server_failed"
        error_msg = ""

        try:
            proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)

            ok, startup_s, models_json = wait_http_ok(
                endpoint.rstrip("/") + "/models",
                timeout_s=args.start_timeout_s,
                poll_s=1.0,
            )
            if not ok:
                status = "start_timeout"
                error_msg = f"Server not ready within {args.start_timeout_s}s"
                terminate_process_tree(proc)
                print(f"  -> FAIL: {status} ({error_msg})")
                if args.keep_logs and log_path:
                    print(f"  -> log: {log_path}")
                results.append(CheckResult(
                    model_key=mk, hf_id=hf_id, port=port, gpu=gpu, tensor_parallel=args.tensor_parallel,
                    status=status, error_msg=error_msg, served_models=[],
                    startup_s=startup_s, completion_s=None, response_preview=""
                ).__dict__)
                continue

            # parse served model ids
            if isinstance(models_json, dict):
                served_models = [d.get("id") for d in models_json.get("data", []) if isinstance(d, dict) and d.get("id")]

            # choose model name for request:
            # if server reports exactly one model id, use it; else try hf_id first.
            model_name = served_models[0] if len(served_models) == 1 else hf_id

            ok2, completion_s, resp, err = try_chat_completion(endpoint, model_name, max_tokens=args.max_tokens)
            if not ok2:
                status = "request_failed"
                error_msg = err
                terminate_process_tree(proc)
                print(f"  -> FAIL: {status} ({error_msg})")
                if args.keep_logs and log_path:
                    print(f"  -> log: {log_path}")
                results.append(CheckResult(
                    model_key=mk, hf_id=hf_id, port=port, gpu=gpu, tensor_parallel=args.tensor_parallel,
                    status=status, error_msg=error_msg, served_models=served_models,
                    startup_s=startup_s, completion_s=completion_s, response_preview=""
                ).__dict__)
                continue

            status = "ok"
            response_preview = (resp or "").strip().replace("\n", " ")[:120]
            terminate_process_tree(proc)
            print(f"  -> OK: startup={startup_s:.2f}s completion={completion_s:.2f}s resp='{response_preview}'")
            if args.keep_logs and log_path:
                print(f"  -> log: {log_path}")

            results.append(CheckResult(
                model_key=mk, hf_id=hf_id, port=port, gpu=gpu, tensor_parallel=args.tensor_parallel,
                status=status, error_msg="", served_models=served_models,
                startup_s=startup_s, completion_s=completion_s, response_preview=response_preview
            ).__dict__)

        except Exception as e:
            error_msg = repr(e)
            if proc is not None:
                terminate_process_tree(proc)
            print(f"  -> FAIL: exception {error_msg}")
            if args.keep_logs and log_path:
                print(f"  -> log: {log_path}")
            results.append(CheckResult(
                model_key=mk, hf_id=hf_id, port=port, gpu=gpu, tensor_parallel=args.tensor_parallel,
                status="server_failed", error_msg=error_msg, served_models=served_models,
                startup_s=startup_s, completion_s=completion_s, response_preview=response_preview
            ).__dict__)
        finally:
            if args.keep_logs and log_path:
                try:
                    stdout.close()  # type: ignore
                except Exception:
                    pass

        print("")

    # Write results
    out = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates_path": args.candidates,
        "base_port": args.base_port,
        "gpus": gpus,
        "tensor_parallel": args.tensor_parallel,
        "max_model_len": args.max_model_len,
        "gpu_mem_util": args.gpu_mem_util,
        "results": results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Summary
    ok_n = sum(1 for r in results if r["status"] == "ok")
    print(f"[summary] ok={ok_n}/{len(results)} wrote {args.out_json}")


if __name__ == "__main__":
    main()
