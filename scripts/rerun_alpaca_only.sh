#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_CFG_DIR="$ROOT/configs/rq2_shards/_alpaca_only_tmp"
TMP_OUT_DIR="$ROOT/data/interaction_logs/grpp_il_v1/shards/_alpaca_only_tmp"
FINAL_OUT_DIR="$ROOT/data/interaction_logs/grpp_il_v1/shards"
LOG_DIR="$ROOT/logs/rerun_alpaca_only"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

# IMPORTANT: GPU 3 is under maintenance on morel. Use GPU 0.
GPU_ID="${GPU_ID:-0}"

mkdir -p "$TMP_CFG_DIR"
mkdir -p "$TMP_OUT_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found at $PYTHON_BIN"
  exit 1
fi

MODELS=(
  "mistral-7b"
  "qwen2.5-7b"
  "llama3.1-8b"
  "qwen2.5-14b"
  "yi-34b"
  "codellama-34b"
  "mixtral-8x7b"
  "llama3.1-70b"
  "qwen2.5-72b"
)

# model_key|hf_id|served_model_name|port|max_len|gpu_mem_util|quantization
MODEL_SPECS=(
  "mistral-7b|RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit|mistral-7b-gptq4|18000|512|0.80|gptq_marlin"
  "qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4|qwen2.5-7b-gptq4|18001|512|0.80|gptq_marlin"
  "llama3.1-8b|hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4|llama3.1-8b-gptq4|18002|512|0.80|gptq_marlin"
  "qwen2.5-14b|Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4|qwen2.5-14b-gptq4|18003|512|0.75|gptq_marlin"
  "yi-34b|TheBloke/Yi-34B-Chat-GPTQ|yi-34b-gptq4|18100|512|0.70|gptq_marlin"
  "codellama-34b|TheBloke/CodeLlama-34B-Instruct-GPTQ|codellama-34b-gptq4|18101|512|0.70|gptq_marlin"
  "mixtral-8x7b|TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ|mixtral-8x7b-gptq4|18102|512|0.70|gptq_marlin"
  "llama3.1-70b|hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4|llama3.1-70b-gptq4|18200|512|0.60|gptq_marlin"
  "qwen2.5-72b|Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4|qwen2.5-72b-gptq4|18201|512|0.55|gptq_marlin"
)

echo "==== RERUN ALPACA ONLY (MOREL / GPU ${GPU_ID}) ===="

cleanup_pid() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping vLLM PID $pid"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 5
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "Force killing PID $pid"
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

wait_for_vllm() {
  local port="$1"
  local max_wait="${2:-600}"
  local waited=0

  echo "Waiting for vLLM on port $port ..."
  until curl -sf "http://localhost:${port}/v1/models" >/dev/null; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= max_wait )); then
      echo "ERROR: vLLM did not become ready on port $port within ${max_wait}s"
      return 1
    fi
  done

  echo "vLLM is ready on port $port"
  return 0
}

for MODEL in "${MODELS[@]}"; do
  SPEC=""
  for row in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model_key hf_id served_name port max_len gpu_mem quant <<< "$row"
    if [[ "$model_key" == "$MODEL" ]]; then
      SPEC="$row"
      break
    fi
  done

  if [[ -z "$SPEC" ]]; then
    echo "ERROR: no model spec found for $MODEL"
    exit 1
  fi

  IFS='|' read -r MODEL_KEY HF_ID SERVED_NAME PORT MAX_LEN GPU_MEM_UTIL QUANTIZATION <<< "$SPEC"

  SRC_CFG="$ROOT/configs/rq2_shards/${MODEL}.yaml"
  TMP_CFG="${TMP_CFG_DIR}/${MODEL}_alpaca_only.yaml"
  OLD_SHARD="${FINAL_OUT_DIR}/train__${MODEL}.jsonl"
  NEW_ALPACA_SHARD="${TMP_OUT_DIR}/alpaca_only__${MODEL}.jsonl"
  MERGED_SHARD="${FINAL_OUT_DIR}/train__${MODEL}.jsonl"
  VLLM_LOG="${LOG_DIR}/${MODEL}.log"

  if [[ ! -f "$SRC_CFG" ]]; then
    echo "ERROR: missing config $SRC_CFG"
    exit 1
  fi

  if [[ ! -f "$OLD_SHARD" ]]; then
    echo "ERROR: missing shard $OLD_SHARD"
    exit 1
  fi

  echo
  echo "--------------------------------------"
  echo "Model: $MODEL"
  echo "HF_ID: $HF_ID"
  echo "Served model name: $SERVED_NAME"
  echo "Port: $PORT"
  echo "GPU_ID: $GPU_ID"
  echo "Source cfg: $SRC_CFG"
  echo "Temp cfg: $TMP_CFG"
  echo "Old shard: $OLD_SHARD"
  echo "New alpaca shard: $NEW_ALPACA_SHARD"
  echo "Merged shard: $MERGED_SHARD"
  echo "vLLM log: $VLLM_LOG"
  echo "--------------------------------------"

  "$PYTHON_BIN" - <<PY
import yaml
from pathlib import Path

src = Path(r"$SRC_CFG")
dst = Path(r"$TMP_CFG")
new_out = r"$NEW_ALPACA_SHARD"
endpoint = r"http://localhost:$PORT/v1"
served_name = r"$SERVED_NAME"

with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

tasks = cfg.get("tasks", {})
if "alpaca" not in tasks:
    raise RuntimeError(f"alpaca task not found in {src}")

cfg["tasks"] = {"alpaca": tasks["alpaca"]}
cfg["out_jsonl"] = new_out
cfg["endpoint"] = endpoint
cfg["served_model_name"] = served_name

with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"Wrote alpaca-only config: {dst}")
print(f"Tasks in temp cfg: {list(cfg.get('tasks', {}).keys())}")
print(f"Endpoint in temp cfg: {cfg.get('endpoint')}")
print(f"served_model_name in temp cfg: {cfg.get('served_model_name')}")
print(f"out_jsonl in temp cfg: {cfg.get('out_jsonl')}")
PY

  VLLM_PID=""
  cleanup_pid "$VLLM_PID" || true

  VLLM_PID="$("$ROOT/scripts/start_vllm.sh" \
    "$MODEL_KEY" \
    "$HF_ID" \
    "$SERVED_NAME" \
    "$PORT" \
    "$MAX_LEN" \
    "$GPU_MEM_UTIL" \
    "$GPU_ID" \
    "$QUANTIZATION" \
    "$VLLM_LOG"
  )"

  echo "Started vLLM with PID $VLLM_PID"

  if ! wait_for_vllm "$PORT" 600; then
    echo "ERROR: vLLM failed to start for $MODEL"
    echo "Check log: $VLLM_LOG"
    cleanup_pid "$VLLM_PID"
    exit 1
  fi

  echo "Endpoint check:"
  curl -sf "http://localhost:${PORT}/v1/models" || {
    echo "ERROR: endpoint health check failed for $MODEL"
    cleanup_pid "$VLLM_PID"
    exit 1
  }
  echo

  "$PYTHON_BIN" datasets/build_rq2_interaction_logs.py \
    --base_config "$TMP_CFG" \
    --config "$TMP_CFG"

  "$PYTHON_BIN" - <<PY
import json
from collections import Counter
from pathlib import Path

old_path = Path(r"$OLD_SHARD")
alpaca_path = Path(r"$NEW_ALPACA_SHARD")
merged_path = Path(r"$MERGED_SHARD")

if not alpaca_path.exists():
    raise RuntimeError(f"Expected alpaca shard not found: {alpaca_path}")

non_alpaca = []
alpaca_new = []

with old_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("task") != "alpaca":
            non_alpaca.append(row)

with alpaca_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("task") == "alpaca":
            alpaca_new.append(row)

if not alpaca_new:
    raise RuntimeError(f"No alpaca rows found in {alpaca_path}")

task_counts = Counter(row.get("task") for row in alpaca_new)
unexpected_tasks = [t for t in task_counts if t != "alpaca"]
if unexpected_tasks:
    raise RuntimeError(
        f"Temp shard contains unexpected tasks: {unexpected_tasks} | counts={dict(task_counts)}"
    )

failed_counts = Counter(row.get("failed") for row in alpaca_new)
errors = Counter((row.get("error") or "").strip() for row in alpaca_new if row.get("error"))
metric_values = Counter(str((row.get("performance") or {}).get("metric")) for row in alpaca_new)

ok_primary = 0
ok_f1 = 0
ok_primary_and_f1 = 0

for row in alpaca_new:
    perf = row.get("performance", {}) or {}
    has_primary = perf.get("primary") is not None
    has_f1 = perf.get("f1") is not None
    if has_primary:
        ok_primary += 1
    if has_f1:
        ok_f1 += 1
    if has_primary and has_f1:
        ok_primary_and_f1 += 1

print(f"New alpaca rows: {len(alpaca_new)}")
print(f"Task counts in new shard: {dict(task_counts)}")
print(f"Failed counts: {dict(failed_counts)}")
print(f"Metric field distribution: {dict(metric_values)}")
print(f"New alpaca rows with primary != None: {ok_primary}")
print(f"New alpaca rows with f1 != None: {ok_f1}")
print(f"New alpaca rows with primary != None and f1 != None: {ok_primary_and_f1}")

if errors:
    print("Top errors:")
    for msg, n in errors.most_common(10):
        print(f"[{n}] {msg}")

if failed_counts.get(True, 0) > 0:
    raise RuntimeError(
        f"Generated alpaca shard still contains failed rows: {failed_counts.get(True, 0)}"
    )

if ok_primary == 0:
    raise RuntimeError(
        f"All alpaca rows have performance.primary=None in {alpaca_path}"
    )

with merged_path.open("w", encoding="utf-8") as f:
    for row in non_alpaca:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
    for row in alpaca_new:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")

print(f"Merged shard written: {merged_path}")
print(f"Kept non-alpaca rows: {len(non_alpaca)}")
print(f"Inserted new alpaca rows: {len(alpaca_new)}")
print(f"Final total rows: {len(non_alpaca) + len(alpaca_new)}")
PY

  cleanup_pid "$VLLM_PID"
done

echo
echo "==== DONE: ALPACA RERUN + MERGE COMPLETED ===="