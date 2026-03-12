#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/start_vllm.sh <model_key> <hf_id> <served_model_name> <port> <max_len> <gpu_mem_util> <cuda_device> <quantization> <logfile>

MODEL_KEY="${1:?model_key}"
HF_ID="${2:?hf_id}"
SERVED_NAME="${3:?served_model_name}"
PORT="${4:?port}"
MAX_LEN="${5:?max_len}"
GPU_MEM_UTIL="${6:?gpu_mem_util}"
CUDA_DEV="${7:?cuda_device}"
QUANTIZATION="${8:?quantization}"
LOGFILE="${9:?logfile}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[ERROR] Python environment not found at: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOGFILE")"

export CUDA_VISIBLE_DEVICES="$CUDA_DEV"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$HF_ID" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --quantization "$QUANTIZATION" \
  > "$LOGFILE" 2>&1 &

echo $!