#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Uso:"
  echo "  $0 <gpu_id> <port> <served_name> <hf_model_repo> <quantization> [max_model_len] [gpu_mem_util]"
  echo
  echo "Ejemplo:"
  echo "  $0 0 18000 mistral-7b-gptq4 some-org/Mistral-7B-Instruct-v0.3-GPTQ gptq 512 0.90"
  exit 1
fi

GPU_ID="$1"
PORT="$2"
SERVED_NAME="$3"
HF_MODEL_REPO="$4"
QUANTIZATION="$5"
MAX_MODEL_LEN="${6:-512}"
GPU_MEM_UTIL="${7:-0.90}"

# ------------------------------------------------------------
# Resolve repository root dynamically
# ------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[ERROR] Python environment not found at:"
  echo "        $PYTHON_BIN"
  echo "Activate or create the venv first:"
  echo "        python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

LOG_DIR="$ROOT/logs/quant"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SERVED_NAME}.log"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] PYTHON=${PYTHON_BIN}"
echo "[INFO] GPU=${GPU_ID}"
echo "[INFO] PORT=${PORT}"
echo "[INFO] SERVED_NAME=${SERVED_NAME}"
echo "[INFO] HF_MODEL_REPO=${HF_MODEL_REPO}"
echo "[INFO] QUANTIZATION=${QUANTIZATION}"
echo "[INFO] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[INFO] GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[INFO] LOG_FILE=${LOG_FILE}"

"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "${HF_MODEL_REPO}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --dtype auto \
  --quantization "${QUANTIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code \
  2>&1 | tee "${LOG_FILE}"
