set -euo pipefail

# Usage:
#   ./scripts/start_vllm.sh <model_key> <hf_id> <served_model_name> <port> <max_len> <gpu_mem_util> <cuda_device> <logfile>
#
# Example:
#   ./scripts/start_vllm.sh mistral-7b mistralai/Mistral-7B-Instruct-v0.3 mistral-7b 18010 512 0.90 1 logs/vllm_mistral-7b_18010.log

MODEL_KEY="${1:?model_key}"
HF_ID="${2:?hf_id}"
SERVED_NAME="${3:?served_model_name}"
PORT="${4:?port}"
MAX_LEN="${5:?max_len}"
GPU_MEM_UTIL="${6:?gpu_mem_util}"
CUDA_DEV="${7:?cuda_device}"
LOGFILE="${8:?logfile}"

mkdir -p "$(dirname "$LOGFILE")"

export CUDA_VISIBLE_DEVICES="$CUDA_DEV"

# Nota: si quieres evitar fragmentación en runs largos, puedes activar esto:
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Lanza vLLM en background y devuelve PID via stdout
python -m vllm.entrypoints.openai.api_server \
  --model "$HF_ID" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  > "$LOGFILE" 2>&1 &

echo $!