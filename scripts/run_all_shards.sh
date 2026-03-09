#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

READY_TIMEOUT_S=900
READY_POLL_S=2
COOLDOWN_S=10

SHARD_DIR="configs/rq2_shards"

# Fijamos GPU 1 por consistencia con enexa2
GPU_ID="${GPU_ID:-1}"

MODELS=(
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

wait_ready () {
  local endpoint="$1"
  local deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
  while true; do
    if curl -sSf "$endpoint/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      return 1
    fi
    sleep "$READY_POLL_S"
  done
}

kill_pid () {
  local pid="$1"
  if [ -z "$pid" ]; then return 0; fi
  if ! ps -p "$pid" >/dev/null 2>&1; then return 0; fi

  echo "Stopping vLLM PID=$pid ..."
  kill "$pid" >/dev/null 2>&1 || true

  for _ in $(seq 1 30); do
    if ! ps -p "$pid" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "PID $pid did not exit, sending SIGKILL..."
  kill -9 "$pid" >/dev/null 2>&1 || true
}

print_gpu () {
  echo "GPU snapshot:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
}

echo "Starting shard pipeline (sequential, quantized)..."
echo "Using GPU: $GPU_ID"
print_gpu
echo

for row in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_KEY HF_ID SERVED_NAME PORT MAX_LEN GPU_MEM_UTIL QUANTIZATION <<< "$row"

  OUTPUT_FILE="data/interaction_logs/grpp_il_v1/shards/train__${MODEL_KEY}.jsonl"
  SHARD_YAML="$SHARD_DIR/${MODEL_KEY}.yaml"
  ENDPOINT="http://localhost:${PORT}"

  if [ -f "$OUTPUT_FILE" ]; then
    echo "Skipping $MODEL_KEY (already exists)"
    continue
  fi

  if [ ! -f "$SHARD_YAML" ]; then
    echo "ERROR: Missing shard yaml: $SHARD_YAML"
    exit 1
  fi

  VLLM_LOG="$LOGDIR/vllm_${MODEL_KEY}_${PORT}.log"
  RUN_LOG="$LOGDIR/run_${MODEL_KEY}_${PORT}.log"

  echo "--------------------------------------"
  echo "Model:        $MODEL_KEY"
  echo "HF:           $HF_ID"
  echo "Served:       $SERVED_NAME"
  echo "Port:         $PORT"
  echo "GPU:          $GPU_ID"
  echo "Quantization: $QUANTIZATION"
  echo "YAML:         $SHARD_YAML"
  echo "--------------------------------------"

  VLLM_PID="$(./scripts/start_vllm.sh \
    "$MODEL_KEY" \
    "$HF_ID" \
    "$SERVED_NAME" \
    "$PORT" \
    "$MAX_LEN" \
    "$GPU_MEM_UTIL" \
    "$GPU_ID" \
    "$QUANTIZATION" \
    "$VLLM_LOG")"

  echo "vLLM started with PID=$VLLM_PID"
  echo "vLLM log: $VLLM_LOG"

  echo "Waiting for readiness: $ENDPOINT/v1/models ..."
  if ! wait_ready "$ENDPOINT"; then
    echo "ERROR: vLLM not ready within ${READY_TIMEOUT_S}s"
    tail -n 120 "$VLLM_LOG" || true
    kill_pid "$VLLM_PID"
    exit 1
  fi

  echo "Ready ✅"

  echo "Running shard builder..."
  set +e
  python datasets/build_rq2_interaction_logs.py --config "$SHARD_YAML" 2>&1 | tee "$RUN_LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "ERROR: shard builder failed for $MODEL_KEY"
    kill_pid "$VLLM_PID"
    exit 1
  fi

  echo "Shard builder done ✅"

  kill_pid "$VLLM_PID"

  sleep "$COOLDOWN_S"
  print_gpu
  echo
done

echo "All shards completed ✅"