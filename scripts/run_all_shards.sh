set -euo pipefail

# Recomendación: ejecuta desde la raíz del repo (graph-router-2)
# y con el venv activado (.venv).

ROOT="$(pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# -------- Config global --------
CUDA_DEV=1

# Puertos por modelo (evita colisiones si se queda algo vivo)
# Puedes mantener un port base e ir sumando.
PORT_BASE=18010

# Espera readiness
READY_TIMEOUT_S=600
READY_POLL_S=2

# Descanso para que CUDA libere memoria entre modelos
COOLDOWN_S=10

# Ruta shard yamls
SHARD_DIR="configs/rq2_shards"

# -------- Tabla de modelos (model_key | hf_id | served_name | max_len | gpu_mem_util) --------
# Ajusta max_len y gpu_mem_util si un modelo da OOM.
MODELS=(
  "mistral-7b|mistralai/Mistral-7B-Instruct-v0.3|mistral-7b|512|0.90"
  "qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct|qwen2.5-7b|512|0.90"
  #llama gated for now
  #"llama3.1-8b|meta-llama/Llama-3.1-8B-Instruct|llama3.1-8b|512|0.90"
  "qwen2.5-14b|Qwen/Qwen2.5-14B-Instruct|qwen2.5-14b|512|0.90"
  "yi-34b|01-ai/Yi-34B-Chat|yi-34b|2048|0.92"
  "mixtral-8x7b|mistralai/Mixtral-8x7B-Instruct-v0.1|mixtral-8x7b|512|0.90"
  "codellama-34b|codellama/CodeLlama-34b-Instruct-hf|codellama-34b|2048|0.92"
)

# -------- Helpers --------
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

  # Espera a que muera
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

# -------- Main --------
echo "Starting shard pipeline (sequential)..."
print_gpu
echo

i=0
for row in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_KEY HF_ID SERVED_NAME MAX_LEN GPU_MEM_UTIL <<< "$row"

  SHARD_YAML="$SHARD_DIR/${MODEL_KEY}.yaml"
  if [ ! -f "$SHARD_YAML" ]; then
    echo "ERROR: Missing shard yaml: $SHARD_YAML"
    exit 1
  fi

  PORT=$((PORT_BASE + i))
  i=$((i+1))
  ENDPOINT="http://localhost:${PORT}"

  VLLM_LOG="$LOGDIR/vllm_${MODEL_KEY}_${PORT}.log"
  RUN_LOG="$LOGDIR/run_${MODEL_KEY}_${PORT}.log"

  echo "--------------------------------------"
  echo "Model:  $MODEL_KEY"
  echo "HF:     $HF_ID"
  echo "Served: $SERVED_NAME"
  echo "Port:   $PORT"
  echo "YAML:   $SHARD_YAML"
  echo "--------------------------------------"


  # 1) Start vLLM
  VLLM_PID="$(./scripts/start_vllm.sh "$MODEL_KEY" "$HF_ID" "$SERVED_NAME" "$PORT" "$MAX_LEN" "$GPU_MEM_UTIL" "$CUDA_DEV" "$VLLM_LOG")"
  echo "vLLM started with PID=$VLLM_PID"
  echo "vLLM log: $VLLM_LOG"

  # 2) Wait ready
  echo "Waiting for readiness: $ENDPOINT/v1/models ..."
  if ! wait_ready "$ENDPOINT"; then
    echo "ERROR: vLLM not ready within ${READY_TIMEOUT_S}s"
    echo "Last 80 lines of vLLM log:"
    tail -n 80 "$VLLM_LOG" || true
    kill_pid "$VLLM_PID"
    exit 1
  fi
  echo "Ready ✅"

  # 3) Run shard builder
  echo "Running shard builder..."
  set +e
  python datasets/build_rq2_interaction_logs.py --config "$SHARD_YAML" 2>&1 | tee "$RUN_LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "ERROR: shard builder failed for $MODEL_KEY (rc=$rc)"
    echo "See: $RUN_LOG"
    kill_pid "$VLLM_PID"
    exit 1
  fi
  echo "Shard builder done ✅"
  echo "Run log: $RUN_LOG"

  # 4) Stop vLLM
  kill_pid "$VLLM_PID"
  sleep "$COOLDOWN_S"
  print_gpu
  echo
done

echo "All shards completed ✅"