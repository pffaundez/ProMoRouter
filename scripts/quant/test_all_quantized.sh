#!/usr/bin/env bash
set -euo pipefail

# Lista de modelos cuantizados a probar (formato: served_name:hf_repo)
MODELS=(
  "mistral-7b-gptq4:RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit"
  "qwen2.5-7b-gptq4:Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
  "llama3.1-8b-gptq4:hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
  "qwen2.5-14b-gptq4:Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
  "yi-34b-gptq4:TheBloke/Yi-34B-Chat-GPTQ"
  "mixtral-8x7b-gptq4:TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
  "codellama-34b-gptq4:TheBloke/CodeLlama-34B-Instruct-GPTQ"
  "llama3.1-70b-gptq4:hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
  "qwen2.5-72b-gptq4:Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
  "mixtral-8x22b-gptq4:jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit"
)

GPU_ID=1
BASE_PORT=18000
QUANTIZATION="gptq"
MAX_MODEL_LEN=2048
GPU_MEM_UTIL=0.9

for MODEL_ENTRY in "${MODELS[@]}"; do
  SERVED_NAME=$(echo "$MODEL_ENTRY" | cut -d: -f1)
  HF_REPO=$(echo "$MODEL_ENTRY" | cut -d: -f2)
  PORT=$((BASE_PORT + RANDOM % 100))  # Puerto aleatorio para evitar conflictos

  echo "=========================================="
  echo "[INFO] Probando modelo: ${SERVED_NAME} (${HF_REPO}) en GPU ${GPU_ID}, puerto ${PORT}"
  echo "=========================================="

  # Iniciar servidor en background
  ./scripts/quant/start_quant_vllm.sh "${GPU_ID}" "${PORT}" "${SERVED_NAME}" "${HF_REPO}" "${QUANTIZATION}" "${MAX_MODEL_LEN}" "${GPU_MEM_UTIL}" &
  SERVER_PID=$!

  # Esperar a que esté listo
  if ./scripts/quant/wait_for_vllm.sh "${PORT}" "${SERVED_NAME}" 300; then
    echo "[INFO] Servidor listo, ejecutando smoke test..."
    ./scripts/quant/api_smoke.sh "${PORT}" "${SERVED_NAME}"
    echo "[OK] Smoke test completado para ${SERVED_NAME}"
  else
    echo "[ERROR] Falló al iniciar servidor para ${SERVED_NAME}"
  fi

  # Matar el servidor
  echo "[INFO] Deteniendo servidor (PID: ${SERVER_PID})..."
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true

  echo "[INFO] Modelo ${SERVED_NAME} probado. Continuando al siguiente..."
  # read -r  # Removido para ejecución automática
done

echo "[INFO] Todos los modelos probados."