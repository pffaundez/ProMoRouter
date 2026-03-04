#!/bin/bash

set -u  # fail on undefined vars (pero no en errores de comandos)

GPU_ID=1
MAX_WAIT=120
MIN_FREE_MEM=20000  # MiB
START_GLOBAL=$(date +%s)

MODELS=(
  "mistral-7b|mistralai/Mistral-7B-Instruct-v0.3|18001"
  "qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct|18002"
  "llama3.1-8b|meta-llama/Llama-3.1-8B-Instruct|18003"
  "qwen2.5-14b|Qwen/Qwen2.5-14B-Instruct|18004"
  "yi-34b|01-ai/Yi-34B-Chat|18005"
  "codellama-34b|codellama/CodeLlama-34b-Instruct-hf|18006"
  "mixtral-8x7b|mistralai/Mixtral-8x7B-Instruct-v0.1|18007"
)

TOTAL_MODELS=${#MODELS[@]}
COUNT=0

export CUDA_VISIBLE_DEVICES=$GPU_ID
source .venv/bin/activate

echo "Starting sequential shard generation..."
echo "Total models: $TOTAL_MODELS"
echo "-----------------------------------------"

for ENTRY in "${MODELS[@]}"; do

  COUNT=$((COUNT+1))
  IFS="|" read -r MODEL_KEY HF_ID PORT <<< "$ENTRY"

  echo ""
  echo "[$COUNT/$TOTAL_MODELS] Processing $MODEL_KEY"
  echo "-----------------------------------------"

  START_MODEL=$(date +%s)

  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID)
  echo "Free GPU memory: ${FREE_MEM} MiB"

  if [ "$FREE_MEM" -lt "$MIN_FREE_MEM" ]; then
      echo "⚠ Skipping $MODEL_KEY (low memory)"
      continue
  fi

  # Launch vLLM
  python -m vllm.entrypoints.openai.api_server \
    --model $HF_ID \
    --port $PORT \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.90 \
    --served-model-name $MODEL_KEY \
    > logs_${MODEL_KEY}.txt 2>&1 &

  VLLM_PID=$!
  echo "Launched PID: $VLLM_PID"

  # Wait until ready
  READY=false
  for i in $(seq 1 $MAX_WAIT); do
      sleep 2
      if curl -s http://localhost:$PORT/v1/models > /dev/null; then
          READY=true
          break
      fi
  done

  if [ "$READY" = false ]; then
      echo "❌ $MODEL_KEY failed to start"
      kill $VLLM_PID 2>/dev/null || true
      continue
  fi

  echo "$MODEL_KEY is ready"

  # Run builder (tolerant)
  python datasets/build_rq2_interaction_logs.py \
    --config configs/rq2_shards/${MODEL_KEY}.yaml \
    || echo "⚠ Builder failed for $MODEL_KEY (continuing)"

  # Stop only this server
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true

  END_MODEL=$(date +%s)
  DURATION=$((END_MODEL - START_MODEL))

  echo "Finished $MODEL_KEY in $DURATION seconds"

  # ETA calculation
  ELAPSED=$((END_MODEL - START_GLOBAL))
  AVG=$((ELAPSED / COUNT))
  REMAINING=$((TOTAL_MODELS - COUNT))
  ETA=$((AVG * REMAINING))

  echo "Estimated time remaining: $ETA seconds (~$((ETA/60)) min)"

  sleep 5

done

END_GLOBAL=$(date +%s)
TOTAL_TIME=$((END_GLOBAL - START_GLOBAL))

echo ""
echo "========================================="
echo "All models processed."
echo "Total runtime: $TOTAL_TIME seconds (~$((TOTAL_TIME/60)) min)"
echo "========================================="