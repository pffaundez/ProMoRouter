#!/usr/bin/env bash
set -euo pipefail
threshold_data="${1:-$HOME/repos/graph-router-2/data}"
threshold_device="${2:-cuda:0}"
python train_support_threshold.py \
  --data-path "$threshold_data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl" \
  --query-emb-path "$threshold_data/router/query_embeddings.pt" \
  --task-emb-path "$threshold_data/router/task_embeddings.pt" \
  --prompt-emb-path "$threshold_data/router/prompt_embeddings.pt" \
  --model-emb-path "$threshold_data/router/model_embeddings.pt" \
  --device "$threshold_device"
