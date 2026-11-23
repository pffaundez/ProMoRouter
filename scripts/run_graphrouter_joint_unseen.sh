#!/usr/bin/env bash
set -euo pipefail
joint_data="${1:-$HOME/repos/graph-router-2/data}"
joint_device="${2:-cuda:0}"
python train_graphrouter_joint_unseen.py \
  --data-path "$joint_data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl" \
  --query-emb-path "$joint_data/router/query_embeddings.pt" \
  --task-emb-path "$joint_data/router/task_embeddings.pt" \
  --prompt-emb-path "$joint_data/router/prompt_embeddings.pt" \
  --model-emb-path "$joint_data/router/model_embeddings.pt" \
  --held-out-model qwen2.5-14b --seed 1 --reward-weight 0.9 \
  --epochs 1000 --device "$joint_device"
