#!/usr/bin/env bash
set -euo pipefail
# Run from ProMoRouter, in the existing graph-router-2 Python environment.
graphrouter_data="${1:-$HOME/repos/graph-router-2/data}"
graphrouter_device="${2:-cuda:0}"
python train_graphrouter_unseen.py \
  --data-path "$graphrouter_data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl" \
  --query-emb-path "$graphrouter_data/router/query_embeddings.pt" \
  --task-emb-path "$graphrouter_data/router/task_embeddings.pt" \
  --model-emb-path "$graphrouter_data/router/model_embeddings.pt" \
  --held-out-model qwen2.5-14b --seed 1 --reward-weight 0.9 \
  --small-support 8 --large-support 80 --device "$graphrouter_device"
