#!/usr/bin/env bash
set -euo pipefail
# Physical GPU 0 only; two matched arms in one process, no sweep.
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
episodic_data="${1:-$HOME/repos/graph-router-2/data}"
python -c 'import torch; assert torch.cuda.is_available(), "GPU 0 unavailable: refusing CPU fallback"; print("GPU 0:", torch.cuda.get_device_name(0))'
python train_graphrouter_joint_episodic.py \
  --data-path "$episodic_data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl" \
  --query-emb-path "$episodic_data/router/query_embeddings.pt" \
  --task-emb-path "$episodic_data/router/task_embeddings.pt" \
  --prompt-emb-path "$episodic_data/router/prompt_embeddings.pt" \
  --model-emb-path "$episodic_data/router/model_embeddings.pt" \
  --held-out-model qwen2.5-14b --seed 1 --reward-weight 0.9 \
  --epochs 1000 --device cuda:0
