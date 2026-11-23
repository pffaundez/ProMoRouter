#!/usr/bin/env bash
set -euo pipefail
fixed_data="${1:-$HOME/repos/graph-router-2/data}"
python audit_graphrouter_joint_fixed.py \
  --data-path "$fixed_data/interaction_logs/grpp_il_v1/router_bipartite_qnorm_complete.jsonl"
