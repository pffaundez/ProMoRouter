#!/usr/bin/env bash
set -euo pipefail
reference_device="${1:-cuda:0}"
python prepare_graphrouter_reference.py
python train_graphrouter_reference.py --device "$reference_device" --epochs 1000
