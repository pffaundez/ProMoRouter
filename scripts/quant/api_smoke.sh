#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Uso: $0 <port> <served_name>"
  exit 1
fi

PORT="$1"
SERVED_NAME="$2"

curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Reply with exactly: quant-smoke-ok\"}
    ],
    \"temperature\": 0.0,
    \"max_tokens\": 16
  }" | python -m json.tool