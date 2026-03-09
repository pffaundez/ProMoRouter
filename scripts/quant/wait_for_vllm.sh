#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Uso: $0 <port> <served_name> [timeout_seconds]"
  exit 1
fi

PORT="$1"
SERVED_NAME="$2"
TIMEOUT="${3:-600}"

START_TS=$(date +%s)

echo "[INFO] Esperando servidor en http://localhost:${PORT}/v1/models"

while true; do
  if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${SERVED_NAME}"; then
    echo "[OK] Servidor listo y modelo detectado: ${SERVED_NAME}"
    exit 0
  fi

  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))
  if [ "${ELAPSED}" -ge "${TIMEOUT}" ]; then
    echo "[ERROR] Timeout esperando a ${SERVED_NAME} en puerto ${PORT}"
    exit 1
  fi

  sleep 5
done
