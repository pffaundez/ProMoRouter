#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Uso: $0 <served_name>"
  exit 1
fi

SERVED_NAME="$1"
LOG_FILE="logs/quant/${SERVED_NAME}.log"

echo "===== nvidia-smi watch ====="
echo "Usa Ctrl+C para salir"
while true; do
  clear
  date
  echo
  nvidia-smi
  echo
  echo "===== tail ${LOG_FILE} ====="
  if [ -f "${LOG_FILE}" ]; then
    tail -n 30 "${LOG_FILE}"
  else
    echo "Log aún no existe"
  fi
  sleep 5
done
