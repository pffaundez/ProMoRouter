set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Uso: $0 <ruta_tsv>"
  exit 1
fi

ROUND_FILE="$1"

while IFS=$'\t' read -r GPU_ID PORT SERVED_NAME HF_MODEL_REPO QUANTIZATION MAX_MODEL_LEN GPU_MEM_UTIL; do
  if [[ -z "${GPU_ID}" || "${GPU_ID}" == \#* ]]; then
    continue
  fi

  echo
  echo "============================================================"
  echo "[ROUND] Modelo: ${SERVED_NAME}"
  echo "============================================================"
  echo

  echo "[STEP] Arranca este modelo en una Terminal 1 aparte con:"
  echo "scripts/quant/start_quant_vllm.sh ${GPU_ID} ${PORT} ${SERVED_NAME} ${HF_MODEL_REPO} ${QUANTIZATION} ${MAX_MODEL_LEN} ${GPU_MEM_UTIL}"
  echo
  echo "[STEP] En Terminal 2 corre:"
  echo "scripts/quant/wait_for_vllm.sh ${PORT} ${SERVED_NAME}"
  echo "scripts/quant/api_smoke.sh ${PORT} ${SERVED_NAME}"
  echo
  echo "[STEP] Cuando termines, mata el servidor y presiona ENTER para seguir al siguiente modelo."
  read -r _ < /dev/tty

done < "${ROUND_FILE}"