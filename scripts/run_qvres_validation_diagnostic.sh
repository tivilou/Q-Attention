#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=0
BATCH_SIZE=8
MAX_RECORDS=0
LOG_EVERY_BATCHES=50

usage() {
  cat <<'EOF'
Usage: bash scripts/run_qvres_validation_diagnostic.sh RUN_DIR [options]

Options:
  --gpu N                   GPU index (default: 0)
  --batch-size N            Evaluation batch size (default: 8)
  --max-records N           Validation records; 0 means all (default: 0)
  --log-every-batches N     Progress interval (default: 50)
  -h, --help                Show this help
EOF
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
if [[ "$1" == -h || "$1" == --help ]]; then
  usage
  exit 0
fi
RUN_DIR=$1
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=$2; shift ;;
    --batch-size) BATCH_SIZE=$2; shift ;;
    --max-records) MAX_RECORDS=$2; shift ;;
    --log-every-batches) LOG_EVERY_BATCHES=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer." >&2; exit 2; }
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch size must be positive." >&2; exit 2; }
[[ "${MAX_RECORDS}" =~ ^[0-9]+$ ]] || { echo "Max records must be non-negative." >&2; exit 2; }
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "Log interval must be positive." >&2; exit 2; }

cd "${ROOT}"
RUN_DIR=$(readlink -m "${RUN_DIR}")
[[ -d "${RUN_DIR}" ]] || { echo "Run directory does not exist: ${RUN_DIR}" >&2; exit 1; }
nvidia-smi -i "${GPU}" --query-gpu=name --format=csv,noheader >/dev/null || {
  echo "GPU ${GPU} is not available." >&2
  exit 1
}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT_DIR="${ROOT}/reports/q_vres_relation_transfer/${STAMP}-validation-diagnostic"
echo "[qvres-diagnostic] source=${RUN_DIR} gpu=${GPU} report_dir=${REPORT_DIR}"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="${GPU}" \
"${PYTHON_BIN}" experiments/diagnose_qvres_relation_transfer.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${REPORT_DIR}" \
  --split valid \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --max-records "${MAX_RECORDS}" \
  --log-every-batches "${LOG_EVERY_BATCHES}"

echo "[qvres-diagnostic] complete"
echo "REPORT_DIR=${REPORT_DIR}"
echo "Submit only:"
echo "  ${REPORT_DIR}/diagnostic_summary.json"
echo "  ${REPORT_DIR}/diagnostic_summary.md"
