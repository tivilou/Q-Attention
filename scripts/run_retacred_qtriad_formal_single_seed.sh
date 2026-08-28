#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_SPEC=0
OUTPUT_DIR=
LOG_EVERY_BATCHES=50
REPORT_DIR=
SKIP_PREFLIGHT=0
DRY_RUN=0

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ "${PYTHON_BIN}" == */* ]]; then
      [[ -x "${PYTHON_BIN}" ]] && { printf '%s\n' "${PYTHON_BIN}"; return; }
    elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"; return
    fi
    echo "PYTHON_BIN is not executable or not on PATH: ${PYTHON_BIN}" >&2
    return 1
  fi
  for candidate in python python3; do
    command -v "${candidate}" >/dev/null 2>&1 && { command -v "${candidate}"; return; }
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN." >&2
  return 1
}

usage() { echo "Usage: bash scripts/run_retacred_qtriad_formal_single_seed.sh [--gpu N[,N...]|auto] [--output-dir PATH] [--report-dir PATH] [--log-every-batches N] [--skip-preflight] [--dry-run]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|--gpus|--output-dir|--report-dir|--log-every-batches)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --gpu|--gpus) GPU_SPEC=$2;;
        --output-dir) OUTPUT_DIR=$2;;
        --report-dir) REPORT_DIR=$2;;
        --log-every-batches) LOG_EVERY_BATCHES=$2;;
      esac
      shift 2;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
done

PYTHON_BIN=$(resolve_python_bin)
export PYTHON_BIN
cd "${ROOT}"
[[ "${GPU_SPEC}" == "auto" || "${GPU_SPEC}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--gpu/--gpus must be auto or a comma-separated list of non-negative integers." >&2; exit 2; }
if [[ "${GPU_SPEC}" != "auto" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${GPU_SPEC}"
  declare -A SEEN_GPU_IDS=()
  for GPU_ID in "${GPU_IDS[@]}"; do
    [[ -z "${SEEN_GPU_IDS[${GPU_ID}]:-}" ]] || { echo "Duplicate GPU ID: ${GPU_ID}" >&2; exit 2; }
    SEEN_GPU_IDS[${GPU_ID}]=1
  done
fi
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "--log-every-batches must be positive." >&2; exit 2; }
if [[ "${GPU_SPEC}" != "auto" ]]; then
  nvidia-smi -i "${GPU_SPEC}" --query-gpu=name --format=csv,noheader >/dev/null || { echo "GPU ${GPU_SPEC} is unavailable." >&2; exit 1; }
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${OUTPUT_DIR:-runs/retacred_qtriad_formal_single_seed/${STAMP}_seed13}
RUN_DIR=$(readlink -m "${RUN_DIR}")
case "${RUN_DIR}" in "${ROOT}/runs/"*) ;; *) echo "Output must be under runs/." >&2; exit 2;; esac
[[ "$(basename "${RUN_DIR}")" == *_seed13 ]] || { echo "Output directory must end with _seed13." >&2; exit 2; }
[[ ${DRY_RUN} -eq 1 || ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse output directory." >&2; exit 1; }
[[ ${SKIP_PREFLIGHT} -eq 1 || ${DRY_RUN} -eq 1 ]] || bash scripts/check_retacred_qtriad_formal_single_seed.sh

COMMAND=(
  "${PYTHON_BIN}" experiments/run_qtriad_relation_transfer.py
  --config configs/retacred_qtriad_formal_single_seed.json
  --device cuda
  --gpus "${GPU_SPEC}"
  --seed 13
  --output-dir "${RUN_DIR}"
  --log-every-batches "${LOG_EVERY_BATCHES}"
  --started-at-utc "${STAMP}"
  --python-bin "${PYTHON_BIN}"
)
if [[ "${GPU_SPEC}" == "auto" ]]; then
  COMMAND+=(--hardware-profile auto)
fi
if [[ ${DRY_RUN} -eq 1 ]]; then
  if [[ "${GPU_SPEC}" != "auto" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_SPEC}"
  fi
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

LOG_TMP=$(mktemp)
cleanup_log() {
  if [[ -f "${LOG_TMP}" && -d "${RUN_DIR}" ]]; then
    mkdir -p "${RUN_DIR}/logs"
    mv "${LOG_TMP}" "${RUN_DIR}/logs/run.log"
  else
    rm -f "${LOG_TMP}"
  fi
}
trap cleanup_log EXIT

set +e
if [[ "${GPU_SPEC}" == "auto" ]]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "${COMMAND[@]}" 2>&1 | tee "${LOG_TMP}"
else
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_SPEC}" \
    "${COMMAND[@]}" 2>&1 | tee "${LOG_TMP}"
fi
RUN_STATUS=${PIPESTATUS[0]}
set -e
if [[ ${RUN_STATUS} -ne 0 ]]; then
  exit "${RUN_STATUS}"
fi
[[ -f "${RUN_DIR}/RUN_COMPLETE" && -f "${RUN_DIR}/run_summary.json" && -f "${RUN_DIR}/run_summary.md" ]] || { echo "Run did not produce complete markers and summaries." >&2; exit 1; }
EXPORT_COMMAND=(bash scripts/export_retacred_qtriad_formal_single_seed_report.sh --run-dir "${RUN_DIR}")
if [[ -n "${REPORT_DIR}" ]]; then
  EXPORT_COMMAND+=(--report-dir "${REPORT_DIR}")
fi
"${EXPORT_COMMAND[@]}"
