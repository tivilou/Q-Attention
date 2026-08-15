#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONUNBUFFERED
SEED=13
GPU_SPEC=0
OUTPUT_DIR=
LOG_EVERY_BATCHES=50
STALE_TIMEOUT_MINUTES=45
RUN_TIMEOUT_HOURS=48
DASHBOARD_INTERVAL_SECONDS=10
PROGRESS_FORMAT=both
PARALLEL_MODE=serial
SKIP_PREFLIGHT=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_qvres_relation_transfer_full.sh [options]

Options:
  --seed N                    Random seed (default: 13)
  --gpus SPEC                 GPU indexes, comma list, or auto (default: 0)
  --parallel-mode MODE        serial or selectors (default: serial)
  --output-dir PATH           New output directory under runs/
  --log-every-batches N       Progress interval (default: 50)
  --stale-timeout-minutes N  Stop after no heartbeat (default: 45)
  --run-timeout-hours N       Hard timeout for each stage (default: 48)
  --dashboard-interval-seconds N  Central status refresh interval (default: 10)
  --progress-format MODE      json or both (default: both)
  --skip-preflight            Skip environment/data/test checks
  --dry-run                   Print the command without running it
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED=$2; shift ;;
    --gpus) GPU_SPEC=$2; shift ;;
    --parallel-mode) PARALLEL_MODE=$2; shift ;;
    --output-dir) OUTPUT_DIR=$2; shift ;;
    --log-every-batches) LOG_EVERY_BATCHES=$2; shift ;;
    --stale-timeout-minutes) STALE_TIMEOUT_MINUTES=$2; shift ;;
    --run-timeout-hours) RUN_TIMEOUT_HOURS=$2; shift ;;
    --dashboard-interval-seconds) DASHBOARD_INTERVAL_SECONDS=$2; shift ;;
    --progress-format) PROGRESS_FORMAT=$2; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }
[[ "${PARALLEL_MODE}" == serial || "${PARALLEL_MODE}" == selectors ]] || { echo "Parallel mode must be serial or selectors." >&2; exit 2; }
[[ "${GPU_SPEC}" == auto || "${GPU_SPEC}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid --gpus value: ${GPU_SPEC}" >&2; exit 2; }
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "Log interval must be positive." >&2; exit 2; }
[[ "${STALE_TIMEOUT_MINUTES}" =~ ^[1-9][0-9]*$ ]] || { echo "Stale timeout must be positive." >&2; exit 2; }
[[ "${RUN_TIMEOUT_HOURS}" =~ ^[1-9][0-9]*$ ]] || { echo "Run timeout must be positive." >&2; exit 2; }
[[ "${DASHBOARD_INTERVAL_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "${DASHBOARD_INTERVAL_SECONDS}" != 0 ]] || { echo "Dashboard interval must be positive." >&2; exit 2; }
[[ "${PROGRESS_FORMAT}" == json || "${PROGRESS_FORMAT}" == both ]] || { echo "Progress format must be json or both." >&2; exit 2; }

cd "${ROOT}"
if [[ "${PARALLEL_MODE}" == serial ]]; then
  if [[ "${GPU_SPEC}" == auto ]]; then
    GPU_SPEC=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  fi
  [[ "${GPU_SPEC}" =~ ^[0-9]+$ ]] || { echo "Serial mode requires exactly one GPU." >&2; exit 2; }
  nvidia-smi -i "${GPU_SPEC}" --query-gpu=name --format=csv,noheader >/dev/null || {
    echo "GPU ${GPU_SPEC} is not available." >&2
    exit 1
  }
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [[ "${PARALLEL_MODE}" == selectors ]]; then
  RUN_DIR=${OUTPUT_DIR:-runs/q_vres_relation_transfer_full/${STAMP}_seed${SEED}_selector_parallel}
else
  RUN_DIR=${OUTPUT_DIR:-runs/q_vres_relation_transfer_full/${STAMP}_seed${SEED}}
fi
RUN_DIR=$(readlink -m "${RUN_DIR}")
case "${RUN_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "Output directory must be inside ${ROOT}/runs" >&2; exit 2 ;;
esac
if [[ ${DRY_RUN} -eq 0 && -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${RUN_DIR}" >&2
  exit 1
fi

PREFLIGHT_LOG=
cleanup_preflight() {
  [[ -z "${PREFLIGHT_LOG}" || ! -f "${PREFLIGHT_LOG}" ]] || rm -f "${PREFLIGHT_LOG}"
}
persist_preflight() {
  if [[ -n "${PREFLIGHT_LOG}" && -f "${PREFLIGHT_LOG}" && -d "${RUN_DIR}" ]]; then
    cp "${PREFLIGHT_LOG}" "${RUN_DIR}/preflight.log"
  fi
}
trap cleanup_preflight EXIT
if [[ ${DRY_RUN} -eq 0 && ${SKIP_PREFLIGHT} -eq 0 ]]; then
  PREFLIGHT_LOG=$(mktemp)
  set +e
  bash scripts/check_qvres_relation_transfer_full.sh 2>&1 | tee "${PREFLIGHT_LOG}"
  PREFLIGHT_STATUS=${PIPESTATUS[0]}
  set -e
  [[ ${PREFLIGHT_STATUS} -eq 0 ]] || exit "${PREFLIGHT_STATUS}"
fi

if [[ "${PARALLEL_MODE}" == selectors ]]; then
  SELECTOR_COMMAND=(
    "${PYTHON_BIN}" scripts/run_qvres_relation_transfer_selector_parallel.py
    --config configs/q_vres_relation_transfer_full.json
    --seed "${SEED}"
    --gpus "${GPU_SPEC}"
    --output-dir "${RUN_DIR}"
    --log-every-batches "${LOG_EVERY_BATCHES}"
    --stale-timeout-minutes "${STALE_TIMEOUT_MINUTES}"
    --run-timeout-hours "${RUN_TIMEOUT_HOURS}"
    --dashboard-interval-seconds "${DASHBOARD_INTERVAL_SECONDS}"
    --progress-format "${PROGRESS_FORMAT}"
  )
  if [[ ${DRY_RUN} -eq 1 ]]; then
    SELECTOR_COMMAND+=(--dry-run)
  fi
  set +e
  "${SELECTOR_COMMAND[@]}"
  STATUS=$?
  set -e
  persist_preflight
  exit "${STATUS}"
fi

COMMAND=(
  "${PYTHON_BIN}" experiments/run_q_causal_value_evidence_relation_transfer.py
  --config configs/q_vres_relation_transfer_full.json
  --formal-experiment
  --device cuda
  --seed "${SEED}"
  --output-dir "${RUN_DIR}"
  --log_every_batches "${LOG_EVERY_BATCHES}"
)

if [[ ${DRY_RUN} -eq 1 ]]; then
  printf '[dry-run] seed=%s gpu=%s run_dir=%s\n' "${SEED}" "${GPU_SPEC}" "${RUN_DIR}"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_SPEC}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/status"
HEARTBEAT_FILE=${RUN_DIR}/status/run.heartbeat
STATUS_FILE=${RUN_DIR}/status/run.env
LOG_FILE=${RUN_DIR}/logs/run.log
printf 'STATUS=running\nSEED=%s\nGPU_ID=%s\nSTARTED_AT=%s\nHEARTBEAT_FILE=%s\n' \
  "${SEED}" "${GPU_SPEC}" "$(date -Iseconds)" "${HEARTBEAT_FILE}" > "${STATUS_FILE}"
touch "${HEARTBEAT_FILE}"
printf '[%s] START seed=%s gpu=%s run_dir=%s\n' "$(date -Iseconds)" "${SEED}" "${GPU_SPEC}" "${RUN_DIR}" | tee "${LOG_FILE}"

set +e
CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${GPU_SPEC}" \
  Q_ATTENTION_PROGRESS_FORMAT="${PROGRESS_FORMAT}" \
  Q_ATTENTION_HEARTBEAT_FILE="${HEARTBEAT_FILE}" \
  "${PYTHON_BIN}" scripts/run_with_health_watchdog.py \
  --heartbeat-file "${HEARTBEAT_FILE}" \
  --stale-seconds "$((STALE_TIMEOUT_MINUTES * 60))" \
  --timeout-seconds "$((RUN_TIMEOUT_HOURS * 3600))" \
  -- "${COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e
persist_preflight

if [[ ${STATUS} -ne 0 || ! -f "${RUN_DIR}/run_summary.json" || ! -f "${RUN_DIR}/run_summary.md" ]]; then
  printf 'STATUS=failed\nSEED=%s\nGPU_ID=%s\nFAILED_AT=%s\nEXIT_CODE=%s\nHEARTBEAT_FILE=%s\n' \
    "${SEED}" "${GPU_SPEC}" "$(date -Iseconds)" "${STATUS}" "${HEARTBEAT_FILE}" > "${STATUS_FILE}"
  printf '%s\n' "$(date -Iseconds)" > "${RUN_DIR}/RUN_FAILED"
  echo "FAILED seed=${SEED} gpu=${GPU_SPEC} exit=${STATUS}" >&2
  exit "${STATUS:-1}"
fi

printf 'STATUS=complete\nSEED=%s\nGPU_ID=%s\nCOMPLETED_AT=%s\nHEARTBEAT_FILE=%s\n' \
  "${SEED}" "${GPU_SPEC}" "$(date -Iseconds)" "${HEARTBEAT_FILE}" > "${STATUS_FILE}"
printf '%s\n' "$(date -Iseconds)" > "${RUN_DIR}/RUN_COMPLETE"
printf '[%s] COMPLETE seed=%s gpu=%s run_dir=%s\n' "$(date -Iseconds)" "${SEED}" "${GPU_SPEC}" "${RUN_DIR}" | tee -a "${LOG_FILE}"
