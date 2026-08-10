#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
GPU=0
SEED=13
OUTPUT_DIR=""
STALE_TIMEOUT_MINUTES=45
STAGE_TIMEOUT_HOURS=12
LOG_EVERY_BATCHES=25
SKIP_PREFLIGHT=0
DRY_RUN=0
RUN_CONTROLS=always

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_qness_proportional.sh [options]

Options:
  --gpu N                    GPU index (default: 0)
  --seed N                   Random seed (default: 13)
  --output-dir PATH          New output directory; default is timestamped under runs/
  --log-every-batches N      Progress interval (default: 25)
  --run-controls MODE        always or never (default: always)
  --stale-timeout-minutes N Stop if no heartbeat progress (default: 45)
  --stage-timeout-hours N   Maximum whole-gate duration (default: 12)
  --skip-preflight           Skip clean/data/GPU/pytest checks
  --dry-run                  Print all child commands without creating output
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=$2; shift ;;
    --seed) SEED=$2; shift ;;
    --output-dir) OUTPUT_DIR=$2; shift ;;
    --log-every-batches) LOG_EVERY_BATCHES=$2; shift ;;
    --run-controls) RUN_CONTROLS=$2; shift ;;
    --stale-timeout-minutes) STALE_TIMEOUT_MINUTES=$2; shift ;;
    --stage-timeout-hours) STAGE_TIMEOUT_HOURS=$2; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer." >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "Log interval must be positive." >&2; exit 2; }
[[ "${STALE_TIMEOUT_MINUTES}" =~ ^[1-9][0-9]*$ ]] || { echo "Stale timeout must be positive." >&2; exit 2; }
[[ "${STAGE_TIMEOUT_HOURS}" =~ ^[1-9][0-9]*$ ]] || { echo "Stage timeout must be positive." >&2; exit 2; }
[[ "${RUN_CONTROLS}" == always || "${RUN_CONTROLS}" == never ]] || {
  echo "run-controls must be always or never." >&2
  exit 2
}

cd "${ROOT}"
nvidia-smi -i "${GPU}" --query-gpu=name --format=csv,noheader >/dev/null
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${OUTPUT_DIR:-runs/retacred_qness_proportional_${STAMP}_seed${SEED}}
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse output directory: ${RUN_DIR}" >&2; exit 1; }
HEARTBEAT="${RUN_DIR}.heartbeat"

if [[ ${DRY_RUN} -eq 0 && ${SKIP_PREFLIGHT} -eq 0 ]]; then
  mkdir -p "$(dirname "${RUN_DIR}")"
  bash scripts/check_retacred_dual_qres_full.sh 2>&1 | tee "${RUN_DIR}.preflight.log"
fi

echo "RUN_DIR=${RUN_DIR}"
echo "GPU=${GPU} SEED=${SEED} RUN_CONTROLS=${RUN_CONTROLS}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  "${PYTHON_BIN}" experiments/run_relation_qness_proportional_gate.py \
    --output_dir "${RUN_DIR}" --seed "${SEED}" --device cuda \
    --log_every_batches "${LOG_EVERY_BATCHES}" --run_controls "${RUN_CONTROLS}" --dry_run
  exit 0
fi

mkdir -p "$(dirname "${HEARTBEAT}")"
touch "${HEARTBEAT}"
set +e
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
  Q_ATTENTION_HEARTBEAT_FILE="${HEARTBEAT}" \
  "${PYTHON_BIN}" scripts/run_with_health_watchdog.py \
    --heartbeat-file "${HEARTBEAT}" \
    --stale-seconds "$((STALE_TIMEOUT_MINUTES * 60))" \
    --timeout-seconds "$((STAGE_TIMEOUT_HOURS * 3600))" \
    -- "${PYTHON_BIN}" experiments/run_relation_qness_proportional_gate.py \
      --output_dir "${RUN_DIR}" --seed "${SEED}" --device cuda \
      --log_every_batches "${LOG_EVERY_BATCHES}" --run_controls "${RUN_CONTROLS}" \
      2>&1 | tee "${RUN_DIR}.console.log"
STATUS=${PIPESTATUS[0]}
set -e
rm -f "${HEARTBEAT}"
if [[ ${STATUS} -ne 0 ]]; then
  echo "Q-NESS proportional gate failed: ${RUN_DIR}" >&2
  exit "${STATUS}"
fi
echo "Q-NESS proportional gate complete: ${RUN_DIR}"
