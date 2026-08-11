#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONUNBUFFERED
GPU_SPEC=0
GPU_IDS=()
SEED=13
OUTPUT_DIR=""
STALE_TIMEOUT_MINUTES=45
STAGE_TIMEOUT_HOURS=12
LOG_EVERY_BATCHES=25
PROGRESS_FORMAT=both
SKIP_PREFLIGHT=0
DRY_RUN=0
RUN_CONTROLS=always
MIN_COMMIT=f6e2bc5

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_qness_proportional.sh [options]

Options:
  --gpu N                    One GPU index; compatibility alias for --gpus N
  --gpus SPEC                GPU indexes such as 0,1,2 or auto (default: 0)
  --seed N                   Random seed (default: 13)
  --output-dir PATH          New output directory; default is timestamped under runs/
  --log-every-batches N      Progress interval (default: 25)
  --progress-format MODE     json or both (default: both)
  --run-controls MODE        always or never (default: always)
  --stale-timeout-minutes N  Stop if no heartbeat progress (default: 45)
  --stage-timeout-hours N    Maximum whole-gate duration (default: 12)
  --skip-preflight           Skip clean/data/GPU/pytest checks
  --dry-run                  Print all child commands without creating output
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_SPEC=$2; shift ;;
    --gpus) GPU_SPEC=$2; shift ;;
    --seed) SEED=$2; shift ;;
    --output-dir) OUTPUT_DIR=$2; shift ;;
    --log-every-batches) LOG_EVERY_BATCHES=$2; shift ;;
    --progress-format) PROGRESS_FORMAT=$2; shift ;;
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

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "Log interval must be positive." >&2; exit 2; }
[[ "${STALE_TIMEOUT_MINUTES}" =~ ^[1-9][0-9]*$ ]] || { echo "Stale timeout must be positive." >&2; exit 2; }
[[ "${STAGE_TIMEOUT_HOURS}" =~ ^[1-9][0-9]*$ ]] || { echo "Stage timeout must be positive." >&2; exit 2; }
[[ "${RUN_CONTROLS}" == always || "${RUN_CONTROLS}" == never ]] || {
  echo "run-controls must be always or never." >&2
  exit 2
}
[[ "${PROGRESS_FORMAT}" == json || "${PROGRESS_FORMAT}" == both ]] || {
  echo "progress-format must be json or both." >&2
  exit 2
}

cd "${ROOT}"

resolve_gpus() {
  local gpu
  local -A seen=()
  local resolved=()
  if [[ "${GPU_SPEC}" == auto ]]; then
    mapfile -t resolved < <(
      nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' '
    )
  else
    IFS=',' read -r -a resolved <<< "${GPU_SPEC}"
  fi
  for gpu in "${resolved[@]}"; do
    gpu=${gpu//[[:space:]]/}
    [[ "${gpu}" =~ ^[0-9]+$ ]] || {
      echo "GPU indexes must be non-negative integers: ${gpu}" >&2
      exit 2
    }
    [[ -z "${seen[${gpu}]+x}" ]] || {
      echo "GPU index appears more than once: ${gpu}" >&2
      exit 2
    }
    GPU_IDS+=("${gpu}")
    seen[${gpu}]=1
  done
  [[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No GPU was selected." >&2; exit 1; }
  for gpu in "${GPU_IDS[@]}"; do
    nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader >/dev/null || {
      echo "GPU ${gpu} is not available." >&2
      exit 1
    }
  done
}

resolve_gpus
GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${OUTPUT_DIR:-runs/retacred_qness_proportional_${STAMP}_seed${SEED}}
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse output directory: ${RUN_DIR}" >&2; exit 1; }
HEARTBEAT="${RUN_DIR}.heartbeat"

if [[ ${DRY_RUN} -eq 0 && ${SKIP_PREFLIGHT} -eq 0 ]]; then
  mkdir -p "$(dirname "${RUN_DIR}")"
  MIN_COMMIT="${MIN_COMMIT}" bash scripts/check_retacred_dual_qres_full.sh 2>&1 | tee "${RUN_DIR}.preflight.log"
fi

echo "RUN_DIR=${RUN_DIR}"
echo "GPU_IDS=${GPU_LIST} SEED=${SEED} RUN_CONTROLS=${RUN_CONTROLS}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  "${PYTHON_BIN}" experiments/run_relation_qness_proportional_gate.py \
    --output_dir "${RUN_DIR}" --seed "${SEED}" --device cuda \
    --gpus "${GPU_LIST}" --log_every_batches "${LOG_EVERY_BATCHES}" \
    --run_controls "${RUN_CONTROLS}" --dry_run
  exit 0
fi

mkdir -p "$(dirname "${HEARTBEAT}")"
touch "${HEARTBEAT}"
set +e
Q_ATTENTION_HEARTBEAT_FILE="${HEARTBEAT}" \
  Q_ATTENTION_PROGRESS_FORMAT="${PROGRESS_FORMAT}" \
  "${PYTHON_BIN}" scripts/run_with_health_watchdog.py \
    --heartbeat-file "${HEARTBEAT}" \
    --stale-seconds "$((STALE_TIMEOUT_MINUTES * 60))" \
    --timeout-seconds "$((STAGE_TIMEOUT_HOURS * 3600))" \
    -- "${PYTHON_BIN}" experiments/run_relation_qness_proportional_gate.py \
      --output_dir "${RUN_DIR}" --seed "${SEED}" --device cuda \
      --gpus "${GPU_LIST}" --log_every_batches "${LOG_EVERY_BATCHES}" \
      --run_controls "${RUN_CONTROLS}" \
      2>&1 | tee "${RUN_DIR}.console.log"
STATUS=${PIPESTATUS[0]}
set -e
rm -f "${HEARTBEAT}"
if [[ ${STATUS} -ne 0 ]]; then
  echo "Q-NESS proportional gate failed: ${RUN_DIR}" >&2
  exit "${STATUS}"
fi
echo "Q-NESS proportional gate complete: ${RUN_DIR}"
