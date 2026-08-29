#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_SPEC=0
MODEL_PARALLEL_GPU_SPEC=
GPU_SPEC_EXPLICIT=0
OUTPUT_DIR=
RESUME_DIR=
IMPORT_BASELINE_FROM=
LOG_EVERY_BATCHES=50
CHECKPOINT_EVERY_BATCHES=50
HARDWARE_PROFILE=adaptive
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

check_gpu_capacity() {
  local spec="$1"
  local inventory selected index free
  inventory=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits) || {
    echo "Unable to query GPU memory with nvidia-smi." >&2
    return 1
  }
  declare -A FREE_MIB=()
  while IFS=',' read -r index free; do
    [[ "${index}" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || continue
    index="${index//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    FREE_MIB["${index}"]="${free}"
  done <<< "${inventory}"
  if [[ "${spec}" == "auto" ]]; then
    selected=()
    for index in "${!FREE_MIB[@]}"; do
      (( FREE_MIB["${index}"] >= 8192 )) && selected+=("${index}")
    done
  else
    IFS=',' read -r -a selected <<< "${spec}"
  fi
  [[ ${#selected[@]} -gt 0 ]] || { echo "No GPU selected." >&2; return 1; }
  local unsafe=0
  for index in "${selected[@]}"; do
    if [[ -z "${FREE_MIB[${index}]+present}" ]]; then
      echo "Requested GPU ${index} is unavailable." >&2
      unsafe=1
    elif (( FREE_MIB["${index}"] < 8192 )); then
      echo "GPU ${index} has only ${FREE_MIB[${index}]} MiB free; at least 8192 MiB is required." >&2
      unsafe=1
    fi
  done
  if (( unsafe )); then
    echo "Competing CUDA processes:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >&2 || true
    echo "Stop or wait for the competing process, then rerun the unchanged contract." >&2
    return 1
  fi
}

usage() { echo "Usage: bash scripts/run_retacred_qtriad_formal_single_seed.sh [--gpu N[,N...]|auto] [--model-parallel-gpus N,N] [--hardware-profile config|auto|adaptive|low_memory|balanced|high_memory] [--output-dir PATH | --resume RUN_DIR | --import-baseline-from OLD_RUN_DIR] [--report-dir PATH] [--log-every-batches N] [--checkpoint-every-batches N] [--skip-preflight] [--dry-run]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|--gpus|--model-parallel-gpus|--hardware-profile|--output-dir|--resume|--import-baseline-from|--report-dir|--log-every-batches|--checkpoint-every-batches)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --gpu|--gpus) GPU_SPEC=$2; GPU_SPEC_EXPLICIT=1;;
        --model-parallel-gpus) MODEL_PARALLEL_GPU_SPEC=$2;;
        --hardware-profile) HARDWARE_PROFILE=$2;;
        --output-dir) OUTPUT_DIR=$2;;
        --resume) RESUME_DIR=$2;;
        --import-baseline-from) IMPORT_BASELINE_FROM=$2;;
        --report-dir) REPORT_DIR=$2;;
        --log-every-batches) LOG_EVERY_BATCHES=$2;;
        --checkpoint-every-batches) CHECKPOINT_EVERY_BATCHES=$2;;
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
if [[ -n "${MODEL_PARALLEL_GPU_SPEC}" ]]; then
  [[ "${MODEL_PARALLEL_GPU_SPEC}" =~ ^[0-9]+(,[0-9]+)+$ ]] || { echo "--model-parallel-gpus must contain at least two comma-separated GPU IDs." >&2; exit 2; }
  IFS=',' read -r -a MODEL_PARALLEL_GPU_IDS <<< "${MODEL_PARALLEL_GPU_SPEC}"
  declare -A SEEN_MODEL_PARALLEL_GPU_IDS=()
  for GPU_ID in "${MODEL_PARALLEL_GPU_IDS[@]}"; do
    [[ -z "${SEEN_MODEL_PARALLEL_GPU_IDS[${GPU_ID}]:-}" ]] || { echo "Duplicate model-parallel GPU ID: ${GPU_ID}" >&2; exit 2; }
    SEEN_MODEL_PARALLEL_GPU_IDS[${GPU_ID}]=1
  done
  [[ "${GPU_SPEC_EXPLICIT}" -eq 0 || "${GPU_SPEC}" == "${MODEL_PARALLEL_GPU_SPEC}" ]] || { echo "--gpu/--gpus and --model-parallel-gpus must name the same GPUs." >&2; exit 2; }
  GPU_SPEC="${MODEL_PARALLEL_GPU_SPEC}"
fi
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "--log-every-batches must be positive." >&2; exit 2; }
[[ "${CHECKPOINT_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "--checkpoint-every-batches must be positive." >&2; exit 2; }
[[ "${HARDWARE_PROFILE}" =~ ^(config|auto|adaptive|low_memory|balanced|high_memory)$ ]] || { echo "Invalid --hardware-profile." >&2; exit 2; }
[[ -z "${RESUME_DIR}" || -z "${OUTPUT_DIR}" ]] || { echo "--resume and --output-dir are mutually exclusive." >&2; exit 2; }
[[ -z "${RESUME_DIR}" || -z "${IMPORT_BASELINE_FROM}" ]] || { echo "--resume and --import-baseline-from are mutually exclusive." >&2; exit 2; }
if [[ "${GPU_SPEC}" != "auto" ]]; then
  nvidia-smi -i "${GPU_SPEC}" --query-gpu=name --format=csv,noheader >/dev/null || { echo "GPU ${GPU_SPEC} is unavailable." >&2; exit 1; }
fi
check_gpu_capacity "${GPU_SPEC}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${RESUME_DIR:-${OUTPUT_DIR:-runs/retacred_qtriad_formal_single_seed/${STAMP}_seed13}}
RUN_DIR=$(readlink -m "${RUN_DIR}")
case "${RUN_DIR}" in "${ROOT}/runs/"*) ;; *) echo "Output must be under runs/." >&2; exit 2;; esac
[[ "$(basename "${RUN_DIR}")" == *_seed13 ]] || { echo "Output directory must end with _seed13." >&2; exit 2; }
if [[ -n "${RESUME_DIR}" ]]; then
  [[ -d "${RUN_DIR}" && -f "${RUN_DIR}/run_manifest.json" ]] || { echo "--resume requires an existing run with run_manifest.json." >&2; exit 1; }
else
  [[ ${DRY_RUN} -eq 1 || ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse output directory." >&2; exit 1; }
fi
[[ ${SKIP_PREFLIGHT} -eq 1 || ${DRY_RUN} -eq 1 ]] || bash scripts/check_retacred_qtriad_formal_single_seed.sh

COMMAND=(
  "${PYTHON_BIN}" experiments/run_qtriad_relation_transfer.py
  --config configs/retacred_qtriad_formal_single_seed.json
  --device cuda
  --seed 13
  --log-every-batches "${LOG_EVERY_BATCHES}"
  --checkpoint-every-batches "${CHECKPOINT_EVERY_BATCHES}"
  --python-bin "${PYTHON_BIN}"
)
if [[ -n "${RESUME_DIR}" ]]; then
  COMMAND+=(--resume "${RUN_DIR}")
else
  COMMAND+=(--output-dir "${RUN_DIR}")
  COMMAND+=(--started-at-utc "${STAMP}")
fi
if [[ -n "${IMPORT_BASELINE_FROM}" ]]; then
  COMMAND+=(--import-baseline-from "${IMPORT_BASELINE_FROM}")
fi
if [[ -n "${MODEL_PARALLEL_GPU_SPEC}" ]]; then
  [[ "${HARDWARE_PROFILE}" != "adaptive" ]] || { echo "--hardware-profile adaptive is supported for selector workers only; choose config/auto/low_memory/balanced/high_memory with --model-parallel-gpus." >&2; exit 2; }
  COMMAND+=(--model-parallel-gpus "${MODEL_PARALLEL_GPU_SPEC}")
  COMMAND+=(--hardware-profile "${HARDWARE_PROFILE}")
elif [[ "${GPU_SPEC}" == "auto" ]]; then
  COMMAND+=(--gpus "${GPU_SPEC}")
  COMMAND+=(--hardware-profile "${HARDWARE_PROFILE/config/auto}")
else
  COMMAND+=(--gpus "${GPU_SPEC}")
  COMMAND+=(--hardware-profile "${HARDWARE_PROFILE}")
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
    if [[ -n "${RESUME_DIR}" ]]; then
      mv "${LOG_TMP}" "${RUN_DIR}/logs/run.resume-$(date -u +%Y%m%dT%H%M%SZ).log"
    else
      mv "${LOG_TMP}" "${RUN_DIR}/logs/run.log"
    fi
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
if [[ ${RUN_STATUS} -eq 75 ]]; then
  echo "Run safely paused at a batch checkpoint; do not run the exporter. Resume with --resume '${RUN_DIR}'." >&2
  exit 75
fi
if [[ ${RUN_STATUS} -ne 0 ]]; then
  exit "${RUN_STATUS}"
fi
[[ -f "${RUN_DIR}/RUN_COMPLETE" && -f "${RUN_DIR}/run_summary.json" && -f "${RUN_DIR}/run_summary.md" ]] || { echo "Run did not produce complete markers and summaries." >&2; exit 1; }
EXPORT_COMMAND=(bash scripts/export_retacred_qtriad_formal_single_seed_report.sh --run-dir "${RUN_DIR}")
if [[ -n "${REPORT_DIR}" ]]; then
  EXPORT_COMMAND+=(--report-dir "${REPORT_DIR}")
fi
"${EXPORT_COMMAND[@]}"
