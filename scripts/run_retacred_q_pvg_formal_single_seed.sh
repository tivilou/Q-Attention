#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN=python3
RUN_DIR=""
REPORT_DIR=""
GPU_SPEC=auto
HARDWARE_PROFILE=adaptive
OUTPUT_DIR=""
RESUME_DIR=""
IMPORT_BASELINE_FROM=""
DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|--gpus) GPU_SPEC=$2; shift 2;;
    --hardware-profile) HARDWARE_PROFILE=$2; shift 2;;
    --output-dir) OUTPUT_DIR=$2; shift 2;;
    --resume) RESUME_DIR=$2; shift 2;;
    --import-baseline-from) IMPORT_BASELINE_FROM=$2; shift 2;;
    --report-dir) REPORT_DIR=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --allow-gpu-topology-change|--allow-code-update) EXTRA_ARGS+=("$1"); shift;;
    --log-every-batches|--checkpoint-every-batches) EXTRA_ARGS+=("$1" "$2"); shift 2;;
    --help|-h) exec "${PYTHON_BIN}" "${ROOT}/experiments/run_q_pvg_formal_single_seed.py" --help;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
cd "${ROOT}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -n "${RESUME_DIR}" && -n "${OUTPUT_DIR}" ]]; then echo "--resume and --output-dir are mutually exclusive" >&2; exit 2; fi
if [[ -n "${RESUME_DIR}" && -n "${IMPORT_BASELINE_FROM}" ]]; then echo "--resume and --import-baseline-from are mutually exclusive" >&2; exit 2; fi
RUN_DIR="${RESUME_DIR:-${OUTPUT_DIR:-runs/retacred_q_pvg_formal_single_seed/${STAMP}_seed13}}"
if [[ -z "${RESUME_DIR}" ]]; then
  "${PYTHON_BIN}" scripts/check_retacred_q_pvg_formal_single_seed.py --config configs/retacred_q_pvg_formal_single_seed.json --fresh-run
fi
COMMAND=("${PYTHON_BIN}" experiments/run_q_pvg_formal_single_seed.py --config configs/retacred_q_pvg_formal_single_seed.json --device cuda --gpus "${GPU_SPEC}" --hardware-profile "${HARDWARE_PROFILE}" --python-bin "${PYTHON_BIN}")
if [[ -n "${RESUME_DIR}" ]]; then COMMAND+=(--resume "${RUN_DIR}"); else COMMAND+=(--output-dir "${RUN_DIR}" --started-at-utc "${STAMP}"); fi
[[ -n "${IMPORT_BASELINE_FROM}" ]] && COMMAND+=(--import-baseline-from "${IMPORT_BASELINE_FROM}")
COMMAND+=("${EXTRA_ARGS[@]}")
if [[ ${DRY_RUN} -eq 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
"${COMMAND[@]}"
[[ -f "${RUN_DIR}/RUN_COMPLETE" ]] || { echo "formal run did not complete; exporter not started" >&2; exit 1; }
EXPORT_ARGS=(--run-dir "${RUN_DIR}")
[[ -n "${REPORT_DIR}" ]] && EXPORT_ARGS+=(--report-dir "${REPORT_DIR}")
exec scripts/export_retacred_q_pvg_formal_single_seed_report.sh "${EXPORT_ARGS[@]}"
