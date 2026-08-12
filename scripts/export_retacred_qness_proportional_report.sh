#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR=${1:-}
REPORT_ROOT=${2:-reports/retacred}

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: bash scripts/export_retacred_qness_proportional_report.sh RUN_DIR [REPORT_ROOT]" >&2
  exit 2
fi

cd "${ROOT}"
[[ -d "${RUN_DIR}" ]] || { echo "Run directory not found: ${RUN_DIR}" >&2; exit 1; }
for filename in run_summary.json run_summary.md run_manifest.json; do
  [[ -f "${RUN_DIR}/${filename}" ]] || {
    echo "Missing required run output: ${RUN_DIR}/${filename}" >&2
    exit 1
  }
done

SEED=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["config"]["seed"])' "${RUN_DIR}/run_summary.json")
STAMP=$(date -u +%Y%m%d-%H%M%S)
REPORT_DIR=${REPORT_ROOT}/${STAMP}-qness-proportional-seed${SEED}
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to reuse report directory: ${REPORT_DIR}" >&2; exit 1; }
mkdir -p "${REPORT_DIR}"

copy_file() {
  local source=$1
  local relative=$2
  [[ -f "${source}" ]] || return 0
  mkdir -p "${REPORT_DIR}/$(dirname "${relative}")"
  cp "${source}" "${REPORT_DIR}/${relative}"
}

for filename in run_summary.json run_summary.md run_manifest.json run_config.json gpu_assignments.json; do
  copy_file "${RUN_DIR}/${filename}" "${filename}"
done

if compgen -G "${RUN_DIR}/status/*.env" >/dev/null; then
  mkdir -p "${REPORT_DIR}/status"
  for status_file in "${RUN_DIR}"/status/*.env; do
    cp "${status_file}" "${REPORT_DIR}/status/$(basename "${status_file}")"
  done
fi

STAGES=(
  baseline
  core/quantum
  selector/qness
  selector/qness_classical
  selector/qness_commuting
  selector/qness_separable
  selector/qness_phase_scrambled
  selector/qness_dephased
)
for stage in "${STAGES[@]}"; do
  copy_file "${RUN_DIR}/${stage}/metrics.json" "${stage}/metrics.json"
  copy_file "${RUN_DIR}/${stage}/diagnostics.json" "${stage}/diagnostics.json"
done

if compgen -G "${RUN_DIR}/logs/*.log" >/dev/null; then
  mkdir -p "${REPORT_DIR}/logs"
  for log_file in "${RUN_DIR}"/logs/*.log; do
    tail -n 1000 "${log_file}" > "${REPORT_DIR}/logs/$(basename "${log_file}").tail.txt"
  done
fi

printf '%s\n' "$(basename "${RUN_DIR}")" > "${REPORT_DIR}/source_run.txt"
find "${REPORT_DIR}" -type f -printf '%P\n' | sort > "${REPORT_DIR}/REPORT_MANIFEST.txt"

echo "REPORT_DIR=${REPORT_DIR}"
echo "Only whitelist summaries, GPU assignments, stage status, metrics, diagnostics, and log tails were exported."
