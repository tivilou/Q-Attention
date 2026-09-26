#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=""
REPORT_DIR=""
NO_COMMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR=$2; shift 2;;
    --report-dir) REPORT_DIR=$2; shift 2;;
    --no-commit) NO_COMMIT=1; shift;;
    -h|--help) echo "Usage: $0 --run-dir PATH [--report-dir PATH] [--no-commit]"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "${RUN_DIR}" ]] || { echo "--run-dir is required." >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-python}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN=python3
cd "${ROOT}"
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Exporter must run on branch 1.1." >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Working tree must be clean before export." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "origin/1.1 must be an ancestor of HEAD." >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "origin/main must be an ancestor of HEAD." >&2; exit 1; }
RUN_DIR=$(readlink -f "${RUN_DIR}")
RUN_NAME=$(basename "${RUN_DIR}")
[[ "${RUN_NAME}" == *_seed13 ]] || { echo "Run directory must end with _seed13." >&2; exit 1; }
"${PYTHON_BIN}" scripts/check_retacred_q_epvg_formal_single_seed.py --config configs/retacred_q_epvg_formal_single_seed.json --run-dir "${RUN_DIR}"
DEFAULT_REPORT_DIR="reports/retacred_q_epvg_formal_single_seed/${RUN_NAME}"
REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}
[[ "${REPORT_DIR}" = /* ]] || REPORT_DIR="${ROOT}/${REPORT_DIR}"
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in
  "${ROOT}/reports/retacred_q_epvg_formal_single_seed/"*) ;;
  *) echo "Report must be under the Q-EPVG report root." >&2; exit 1;;
esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite report directory." >&2; exit 1; }
EXPORT_ARGS=(
  scripts/q_epvg_report_export.py
  --run-dir "${RUN_DIR}"
  --report-dir "${REPORT_DIR}"
  --config configs/retacred_q_epvg_formal_single_seed.json
  --reporting-commit "$(git rev-parse HEAD)"
)
if [[ -n "${Q_EPVG_EXPORT_INJECT_FAILURE_AFTER:-}" ]]; then
  EXPORT_ARGS+=(--inject-failure-after "${Q_EPVG_EXPORT_INJECT_FAILURE_AFTER}")
fi
if [[ "${Q_EPVG_EXPORT_INJECT_VALIDATION_FAILURE:-0}" == 1 ]]; then
  EXPORT_ARGS+=(--inject-validation-failure)
fi
"${PYTHON_BIN}" "${EXPORT_ARGS[@]}"
REPORT_REL=${REPORT_DIR#"${ROOT}/"}
git add -- "${REPORT_REL}"
git diff --cached --check
if [[ ${NO_COMMIT} -eq 1 ]]; then echo "REPORT_DIR=${REPORT_REL}"; exit 0; fi
git commit -m "Add Q-EPVG formal single-seed report"
git push origin 1.1
echo "REPORT_DIR=${REPORT_REL}"
