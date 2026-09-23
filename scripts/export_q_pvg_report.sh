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
"${PYTHON_BIN}" scripts/check_retacred_q_pvg_formal_single_seed.py --config configs/retacred_q_pvg_formal_single_seed.json --run-dir "${RUN_DIR}"
DEFAULT_REPORT_DIR="reports/retacred_q_pvg_formal_single_seed/${RUN_NAME}"
REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}
[[ "${REPORT_DIR}" = /* ]] || REPORT_DIR="${ROOT}/${REPORT_DIR}"
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in
  "${ROOT}/reports/retacred_q_pvg_formal_single_seed/"*) ;;
  *) echo "Report must be under the Q-PVG report root." >&2; exit 1;;
esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite report directory." >&2; exit 1; }
mkdir -p "${REPORT_DIR}/metrics" "${REPORT_DIR}/case_study"
cp "${RUN_DIR}/RUN_COMPLETE" "${RUN_DIR}/run_summary.json" "${RUN_DIR}/run_summary.md" "${REPORT_DIR}/"
cp "${RUN_DIR}/gpu_assignments.json" "${REPORT_DIR}/gpu_assignments.json"
cp configs/retacred_q_pvg_formal_single_seed.json "${REPORT_DIR}/run_config.json"
cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/metrics/baseline.json"
for selector in q_pvg_phase_value q_pvg_real_only q_pvg_classical_complex q_pvg_random_phase q_pvg_score_value; do
  cp "${RUN_DIR}/selectors/${selector}/metrics.json" "${REPORT_DIR}/metrics/${selector}.json"
  cp "${RUN_DIR}/selectors/${selector}/case_study.json" "${REPORT_DIR}/case_study/${selector}.json"
  cp "${RUN_DIR}/selectors/${selector}/sample_trace.json" "${REPORT_DIR}/case_study/${selector}.sample-trace.json"
done
printf '%s\n' "$(git rev-parse HEAD)" > "${REPORT_DIR}/reporting_commit.txt"
for split in train valid test; do
  src="${RUN_DIR}/data/${split}.jsonl"
  [[ -f "${src}" ]] || { echo "Missing materialized ${split} data." >&2; exit 1; }
  printf '%s %s\n' "${src}" "$(wc -l < "${src}")"
done > "${REPORT_DIR}/data_counts.txt"
sha256sum "${RUN_DIR}/data/train.jsonl" "${RUN_DIR}/data/valid.jsonl" "${RUN_DIR}/data/test.jsonl" > "${REPORT_DIR}/data.sha256"
if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.jsonl' \) | grep -q .; then echo "Forbidden private artifact found in report." >&2; exit 1; fi
REPORT_REL=${REPORT_DIR#"${ROOT}/"}
git add -- "${REPORT_REL}"
git diff --cached --check
if [[ ${NO_COMMIT} -eq 1 ]]; then echo "REPORT_DIR=${REPORT_REL}"; exit 0; fi
git commit -m "Add Q-PVG formal single-seed report"
git push origin 1.1
echo "REPORT_DIR=${REPORT_REL}"
