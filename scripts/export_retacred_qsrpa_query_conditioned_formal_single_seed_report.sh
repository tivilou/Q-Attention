#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); RUN_DIR=; REPORT_DIR=; NO_COMMIT=0
resolve_python_bin(){
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
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"; return
    fi
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN" >&2
  return 1
}
PYTHON_BIN=$(resolve_python_bin); export PYTHON_BIN
while [[ $# -gt 0 ]]; do case "$1" in --run-dir) RUN_DIR=$2; shift 2;; --report-dir) REPORT_DIR=$2; shift 2;; --no-commit) NO_COMMIT=1; shift;; -h|--help) echo "Usage: $0 --run-dir PATH [--report-dir PATH] [--no-commit]"; exit 0;; *) exit 2;; esac; done
cd "${ROOT}"; [[ "$(git branch --show-current)" == "1.1" ]] || { echo "Exporter must run on branch 1.1" >&2; exit 1; }; [[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Working tree must be clean" >&2; exit 1; }; git merge-base --is-ancestor origin/1.1 HEAD || { echo "HEAD must include origin/1.1 before report export" >&2; exit 1; }; git merge-base --is-ancestor origin/main HEAD || { echo "HEAD must include origin/main before report export" >&2; exit 1; }
[[ -n "${RUN_DIR}" ]] || { echo "--run-dir is required" >&2; exit 2; }; [[ "${RUN_DIR}" = /* ]] || RUN_DIR="${ROOT}/${RUN_DIR}"; RUN_DIR=$(readlink -f "${RUN_DIR}")
for file in RUN_COMPLETE run_summary.json run_summary.md provenance.env baseline/metrics.json quantum_query_conditioned_soft_role_pair/valid/metrics.json quantum_query_conditioned_soft_role_pair/test/metrics.json classical_query_conditioned_soft_role_pair/valid/metrics.json classical_query_conditioned_soft_role_pair/test/metrics.json; do [[ -f "${RUN_DIR}/${file}" ]] || { echo "Missing ${file}" >&2; exit 1; }; done
provenance_value(){
  local key=$1
  awk -F= -v key="${key}" '$1 == key {print substr($0, index($0, "=") + 1); exit}' "${RUN_DIR}/provenance.env"
}
[[ "$(provenance_value PROVENANCE_STATUS)" == "recorded_by_runner" ]] || { echo "provenance.env was not recorded by the runner" >&2; exit 1; }
EXECUTION_COMMIT=$(provenance_value GIT_COMMIT)
EXECUTION_BRANCH=$(provenance_value GIT_BRANCH)
[[ "${EXECUTION_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "provenance.env has no valid GIT_COMMIT" >&2; exit 1; }
[[ "${EXECUTION_BRANCH}" == "1.1" ]] || { echo "provenance.env GIT_BRANCH must be 1.1" >&2; exit 1; }
git cat-file -e "${EXECUTION_COMMIT}^{commit}" || { echo "execution commit is unavailable in this checkout" >&2; exit 1; }
git merge-base --is-ancestor "${EXECUTION_COMMIT}" HEAD || { echo "execution commit is not an ancestor of exporter HEAD" >&2; exit 1; }
"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
p=json.loads((root/'run_summary.json').read_text(encoding='utf-8'))
assert p.get('seed') == 13
assert p.get('test_used_for_training_or_selection') is False
assert 'quantum_query_conditioned_soft_role_pair' in p.get('methods', {})
assert 'classical_query_conditioned_soft_role_pair' in p.get('methods', {})
assert 'candidate_minus_matched' in p and 'candidate_minus_disabled' in p
for method in ('quantum_query_conditioned_soft_role_pair', 'classical_query_conditioned_soft_role_pair'):
    for split in ('valid', 'test'):
        metrics = json.loads((root / method / split / 'metrics.json').read_text(encoding='utf-8'))
        contract = metrics.get('checkpoint_metadata', {}).get('action_contract', {})
        assert contract == {
            'protocol': 'label_free_query_conditioned_soft_role_pair',
            'action_uses_subject_object_masks': False,
            'subject_object_spans_allowed_for_action': False,
            'subject_object_spans_allowed_for_offline_evaluation': True,
        }
PY
RUN_NAME=$(basename "${RUN_DIR}"); case "${RUN_NAME}" in *_seed13) ;; *) echo "Run directory must end with _seed13: ${RUN_NAME}" >&2; exit 1;; esac
EXPECTED_TIMESTAMP=${RUN_NAME%_seed13}
[[ "$(provenance_value RUN_TIMESTAMP)" == "${EXPECTED_TIMESTAMP}" ]] || { echo "provenance.env RUN_TIMESTAMP does not match the raw run directory" >&2; exit 1; }
[[ -n "$(provenance_value PYTHON_BIN)" ]] || { echo "provenance.env is missing PYTHON_BIN" >&2; exit 1; }
[[ -n "$(provenance_value PYTHON_VERSION)" ]] || { echo "provenance.env is missing PYTHON_VERSION" >&2; exit 1; }
DEFAULT_REPORT_DIR="reports/retacred_qsrpa_query_conditioned_formal_single_seed/${RUN_NAME}"
REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}; [[ "${REPORT_DIR}" = /* ]] || REPORT_DIR="${ROOT}/${REPORT_DIR}"; REPORT_DIR=$(readlink -m "${REPORT_DIR}"); case "${REPORT_DIR}" in "${ROOT}/reports/retacred_qsrpa_query_conditioned_formal_single_seed/"*) ;; *) exit 1;; esac; [[ ! -e "${REPORT_DIR}" ]] || exit 1
mkdir -p "${REPORT_DIR}/metrics"; cp "${RUN_DIR}/RUN_COMPLETE" "${RUN_DIR}/run_summary.json" "${RUN_DIR}/run_summary.md" "${REPORT_DIR}/"; cp configs/retacred_qsrpa_query_conditioned_formal_single_seed.json "${REPORT_DIR}/run_config.json"; cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/metrics/baseline.json"
for name in quantum_global_context classical_global_context quantum_soft_role_pair classical_soft_role_pair quantum_query_conditioned_soft_role_pair classical_query_conditioned_soft_role_pair; do cp "${RUN_DIR}/${name}/valid/metrics.json" "${REPORT_DIR}/metrics/${name}_valid.json"; cp "${RUN_DIR}/${name}/test/metrics.json" "${REPORT_DIR}/metrics/${name}_test.json"; done
if [[ -f "${RUN_DIR}/provenance.env" ]]; then cp "${RUN_DIR}/provenance.env" "${REPORT_DIR}/"; else echo PROVENANCE_STATUS=not_recorded_by_runner > "${REPORT_DIR}/provenance.env"; fi
printf '%s\n' "$(git rev-parse HEAD)" > "${REPORT_DIR}/reporting_commit.txt"; wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data_counts.txt"; sha256sum data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data.sha256"
find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.jsonl' \) | grep -q . && { echo "Forbidden private artifact" >&2; exit 1; } || true
REPORT_REL=${REPORT_DIR#"${ROOT}/"}; git add -- "${REPORT_REL}"; git diff --cached --check; [[ ${NO_COMMIT} -eq 1 ]] && { echo "REPORT_DIR=${REPORT_REL}"; exit 0; }; git commit -m "Add query-conditioned Q-SRPA formal single-seed report"; git push origin 1.1; echo "REPORT_DIR=${REPORT_REL}"
