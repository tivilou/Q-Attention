#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); PYTHON_BIN=${PYTHON_BIN:-python}; RUN_DIR=; REPORT_DIR=; NO_COMMIT=0
while [[ $# -gt 0 ]]; do case "$1" in --run-dir) RUN_DIR=$2; shift 2;; --report-dir) REPORT_DIR=$2; shift 2;; --no-commit) NO_COMMIT=1; shift;; -h|--help) echo "Usage: $0 --run-dir PATH [--report-dir PATH] [--no-commit]"; exit 0;; *) exit 2;; esac; done
cd "${ROOT}"; [[ "$(git branch --show-current)" == "1.1" ]] || { echo "Exporter must run on branch 1.1" >&2; exit 1; }; [[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Working tree must be clean" >&2; exit 1; }; git merge-base --is-ancestor origin/main HEAD
[[ -n "${RUN_DIR}" ]] || { echo "--run-dir is required" >&2; exit 2; }; [[ "${RUN_DIR}" = /* ]] || RUN_DIR="${ROOT}/${RUN_DIR}"; RUN_DIR=$(readlink -f "${RUN_DIR}")
for file in RUN_COMPLETE run_summary.json run_summary.md baseline/metrics.json quantum_query_conditioned_soft_role_pair/valid/metrics.json quantum_query_conditioned_soft_role_pair/test/metrics.json classical_query_conditioned_soft_role_pair/valid/metrics.json classical_query_conditioned_soft_role_pair/test/metrics.json; do [[ -f "${RUN_DIR}/${file}" ]] || { echo "Missing ${file}" >&2; exit 1; }; done
"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
p=json.loads((Path(sys.argv[1])/'run_summary.json').read_text(encoding='utf-8'))
assert p.get('seed') == 13
assert p.get('test_used_for_training_or_selection') is False
assert 'quantum_query_conditioned_soft_role_pair' in p.get('methods', {})
assert 'classical_query_conditioned_soft_role_pair' in p.get('methods', {})
assert 'candidate_minus_matched' in p and 'candidate_minus_disabled' in p
PY
REPORT_DIR=${REPORT_DIR:-reports/retacred_qsrpa_query_conditioned_formal_single_seed/$(date -u +%Y%m%d-%H%M%S)-full-single-seed13}; [[ "${REPORT_DIR}" = /* ]] || REPORT_DIR="${ROOT}/${REPORT_DIR}"; REPORT_DIR=$(readlink -m "${REPORT_DIR}"); case "${REPORT_DIR}" in "${ROOT}/reports/retacred_qsrpa_query_conditioned_formal_single_seed/"*) ;; *) exit 1;; esac; [[ ! -e "${REPORT_DIR}" ]] || exit 1
mkdir -p "${REPORT_DIR}/metrics"; cp "${RUN_DIR}/RUN_COMPLETE" "${RUN_DIR}/run_summary.json" "${RUN_DIR}/run_summary.md" "${REPORT_DIR}/"; cp configs/retacred_qsrpa_query_conditioned_formal_single_seed.json "${REPORT_DIR}/run_config.json"; cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/metrics/baseline.json"
for name in quantum_global_context classical_global_context quantum_soft_role_pair classical_soft_role_pair quantum_query_conditioned_soft_role_pair classical_query_conditioned_soft_role_pair; do cp "${RUN_DIR}/${name}/valid/metrics.json" "${REPORT_DIR}/metrics/${name}_valid.json"; cp "${RUN_DIR}/${name}/test/metrics.json" "${REPORT_DIR}/metrics/${name}_test.json"; done
if [[ -f "${RUN_DIR}/provenance.env" ]]; then cp "${RUN_DIR}/provenance.env" "${REPORT_DIR}/"; else echo PROVENANCE_STATUS=not_recorded_by_runner > "${REPORT_DIR}/provenance.env"; fi
printf '%s\n' "$(git rev-parse HEAD)" > "${REPORT_DIR}/reporting_commit.txt"; wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data_counts.txt"; sha256sum data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data.sha256"
find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.jsonl' \) | grep -q . && { echo "Forbidden private artifact" >&2; exit 1; } || true
REPORT_REL=${REPORT_DIR#"${ROOT}/"}; git add -- "${REPORT_REL}"; git diff --cached --check; [[ ${NO_COMMIT} -eq 1 ]] && { echo "REPORT_DIR=${REPORT_REL}"; exit 0; }; git commit -m "Add query-conditioned Q-SRPA formal single-seed report"; git push origin 1.1; echo "REPORT_DIR=${REPORT_REL}"
