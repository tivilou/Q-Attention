#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "${ROOT}"
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
command -v nvidia-smi >/dev/null
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Formal run must execute on branch 1.1" >&2; exit 1; }
GIT_STATUS=$(git status --porcelain --untracked-files=all)
if [[ -n "${GIT_STATUS}" ]]; then
  echo "Repository is dirty. Commit or isolate local changes before the formal run:" >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 1
fi
git merge-base --is-ancestor origin/1.1 HEAD || { echo "HEAD must include origin/1.1 before the formal run" >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "HEAD must include origin/main before the formal run" >&2; exit 1; }
for FILE in configs/retacred_qsrpa_query_conditioned_formal_single_seed.json experiments/train_relation_baseline.py experiments/train_relation_attention_score_kernel.py experiments/eval_relation_attention_score_kernel.py scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh scripts/export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh scripts/summarize_retacred_qsrpa_query_conditioned_formal_single_seed.py; do [[ -f "${FILE}" ]] || { echo "Missing ${FILE}" >&2; exit 1; }; done
[[ $(wc -l < data/relation/retacred/train.jsonl) -eq 58465 ]] || exit 1
[[ $(wc -l < data/relation/retacred/valid.jsonl) -eq 19584 ]] || exit 1
[[ $(wc -l < data/relation/retacred/test.jsonl) -eq 13418 ]] || exit 1
"${PYTHON_BIN}" -c 'import json; p=json.load(open("configs/retacred_qsrpa_query_conditioned_formal_single_seed.json")); assert p["formal_experiment"] and p["seed"]==13 and p["candidate"]=="quantum_query_conditioned_soft_role_pair"; print("formal Q-SRPA query-conditioned config=OK")'
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m pytest -q tests/test_attention_score_kernel.py tests/test_retacred_qsrpa_query_conditioned_formal_single_seed.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; echo "Preflight OK"; git rev-parse HEAD
