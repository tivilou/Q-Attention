#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); PYTHON_BIN=${PYTHON_BIN:-python}; cd "${ROOT}"
command -v "${PYTHON_BIN}" >/dev/null; command -v nvidia-smi >/dev/null
GIT_STATUS=$(git status --porcelain --untracked-files=all)
if [[ -n "${GIT_STATUS}" ]]; then
  echo "Repository is dirty. Commit or isolate local changes before the formal run:" >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 1
fi
for FILE in configs/retacred_qsrpa_query_conditioned_formal_single_seed.json experiments/train_relation_baseline.py experiments/train_relation_attention_score_kernel.py experiments/eval_relation_attention_score_kernel.py scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh scripts/summarize_retacred_qsrpa_query_conditioned_formal_single_seed.py; do [[ -f "${FILE}" ]] || { echo "Missing ${FILE}" >&2; exit 1; }; done
[[ $(wc -l < data/relation/retacred/train.jsonl) -eq 58465 ]] || exit 1
[[ $(wc -l < data/relation/retacred/valid.jsonl) -eq 19584 ]] || exit 1
[[ $(wc -l < data/relation/retacred/test.jsonl) -eq 13418 ]] || exit 1
"${PYTHON_BIN}" -c 'import json; p=json.load(open("configs/retacred_qsrpa_query_conditioned_formal_single_seed.json")); assert p["formal_experiment"] and p["seed"]==13 and p["candidate"]=="quantum_query_conditioned_soft_role_pair"; print("formal Q-SRPA query-conditioned config=OK")'
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m pytest -q tests/test_attention_score_kernel.py tests/test_retacred_qsrpa_query_conditioned_formal_single_seed.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; echo "Preflight OK"; git rev-parse HEAD
