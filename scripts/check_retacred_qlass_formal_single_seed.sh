#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_TESTS=1
ALLOW_DIRTY=0

usage() {
  echo "Usage: bash scripts/check_retacred_qlass_formal_single_seed.sh [--skip-tests] [--allow-dirty]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests) RUN_TESTS=0 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "${ROOT}"
command -v "${PYTHON_BIN}" >/dev/null
command -v git >/dev/null
command -v nvidia-smi >/dev/null

for FILE in \
  configs/retacred_qlass_formal_single_seed.json \
  experiments/train_relation_baseline.py \
  experiments/train_relation_attention_score_kernel.py \
  experiments/eval_relation_attention_score_kernel.py \
  scripts/run_retacred_qlass_formal_single_seed.sh \
  scripts/summarize_retacred_qlass_formal_single_seed.py; do
  [[ -f "${FILE}" ]] || { echo "Missing ${FILE}" >&2; exit 1; }
done

if [[ ${ALLOW_DIRTY} -eq 0 ]]; then
  GIT_STATUS=$(git status --porcelain --untracked-files=all)
  if [[ -n "${GIT_STATUS}" ]]; then
    echo "Repository is dirty. Commit or isolate local changes before the formal run:" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 1
  fi
fi

TRAIN_PATH=data/relation/retacred/train.jsonl
VALID_PATH=data/relation/retacred/valid.jsonl
TEST_PATH=data/relation/retacred/test.jsonl
[[ -s "${TRAIN_PATH}" ]] || { echo "Missing ${TRAIN_PATH}" >&2; exit 1; }
[[ -s "${VALID_PATH}" ]] || { echo "Missing ${VALID_PATH}" >&2; exit 1; }
[[ -s "${TEST_PATH}" ]] || { echo "Missing ${TEST_PATH}" >&2; exit 1; }

TRAIN_COUNT=$(wc -l < "${TRAIN_PATH}")
VALID_COUNT=$(wc -l < "${VALID_PATH}")
TEST_COUNT=$(wc -l < "${TEST_PATH}")
[[ "${TRAIN_COUNT}" -eq 58465 ]] || { echo "Unexpected train count: ${TRAIN_COUNT}" >&2; exit 1; }
[[ "${VALID_COUNT}" -eq 19584 ]] || { echo "Unexpected valid count: ${VALID_COUNT}" >&2; exit 1; }
[[ "${TEST_COUNT}" -eq 13418 ]] || { echo "Unexpected test count: ${TEST_COUNT}" >&2; exit 1; }

"${PYTHON_BIN}" -c '
import json
p = json.load(open("configs/retacred_qlass_formal_single_seed.json", encoding="utf-8"))
assert p["formal_experiment"] is True
assert p["seed"] == 13
assert p["selection_metric"] == "macro_f1_then_loss"
assert p["kernel"]["input_encoding"] == "joint"
assert p["kernel"]["query_scope"] == "all"
assert p["kernel"]["relation_anchor_mode"] == "global_context"
assert p["expected_records"] == {"train": 58465, "valid": 19584, "test": 13418}
print("formal config=OK")
'
"${PYTHON_BIN}" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)'
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c 'from q_attention.plugins import RelationScoreKernelConfig; print("Q-LASS import=OK")'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ${RUN_TESTS} -eq 1 ]]; then
  "${PYTHON_BIN}" -m pytest -q tests/test_attention_score_kernel.py tests/test_retacred_qlass_formal_single_seed.py
fi

echo "Preflight OK"
echo "commit=$(git rev-parse HEAD)"
echo "train_records=${TRAIN_COUNT}"
echo "valid_records=${VALID_COUNT}"
echo "test_records=${TEST_COUNT}"
