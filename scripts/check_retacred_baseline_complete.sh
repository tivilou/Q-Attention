#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Neither python nor python3 is available; set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi
RUN_TESTS=1
ALLOW_DIRTY=0

usage() {
  echo "Usage: bash scripts/check_retacred_baseline_complete.sh [--skip-tests] [--allow-dirty]"
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
  configs/retacred_baseline_complete.json \
  experiments/train_relation_baseline.py \
  experiments/evaluate_relation_baseline.py \
  src/q_attention/metrics.py \
  scripts/run_retacred_baseline_complete.sh \
  scripts/check_retacred_baseline_publish_preflight.sh \
  scripts/validate_retacred_baseline_checkpoint.py \
  scripts/publish_retacred_baseline_report.sh \
  scripts/export_retacred_baseline_report.sh; do
  [[ -f "${FILE}" ]] || { echo "Missing ${FILE}" >&2; exit 1; }
done

if [[ ${ALLOW_DIRTY} -eq 0 ]]; then
  GIT_STATUS=$(git status --porcelain --untracked-files=all)
  [[ -z "${GIT_STATUS}" ]] || {
    echo "Repository is dirty; commit or isolate local changes before the formal run:" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 1
  }
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
p = json.load(open("configs/retacred_baseline_complete.json", encoding="utf-8"))
assert p["formal_experiment"] is True
assert p["expected_records"] == {"train": 58465, "valid": 19584, "test": 13418}
assert p["evaluation"]["test_used_for_training_or_selection"] is False
required = {"micro_precision", "micro_recall", "micro_f1", "accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"}
assert required.issubset(p["report_contract"]["metrics"])
print("baseline complete config=OK")
'
"${PYTHON_BIN}" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ${RUN_TESTS} -eq 1 ]]; then
  "${PYTHON_BIN}" -m pytest -q tests/test_metrics.py tests/test_relation_data.py
fi

echo "Baseline preflight OK"
echo "commit=$(git rev-parse HEAD)"
echo "train_records=${TRAIN_COUNT}"
echo "valid_records=${VALID_COUNT}"
echo "test_records=${TEST_COUNT}"
