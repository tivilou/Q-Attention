#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
MIN_COMMIT=${MIN_COMMIT:-b8d794f}
RUN_TESTS=1
ALLOW_DIRTY=0

usage() {
  echo "Usage: bash scripts/check_retacred_dual_qres_full.sh [--skip-tests] [--allow-dirty]"
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

if ! git merge-base --is-ancestor "${MIN_COMMIT}" HEAD; then
  echo "Current code does not contain required commit ${MIN_COMMIT}." >&2
  exit 1
fi

GIT_STATUS=$(git status --porcelain)
if [[ ${ALLOW_DIRTY} -eq 0 && -n "${GIT_STATUS}" ]]; then
  echo "Repository is dirty. Commit, restore, or remove local changes before running:" >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 1
fi

TRAIN_PATH=data/relation/retacred/train.jsonl
VALID_PATH=data/relation/retacred/valid.jsonl
[[ -s "${TRAIN_PATH}" ]] || { echo "Missing ${TRAIN_PATH}" >&2; exit 1; }
[[ -s "${VALID_PATH}" ]] || { echo "Missing ${VALID_PATH}" >&2; exit 1; }

TRAIN_COUNT=$(wc -l < "${TRAIN_PATH}")
VALID_COUNT=$(wc -l < "${VALID_PATH}")
[[ "${TRAIN_COUNT}" -eq 58465 ]] || { echo "Unexpected train count: ${TRAIN_COUNT}" >&2; exit 1; }
[[ "${VALID_COUNT}" -eq 19584 ]] || { echo "Unexpected valid count: ${VALID_COUNT}" >&2; exit 1; }

"${PYTHON_BIN}" -c "import torch; print('torch=' + torch.__version__); print('cuda=' + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ${RUN_TESTS} -eq 1 ]]; then
  "${PYTHON_BIN}" -m pytest -q
fi

echo "Preflight OK"
echo "commit=$(git rev-parse HEAD)"
echo "train_records=${TRAIN_COUNT}"
echo "valid_records=${VALID_COUNT}"
