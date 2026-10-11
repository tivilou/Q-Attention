#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXPECTED_COMMIT=fe00781116ab4c34cbcd9a2ae5b02513afae87ee
UPSTREAM_DIR=

usage() {
  cat <<'EOF'
Usage: bash scripts/check_retacred_reference_improved_baseline_source.sh \
  --upstream-dir PATH

This is a read-only source/provenance gate. It does not train or download a
checkpoint.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir)
      [[ $# -ge 2 ]] || { echo "--upstream-dir requires a value." >&2; exit 2; }
      UPSTREAM_DIR=$2
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -n "${UPSTREAM_DIR}" ]] || { echo "--upstream-dir is required." >&2; exit 2; }
[[ -d "${UPSTREAM_DIR}/.git" ]] || { echo "Not a Git checkout: ${UPSTREAM_DIR}" >&2; exit 1; }

ACTUAL_COMMIT=$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || {
  echo "Unexpected upstream commit: ${ACTUAL_COMMIT}; expected ${EXPECTED_COMMIT}." >&2
  exit 1
}

REMOTE_URL=$(git -C "${UPSTREAM_DIR}" remote get-url origin 2>/dev/null || true)
[[ "${REMOTE_URL}" == "https://github.com/wzhouad/RE_improved_baseline.git" ]] || {
  echo "Unexpected upstream origin: ${REMOTE_URL}" >&2
  exit 1
}

for FILE in train_retacred.py prepro.py model.py utils.py evaluation.py; do
  [[ -f "${UPSTREAM_DIR}/${FILE}" ]] || {
    echo "Missing upstream source file: ${FILE}" >&2
    exit 1
  }
done

echo "improved baseline source=OK"
echo "upstream_commit=${ACTUAL_COMMIT}"
echo "upstream_origin=${REMOTE_URL}"
echo "checkpoint=not-locked"
echo "status=planned"
