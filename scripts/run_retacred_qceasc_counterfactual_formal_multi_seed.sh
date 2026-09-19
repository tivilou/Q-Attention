#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ "${PYTHON_BIN}" == */* ]]; then
      [[ -x "${PYTHON_BIN}" ]] && { printf '%s\n' "${PYTHON_BIN}"; return; }
    elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"
      return
    fi
    echo "PYTHON_BIN is not executable or not on PATH: ${PYTHON_BIN}" >&2
    return 1
  fi
  for candidate in python python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN." >&2
  return 1
}

PYTHON_BIN=$(resolve_python_bin)
export PYTHON_BIN
cd "${ROOT}"
CONFIG_ARGS=(
  --config configs/retacred_qceasc_counterfactual_formal_single_seed.json
)
IMPORT_ARGS=(
  --import-seed13-report
  reports/retacred_qceasc_counterfactual_formal_single_seed/20260918T023816Z_seed13
)
for arg in "$@"; do
  case "${arg}" in
    --resume-group|--resume-group=*)
      IMPORT_ARGS=()
      break
      ;;
  esac
done
exec "${PYTHON_BIN}" scripts/run_retacred_qceasc_formal_multi_seed.py \
  "${CONFIG_ARGS[@]}" \
  "${IMPORT_ARGS[@]}" \
  "$@"
