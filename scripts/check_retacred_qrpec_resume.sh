#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

resolve_python_bin() {
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
    command -v "${candidate}" >/dev/null 2>&1 && { command -v "${candidate}"; return; }
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN." >&2
  return 1
}

PYTHON_BIN=$(resolve_python_bin)
export PYTHON_BIN
cd "${ROOT}"
exec "${PYTHON_BIN}" "${ROOT}/scripts/check_retacred_qrpec_resume.py" "$@"
