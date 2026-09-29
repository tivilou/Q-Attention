#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "Python 3 is required (set PYTHON_BIN to an interpreter)." >&2
  exit 2
fi

exec "${PYTHON_CMD[@]}" "$SCRIPT_DIR/upload_qepvg_case_study_recovery.py" "$@"
