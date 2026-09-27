#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python3)"
else
  echo "ERROR: python or python3 is required" >&2
  exit 2
fi

exec "$PYTHON_CMD" "$SCRIPT_DIR/check_q_epvg_case_study_structure.py" "$@"
