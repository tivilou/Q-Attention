#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n "${PYTHON_BIN:-}" ]]; then command -v "$PYTHON_BIN" >/dev/null
elif command -v python >/dev/null 2>&1; then PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3
else echo 'Activate the documented Python environment first' >&2; exit 2; fi
cd "$ROOT"
exec "$PYTHON_BIN" -u -m baselines.retacred.reference_complete "$@"
