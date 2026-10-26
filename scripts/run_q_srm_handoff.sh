#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then PYTHON_EXEC="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then PYTHON_EXEC="$(command -v python)"
else PYTHON_EXEC="$(command -v python3)"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PYTHON_EXEC" "$ROOT/experiments/q_srm_handoff.py" run "$@"
