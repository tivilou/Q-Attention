#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  runner_python="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  runner_python="$(command -v python)"
else
  runner_python="$(command -v python3)"
fi
cd "$ROOT"
exec "$runner_python" "$ROOT/experiments/run_ecpa_replication.py" "$@"
