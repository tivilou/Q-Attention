#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  PYTHON="$(command -v python3)"
fi
cd "$ROOT"
exec "$PYTHON" -u -m baselines.retacred.luke_runtime_fidelity "$@"
