#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${PYTHON_BIN:-}" ]]; then PY="$PYTHON_BIN"; elif command -v python >/dev/null; then PY=python; else PY=python3; fi
exec "$PY" -u -m baselines.retacred.reference_preflight "$@"
