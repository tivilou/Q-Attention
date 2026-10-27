#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${PYTHON_BIN:-}" ]]; then py="$PYTHON_BIN"; elif command -v python >/dev/null; then py=python; else py=python3; fi
exec "$py" -B experiments/q_rmea_confirmation_handoff.py export "$@"
