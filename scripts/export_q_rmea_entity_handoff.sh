#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -n "${PYTHON_BIN:-}" ]]; then py="$PYTHON_BIN"; elif command -v python >/dev/null 2>&1; then py=python; else py=python3; fi
exec "$py" -B experiments/q_rmea_entity_handoff.py export "$@"
