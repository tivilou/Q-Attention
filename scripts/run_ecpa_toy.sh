#!/usr/bin/env bash
set -euo pipefail
ECPA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-python3}" -u "$ECPA_ROOT/experiments/run_ecpa_toy.py" "$@"
