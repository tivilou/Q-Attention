#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
cd "${ROOT}"
exec "${PYTHON_BIN}" scripts/run_retacred_dual_qres_multi_seed.py "$@"
