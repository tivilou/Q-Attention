#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
cd "${ROOT}"
exec "${PYTHON_BIN}" scripts/run_qvres_relation_transfer_multi_seed.py "$@"
