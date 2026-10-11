#!/usr/bin/env bash
# Paper/config guard only: no downloading, model loading, training or publication.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ "${1:-}" == "--describe" ]]; then
  printf '%s\n' '{"scope":"MetaEntail paper recipe and declared-common guard","training":false,"network":false,"backend_imports":false,"pending_exit":2}'
  exit 0
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON=$PYTHON_BIN
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  PYTHON=python3
fi
cd -- "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -m baselines.retacred.metaentail_recipe "$@"
