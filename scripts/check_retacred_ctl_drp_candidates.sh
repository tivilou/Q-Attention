#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--describe" && "$#" == 1 ]]; then
  printf '%s\n' '{"scope":"CTL-DRP cited RECENT/train-only candidate primitive","training":false,"network":false,"weights":false,"pending_exit":2}'
  exit 0
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  PYTHON="$(command -v python3)"
fi
cd -- "$ROOT"
exec "$PYTHON" -m baselines.retacred.ctl_drp_candidates "$@"
