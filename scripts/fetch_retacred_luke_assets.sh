#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
PY="${PYTHON_BIN:-}"
if [[ -z "$PY" ]]; then
  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PY="$candidate"
      break
    fi
  done
fi
if [[ -z "$PY" ]] || ! command -v "$PY" >/dev/null 2>&1; then
  printf '%s\n' 'Set PYTHON_BIN to the current environment Python.' >&2
  exit 2
fi
exec "$PY" -m baselines.retacred.luke_assets fetch "$@"
