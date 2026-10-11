#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${1:-}" == "--describe" && $# -eq 1 ]]; then
  printf '%s\n' 'CTL-DRP: explicit tiny or hash-locked pretrained single-GPU FP32 synthetic diagnostic; no full-data training, network, Git action or formal admission.'
  exit 0
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"
  else printf '%s\n' 'Python interpreter required' >&2; exit 1; fi
fi
cd -- "$ROOT"
exec "$PYTHON_BIN" -m baselines.retacred.ctl_drp_cuda_smoke "$@"
