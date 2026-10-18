#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/run_retacred_q_ecpa_formal_multi_seed.sh" --check "$@"
