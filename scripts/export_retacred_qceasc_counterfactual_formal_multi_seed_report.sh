#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"
exec bash scripts/export_retacred_qceasc_formal_multi_seed_report.sh "$@"
