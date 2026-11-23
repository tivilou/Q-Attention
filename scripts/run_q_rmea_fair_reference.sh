#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -n "${PYTHON_BIN:-}" ]]; then
    qattn_python="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
    qattn_python="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    qattn_python="$(command -v python3)"
else
    echo '请先激活已准备的 Python 环境，或设置 PYTHON_BIN。' >&2
    exit 2
fi
exec "$qattn_python" -B -m baselines.retacred.rmea_fair_reference_handoff "$@"
