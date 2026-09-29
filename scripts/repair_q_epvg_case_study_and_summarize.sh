#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python3)"
else
  echo "ERROR: python or python3 is required" >&2
  exit 2
fi

if [[ "$#" -ne 2 || "$1" != "--group-dir" ]]; then
  echo "usage: $0 --group-dir RUN_DIR" >&2
  exit 2
fi

GROUP_DIR="$2"
if [[ "$GROUP_DIR" != /* ]]; then
  GROUP_DIR="$ROOT_DIR/$GROUP_DIR"
fi

echo "[repair] preflight (read-only): $GROUP_DIR"
"$PYTHON_CMD" "$SCRIPT_DIR/repair_q_epvg_case_study_stages.py" \
  --group-dir "$GROUP_DIR"

echo "[repair] applying checksum-verified reconstruction: $GROUP_DIR"
"$PYTHON_CMD" "$SCRIPT_DIR/repair_q_epvg_case_study_stages.py" \
  --group-dir "$GROUP_DIR" --apply

"$PYTHON_CMD" "$ROOT_DIR/scripts/summarize_retacred_q_epvg_formal_multi_seed.py" \
  --group-dir "$GROUP_DIR" \
  --output-json "$GROUP_DIR/multi_seed_summary.json" \
  --output-md "$GROUP_DIR/multi_seed_summary.md"

if [[ ! -f "$GROUP_DIR/MULTI_SEED_COMPLETE" ]]; then
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$GROUP_DIR/MULTI_SEED_COMPLETE"
fi
echo "[repair] summary and completion marker written: $GROUP_DIR"
