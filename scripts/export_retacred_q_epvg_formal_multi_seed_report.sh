#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
GROUP_DIR=
REPORT_DIR=
NO_COMMIT=0
resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ "${PYTHON_BIN}" == */* ]]; then
      [[ -x "${PYTHON_BIN}" ]] && { printf '%s\n' "${PYTHON_BIN}"; return; }
    elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"
      return
    fi
    echo "PYTHON_BIN is not executable or not on PATH: ${PYTHON_BIN}" >&2
    return 1
  fi
  for candidate in python python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN." >&2
  return 1
}

PYTHON_BIN=$(resolve_python_bin)

while [ $# -gt 0 ]; do
  case "$1" in
    --group-dir|--report-dir)
      [ $# -ge 2 ] || { echo "Missing value for $1." >&2; exit 2; }
      if [ "$1" = "--group-dir" ]; then GROUP_DIR=$2; else REPORT_DIR=$2; fi
      shift 2
      ;;
    --no-commit) NO_COMMIT=1; shift ;;
    -h|--help)
      echo "Usage: $0 --group-dir PATH [--report-dir PATH] [--no-commit]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
[ "$(git branch --show-current)" = "1.1" ] || { echo "Exporter must run on branch 1.1." >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "Working tree must be clean before export." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "origin/1.1 must be an ancestor of HEAD." >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "origin/main must be an ancestor of HEAD." >&2; exit 1; }
[ -n "$GROUP_DIR" ] || { echo "--group-dir is required." >&2; exit 2; }
[[ "$GROUP_DIR" == /* ]] || GROUP_DIR="$ROOT/$GROUP_DIR"
GROUP_DIR=$(readlink -f "$GROUP_DIR")
case "$GROUP_DIR" in
  "$ROOT/runs/retacred_q_epvg_formal_multi_seed/"*) ;;
  *) echo "Group directory is outside the Q-EPVG multi-seed root." >&2; exit 1 ;;
esac
[ -f "$GROUP_DIR/MULTI_SEED_COMPLETE" ] || { echo "Missing MULTI_SEED_COMPLETE." >&2; exit 1; }
[ -f "$GROUP_DIR/multi_seed_manifest.json" ] && [ -f "$GROUP_DIR/multi_seed_status.json" ] || { echo "Missing multi-seed manifest/status." >&2; exit 1; }
if [ -z "$REPORT_DIR" ]; then REPORT_DIR="$ROOT/reports/retacred_q_epvg_formal_multi_seed/$(basename "$GROUP_DIR")"; fi
[[ "$REPORT_DIR" == /* ]] || REPORT_DIR="$ROOT/$REPORT_DIR"
REPORT_DIR=$(readlink -m "$REPORT_DIR")
case "$REPORT_DIR" in
  "$ROOT/reports/retacred_q_epvg_formal_multi_seed/"*) ;;
  *) echo "Report must be under the Q-EPVG multi-seed report root." >&2; exit 1 ;;
esac
[ ! -e "$REPORT_DIR" ] || { echo "Refusing to overwrite report directory." >&2; exit 1; }
REPORTING_COMMIT=$(git rev-parse HEAD)
"$PYTHON_BIN" scripts/q_epvg_multi_seed_report_export.py \
  --group-dir "$GROUP_DIR" \
  --report-dir "$REPORT_DIR" \
  --reporting-commit "$REPORTING_COMMIT"
REPORT_REL=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$REPORT_DIR" "$ROOT")
git add -- "$REPORT_REL"
git diff --cached --check
if [ "$NO_COMMIT" -eq 1 ]; then
  echo "REPORT_DIR=$REPORT_REL"
  exit 0
fi
git commit -m "Add Q-EPVG Re-TACRED formal multi-seed report"
git push origin 1.1
echo "REPORT_DIR=$REPORT_REL"
