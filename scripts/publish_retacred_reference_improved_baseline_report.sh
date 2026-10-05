#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=
NO_PUSH=0
usage() { echo "Usage: bash scripts/publish_retacred_reference_improved_baseline_report.sh --run-dir PATH [--no-push]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; RUN_DIR=$2; shift ;;
    --no-push) NO_PUSH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$RUN_DIR" ]] || { usage >&2; exit 2; }
cd "$ROOT"
RUN_DIR=$(readlink -f -- "$RUN_DIR")
case "$RUN_DIR" in "$ROOT"/runs/*) ;; *) echo "RUN_DIR must be inside $ROOT/runs." >&2; exit 1 ;; esac
[[ -f "$RUN_DIR/RUN_COMPLETE" ]] || { echo "Run is not complete." >&2; exit 1; }
[[ -f "$RUN_DIR/EXPORT_COMPLETE" ]] || { echo "Report export is not complete." >&2; exit 1; }
REPORT_REL="reports/retacred_reference_improved_baseline/$(basename "$RUN_DIR")"
REPORT_DIR="$ROOT/$REPORT_REL"
[[ -d "$REPORT_DIR" ]] || { echo "Exported report is missing: $REPORT_DIR" >&2; exit 1; }
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Publishing requires branch 1.1." >&2; exit 1; }
for ref in origin/main origin/1.1; do git rev-parse --verify "$ref^{commit}" >/dev/null 2>&1 || { echo "Missing $ref; fetch origin before retrying." >&2; exit 1; }; done
git merge-base --is-ancestor origin/main HEAD || { echo "HEAD does not contain origin/main." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "HEAD does not contain origin/1.1." >&2; exit 1; }

if [[ -f "$RUN_DIR/COMMIT_COMPLETE" ]]; then
  COMMIT_SHA=$(tr -d '\r\n' < "$RUN_DIR/COMMIT_COMPLETE")
  [[ "$(git rev-parse HEAD)" == "$COMMIT_SHA" ]] || { echo "HEAD differs from COMMIT_COMPLETE." >&2; exit 1; }
else
  git add -- "$REPORT_REL"
  STAGED_FILES=$(git diff --cached --name-only)
  [[ -n "$STAGED_FILES" ]] || { echo "No report files staged." >&2; exit 1; }
  while IFS= read -r staged; do
    [[ -z "$staged" ]] && continue
    [[ "$staged" == "$REPORT_REL" || "$staged" == "$REPORT_REL"/* ]] || { echo "Staged path is outside report: $staged" >&2; exit 1; }
  done <<< "$STAGED_FILES"
  git diff --cached --check
  git commit -m "report: add Improved Baseline Re-TACRED reference ($(basename "$RUN_DIR"))" 2>&1 | tee "$RUN_DIR/logs/report_commit.log"
  COMMIT_SHA=$(git rev-parse HEAD)
  printf '%s\n' "$COMMIT_SHA" > "$RUN_DIR/COMMIT_COMPLETE"
fi

if [[ "$NO_PUSH" -eq 1 ]]; then
  printf 'STATUS=push-skipped\nCOMMIT=%s\nAT=%s\n' "$COMMIT_SHA" "$(date -Iseconds)" > "$RUN_DIR/status/publish.env"
  echo "Reference report committed locally ($COMMIT_SHA); push skipped."
  exit 0
fi
git push origin HEAD:1.1 2>&1 | tee "$RUN_DIR/logs/report_push.log"
printf '%s\n' "$(date -Iseconds)" > "$RUN_DIR/PUSH_COMPLETE"
printf 'STATUS=push-complete\nCOMMIT=%s\nAT=%s\n' "$COMMIT_SHA" "$(date -Iseconds)" > "$RUN_DIR/status/publish.env"
echo "Reference report committed and pushed: $REPORT_DIR"
