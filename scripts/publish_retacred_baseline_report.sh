#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=
NO_PUSH=0
usage() { echo "Usage: bash scripts/publish_retacred_baseline_report.sh --run-dir PATH [--no-push]"; }
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
case "$RUN_DIR" in "${ROOT}"/runs/*) ;; *) echo "RUN_DIR must be inside $ROOT/runs." >&2; exit 1 ;; esac
[[ -f "$RUN_DIR/RUN_COMPLETE" ]] || { echo "Run is not complete." >&2; exit 1; }
[[ -f "$RUN_DIR/EXPORT_COMPLETE" ]] || { echo "Report export is not complete." >&2; exit 1; }
REPORT_DIR="$ROOT/reports/retacred_baseline_complete/$(basename "$RUN_DIR")"
REPORT_REL="reports/retacred_baseline_complete/$(basename "$RUN_DIR")"
[[ -d "$REPORT_DIR" ]] || { echo "Exported report is missing: $REPORT_DIR" >&2; exit 1; }
record_publish() {
  local status=$1 stage=$2 reason="" at
  [[ $# -lt 3 ]] || reason=$3
  at=$(date -Iseconds)
  { printf 'STATUS=%s\nSTAGE=%s\nAT=%s\n' "$status" "$stage" "$at"; [[ -z "$reason" ]] || printf 'REASON=%s\n' "$reason"; [[ ! -f "$RUN_DIR/COMMIT_COMPLETE" ]] || printf 'COMMIT=%s\n' "$(cat "$RUN_DIR/COMMIT_COMPLETE")"; } > "$RUN_DIR/status/publish.env"
  printf '%s status=%s stage=%s at=%s' "$at" "$status" "$stage" >> "$RUN_DIR/status/history.log"
  [[ -z "$reason" ]] || printf ' reason=%s' "$reason" >> "$RUN_DIR/status/history.log"
  printf '\n' >> "$RUN_DIR/status/history.log"
}
fail_publish() { record_publish failed "$1" "$2"; echo "$3" >&2; exit 1; }
BRANCH=$(git branch --show-current)
[[ "$BRANCH" == "1.1" ]] || fail_publish preflight wrong-branch "Publishing requires branch 1.1."
for REF in origin/main origin/1.1; do git rev-parse --verify "$REF^{commit}" >/dev/null 2>&1 || fail_publish preflight missing-ref "Missing $REF; fetch origin before retrying."; done
git merge-base --is-ancestor origin/main HEAD || fail_publish preflight main-ancestry "HEAD no longer contains origin/main."
git merge-base --is-ancestor origin/1.1 HEAD || fail_publish preflight release-ancestry "HEAD no longer contains origin/1.1."
if [[ -f "$RUN_DIR/COMMIT_COMPLETE" ]]; then
  COMMIT_SHA=$(tr -d '\r\n' < "$RUN_DIR/COMMIT_COMPLETE")
  [[ "$(git rev-parse HEAD)" == "$COMMIT_SHA" ]] || fail_publish preflight head-moved "HEAD differs from this run's COMMIT_COMPLETE."
else
  git add -- "$REPORT_REL" || fail_publish stage git-add "Failed to stage the exported report."
  STAGED_FILES=$(git diff --cached --name-only)
  [[ -n "$STAGED_FILES" ]] || fail_publish stage empty-index "No report files are staged."
  while IFS= read -r STAGED_FILE; do
    [[ -z "$STAGED_FILE" ]] && continue
    [[ "$STAGED_FILE" == "$REPORT_REL" || "$STAGED_FILE" == "$REPORT_REL"/* ]] || fail_publish stage unexpected-staged-file "Staged path is outside the report: $STAGED_FILE"
  done <<< "$STAGED_FILES"
  git diff --cached --check || fail_publish stage staged-diff-check "Staged report failed git diff --cached --check."
  COMMIT_MESSAGE="report: add complete Re-TACRED baseline evaluation ($(basename "$RUN_DIR"))"
  if ! git commit -m "$COMMIT_MESSAGE" 2>&1 | tee "$RUN_DIR/logs/report_commit.log"; then
    fail_publish commit commit-failed "Commit failed. Rerun this script with the same --run-dir after fixing Git configuration."
  fi
  COMMIT_SHA=$(git rev-parse HEAD)
  printf '%s\n' "$COMMIT_SHA" > "$RUN_DIR/COMMIT_COMPLETE"
fi
if [[ "$NO_PUSH" -eq 1 ]]; then
  record_publish push-skipped push
  echo "Report committed locally ($COMMIT_SHA); push skipped."
  exit 0
fi
if ! git push origin HEAD:1.1 2>&1 | tee "$RUN_DIR/logs/report_push.log"; then
  record_publish push-failed push
  echo "Report commit $COMMIT_SHA is retained; rerun this script with the same --run-dir. Training will not repeat." >&2
  exit 1
fi
printf '%s\n' "$(date -Iseconds)" > "$RUN_DIR/PUSH_COMPLETE"
record_publish push-complete push
echo "Report exported, committed, and pushed: $REPORT_DIR"
