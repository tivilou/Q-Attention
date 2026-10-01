#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=
REPORT_DIR=
usage() { echo "Usage: bash scripts/check_retacred_baseline_publish_preflight.sh --run-dir PATH --report-dir PATH"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; RUN_DIR=$2; shift ;;
    --report-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; REPORT_DIR=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$RUN_DIR" && -n "$REPORT_DIR" ]] || { usage >&2; exit 2; }
cd "$ROOT"
RUN_DIR=$(readlink -m -- "$RUN_DIR")
REPORT_DIR=$(readlink -m -- "$REPORT_DIR")
case "$RUN_DIR" in "${ROOT}"/runs/*) ;; *) echo "Planned run directory must be inside $ROOT/runs." >&2; exit 1 ;; esac
case "$REPORT_DIR" in "${ROOT}"/reports/retacred_baseline_complete/*) ;; *) echo "Planned report directory is outside reports/retacred_baseline_complete." >&2; exit 1 ;; esac
[[ ! -e "$RUN_DIR" ]] || { echo "Planned run directory already exists: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$REPORT_DIR" ]] || { echo "Planned report directory already exists: $REPORT_DIR" >&2; exit 1; }
BRANCH=$(git branch --show-current)
[[ "$BRANCH" == "1.1" ]] || { echo "Publishing requires branch 1.1 before training; current branch is $BRANCH." >&2; exit 1; }
for REF in origin/main origin/1.1; do
  git rev-parse --verify "$REF^{commit}" >/dev/null 2>&1 || { echo "Missing $REF; run: git fetch origin --prune" >&2; exit 1; }
done
git merge-base --is-ancestor origin/main HEAD || { echo "Current branch must include origin/main; merge it before training." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "Current branch must include origin/1.1; merge it before training." >&2; exit 1; }
GIT_STATUS=$(git status --porcelain --untracked-files=all)
[[ -z "$GIT_STATUS" ]] || { echo "Publishing requires a clean worktree before training:" >&2; printf '%s\n' "$GIT_STATUS" >&2; exit 1; }
git diff --cached --quiet || { echo "Publishing requires an empty Git index before training." >&2; exit 1; }
REPORT_REL="reports/retacred_baseline_complete/$(basename "$REPORT_DIR")"
if git check-ignore -q -- "$REPORT_REL"; then
  echo "Planned report directory is ignored by Git: $REPORT_REL" >&2
  exit 1
fi
echo "Baseline publish preflight OK: branch=$BRANCH run=$RUN_DIR report=$REPORT_DIR"
