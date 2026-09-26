#!/usr/bin/env bash
set -euo pipefail

# Re-export one immutable, completed Q-EPVG seed-13 run with lifecycle
# provenance. This intentionally does not train, resume, delete, or overwrite
# the historical report.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

RUN_DIR="runs/retacred_q_epvg_formal_single_seed/20260926T011732Z_seed13"
REPORT_DIR="reports/retacred_q_epvg_formal_single_seed/20260926T011732Z_seed13_retry"
EXPORTER="scripts/export_retacred_q_epvg_formal_single_seed_report.sh"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$(git branch --show-current)" == "1.1" ]] || fail "run this script on branch 1.1"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "${PYTHON_BIN}" > /dev/null 2>&1; then
  PYTHON_BIN=python3
fi
command -v "${PYTHON_BIN}" > /dev/null 2>&1 || fail "python or python3 is required"

REPORT_PARENT=$(dirname "${REPORT_DIR}")
REPORT_NAME=$(basename "${REPORT_DIR}")
STAGING_GLOB="${REPORT_PARENT}/.${REPORT_NAME}.staging-*"

allow_report_only_dirty_tree() {
  local status_line staged_path
  while IFS= read -r status_line; do
    [[ -z "${status_line}" ]] && continue
    staged_path=${status_line:3}
    case "${staged_path}" in
      "${REPORT_DIR}/"*) ;;
      *) fail "working tree contains unrelated changes: ${status_line}" ;;
    esac
  done < <(git status --short --untracked-files=all)
  git reset --quiet -- "${REPORT_DIR}" || true
}

allow_report_only_dirty_tree

git fetch origin --prune
git pull --ff-only origin 1.1
git merge --no-edit origin/main
git merge-base --is-ancestor origin/1.1 HEAD || fail "origin/1.1 must be an ancestor of HEAD"
git merge-base --is-ancestor origin/main HEAD || fail "origin/main must be an ancestor of HEAD"

allow_report_only_dirty_tree

[[ -f "${RUN_DIR}/RUN_COMPLETE" ]] || fail "completed run marker is missing: ${RUN_DIR}/RUN_COMPLETE"
[[ -s "${RUN_DIR}/run_summary.data" ]] || fail "completed run summary is missing or empty: ${RUN_DIR}/run_summary.data"
[[ -x "${EXPORTER}" ]] || fail "exporter is missing or not executable: ${EXPORTER}"
validate_manifest() {
  [[ -s "${REPORT_DIR}/export_manifest.json" ]] || fail "export_manifest.json is missing or empty"
  "${PYTHON_BIN}" - "${REPORT_DIR}/export_manifest.json" "${RUN_DIR}" "${REPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
expected_source = Path(sys.argv[2]).resolve()
expected_report = Path(sys.argv[3]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest.get("status") == "complete", manifest
assert int(manifest.get("attempt_count", 0)) >= 2, manifest
assert int(manifest.get("retry_count", 0)) >= 1, manifest
assert Path(manifest["source_run"]).resolve() == expected_source, manifest
assert Path(manifest["report_identity"]).resolve() == expected_report, manifest
assert manifest.get("failure_reason"), manifest
print(json.dumps({
    "status": manifest["status"],
    "attempt_count": manifest["attempt_count"],
    "retry_count": manifest["retry_count"],
    "source_run": manifest["source_run"],
    "report_identity": manifest["report_identity"],
    "failure_reason": manifest["failure_reason"],
}, ensure_ascii=False, indent=2))
PY
}

if [[ -e "${REPORT_DIR}" ]]; then
  echo "[1/5] Existing complete retry report found; skipping export and resuming publication."
  validate_manifest
else
  CANARY_LOG=$(mktemp)
  trap 'rm -f "${CANARY_LOG}"' EXIT
  echo "[1/5] Running the required exporter cleanup canary (expected failure)..."
  set +e
  Q_EPVG_EXPORT_INJECT_FAILURE_AFTER=2 \
    bash "${EXPORTER}" \
      --run-dir "${RUN_DIR}" \
      --report-dir "${REPORT_DIR}" \
      --no-commit 2>&1 | tee "${CANARY_LOG}"
  CANARY_STATUS=${PIPESTATUS[0]}
  set -e
  [[ ${CANARY_STATUS} -ne 0 ]] || fail "the injected exporter failure unexpectedly succeeded"
  grep -Fq "injected copy failure" "${CANARY_LOG}" || fail "exporter failed before the injected copy-failure checkpoint"
  [[ ! -e "${REPORT_DIR}" ]] || fail "failed export left a final report directory"
  if compgen -G "${STAGING_GLOB}" > /dev/null; then
    fail "failed export left a staging directory"
  fi

  echo "[2/5] Retrying export from the same immutable completed run..."
  bash "${EXPORTER}" \
    --run-dir "${RUN_DIR}" \
    --report-dir "${REPORT_DIR}" \
    --no-commit
  validate_manifest
fi

echo "[3/5] Checking that only the exporter-approved report is staged..."
git add -- "${REPORT_DIR}"
git diff --cached --check
mapfile -t STAGED_FILES < <(git diff --cached --name-only)
[[ ${#STAGED_FILES[@]} -gt 0 ]] || fail "exporter staged no report files"
for staged in "${STAGED_FILES[@]}"; do
  case "${staged}" in
    "${REPORT_DIR}/"*) ;;
    *) fail "unexpected staged path: ${staged}" ;;
  esac
done

echo "[4/5] Committing and pushing only the retry report to 1.1..."
if git diff --cached --quiet; then
  echo "No new staged files; the retry report is already committed locally."
else
  git commit -m "Re-export Q-EPVG seed13 report with lifecycle manifest"
fi

echo "[5/5] Pushing the report commit to 1.1..."
git push origin 1.1
git status --short --branch
git log -1 --oneline
echo "REPORT_DIR=${REPORT_DIR}"
