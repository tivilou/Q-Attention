#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Neither python nor python3 is available; set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

usage() { echo "Usage: bash scripts/export_retacred_baseline_report.sh RUN_DIR [REPORT_DIR]"; }
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
cd "${ROOT}"
[[ "$(git branch --show-current)" == "1.1" ]] || {
  echo "Exporter must run on branch 1.1; no report was committed." >&2
  exit 1
}
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "Working tree must be clean before export; no report was committed." >&2
  git status --short >&2
  exit 1
}
git merge-base --is-ancestor origin/main HEAD || {
  echo "Branch 1.1 must include origin/main before export." >&2
  exit 1
}
git merge-base --is-ancestor origin/1.1 HEAD || {
  echo "Branch 1.1 must include origin/1.1 before export." >&2
  exit 1
}
[[ -z "$(git diff --cached --name-only)" ]] || {
  echo "Index is not empty; clear unrelated staged files before export." >&2
  exit 1
}
RUN_DIR=$(readlink -f "$1")
case "${RUN_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "RUN_DIR must be inside ${ROOT}/runs" >&2; exit 1 ;;
esac

for FILE in RUN_COMPLETE run_summary.json run_summary.data run_summary.md run_config.json metrics.json; do
  [[ -f "${RUN_DIR}/${FILE}" ]] || { echo "Missing ${RUN_DIR}/${FILE}" >&2; exit 1; }
done
[[ ! -e "${RUN_DIR}/RUN_FAILED" ]] || { echo "Run contains RUN_FAILED." >&2; exit 1; }

IFS=$'\t' read -r RUN_COMMIT SEED STATUS < <(
  "${PYTHON_BIN}" - "${RUN_DIR}/run_summary.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload.get("schema") == "q-attention.retacred-baseline-evaluation.v1"
assert payload.get("status") == "complete"
assert payload.get("test_used_for_training_or_selection") is False
assert payload.get("provenance", {}).get("git_dirty") is False
required = {"micro_precision", "micro_recall", "micro_f1", "accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"}
for split in ("train", "valid", "test"):
    metrics = payload["splits"][split]["metrics"]
    assert required.issubset(metrics), (split, sorted(required - set(metrics)))
    assert payload["splits"][split]["per_class"]
print(payload["provenance"].get("git_commit") or "", payload["seed"], payload["status"], sep="\t")
PY
)
[[ -s "${RUN_DIR}/run_summary.data" ]] || { echo "run_summary.data is empty." >&2; exit 1; }
grep -q '^schema=q-attention\.retacred-baseline-run-summary-data\.v1$' "${RUN_DIR}/run_summary.data" || {
  echo "run_summary.data has an unexpected schema." >&2
  exit 1
}
grep -q '^status=complete$' "${RUN_DIR}/run_summary.data" || {
  echo "run_summary.data does not record status=complete." >&2
  exit 1
}
HEAD=$(git rev-parse HEAD)
[[ -z "${RUN_COMMIT}" || "${HEAD}" == "${RUN_COMMIT}" ]] || {
  echo "Run commit ${RUN_COMMIT} does not match HEAD ${HEAD}." >&2
  exit 1
}

REPORT_DIR=${2:-reports/retacred_baseline_complete/$(basename "${RUN_DIR}")}
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in
  "${ROOT}"/reports/retacred_baseline_complete/*) ;;
  *) echo "REPORT_DIR must be inside reports/retacred_baseline_complete." >&2; exit 1 ;;
esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite ${REPORT_DIR}." >&2; exit 1; }

mkdir -p "${ROOT}/reports"
TMP_DIR=$(mktemp -d "${ROOT}/reports/.baseline-export.XXXXXX")
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT
mkdir -p "${TMP_DIR}/logs"
cp "${RUN_DIR}/RUN_COMPLETE" "${TMP_DIR}/"
cp "${RUN_DIR}/run_summary.json" "${TMP_DIR}/"
cp "${RUN_DIR}/run_summary.data" "${TMP_DIR}/"
cp "${RUN_DIR}/run_summary.md" "${TMP_DIR}/"
cp "${RUN_DIR}/run_config.json" "${TMP_DIR}/"
cp "${RUN_DIR}/metrics.json" "${TMP_DIR}/baseline_metrics.json"
tail -n 1000 "${RUN_DIR}/logs/baseline_train.log" > "${TMP_DIR}/logs/baseline_train.log.tail.txt" 2>/dev/null || true
tail -n 1000 "${RUN_DIR}/logs/baseline_evaluation.log" > "${TMP_DIR}/logs/baseline_evaluation.log.tail.txt" 2>/dev/null || true
{
  wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl
} > "${TMP_DIR}/data_counts.txt"
sha256sum data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${TMP_DIR}/data.sha256"
printf '%s\n' "${HEAD}" > "${TMP_DIR}/export_commit.txt"
"${PYTHON_BIN}" - "${TMP_DIR}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

stage = Path(sys.argv[1])
summary = json.loads((stage / "run_summary.json").read_text(encoding="utf-8"))
expected = {"train": 58465, "valid": 19584, "test": 13418}
counts = {}
for line in (stage / "data_counts.txt").read_text(encoding="utf-8").splitlines():
    match = re.match(r"\s*(\d+)\s+.*\/(train|valid|test)\.jsonl$", line)
    if match:
        counts[match.group(2)] = int(match.group(1))
assert counts == expected, counts
hash_rows = {}
for line in (stage / "data.sha256").read_text(encoding="utf-8").splitlines():
    digest, filename = line.split(None, 1)
    hash_rows[Path(filename).stem] = digest
for split, expected_count in expected.items():
    assert summary["data"][split]["records"] == expected_count
    assert summary["data"][split]["sha256"] == hash_rows[split]
data_text = (stage / "run_summary.data").read_text(encoding="utf-8")
assert "schema=q-attention.retacred-baseline-run-summary-data.v1" in data_text
assert "status=complete" in data_text
print("baseline report identity=OK")
PY
if find "${TMP_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.jsonl' -o -name 'predictions*' \) | grep -q .; then
  echo "Forbidden private artifact detected in report staging." >&2
  exit 1
fi
mkdir -p "$(dirname "${REPORT_DIR}")"
mv "${TMP_DIR}" "${REPORT_DIR}"
trap - EXIT
echo "Baseline report ready: ${REPORT_DIR}"
echo "Standalone export complete; the runner adds, commits, and pushes this report automatically."
