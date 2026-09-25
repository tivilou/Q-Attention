#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR=
REPORT_DIR=
NO_COMMIT=0

usage() {
  cat <<'EOF'
Usage: bash scripts/export_retacred_qlass_formal_single_seed_report.sh [options]

Export one completed Q-LASS raw run to reports/ and commit it to branch 1.1.

Options:
  --run-dir PATH       Raw run directory; default is the newest *_seed13 run
  --report-dir PATH    Destination under reports/retacred_qlass_formal_single_seed/
  --no-commit          Generate and stage the report without commit or push
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; RUN_DIR=$2; shift 2 ;;
    --report-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; REPORT_DIR=$2; shift 2 ;;
    --no-commit) NO_COMMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${ROOT}"
[[ "$(git branch --show-current)" == "1.1" ]] || {
  echo "Run this exporter from branch 1.1; no commit or push was made." >&2
  exit 1
}
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "Working tree must be clean before export; no commit or push was made." >&2
  git status --short >&2
  exit 1
}
git merge-base --is-ancestor origin/main HEAD || {
  echo "Branch 1.1 must include origin/main before export; run git pull/merge first." >&2
  exit 1
}
[[ -z "$(git diff --cached --name-only)" ]] || {
  echo "Index is not empty; clear unrelated staged files before export." >&2
  exit 1
}

if [[ -z "${RUN_DIR}" ]]; then
  mapfile -t RUN_CANDIDATES < <(find "${ROOT}/runs/retacred_qlass_formal_single_seed" \
    -mindepth 1 -maxdepth 1 -type d -name '*_seed13' -print 2>/dev/null | sort)
  [[ ${#RUN_CANDIDATES[@]} -gt 0 ]] || {
    echo "No *_seed13 raw run found under runs/retacred_qlass_formal_single_seed." >&2
    exit 1
  }
  RUN_DIR=${RUN_CANDIDATES[${#RUN_CANDIDATES[@]}-1]}
fi
[[ "${RUN_DIR}" = /* ]] || RUN_DIR="${ROOT}/${RUN_DIR}"
RUN_DIR=$(readlink -f "${RUN_DIR}")

required_files=(
  RUN_COMPLETE
  run_summary.json
  run_summary.md
  baseline/metrics.json
  quantum_global_context/metrics.json
  quantum_global_context/valid/metrics.json
  quantum_global_context/test/metrics.json
  classical_global_context/metrics.json
  classical_global_context/valid/metrics.json
  classical_global_context/test/metrics.json
)
[[ -d "${RUN_DIR}" ]] || { echo "Missing raw run: ${RUN_DIR}" >&2; exit 1; }
[[ ! -e "${RUN_DIR}/RUN_FAILED" ]] || { echo "Raw run contains RUN_FAILED." >&2; exit 1; }
for file in "${required_files[@]}"; do
  [[ -f "${RUN_DIR}/${file}" ]] || { echo "Missing ${RUN_DIR}/${file}" >&2; exit 1; }
done

"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
assert summary.get("seed") == 13, summary.get("seed")
contract = summary.get("data_split_contract", {})
assert contract.get("test_used_for_training_or_selection") is False, contract
for split in ("valid", "test"):
    assert split in summary, split
    for method in ("quantum", "classical"):
        assert method in summary[split], f"{split}/{method}"
assert "quantum_minus_classical" in summary
print("Q-LASS summary contract=OK")
PY

if [[ -z "${REPORT_DIR}" ]]; then
  REPORT_DIR="reports/retacred_qlass_formal_single_seed/$(date -u +%Y%m%d-%H%M%S)-full-single-seed13"
fi
[[ "${REPORT_DIR}" = /* ]] || REPORT_DIR="${ROOT}/${REPORT_DIR}"
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in
  "${ROOT}/reports/retacred_qlass_formal_single_seed/"*) ;;
  *) echo "Report directory must be under reports/retacred_qlass_formal_single_seed/." >&2; exit 1 ;;
esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite ${REPORT_DIR}." >&2; exit 1; }

mkdir -p "${REPORT_DIR}/metrics" "${REPORT_DIR}/status" "${REPORT_DIR}/logs"
cp "${RUN_DIR}/RUN_COMPLETE" "${REPORT_DIR}/"
cp "${RUN_DIR}/run_summary.json" "${REPORT_DIR}/"
cp "${RUN_DIR}/run_summary.md" "${REPORT_DIR}/"
cp "configs/retacred_qlass_formal_single_seed.json" "${REPORT_DIR}/run_config.json"
cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/metrics/baseline.json"
cp "${RUN_DIR}/quantum_global_context/valid/metrics.json" "${REPORT_DIR}/metrics/quantum_valid.json"
cp "${RUN_DIR}/quantum_global_context/test/metrics.json" "${REPORT_DIR}/metrics/quantum_test.json"
cp "${RUN_DIR}/classical_global_context/valid/metrics.json" "${REPORT_DIR}/metrics/classical_valid.json"
cp "${RUN_DIR}/classical_global_context/test/metrics.json" "${REPORT_DIR}/metrics/classical_test.json"

if [[ -f "${RUN_DIR}/provenance.env" ]]; then
  cp "${RUN_DIR}/provenance.env" "${REPORT_DIR}/provenance.env"
else
  printf '%s\n' 'PROVENANCE_STATUS=not_recorded_by_runner' > "${REPORT_DIR}/provenance.env"
fi
printf '%s\n' "$(git rev-parse HEAD)" > "${REPORT_DIR}/reporting_commit.txt"
wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data_counts.txt"
sha256sum data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data.sha256"

for stage in baseline quantum_global_context classical_global_context quantum_valid_eval quantum_test_eval classical_valid_eval classical_test_eval; do
  if [[ -f "${RUN_DIR}/logs/${stage}.log" ]]; then
    tail -n 1000 "${RUN_DIR}/logs/${stage}.log" > "${REPORT_DIR}/logs/${stage}.log.tail.txt"
  fi
done
for status in "${RUN_DIR}/status/"*.env; do
  [[ -f "${status}" ]] || continue
  cp "${status}" "${REPORT_DIR}/status/"
done

if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.jsonl' \) | grep -q .; then
  echo "Forbidden private artifact detected in report." >&2
  exit 1
fi

REPORT_REL=${REPORT_DIR#"${ROOT}/"}
git add -- "${REPORT_REL}"
git diff --cached --check
git diff --cached --name-only
if [[ ${NO_COMMIT} -eq 1 ]]; then
  echo "REPORT_DIR=${REPORT_REL}"
  echo "Staged only; no commit or push was made."
  exit 0
fi

git commit -m "Add Q-LASS Re-TACRED formal single-seed report"
git push origin 1.1
echo "REPORT_DIR=${REPORT_REL}"
echo "Pushed report commit to origin/1.1."
