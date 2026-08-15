#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

usage() {
  echo "Usage: bash scripts/export_qvres_relation_transfer_pilot_report.sh RUN_DIR [REPORT_DIR]"
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
cd "${ROOT}"
RUN_DIR=$(readlink -f "$1")
case "${RUN_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "RUN_DIR must be inside ${ROOT}/runs" >&2; exit 1 ;;
esac

for FILE in \
  RUN_COMPLETE \
  run_summary.json \
  run_summary.md \
  run_config.json \
  selector_parallel_manifest.json \
  selector_parallel_summary.json \
  preflight.log \
  baseline/metrics.json; do
  [[ -f "${RUN_DIR}/${FILE}" ]] || { echo "Missing ${RUN_DIR}/${FILE}" >&2; exit 1; }
done
[[ ! -e "${RUN_DIR}/RUN_FAILED" ]] || { echo "Pilot run also contains RUN_FAILED." >&2; exit 1; }

GIT_STATUS=$(git status --porcelain --untracked-files=all)
[[ -z "${GIT_STATUS}" ]] || {
  echo "Repository must be clean before export:" >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 1
}

IFS=$'\t' read -r RUN_COMMIT SEED < <(
  "${PYTHON_BIN}" -c '
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
required = ["disabled", "q_causal_transport", "classical_causal_transport", "q_causal_key_only"]
assert p.get("formal_experiment") is True
assert p.get("partial_selector_run") is False
assert p.get("parallel_mode") == "selectors"
assert p.get("status") == "pass"
assert p.get("selectors") == required
assert p.get("pilot_validation_gate", {}).get("selection_split") == "valid"
assert p.get("provenance", {}).get("git_dirty") is False
print(p["provenance"]["git_commit"], p["seed"], sep="\t")
' "${RUN_DIR}/run_summary.json"
)
HEAD=$(git rev-parse HEAD)
[[ "${HEAD}" == "${RUN_COMMIT}" ]] || {
  echo "Run commit ${RUN_COMMIT} does not match HEAD ${HEAD}." >&2
  exit 1
}

REPORT_DIR=${2:-reports/q_vres_relation_transfer/$(date -u +%Y%m%d-%H%M%S)-full-pilot-seed${SEED}}
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in
  "${ROOT}"/reports/q_vres_relation_transfer/*) ;;
  *) echo "REPORT_DIR must be inside reports/q_vres_relation_transfer." >&2; exit 1 ;;
esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite ${REPORT_DIR}." >&2; exit 1; }

mkdir -p "${REPORT_DIR}/selectors" "${REPORT_DIR}/logs" "${REPORT_DIR}/status"
cp "${RUN_DIR}/RUN_COMPLETE" "${REPORT_DIR}/"
cp "${RUN_DIR}/run_summary.json" "${REPORT_DIR}/"
cp "${RUN_DIR}/run_summary.md" "${REPORT_DIR}/"
cp "${RUN_DIR}/run_config.json" "${REPORT_DIR}/"
cp "${RUN_DIR}/selector_parallel_manifest.json" "${REPORT_DIR}/"
cp "${RUN_DIR}/selector_parallel_summary.json" "${REPORT_DIR}/"
tail -n 1000 "${RUN_DIR}/preflight.log" > "${REPORT_DIR}/preflight.log.tail.txt"
cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/baseline_metrics.json"
cp "${RUN_DIR}/status/selector_parallel_status.json" "${REPORT_DIR}/status/"
cp "${RUN_DIR}/status/"*.env "${REPORT_DIR}/status/"

for SELECTOR in disabled q_causal_transport classical_causal_transport q_causal_key_only; do
  SOURCE="${RUN_DIR}/selectors/${SELECTOR}/metrics.json"
  [[ -f "${SOURCE}" ]] || { echo "Missing selector metrics: ${SOURCE}" >&2; exit 1; }
  cp "${SOURCE}" "${REPORT_DIR}/selectors/${SELECTOR}.json"
done
for STAGE in baseline q_causal_transport classical_causal_transport q_causal_key_only; do
  SOURCE="${RUN_DIR}/logs/${STAGE}.log"
  [[ -f "${SOURCE}" ]] || { echo "Missing stage log: ${SOURCE}" >&2; exit 1; }
  tail -n 1000 "${SOURCE}" > "${REPORT_DIR}/logs/${STAGE}.log.tail.txt"
done
[[ -f "${RUN_DIR}/baseline_train.log" ]] || { echo "Missing baseline_train.log" >&2; exit 1; }
tail -n 1000 "${RUN_DIR}/baseline_train.log" > "${REPORT_DIR}/logs/baseline_train.log.tail.txt"
printf '%s\n' "${HEAD}" > "${REPORT_DIR}/export_commit.txt"
wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data_counts.txt"

if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.jsonl' \) | grep -q .; then
  echo "Forbidden private artifact detected in report." >&2
  exit 1
fi

echo "Pilot report ready: ${REPORT_DIR}"
echo "Next: git add ${REPORT_DIR} && git diff --cached --check"
