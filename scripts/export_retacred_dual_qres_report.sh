#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  echo "Usage: bash scripts/export_retacred_dual_qres_report.sh RUN_DIR [REPORT_DIR]"
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
cd "${ROOT}"

RUN_DIR=$(readlink -f "$1")
case "${RUN_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "RUN_DIR must be inside ${ROOT}/runs" >&2; exit 1 ;;
esac

[[ -f "${RUN_DIR}/RUN_COMPLETE" ]] || { echo "Run is incomplete: ${RUN_DIR}" >&2; exit 1; }
[[ -f "${RUN_DIR}/run_manifest.env" ]] || { echo "Missing run manifest." >&2; exit 1; }

REQUIRED=(
  "${RUN_DIR}/baseline/metrics.json"
  "${RUN_DIR}/core/quantum/metrics.json"
  "${RUN_DIR}/core/quantum/diagnostics.json"
  "${RUN_DIR}/core/classical/metrics.json"
  "${RUN_DIR}/core/classical/diagnostics.json"
  "${RUN_DIR}/selector/quantum/metrics.json"
  "${RUN_DIR}/selector/quantum/diagnostics.json"
  "${RUN_DIR}/selector/classical/metrics.json"
  "${RUN_DIR}/selector/classical/diagnostics.json"
  "${RUN_DIR}/selector/classical_strong/metrics.json"
  "${RUN_DIR}/selector/classical_strong/diagnostics.json"
)
for FILE in "${REQUIRED[@]}"; do
  [[ -f "${FILE}" ]] || { echo "Missing required result: ${FILE}" >&2; exit 1; }
done

GIT_STATUS=$(git status --porcelain)
[[ -z "${GIT_STATUS}" ]] || {
  echo "Repository must be clean before report export:" >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 1
}

REPORT_DIR=${2:-reports/retacred/$(date +%Y%m%d-%H%M%S)-dual-qres-full}
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite ${REPORT_DIR}" >&2; exit 1; }

mkdir -p "${REPORT_DIR}"/{baseline,core/quantum,core/classical,selector/quantum,selector/classical,selector/classical_strong,logs}
cp "${RUN_DIR}/run_manifest.env" "${REPORT_DIR}/"
cp "${RUN_DIR}/RUN_COMPLETE" "${REPORT_DIR}/"
git rev-parse HEAD > "${REPORT_DIR}/export_commit.txt"
: > "${REPORT_DIR}/git_status.txt"
wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl > "${REPORT_DIR}/data_counts.txt"

cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/baseline/"
for FAMILY in quantum classical; do
  cp "${RUN_DIR}/core/${FAMILY}/metrics.json" "${REPORT_DIR}/core/${FAMILY}/"
  cp "${RUN_DIR}/core/${FAMILY}/diagnostics.json" "${REPORT_DIR}/core/${FAMILY}/"
done
for METHOD in quantum classical classical_strong; do
  cp "${RUN_DIR}/selector/${METHOD}/metrics.json" "${REPORT_DIR}/selector/${METHOD}/"
  cp "${RUN_DIR}/selector/${METHOD}/diagnostics.json" "${REPORT_DIR}/selector/${METHOD}/"
done
for LOG in "${RUN_DIR}"/logs/*.log; do
  tail -n 1000 "${LOG}" > "${REPORT_DIR}/logs/$(basename "${LOG}").tail.txt"
done

if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.jsonl' \) | grep -q .; then
  echo "Forbidden private artifact detected in report." >&2
  exit 1
fi

echo "Report ready: ${REPORT_DIR}"
echo "Next: git add \"${REPORT_DIR}\" && git diff --cached --check"
