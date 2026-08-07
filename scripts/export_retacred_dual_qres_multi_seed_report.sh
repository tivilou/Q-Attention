#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  echo "Usage: bash scripts/export_retacred_dual_qres_multi_seed_report.sh GROUP_DIR [REPORT_DIR]"
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
cd "${ROOT}"

GROUP_DIR=$(readlink -f "$1")
case "${GROUP_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "GROUP_DIR must be inside ${ROOT}/runs" >&2; exit 1 ;;
esac

[[ -f "${GROUP_DIR}/MULTI_SEED_COMPLETE" ]] || {
  echo "Multi-seed run is incomplete: ${GROUP_DIR}" >&2
  exit 1
}
[[ -f "${GROUP_DIR}/multi_seed_manifest.json" ]] || { echo "Missing multi-seed manifest." >&2; exit 1; }
[[ -f "${GROUP_DIR}/multi_seed_summary.json" ]] || { echo "Missing multi-seed summary." >&2; exit 1; }
[[ -f "${GROUP_DIR}/preflight.log" ]] || { echo "Missing central preflight log." >&2; exit 1; }

GIT_STATUS=$(git status --porcelain)
[[ -z "${GIT_STATUS}" ]] || {
  echo "Repository must be clean before report export:" >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 1
}

REPORT_DIR=${2:-reports/retacred/$(date +%Y%m%d-%H%M%S)-dual-qres-multiseed}
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite ${REPORT_DIR}" >&2; exit 1; }
mkdir -p "${REPORT_DIR}"
cp "${GROUP_DIR}/multi_seed_manifest.json" "${REPORT_DIR}/"
cp "${GROUP_DIR}/multi_seed_summary.json" "${REPORT_DIR}/"
cp "${GROUP_DIR}/MULTI_SEED_COMPLETE" "${REPORT_DIR}/"
tail -n 1000 "${GROUP_DIR}/preflight.log" > "${REPORT_DIR}/preflight.log.tail.txt"
git rev-parse HEAD > "${REPORT_DIR}/export_commit.txt"
: > "${REPORT_DIR}/git_status.txt"
wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl > "${REPORT_DIR}/data_counts.txt"

mapfile -t SEED_DIRS < <(find "${GROUP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort -V)
[[ ${#SEED_DIRS[@]} -gt 0 ]] || { echo "No seed run directories found." >&2; exit 1; }

FORMAL_LOGS=(
  baseline
  core_quantum
  core_classical
  selector_quantum
  selector_classical
  selector_classical_strong
)

for SEED_DIR in "${SEED_DIRS[@]}"; do
  SEED_NAME=$(basename "${SEED_DIR}")
  DEST="${REPORT_DIR}/${SEED_NAME}"
  [[ -f "${SEED_DIR}/RUN_COMPLETE" ]] || { echo "Incomplete seed run: ${SEED_DIR}" >&2; exit 1; }
  REQUIRED=(
    "${SEED_DIR}/run_manifest.env"
    "${SEED_DIR}/baseline/metrics.json"
    "${SEED_DIR}/core/quantum/metrics.json"
    "${SEED_DIR}/core/quantum/diagnostics.json"
    "${SEED_DIR}/core/classical/metrics.json"
    "${SEED_DIR}/core/classical/diagnostics.json"
    "${SEED_DIR}/selector/quantum/metrics.json"
    "${SEED_DIR}/selector/quantum/diagnostics.json"
    "${SEED_DIR}/selector/classical/metrics.json"
    "${SEED_DIR}/selector/classical/diagnostics.json"
    "${SEED_DIR}/selector/classical_strong/metrics.json"
    "${SEED_DIR}/selector/classical_strong/diagnostics.json"
  )
  for FILE in "${REQUIRED[@]}"; do
    [[ -f "${FILE}" ]] || { echo "Missing required result: ${FILE}" >&2; exit 1; }
  done

  mkdir -p "${DEST}"/{baseline,core/quantum,core/classical,selector/quantum,selector/classical,selector/classical_strong,logs}
  cp "${SEED_DIR}/run_manifest.env" "${DEST}/"
  cp "${SEED_DIR}/RUN_COMPLETE" "${DEST}/"
  cp "${SEED_DIR}/baseline/metrics.json" "${DEST}/baseline/"
  for FAMILY in quantum classical; do
    cp "${SEED_DIR}/core/${FAMILY}/metrics.json" "${DEST}/core/${FAMILY}/"
    cp "${SEED_DIR}/core/${FAMILY}/diagnostics.json" "${DEST}/core/${FAMILY}/"
  done
  for METHOD in quantum classical classical_strong; do
    cp "${SEED_DIR}/selector/${METHOD}/metrics.json" "${DEST}/selector/${METHOD}/"
    cp "${SEED_DIR}/selector/${METHOD}/diagnostics.json" "${DEST}/selector/${METHOD}/"
  done
  for NAME in "${FORMAL_LOGS[@]}"; do
    LOG="${SEED_DIR}/logs/${NAME}.log"
    [[ -f "${LOG}" ]] || { echo "Missing formal stage log: ${LOG}" >&2; exit 1; }
    tail -n 1000 "${LOG}" > "${DEST}/logs/${NAME}.log.tail.txt"
  done
done

if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.jsonl' \) | grep -q .; then
  echo "Forbidden private artifact detected in report." >&2
  exit 1
fi

echo "Report ready: ${REPORT_DIR}"
echo "Next: git add \"${REPORT_DIR}\" && git diff --cached --check"
