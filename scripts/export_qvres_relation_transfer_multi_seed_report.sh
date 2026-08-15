#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

usage() {
  echo "Usage: bash scripts/export_qvres_relation_transfer_multi_seed_report.sh GROUP_DIR [REPORT_DIR]"
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
cd "${ROOT}"
GROUP_DIR=$(readlink -f "$1")
case "${GROUP_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "GROUP_DIR must be inside ${ROOT}/runs" >&2; exit 1 ;;
esac

[[ -f "${GROUP_DIR}/MULTI_SEED_COMPLETE" ]] || { echo "Multi-seed run is incomplete." >&2; exit 1; }
[[ -f "${GROUP_DIR}/multi_seed_manifest.json" ]] || { echo "Missing multi_seed_manifest.json." >&2; exit 1; }
[[ -f "${GROUP_DIR}/multi_seed_summary.json" ]] || { echo "Missing multi_seed_summary.json." >&2; exit 1; }
[[ -f "${GROUP_DIR}/preflight.log" ]] || { echo "Missing preflight.log." >&2; exit 1; }
GIT_STATUS=$(git status --porcelain --untracked-files=all)
[[ -z "${GIT_STATUS}" ]] || { echo "Repository must be clean before export:" >&2; printf '%s\n' "${GIT_STATUS}" >&2; exit 1; }

HEAD=$(git rev-parse HEAD)
RUN_COMMIT=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${GROUP_DIR}/multi_seed_manifest.json'))['git_commit'])")
[[ "${HEAD}" == "${RUN_COMMIT}" ]] || { echo "Run commit ${RUN_COMMIT} does not match HEAD ${HEAD}." >&2; exit 1; }

REPORT_DIR=${2:-reports/q_vres_relation_transfer/$(date -u +%Y%m%d-%H%M%S)-full-multiseed}
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in
  "${ROOT}"/reports/q_vres_relation_transfer/*) ;;
  *) echo "REPORT_DIR must be inside reports/q_vres_relation_transfer." >&2; exit 1 ;;
esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite ${REPORT_DIR}." >&2; exit 1; }
mkdir -p "${REPORT_DIR}"

cp "${GROUP_DIR}/multi_seed_manifest.json" "${REPORT_DIR}/"
cp "${GROUP_DIR}/multi_seed_summary.json" "${REPORT_DIR}/"
cp "${GROUP_DIR}/MULTI_SEED_COMPLETE" "${REPORT_DIR}/"
tail -n 1000 "${GROUP_DIR}/preflight.log" > "${REPORT_DIR}/preflight.log.tail.txt"
printf '%s\n' "${HEAD}" > "${REPORT_DIR}/export_commit.txt"
wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data_counts.txt"

mapfile -t SEED_DIRS < <(find "${GROUP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort -V)
[[ ${#SEED_DIRS[@]} -gt 0 ]] || { echo "No seed directories found." >&2; exit 1; }
for SEED_DIR in "${SEED_DIRS[@]}"; do
  SEED_NAME=$(basename "${SEED_DIR}")
  DEST="${REPORT_DIR}/${SEED_NAME}"
  [[ -f "${SEED_DIR}/RUN_COMPLETE" ]] || { echo "Incomplete seed: ${SEED_DIR}" >&2; exit 1; }
  for FILE in run_summary.json run_summary.md run_config.json baseline/metrics.json; do
    [[ -f "${SEED_DIR}/${FILE}" ]] || { echo "Missing ${SEED_DIR}/${FILE}" >&2; exit 1; }
  done
  for SELECTOR in disabled q_causal_transport classical_causal_transport q_causal_key_only; do
    [[ -f "${SEED_DIR}/selectors/${SELECTOR}/metrics.json" ]] || {
      echo "Missing selector metrics: ${SEED_DIR}/selectors/${SELECTOR}/metrics.json" >&2
      exit 1
    }
  done
  mkdir -p "${DEST}/selectors" "${DEST}/logs"
  cp "${SEED_DIR}/RUN_COMPLETE" "${DEST}/"
  cp "${SEED_DIR}/run_summary.json" "${DEST}/"
  cp "${SEED_DIR}/run_summary.md" "${DEST}/"
  cp "${SEED_DIR}/run_config.json" "${DEST}/"
  cp "${SEED_DIR}/baseline/metrics.json" "${DEST}/baseline_metrics.json"
  for SELECTOR in disabled q_causal_transport classical_causal_transport q_causal_key_only; do
    cp "${SEED_DIR}/selectors/${SELECTOR}/metrics.json" "${DEST}/selectors/${SELECTOR}.json"
  done
  tail -n 1000 "${SEED_DIR}/logs/run.log" > "${DEST}/logs/run.log.tail.txt"
  tail -n 1000 "${SEED_DIR}/baseline_train.log" > "${DEST}/logs/baseline_train.log.tail.txt"
done

"${PYTHON_BIN}" scripts/summarize_qvres_relation_transfer_multi_seed.py \
  --group-dir "${GROUP_DIR}" \
  --output-json "${REPORT_DIR}/aggregate_summary.json" \
  --output-md "${REPORT_DIR}/aggregate_summary.md"

if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.jsonl' \) | grep -q .; then
  echo "Forbidden private artifact detected in report." >&2
  exit 1
fi

echo "Report ready: ${REPORT_DIR}"
echo "Next: git add ${REPORT_DIR} && git diff --cached --check"
