#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GROUP_DIR=
REPORT_DIR=
NO_COMMIT=0
PROTOCOL=
SELECTOR_CANDIDATE=
SELECTOR_MATCHED=
REPORT_ROOT=

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ "${PYTHON_BIN}" == */* ]]; then
      [[ -x "${PYTHON_BIN}" ]] && { printf '%s\n' "${PYTHON_BIN}"; return; }
    elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"; return
    fi
    echo "PYTHON_BIN is not executable or not on PATH: ${PYTHON_BIN}" >&2
    return 1
  fi
  for candidate in python python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN." >&2
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group-dir|--report-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1." >&2; exit 2; }
      if [[ "$1" == "--group-dir" ]]; then
        GROUP_DIR=$2
      else
        REPORT_DIR=$2
      fi
      shift 2
      ;;
    --no-commit) NO_COMMIT=1; shift ;;
    -h|--help)
      echo "Usage: $0 --group-dir PATH [--report-dir PATH] [--no-commit]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHON_BIN=$(resolve_python_bin)
export PYTHON_BIN
cd "${ROOT}"
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Exporter must run on branch 1.1." >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Working tree must be clean before export." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "origin/1.1 must be an ancestor of HEAD." >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "origin/main must be an ancestor of HEAD." >&2; exit 1; }
[[ -n "${GROUP_DIR}" ]] || { echo "--group-dir is required." >&2; exit 2; }
[[ "${GROUP_DIR}" = /* ]] || GROUP_DIR="${ROOT}/${GROUP_DIR}"
GROUP_DIR=$(readlink -f "${GROUP_DIR}")
case "${GROUP_DIR}" in
  "${ROOT}/runs/retacred_qceasc_formal_multi_seed/"*|"${ROOT}/runs/retacred_qceasc_counterfactual_formal_multi_seed/"*) ;;
  *) echo "Group directory is outside a supported Q-CEASC multi-seed root." >&2; exit 1 ;;
esac
[[ -f "${GROUP_DIR}/MULTI_SEED_COMPLETE" ]] || { echo "Missing MULTI_SEED_COMPLETE." >&2; exit 1; }
[[ -f "${GROUP_DIR}/multi_seed_manifest.json" && -f "${GROUP_DIR}/multi_seed_status.json" ]] || { echo "Missing multi-seed manifest/status." >&2; exit 1; }
PROTOCOL=$(${PYTHON_BIN} - "${GROUP_DIR}/multi_seed_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8')).get('protocol', 'qceasc'))
PY
)
if [[ "${PROTOCOL}" == "qceasc_counterfactual" ]]; then
  SELECTOR_CANDIDATE=q_ceasc_counterfactual
  SELECTOR_MATCHED=classical_counterfactual
  REPORT_ROOT="${ROOT}/reports/retacred_qceasc_counterfactual_formal_multi_seed"
else
  SELECTOR_CANDIDATE=q_ceasc
  SELECTOR_MATCHED=classical_ceasc
  REPORT_ROOT="${ROOT}/reports/retacred_qceasc_formal_multi_seed"
fi
"${PYTHON_BIN}" scripts/summarize_retacred_qceasc_formal_multi_seed.py \
  --group-dir "${GROUP_DIR}" \
  --output-json "${GROUP_DIR}/multi_seed_summary.json" \
  --output-md "${GROUP_DIR}/multi_seed_summary.md"

SEEDS=$(${PYTHON_BIN} - "${GROUP_DIR}/multi_seed_manifest.json" <<'PY'
import json, sys
seeds = json.load(open(sys.argv[1], encoding='utf-8'))['seeds']
print(' '.join(str(int(seed)) for seed in seeds))
PY
)
for seed in ${SEEDS}; do
  SEED_DIR="${GROUP_DIR}/seed_${seed}"
  for file in RUN_COMPLETE run_summary.json run_summary.md run_config.json baseline/metrics.json; do
    [[ -f "${SEED_DIR}/${file}" ]] || { echo "Missing seed ${seed}/${file}." >&2; exit 1; }
  done
  for selector in "${SELECTOR_CANDIDATE}" "${SELECTOR_MATCHED}"; do
    for file in metrics.json case_study.json sample_trace.json; do
      [[ -f "${SEED_DIR}/selectors/${selector}/${file}" ]] || { echo "Missing seed ${seed}/${selector}/${file}." >&2; exit 1; }
    done
    CONFIG_SHA256=$(sha256sum "${SEED_DIR}/run_config.json" | awk '{print $1}')
    "${PYTHON_BIN}" scripts/validate_sample_trace.py "${SEED_DIR}/selectors/${selector}/sample_trace.json" --expected-config-sha256 "${CONFIG_SHA256}"
  done
done

DEFAULT_REPORT_DIR="${REPORT_ROOT#"${ROOT}/"}/$(basename "${GROUP_DIR}")"
REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}
[[ "${REPORT_DIR}" = /* ]] || REPORT_DIR="${ROOT}/${REPORT_DIR}"
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in
  "${REPORT_ROOT}/"*) ;;
  *) echo "Report must be under the selected Q-CEASC multi-seed report root." >&2; exit 1 ;;
esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite report directory." >&2; exit 1; }
mkdir -p "${REPORT_DIR}/seeds"
cp "${GROUP_DIR}/MULTI_SEED_COMPLETE" "${GROUP_DIR}/multi_seed_manifest.json" "${GROUP_DIR}/multi_seed_status.json" "${GROUP_DIR}/multi_seed_summary.json" "${GROUP_DIR}/multi_seed_summary.md" "${REPORT_DIR}/"
printf '%s\n' "$(git rev-parse HEAD)" > "${REPORT_DIR}/reporting_commit.txt"
for seed in ${SEEDS}; do
  SEED_DIR="${GROUP_DIR}/seed_${seed}"
  DEST="${REPORT_DIR}/seeds/seed_${seed}"
  mkdir -p "${DEST}/metrics" "${DEST}/case_study"
  cp "${SEED_DIR}/RUN_COMPLETE" "${SEED_DIR}/run_summary.json" "${SEED_DIR}/run_summary.md" "${SEED_DIR}/run_config.json" "${DEST}/"
  cp "${SEED_DIR}/baseline/metrics.json" "${DEST}/metrics/baseline.json"
  cp "${SEED_DIR}/selectors/${SELECTOR_CANDIDATE}/metrics.json" "${DEST}/metrics/${SELECTOR_CANDIDATE}.json"
  cp "${SEED_DIR}/selectors/${SELECTOR_MATCHED}/metrics.json" "${DEST}/metrics/${SELECTOR_MATCHED}.json"
  for selector in "${SELECTOR_CANDIDATE}" "${SELECTOR_MATCHED}"; do
    cp "${SEED_DIR}/selectors/${selector}/case_study.json" "${DEST}/case_study/${selector}.json"
    cp "${SEED_DIR}/selectors/${selector}/sample_trace.json" "${DEST}/case_study/${selector}.sample-trace.json"
  done
  [[ ! -f "${SEED_DIR}/imported_report.json" ]] || cp "${SEED_DIR}/imported_report.json" "${DEST}/imported_report.json"
  [[ ! -f "${SEED_DIR}/data.sha256" ]] || cp "${SEED_DIR}/data.sha256" "${DEST}/data.sha256"
  [[ ! -f "${SEED_DIR}/data_counts.txt" ]] || cp "${SEED_DIR}/data_counts.txt" "${DEST}/data_counts.txt"
  "${PYTHON_BIN}" - "${SEED_DIR}/run_summary.json" "${DEST}/provenance.json" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
provenance = summary.get('provenance')
if not isinstance(provenance, dict):
    raise SystemExit('run summary is missing provenance')
Path(sys.argv[2]).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
done
if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.jsonl' -o -name '*.log' \) | grep -q .; then
  echo "Forbidden private artifact found in report." >&2
  exit 1
fi
REPORT_REL=${REPORT_DIR#"${ROOT}/"}
git add -- "${REPORT_REL}"
git diff --cached --check
if [[ ${NO_COMMIT} -eq 1 ]]; then
  echo "REPORT_DIR=${REPORT_REL}"
  exit 0
fi
git commit -m "Add Q-CEASC Re-TACRED formal multi-seed report"
git push origin 1.1
echo "REPORT_DIR=${REPORT_REL}"
