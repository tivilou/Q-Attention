#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=
REPORT_DIR=
NO_COMMIT=0

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
    command -v "${candidate}" >/dev/null 2>&1 && { command -v "${candidate}"; return; }
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN." >&2
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir|--report-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1." >&2; exit 2; }
      if [[ "$1" == "--run-dir" ]]; then RUN_DIR=$2; else REPORT_DIR=$2; fi
      shift 2;;
    --no-commit) NO_COMMIT=1; shift;;
    -h|--help) echo "Usage: $0 --run-dir PATH [--report-dir PATH] [--no-commit]"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

PYTHON_BIN=$(resolve_python_bin)
export PYTHON_BIN
cd "${ROOT}"
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Exporter must run on branch 1.1." >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Working tree must be clean before export." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "origin/1.1 must be an ancestor of HEAD." >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "origin/main must be an ancestor of HEAD." >&2; exit 1; }
[[ -n "${RUN_DIR}" ]] || { echo "--run-dir is required." >&2; exit 2; }
[[ "${RUN_DIR}" = /* ]] || RUN_DIR="${ROOT}/${RUN_DIR}"
RUN_DIR=$(readlink -f "${RUN_DIR}")
RUN_NAME=$(basename "${RUN_DIR}")
[[ "${RUN_NAME}" == *_seed13 ]] || { echo "Run directory must end with _seed13." >&2; exit 1; }
for file in RUN_COMPLETE run_summary.json run_summary.md gpu_assignments.json baseline/metrics.json; do
  [[ -f "${RUN_DIR}/${file}" ]] || { echo "Missing ${file} in run directory." >&2; exit 1; }
done
for selector in q_triad classical_density_tensor quantum_product; do
  for split in valid test; do
    [[ -f "${RUN_DIR}/selectors/${selector}/metrics.json" ]] || { echo "Missing ${selector} metrics." >&2; exit 1; }
  done
done
"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
p = json.loads((Path(sys.argv[1]) / "run_summary.json").read_text(encoding="utf-8"))
assert p.get("seed") == 13
assert p.get("test_used_for_training_or_selection") is False
assert p.get("candidate") == "q_triad"
assert p.get("matched_control") == "classical_density_tensor"
assert p.get("formal_experiment") is True
assert isinstance(p.get("hardware_profile"), dict), "run summary must include hardware_profile"
provenance = p.get("provenance")
assert isinstance(provenance, dict), "run summary must include provenance"
required = {"git_revision", "git_branch", "started_at_utc", "torch", "cuda_available"}
missing = sorted(required.difference(provenance))
assert not missing, f"run provenance is missing keys: {missing}"
PY

DEFAULT_REPORT_DIR="reports/retacred_qtriad_formal_single_seed/${RUN_NAME}"
REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}
[[ "${REPORT_DIR}" = /* ]] || REPORT_DIR="${ROOT}/${REPORT_DIR}"
REPORT_DIR=$(readlink -m "${REPORT_DIR}")
case "${REPORT_DIR}" in "${ROOT}/reports/retacred_qtriad_formal_single_seed/"*) ;; *) echo "Report must be under the Q-TRIAD report root." >&2; exit 1;; esac
[[ ! -e "${REPORT_DIR}" ]] || { echo "Refusing to overwrite report directory." >&2; exit 1; }
mkdir -p "${REPORT_DIR}/metrics"
cp "${RUN_DIR}/RUN_COMPLETE" "${RUN_DIR}/run_summary.json" "${RUN_DIR}/run_summary.md" "${REPORT_DIR}/"
cp "${RUN_DIR}/gpu_assignments.json" "${REPORT_DIR}/gpu_assignments.json"
cp configs/retacred_qtriad_formal_single_seed.json "${REPORT_DIR}/run_config.json"
cp "${RUN_DIR}/baseline/metrics.json" "${REPORT_DIR}/metrics/baseline.json"
for selector in q_triad classical_density_tensor quantum_product; do
  cp "${RUN_DIR}/selectors/${selector}/metrics.json" "${REPORT_DIR}/metrics/${selector}.json"
done
printf '%s\n' "$(git rev-parse HEAD)" > "${REPORT_DIR}/reporting_commit.txt"
"${PYTHON_BIN}" - "${RUN_DIR}/run_summary.json" "${REPORT_DIR}/provenance.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
provenance = summary.get("provenance")
if not isinstance(provenance, dict):
    raise SystemExit("run summary is missing provenance")
Path(sys.argv[2]).write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
wc -l data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data_counts.txt"
for pair in \
  "data/relation/retacred/train.jsonl 58465" \
  "data/relation/retacred/valid.jsonl 19584" \
  "data/relation/retacred/test.jsonl 13418"; do
  set -- ${pair}
  [[ "$(wc -l < "$1")" -eq "$2" ]] || { echo "Unexpected line count for $1." >&2; exit 1; }
done
sha256sum data/relation/retacred/train.jsonl data/relation/retacred/valid.jsonl data/relation/retacred/test.jsonl > "${REPORT_DIR}/data.sha256"
if find "${REPORT_DIR}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.jsonl' \) | grep -q .; then
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
git commit -m "Add Q-TRIAD Re-TACRED formal single-seed report"
git push origin 1.1
echo "REPORT_DIR=${REPORT_REL}"
