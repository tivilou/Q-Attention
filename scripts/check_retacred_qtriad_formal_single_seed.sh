#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

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

PYTHON_BIN=$(resolve_python_bin)
export PYTHON_BIN
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Run this check on branch 1.1." >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Working tree must be clean." >&2; git status --short; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "origin/1.1 must be an ancestor of HEAD; merge it before running." >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "origin/main must be an ancestor of HEAD; merge it before running." >&2; exit 1; }

for file in \
    configs/retacred_qtriad_formal_single_seed.json \
    experiments/run_qtriad_relation_transfer.py \
    experiments/run_qtriad_selector_worker.py \
    experiments/train_relation_baseline.py \
    src/q_attention/models/relation_transformer.py \
    src/q_attention/experiments/relation_steering.py \
    src/q_attention/plugins/q_triad.py \
    experiments/run_q_causal_value_evidence_relation_smoke.py \
    experiments/run_q_causal_value_evidence_relation_transfer.py \
    scripts/run_retacred_qtriad_formal_single_seed.sh \
    scripts/export_retacred_qtriad_formal_single_seed_report.sh \
    docs/current/retacred_qtriad_formal_single_seed_zh.md; do
  [[ -f "${file}" ]] || { echo "Missing ${file}" >&2; exit 1; }
done

for pair in \
  "data/relation/retacred/train.jsonl 58465" \
  "data/relation/retacred/valid.jsonl 19584" \
  "data/relation/retacred/test.jsonl 13418"; do
  set -- ${pair}
  [[ -f "$1" ]] || { echo "Missing $1" >&2; exit 1; }
  [[ "$(wc -l < "$1")" -eq "$2" ]] || { echo "Unexpected line count for $1" >&2; exit 1; }
done

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("configs/retacred_qtriad_formal_single_seed.json").read_text(encoding="utf-8"))
assert p["formal_experiment"] is True
assert p["seed"] == 13
assert p["candidate"] == "q_triad"
assert p["matched_control"] == "classical_density_tensor"
assert p["gates"]["test_used_for_training_or_selection"] is False
print("Q-TRIAD formal config=OK")
PY

PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m pytest -q \
  tests/test_q_triad_attention_score_kernel.py \
  tests/test_qtriad_memory_multigpu.py \
  tests/test_model_parallel.py \
  tests/test_retacred_qtriad_formal_single_seed.py
"${PYTHON_BIN}" -m py_compile experiments/run_qtriad_relation_transfer.py experiments/run_qtriad_selector_worker.py src/q_attention/plugins/q_triad.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Q-TRIAD formal preflight=OK"
git rev-parse HEAD
