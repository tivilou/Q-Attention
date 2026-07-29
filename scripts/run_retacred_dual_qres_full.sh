#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=13
OUTPUT_DIR=
DRY_RUN=0
SKIP_PREFLIGHT=0
CURRENT_STAGE=initialization

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_dual_qres_full.sh [options]

Options:
  --seed N             Random seed (default: 13)
  --output-dir PATH    Explicit new output directory
  --skip-preflight     Skip environment/data/tests preflight
  --dry-run            Print commands without running training
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED=$2; shift ;;
    --output-dir) OUTPUT_DIR=$2; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }

cd "${ROOT}"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${OUTPUT_DIR:-runs/retacred_dual_projector_full_${STAMP}_seed${SEED}}

on_error() {
  status=$?
  echo "FAILED stage=${CURRENT_STAGE} exit=${status}" >&2
  exit "${status}"
}
trap on_error ERR

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_stage() {
  local name=$1
  shift
  CURRENT_STAGE=${name}
  echo "[$(date -Iseconds)] START ${name}"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    print_command "$@"
  else
    "$@" 2>&1 | tee "${RUN_DIR}/logs/${name}.log"
  fi
  echo "[$(date -Iseconds)] END ${name}"
}

require_file() {
  [[ ${DRY_RUN} -eq 1 ]] && return 0
  [[ -f "$1" ]] || { echo "Missing expected output: $1" >&2; return 1; }
}

if [[ ${DRY_RUN} -eq 0 ]]; then
  [[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse output directory: ${RUN_DIR}" >&2; exit 1; }
  mkdir -p "${RUN_DIR}/logs"
  if [[ ${SKIP_PREFLIGHT} -eq 0 ]]; then
    CURRENT_STAGE=preflight
    bash scripts/check_retacred_dual_qres_full.sh 2>&1 | tee "${RUN_DIR}/logs/preflight.log"
  fi
  {
    echo "SEED=${SEED}"
    echo "GIT_COMMIT=$(git rev-parse HEAD)"
    echo "STARTED_AT=$(date -Iseconds)"
    echo "PYTHON_BIN=$(command -v "${PYTHON_BIN}")"
    echo "GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';' -)"
    echo "RUN_DIR=${RUN_DIR}"
  } > "${RUN_DIR}/run_manifest.env"
else
  echo "DRY RUN: no files will be written"
fi

echo "RUN_DIR=${RUN_DIR}"

run_stage baseline \
  "${PYTHON_BIN}" experiments/train_relation_baseline.py \
  --train_path data/relation/retacred/train.jsonl \
  --valid_path data/relation/retacred/valid.jsonl \
  --output_dir "${RUN_DIR}/baseline" \
  --epochs 12 --batch_size 128 --lr 0.0005 \
  --dim 128 --num_layers 4 --num_heads 8 --ff_dim 256 \
  --dropout 0.1 --max_length 128 \
  --selection_metric valid_loss --seed "${SEED}" --device cuda
require_file "${RUN_DIR}/baseline/metrics.json"
require_file "${RUN_DIR}/baseline/model.pt"

for FAMILY in quantum classical; do
  run_stage "core_${FAMILY}" \
    "${PYTHON_BIN}" experiments/train_relation_attention_score_kernel.py \
    --model_dir "${RUN_DIR}/baseline" \
    --train_path data/relation/retacred/train.jsonl \
    --valid_path data/relation/retacred/valid.jsonl \
    --output_dir "${RUN_DIR}/core/${FAMILY}" \
    --kernel_type "${FAMILY}" \
    --num_qubits 4 --depth 2 --angle_scale 1.0 \
    --score_readout observable --input_encoding factorized_shared \
    --query_scope all --epochs 4 --batch_size 128 --lr 0.001 \
    --selection_metric valid_loss --diagnostic_batches 64 \
    --seed "${SEED}" --device cuda
  require_file "${RUN_DIR}/core/${FAMILY}/metrics.json"
  require_file "${RUN_DIR}/core/${FAMILY}/diagnostics.json"
  require_file "${RUN_DIR}/core/${FAMILY}/attention_score_kernel.pt"
done

for METHOD in quantum classical classical_strong; do
  if [[ "${METHOD}" == quantum ]]; then CORE=quantum; else CORE=classical; fi
  run_stage "selector_${METHOD}" \
    "${PYTHON_BIN}" experiments/train_relation_counterfactual_evidence.py \
    --model_dir "${RUN_DIR}/baseline" \
    --core_checkpoint "${RUN_DIR}/core/${CORE}/attention_score_kernel.pt" \
    --train_path data/relation/retacred/train.jsonl \
    --valid_path data/relation/retacred/valid.jsonl \
    --output_dir "${RUN_DIR}/selector/${METHOD}" \
    --evidence_type "${METHOD}" \
    --num_qubits 4 --depth 2 --angle_scale 1.0 \
    --evidence_gate_calibration context_budget \
    --evidence_view_score_mode positive \
    --evidence_task_readout dual \
    --evidence_readout connected_relation_token \
    --evidence_correlation_mode phase_selective \
    --evidence_weight_mode signed_centered_l1 \
    --evidence_measurement_mode entanglement_phase_offset \
    --intervention_mode direct_bias --direct_bias_mode centered \
    --evidence_budget 0.35 --diagnostic_batches 64 \
    --epochs 10 --batch_size 64 --lr 0.01 \
    --seed "${SEED}" --device cuda
  require_file "${RUN_DIR}/selector/${METHOD}/metrics.json"
  require_file "${RUN_DIR}/selector/${METHOD}/diagnostics.json"
done

if [[ ${DRY_RUN} -eq 0 ]]; then
  CURRENT_STAGE=finalize
  date -Iseconds > "${RUN_DIR}/RUN_COMPLETE"
  printf '%s\n' "${RUN_DIR}" > runs/latest_dual_qres_full_run.txt
  echo "Completed: ${RUN_DIR}"
  echo "Export with: bash scripts/export_retacred_dual_qres_report.sh \"${RUN_DIR}\""
fi
