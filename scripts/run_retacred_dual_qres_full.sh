#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONUNBUFFERED
SEED=13
LOG_EVERY_BATCHES=50
OUTPUT_DIR=
DRY_RUN=0
SKIP_PREFLIGHT=0
SKIP_CANARY=0
CANARY_ONLY=0
STALE_TIMEOUT_MINUTES=45
STAGE_TIMEOUT_HOURS=24
CURRENT_STAGE=initialization

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_dual_qres_full.sh [options]

Options:
  --seed N             Random seed (default: 13)
  --log-every-batches N  Progress interval (default: 50)
  --output-dir PATH    Explicit new output directory
  --skip-preflight     Skip environment/data/tests preflight
  --skip-canary        Skip the real-data numerical canary
  --canary-only        Run only the real-data numerical canary
  --stale-timeout-minutes N  Stop a stage without progress (default: 45)
  --stage-timeout-hours N    Maximum duration of one stage (default: 24)
  --dry-run            Print commands without running training
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED=$2; shift ;;
    --log-every-batches) LOG_EVERY_BATCHES=$2; shift ;;
    --output-dir) OUTPUT_DIR=$2; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --skip-canary) SKIP_CANARY=1 ;;
    --canary-only) CANARY_ONLY=1 ;;
    --stale-timeout-minutes) STALE_TIMEOUT_MINUTES=$2; shift ;;
    --stage-timeout-hours) STAGE_TIMEOUT_HOURS=$2; shift ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Log interval must be a positive integer." >&2
  exit 2
}
[[ "${STALE_TIMEOUT_MINUTES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Stale timeout must be a positive integer." >&2
  exit 2
}
[[ "${STAGE_TIMEOUT_HOURS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Stage timeout must be a positive integer." >&2
  exit 2
}

cd "${ROOT}"
STAMP=$(date +%Y%m%d_%H%M%S)
if [[ ${CANARY_ONLY} -eq 1 ]]; then
  RUN_DIR=${OUTPUT_DIR:-runs/retacred_dual_projector_canary_${STAMP}_seed${SEED}}
else
  RUN_DIR=${OUTPUT_DIR:-runs/retacred_dual_projector_full_${STAMP}_seed${SEED}}
fi

on_error() {
  status=$?
  if [[ ${DRY_RUN} -eq 0 && -d "${RUN_DIR}" ]]; then
    {
      echo "FAILED_AT=$(date -Iseconds)"
      echo "FAILED_STAGE=${CURRENT_STAGE}"
      echo "EXIT_CODE=${status}"
    } > "${RUN_DIR}/RUN_FAILED"
    printf '%s\n' "${RUN_DIR}" > runs/latest_dual_qres_failed_run.txt
  fi
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
  local log_file
  local heartbeat_file
  local status_file
  local status
  shift
  CURRENT_STAGE=${name}
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[$(date -Iseconds)] START ${name}"
    print_command "$@"
    echo "[$(date -Iseconds)] END ${name}"
  else
    log_file="${RUN_DIR}/logs/${name}.log"
    heartbeat_file="${RUN_DIR}/status/${name}.heartbeat"
    status_file="${RUN_DIR}/status/${name}.env"
    mkdir -p "${RUN_DIR}/status"
    printf 'STATUS=running\nSTARTED_AT=%s\nHEARTBEAT_FILE=%s\n' \
      "$(date -Iseconds)" "${heartbeat_file}" > "${status_file}"
    touch "${heartbeat_file}"
    echo "[$(date -Iseconds)] START ${name}" | tee "${log_file}"
    set +e
    Q_ATTENTION_HEARTBEAT_FILE="${heartbeat_file}" \
      "${PYTHON_BIN}" scripts/run_with_health_watchdog.py \
      --heartbeat-file "${heartbeat_file}" \
      --stale-seconds "$((STALE_TIMEOUT_MINUTES * 60))" \
      --timeout-seconds "$((STAGE_TIMEOUT_HOURS * 3600))" \
      -- "$@" 2>&1 | tee -a "${log_file}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ ${status} -ne 0 ]]; then
      printf 'STATUS=failed\nFAILED_AT=%s\nEXIT_CODE=%s\nHEARTBEAT_FILE=%s\n' \
        "$(date -Iseconds)" "${status}" "${heartbeat_file}" > "${status_file}"
      return "${status}"
    fi
    printf 'STATUS=complete\nCOMPLETED_AT=%s\nHEARTBEAT_FILE=%s\n' \
      "$(date -Iseconds)" "${heartbeat_file}" > "${status_file}"
    echo "[$(date -Iseconds)] END ${name}" | tee -a "${log_file}"
  fi
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
    echo "LOG_EVERY_BATCHES=${LOG_EVERY_BATCHES}"
    echo "SKIP_CANARY=${SKIP_CANARY}"
    echo "CANARY_ONLY=${CANARY_ONLY}"
    echo "STALE_TIMEOUT_MINUTES=${STALE_TIMEOUT_MINUTES}"
    echo "STAGE_TIMEOUT_HOURS=${STAGE_TIMEOUT_HOURS}"
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

if [[ ${SKIP_CANARY} -eq 0 ]]; then
  run_stage canary_prepare \
    "${PYTHON_BIN}" experiments/prepare_relation_data.py \
    --format project_jsonl \
    --dataset_name retacred_canary \
    --train_path data/relation/retacred/train.jsonl \
    --valid_path data/relation/retacred/valid.jsonl \
    --output_dir "${RUN_DIR}/canary/data" \
    --train_limit 256 --valid_limit 128 --seed "${SEED}"
  require_file "${RUN_DIR}/canary/data/data_config.json"

  run_stage canary_baseline \
    "${PYTHON_BIN}" experiments/train_relation_baseline.py \
    --train_path "${RUN_DIR}/canary/data/train.jsonl" \
    --valid_path "${RUN_DIR}/canary/data/valid.jsonl" \
    --output_dir "${RUN_DIR}/canary/baseline" \
    --epochs 1 --batch_size 64 --lr 0.0005 \
    --log_every_batches 1 \
    --dim 128 --num_layers 4 --num_heads 8 --ff_dim 256 \
    --dropout 0.1 --max_length 128 \
    --selection_metric valid_loss --seed "${SEED}" --device cuda
  require_file "${RUN_DIR}/canary/baseline/metrics.json"
  require_file "${RUN_DIR}/canary/baseline/model.pt"

  run_stage canary_core_quantum \
    "${PYTHON_BIN}" experiments/train_relation_attention_score_kernel.py \
    --model_dir "${RUN_DIR}/canary/baseline" \
    --train_path "${RUN_DIR}/canary/data/train.jsonl" \
    --valid_path "${RUN_DIR}/canary/data/valid.jsonl" \
    --output_dir "${RUN_DIR}/canary/core/quantum" \
    --kernel_type quantum \
    --num_qubits 4 --depth 2 --angle_scale 1.0 \
    --score_readout observable --input_encoding factorized_shared \
    --query_scope all --epochs 1 --batch_size 64 --lr 0.001 \
    --log_every_batches 1 --selection_metric valid_loss \
    --diagnostic_batches 2 --seed "${SEED}" --device cuda
  require_file "${RUN_DIR}/canary/core/quantum/metrics.json"
  require_file "${RUN_DIR}/canary/core/quantum/diagnostics.json"
  require_file "${RUN_DIR}/canary/core/quantum/attention_score_kernel.pt"

  run_stage canary_selector_quantum \
    "${PYTHON_BIN}" experiments/train_relation_counterfactual_evidence.py \
    --model_dir "${RUN_DIR}/canary/baseline" \
    --core_checkpoint "${RUN_DIR}/canary/core/quantum/attention_score_kernel.pt" \
    --train_path "${RUN_DIR}/canary/data/train.jsonl" \
    --valid_path "${RUN_DIR}/canary/data/valid.jsonl" \
    --output_dir "${RUN_DIR}/canary/selector/quantum" \
    --evidence_type quantum \
    --num_qubits 4 --depth 2 --angle_scale 1.0 \
    --evidence_gate_calibration context_budget \
    --evidence_view_score_mode positive --evidence_task_readout dual \
    --evidence_readout connected_relation_token \
    --evidence_correlation_mode phase_selective \
    --evidence_weight_mode signed_centered_l1 \
    --evidence_measurement_mode entanglement_phase_offset \
    --intervention_mode direct_bias --direct_bias_mode centered \
    --evidence_budget 0.35 --diagnostic_batches 2 \
    --random_repeats 1 --epochs 1 --batch_size 64 --lr 0.01 \
    --log_every_batches 1 --seed "${SEED}" --device cuda
  require_file "${RUN_DIR}/canary/selector/quantum/metrics.json"
  require_file "${RUN_DIR}/canary/selector/quantum/diagnostics.json"
else
  echo "Skipping real-data canary by request."
fi

if [[ ${CANARY_ONLY} -eq 1 ]]; then
  if [[ ${SKIP_CANARY} -eq 1 ]]; then
    echo "--canary-only cannot be combined with --skip-canary." >&2
    exit 2
  fi
  if [[ ${DRY_RUN} -eq 0 ]]; then
    CURRENT_STAGE=canary_finalize
    date -Iseconds > "${RUN_DIR}/CANARY_COMPLETE"
    printf '%s\n' "${RUN_DIR}" > runs/latest_dual_qres_canary.txt
    echo "Canary passed: ${RUN_DIR}"
  fi
  exit 0
fi

run_stage baseline \
  "${PYTHON_BIN}" experiments/train_relation_baseline.py \
  --train_path data/relation/retacred/train.jsonl \
  --valid_path data/relation/retacred/valid.jsonl \
  --output_dir "${RUN_DIR}/baseline" \
  --epochs 12 --batch_size 128 --lr 0.0005 \
  --log_every_batches "${LOG_EVERY_BATCHES}" \
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
    --log_every_batches "${LOG_EVERY_BATCHES}" \
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
    --log_every_batches "${LOG_EVERY_BATCHES}" \
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
