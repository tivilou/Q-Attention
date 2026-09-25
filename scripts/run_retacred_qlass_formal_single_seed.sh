#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=13
GPU_SPEC=0
OUTPUT_DIR=
LOG_EVERY_BATCHES=50
STALE_TIMEOUT_MINUTES=45
STAGE_TIMEOUT_HOURS=48
SKIP_PREFLIGHT=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_qlass_formal_single_seed.sh [options]

Options:
  --seed N                    Fixed seed (default: 13)
  --gpu N                     One physical GPU index (default: 0)
  --output-dir PATH           New output directory under runs/
  --log-every-batches N       Progress interval (default: 50)
  --stale-timeout-minutes N  Stop a stage without a heartbeat (default: 45)
  --stage-timeout-hours N     Maximum duration of one stage (default: 48)
  --skip-preflight             Skip environment, data, and tests checks
  --dry-run                    Print the serial commands without running them
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED=$2; shift ;;
    --gpu) GPU_SPEC=$2; shift ;;
    --output-dir) OUTPUT_DIR=$2; shift ;;
    --log-every-batches) LOG_EVERY_BATCHES=$2; shift ;;
    --stale-timeout-minutes) STALE_TIMEOUT_MINUTES=$2; shift ;;
    --stage-timeout-hours) STAGE_TIMEOUT_HOURS=$2; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }
[[ "${GPU_SPEC}" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer." >&2; exit 2; }
[[ "${LOG_EVERY_BATCHES}" =~ ^[1-9][0-9]*$ ]] || { echo "Log interval must be positive." >&2; exit 2; }
[[ "${STALE_TIMEOUT_MINUTES}" =~ ^[1-9][0-9]*$ ]] || { echo "Stale timeout must be positive." >&2; exit 2; }
[[ "${STAGE_TIMEOUT_HOURS}" =~ ^[1-9][0-9]*$ ]] || { echo "Stage timeout must be positive." >&2; exit 2; }
[[ "${SEED}" == 13 ]] || { echo "This declared formal run is fixed to seed 13." >&2; exit 2; }

cd "${ROOT}"
nvidia-smi -i "${GPU_SPEC}" --query-gpu=name --format=csv,noheader >/dev/null
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${OUTPUT_DIR:-runs/retacred_qlass_formal_single_seed/${STAMP}_seed${SEED}}
RUN_DIR=$(readlink -m "${RUN_DIR}")
case "${RUN_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "Output directory must be inside ${ROOT}/runs" >&2; exit 2 ;;
esac
if [[ ${DRY_RUN} -eq 0 && -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${RUN_DIR}" >&2
  exit 1
fi

on_error() {
  local status=$?
  if [[ ${DRY_RUN} -eq 0 ]]; then
    mkdir -p "${RUN_DIR}"
    printf 'FAILED_AT=%s\nEXIT_CODE=%s\nSEED=%s\nGPU_ID=%s\n' "$(date -Iseconds)" "${status}" "${SEED}" "${GPU_SPEC}" > "${RUN_DIR}/RUN_FAILED"
  fi
  exit "${status}"
}
trap on_error ERR

if [[ ${DRY_RUN} -eq 0 && ${SKIP_PREFLIGHT} -eq 0 ]]; then
  bash scripts/check_retacred_qlass_formal_single_seed.sh
fi

run_stage() {
  local name=$1
  shift
  local log_file="${RUN_DIR}/logs/${name}.log"
  local heartbeat_file="${RUN_DIR}/status/${name}.heartbeat"
  local status_file="${RUN_DIR}/status/${name}.env"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '[dry-run] stage=%s gpu=%s ' "${name}" "${GPU_SPEC}"
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/status"
  touch "${heartbeat_file}"
  printf 'STATUS=running\nSTAGE=%s\nGPU_ID=%s\nSTARTED_AT=%s\n' "${name}" "${GPU_SPEC}" "$(date -Iseconds)" > "${status_file}"
  set +e
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_SPEC}" \
    Q_ATTENTION_HEARTBEAT_FILE="${heartbeat_file}" \
    "${PYTHON_BIN}" scripts/run_with_health_watchdog.py \
    --heartbeat-file "${heartbeat_file}" \
    --stale-seconds "$((STALE_TIMEOUT_MINUTES * 60))" \
    --timeout-seconds "$((STAGE_TIMEOUT_HOURS * 3600))" \
    -- "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    printf 'STATUS=failed\nSTAGE=%s\nGPU_ID=%s\nFAILED_AT=%s\nEXIT_CODE=%s\n' "${name}" "${GPU_SPEC}" "$(date -Iseconds)" "${status}" > "${status_file}"
    return "${status}"
  fi
  printf 'STATUS=complete\nSTAGE=%s\nGPU_ID=%s\nCOMPLETED_AT=%s\n' "${name}" "${GPU_SPEC}" "$(date -Iseconds)" > "${status_file}"
}

BASELINE_DIR="${RUN_DIR}/baseline"
QUANTUM_DIR="${RUN_DIR}/quantum_global_context"
CLASSICAL_DIR="${RUN_DIR}/classical_global_context"

run_stage baseline \
  "${PYTHON_BIN}" experiments/train_relation_baseline.py \
  --train_path data/relation/retacred/train.jsonl --valid_path data/relation/retacred/valid.jsonl \
  --output_dir "${BASELINE_DIR}" --epochs 12 --batch_size 128 --log_every_batches "${LOG_EVERY_BATCHES}" \
  --lr 0.0005 --dim 128 --num_layers 4 --num_heads 8 --ff_dim 256 --dropout 0.1 --max_length 128 \
  --seed "${SEED}" --selection_metric macro_f1_then_loss --device cuda

run_stage quantum_global_context \
  "${PYTHON_BIN}" experiments/train_relation_attention_score_kernel.py \
  --model_dir "${BASELINE_DIR}" --train_path data/relation/retacred/train.jsonl --valid_path data/relation/retacred/valid.jsonl \
  --output_dir "${QUANTUM_DIR}" --kernel_type quantum --num_qubits 4 --depth 2 --angle_scale 1.0 \
  --max_gain 0.5 --initial_gain 0.02 --score_readout fidelity --input_encoding joint --query_scope all \
  --relation_anchor_mode global_context --epochs 12 --batch_size 256 --lr 0.001 --diagnostic_batches 0 \
  --log_every_batches "${LOG_EVERY_BATCHES}" --seed "${SEED}" --selection_metric macro_f1_then_loss --device cuda

run_stage classical_global_context \
  "${PYTHON_BIN}" experiments/train_relation_attention_score_kernel.py \
  --model_dir "${BASELINE_DIR}" --train_path data/relation/retacred/train.jsonl --valid_path data/relation/retacred/valid.jsonl \
  --output_dir "${CLASSICAL_DIR}" --kernel_type classical --num_qubits 4 --depth 2 --angle_scale 1.0 \
  --max_gain 0.5 --initial_gain 0.02 --score_readout fidelity --input_encoding joint --query_scope all \
  --relation_anchor_mode global_context --epochs 12 --batch_size 256 --lr 0.001 --diagnostic_batches 0 \
  --log_every_batches "${LOG_EVERY_BATCHES}" --seed "${SEED}" --selection_metric macro_f1_then_loss --device cuda

for METHOD in quantum classical; do
  if [[ "${METHOD}" == quantum ]]; then KERNEL_DIR="${QUANTUM_DIR}"; else KERNEL_DIR="${CLASSICAL_DIR}"; fi
  run_stage "${METHOD}_valid_eval" "${PYTHON_BIN}" experiments/eval_relation_attention_score_kernel.py \
    --model_dir "${BASELINE_DIR}" --checkpoint "${KERNEL_DIR}/attention_score_kernel.pt" \
    --data_path data/relation/retacred/valid.jsonl --output_dir "${KERNEL_DIR}/valid" --batch_size 256 \
    --random_repeats 4 --random_seed 101 --device cuda
  run_stage "${METHOD}_test_eval" "${PYTHON_BIN}" experiments/eval_relation_attention_score_kernel.py \
    --model_dir "${BASELINE_DIR}" --checkpoint "${KERNEL_DIR}/attention_score_kernel.pt" \
    --data_path data/relation/retacred/test.jsonl --output_dir "${KERNEL_DIR}/test" --batch_size 256 \
    --random_repeats 4 --random_seed 101 --device cuda
done

if [[ ${DRY_RUN} -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/summarize_retacred_qlass_formal_single_seed.py --run-dir "${RUN_DIR}"
  date -Iseconds > "${RUN_DIR}/RUN_COMPLETE"
  printf 'STATUS=complete\nSEED=%s\nGPU_ID=%s\nCOMPLETED_AT=%s\n' "${SEED}" "${GPU_SPEC}" "$(date -Iseconds)" > "${RUN_DIR}/status/run.env"
  echo "RUN_DIR=${RUN_DIR}"
fi
