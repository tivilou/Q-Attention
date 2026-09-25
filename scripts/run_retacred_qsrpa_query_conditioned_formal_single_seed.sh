#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
resolve_python_bin(){
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
      command -v "${candidate}"; return
    fi
  done
  echo "No Python interpreter found; activate an environment or set PYTHON_BIN" >&2
  return 1
}
PYTHON_BIN=$(resolve_python_bin); export PYTHON_BIN
GPU_SPEC=0; OUTPUT_DIR=; REPORT_DIR=; LOG_EVERY_BATCHES=50; SKIP_PREFLIGHT=0; DRY_RUN=0
usage(){ echo "Usage: bash scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh [--gpu N] [--output-dir PATH] [--report-dir PATH] [--skip-preflight] [--dry-run]"; }
while [[ $# -gt 0 ]]; do case "$1" in --gpu) GPU_SPEC=$2; shift;; --output-dir) OUTPUT_DIR=$2; shift;; --report-dir) REPORT_DIR=$2; shift;; --log-every-batches) LOG_EVERY_BATCHES=$2; shift;; --skip-preflight) SKIP_PREFLIGHT=1;; --dry-run) DRY_RUN=1;; -h|--help) usage; exit 0;; *) usage >&2; exit 2;; esac; shift; done
cd "${ROOT}"
RUN_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${OUTPUT_DIR:-runs/retacred_qsrpa_query_conditioned_formal_single_seed/${RUN_TIMESTAMP}_seed13}
DEFAULT_REPORT_DIR="reports/retacred_qsrpa_query_conditioned_formal_single_seed/${RUN_TIMESTAMP}_seed13"
RUN_DIR=$(readlink -m "${RUN_DIR}")
case "${RUN_DIR}" in "${ROOT}"/runs/*) ;; *) echo "Output must be under runs/" >&2; exit 2;; esac
[[ ${DRY_RUN} -eq 1 || ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse output" >&2; exit 1; }
[[ ${SKIP_PREFLIGHT} -eq 1 || ${DRY_RUN} -eq 1 ]] || bash scripts/check_retacred_qsrpa_query_conditioned_formal_single_seed.sh
record_provenance(){
  local git_commit git_branch python_version
  git_commit=$(git rev-parse HEAD)
  git_branch=$(git branch --show-current)
  python_version=$("${PYTHON_BIN}" --version 2>&1)
  mkdir -p "${RUN_DIR}"
  {
    printf 'PROVENANCE_STATUS=recorded_by_runner\n'
    printf 'RUN_TIMESTAMP=%s\n' "${RUN_TIMESTAMP}"
    printf 'RUN_DIR_BASENAME=%s\n' "$(basename "${RUN_DIR}")"
    printf 'GIT_COMMIT=%s\n' "${git_commit}"
    printf 'GIT_BRANCH=%s\n' "${git_branch}"
    printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
    printf 'PYTHON_VERSION=%s\n' "${python_version}"
  } > "${RUN_DIR}/provenance.env"
}
[[ ${DRY_RUN} -eq 1 ]] || record_provenance
run_stage(){ local name=$1; shift; if [[ ${DRY_RUN} -eq 1 ]]; then printf '[dry-run] %s ' "$name"; printf '%q ' "$@"; echo; return; fi; mkdir -p "${RUN_DIR}/logs"; CUDA_VISIBLE_DEVICES="${GPU_SPEC}" "$@" 2>&1 | tee "${RUN_DIR}/logs/${name}.log"; }
BASELINE_DIR="${RUN_DIR}/baseline"
run_stage baseline "${PYTHON_BIN}" experiments/train_relation_baseline.py --train_path data/relation/retacred/train.jsonl --valid_path data/relation/retacred/valid.jsonl --output_dir "${BASELINE_DIR}" --epochs 12 --batch_size 128 --lr 0.0005 --dim 128 --num_layers 4 --num_heads 8 --ff_dim 256 --dropout 0.1 --max_length 128 --seed 13 --selection_metric macro_f1_then_loss --device cuda
for SPEC in quantum_global_context:quantum:global_context classical_global_context:classical:global_context quantum_soft_role_pair:quantum:soft_role_pair classical_soft_role_pair:classical:soft_role_pair quantum_query_conditioned_soft_role_pair:quantum:query_conditioned_soft_role_pair classical_query_conditioned_soft_role_pair:classical:query_conditioned_soft_role_pair; do IFS=: read -r NAME TYPE ANCHOR <<<"${SPEC}"; run_stage "${NAME}" "${PYTHON_BIN}" experiments/train_relation_attention_score_kernel.py --model_dir "${BASELINE_DIR}" --train_path data/relation/retacred/train.jsonl --valid_path data/relation/retacred/valid.jsonl --output_dir "${RUN_DIR}/${NAME}" --kernel_type "${TYPE}" --num_qubits 4 --depth 2 --angle_scale 1.0 --max_gain 0.5 --initial_gain 0.02 --score_readout fidelity --input_encoding joint --query_scope all --relation_anchor_mode "${ANCHOR}" --role_router_temperature 1.0 --role_entropy_floor 0.35 --role_regularization_weight 0.001 --epochs 12 --batch_size 256 --lr 0.001 --diagnostic_batches 0 --log_every_batches "${LOG_EVERY_BATCHES}" --seed 13 --selection_metric macro_f1_then_loss --device cuda; done
for NAME in quantum_global_context classical_global_context quantum_soft_role_pair classical_soft_role_pair quantum_query_conditioned_soft_role_pair classical_query_conditioned_soft_role_pair; do for SPLIT in valid test; do run_stage "${NAME}_${SPLIT}" "${PYTHON_BIN}" experiments/eval_relation_attention_score_kernel.py --model_dir "${BASELINE_DIR}" --checkpoint "${RUN_DIR}/${NAME}/attention_score_kernel.pt" --data_path "data/relation/retacred/${SPLIT}.jsonl" --output_dir "${RUN_DIR}/${NAME}/${SPLIT}" --batch_size 256 --random_repeats 4 --random_seed 101 --device cuda; done; done
if [[ ${DRY_RUN} -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/summarize_retacred_qsrpa_query_conditioned_formal_single_seed.py --run-dir "${RUN_DIR}"
  date -Iseconds > "${RUN_DIR}/RUN_COMPLETE"
  REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}
  PYTHON_BIN="${PYTHON_BIN}" bash scripts/export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh --run-dir "${RUN_DIR}" --report-dir "${REPORT_DIR}"
  echo "RUN_DIR=${RUN_DIR}"
fi
