#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Neither python nor python3 is available; set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi
SEED=13
GPU_SPEC=0
OUTPUT_DIR=
MODEL_DIR=
SKIP_PREFLIGHT=0
DRY_RUN=0
WRITE_PREDICTIONS=0
SKIP_EXPORT=0
NO_PUSH=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_baseline_complete.sh [options]

Options:
  --seed N              Baseline seed (default: 13)
  --gpu N               Physical GPU index (default: 0)
  --output-dir PATH     New run directory under runs/
  --model-dir PATH      Reuse an existing completed baseline checkpoint; no training
  --skip-preflight      Skip environment/data/test checks
  --write-predictions  Keep private split predictions in the run directory
  --skip-export         Stop after the raw run; do not create or publish a report
  --no-push             Export and commit the report, but do not push to origin/1.1
  --dry-run             Print the planned commands without running them
  -h|--help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED=$2; shift ;;
    --gpu) GPU_SPEC=$2; shift ;;
    --output-dir) OUTPUT_DIR=$2; shift ;;
    --model-dir) MODEL_DIR=$2; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --write-predictions) WRITE_PREDICTIONS=1 ;;
    --skip-export) SKIP_EXPORT=1 ;;
    --no-push) NO_PUSH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }
[[ "${GPU_SPEC}" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer." >&2; exit 2; }

cd "${ROOT}"
if [[ ${DRY_RUN} -eq 0 && ${SKIP_PREFLIGHT} -eq 0 ]]; then
  bash scripts/check_retacred_baseline_complete.sh
fi
if [[ ${DRY_RUN} -eq 0 ]]; then
  nvidia-smi -i "${GPU_SPEC}" --query-gpu=name --format=csv,noheader >/dev/null || {
    echo "GPU ${GPU_SPEC} is not available." >&2
    exit 1
  }
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${OUTPUT_DIR:-runs/retacred_baseline_complete/${STAMP}_seed${SEED}}
RUN_DIR=$(readlink -m "${RUN_DIR}")
case "${RUN_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) echo "Output directory must be inside ${ROOT}/runs" >&2; exit 2 ;;
esac
if [[ ${DRY_RUN} -eq 0 && -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${RUN_DIR}" >&2
  exit 1
fi
RUN_BASENAME=$(basename "${RUN_DIR}")
REPORT_DIR=${ROOT}/reports/retacred_baseline_complete/${RUN_BASENAME}
REPORT_REL=${REPORT_DIR#${ROOT}/}

BASELINE_DIR=${MODEL_DIR:-${RUN_DIR}/baseline}
EVALUATION_DIR=${RUN_DIR}/evaluation
TRAIN_COMMAND=(
  "${PYTHON_BIN}" experiments/train_relation_baseline.py
  --train_path data/relation/retacred/train.jsonl
  --valid_path data/relation/retacred/valid.jsonl
  --output_dir "${BASELINE_DIR}"
  --epochs 12 --batch_size 128 --lr 0.0005
  --dim 128 --num_layers 4 --num_heads 8 --ff_dim 256
  --dropout 0.1 --max_length 128 --seed "${SEED}"
  --selection_metric macro_f1_then_loss --device cuda
)
EVAL_COMMAND=(
  "${PYTHON_BIN}" experiments/evaluate_relation_baseline.py
  --model-dir "${BASELINE_DIR}"
  --train-path data/relation/retacred/train.jsonl
  --valid-path data/relation/retacred/valid.jsonl
  --test-path data/relation/retacred/test.jsonl
  --output-dir "${EVALUATION_DIR}"
  --batch-size 256 --device cuda
)
if [[ ${WRITE_PREDICTIONS} -eq 1 ]]; then
  EVAL_COMMAND+=(--write-predictions)
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  printf '[dry-run] CUDA_VISIBLE_DEVICES=%q ' "${GPU_SPEC}"
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n[dry-run] CUDA_VISIBLE_DEVICES=%q ' "${GPU_SPEC}"
  printf '%q ' "${EVAL_COMMAND[@]}"
  printf '\n'
  if [[ ${SKIP_EXPORT} -eq 0 ]]; then
    printf '[dry-run] bash scripts/export_retacred_baseline_report.sh %q %q\n' "${RUN_DIR}" "${REPORT_DIR}"
    printf '[dry-run] git add -- %q\n' "${REPORT_REL}"
    printf '[dry-run] git diff --cached --check\n'
    printf '[dry-run] git commit -m %q\n' "report: add complete Re-TACRED baseline evaluation (${RUN_BASENAME})"
    if [[ ${NO_PUSH} -eq 0 ]]; then
      printf '[dry-run] git push origin 1.1\n'
    else
      printf '[dry-run] push skipped (--no-push)\n'
    fi
  else
    printf '[dry-run] export and publish skipped (--skip-export)\n'
  fi
  exit 0
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/status"
printf 'STATUS=running\nSEED=%s\nGPU_ID=%s\nSTARTED_AT=%s\n' "${SEED}" "${GPU_SPEC}" "$(date -Iseconds)" > "${RUN_DIR}/status/run.env"
if [[ -z "${MODEL_DIR}" ]]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_SPEC}" \
    "${TRAIN_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/logs/baseline_train.log"
else
  printf 'Reusing baseline checkpoint: %s\n' "${MODEL_DIR}" | tee "${RUN_DIR}/logs/baseline_train.log"
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_SPEC}" \
  "${EVAL_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/logs/baseline_evaluation.log"

cp "${EVALUATION_DIR}/metrics.json" "${RUN_DIR}/metrics.json"
cp "${EVALUATION_DIR}/run_summary.json" "${RUN_DIR}/run_summary.json"
cp "${EVALUATION_DIR}/run_summary.data" "${RUN_DIR}/run_summary.data"
cp "${EVALUATION_DIR}/run_summary.md" "${RUN_DIR}/run_summary.md"
cp "${EVALUATION_DIR}/run_config.json" "${RUN_DIR}/run_config.json"
printf '%s\n' "$(date -Iseconds)" > "${RUN_DIR}/RUN_COMPLETE"
printf 'STATUS=complete\nSEED=%s\nGPU_ID=%s\nCOMPLETED_AT=%s\n' "${SEED}" "${GPU_SPEC}" "$(date -Iseconds)" > "${RUN_DIR}/status/run.env"
echo "RUN_DIR=${RUN_DIR}"

if [[ ${SKIP_EXPORT} -eq 1 ]]; then
  printf 'STATUS=skipped\nREASON=skip-export\nAT=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/status/export.env"
  echo "Report export skipped (--skip-export)."
  exit 0
fi

if ! bash scripts/export_retacred_baseline_report.sh "${RUN_DIR}" "${REPORT_DIR}" 2>&1 | tee "${RUN_DIR}/logs/report_export.log"; then
  printf 'STATUS=failed\nAT=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/status/export.env"
  echo "Report export failed; no commit or push was attempted." >&2
  exit 1
fi
printf 'STATUS=complete\nREPORT_DIR=%s\nAT=%s\n' "${REPORT_DIR}" "$(date -Iseconds)" > "${RUN_DIR}/status/export.env"
printf '%s\n' "$(date -Iseconds)" > "${RUN_DIR}/EXPORT_COMPLETE"

if ! git add -- "${REPORT_REL}"; then
  printf 'STATUS=failed\nREASON=git-add\nAT=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
  echo "Failed to stage the exported report; no commit or push was attempted." >&2
  exit 1
fi
if ! git diff --cached --check; then
  printf 'STATUS=failed\nREASON=staged-diff-check\nAT=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
  echo "Staged report failed git diff --check; no commit or push was attempted." >&2
  exit 1
fi
STAGED_FILES=$(git diff --cached --name-only)
while IFS= read -r STAGED_FILE; do
  [[ -z "${STAGED_FILE}" ]] && continue
  [[ "${STAGED_FILE}" == "${REPORT_REL}" || "${STAGED_FILE}" == "${REPORT_REL}"/* ]] || {
    printf 'STATUS=failed\nREASON=unexpected-staged-file\nAT=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
    echo "Unexpected staged file outside report directory: ${STAGED_FILE}" >&2
    exit 1
  }
done <<< "${STAGED_FILES}"

COMMIT_MESSAGE="report: add complete Re-TACRED baseline evaluation (${RUN_BASENAME})"
if ! git commit -m "${COMMIT_MESSAGE}" 2>&1 | tee "${RUN_DIR}/logs/report_commit.log"; then
  printf 'STATUS=failed\nREASON=commit\nAT=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
  echo "Report export is staged but commit failed; rerun git commit after fixing Git configuration." >&2
  exit 1
fi
COMMIT_SHA=$(git rev-parse HEAD)
printf 'STATUS=commit-complete\nCOMMIT=%s\nAT=%s\n' "${COMMIT_SHA}" "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
printf '%s\n' "${COMMIT_SHA}" > "${RUN_DIR}/COMMIT_COMPLETE"

if [[ ${NO_PUSH} -eq 1 ]]; then
  printf 'STATUS=push-skipped\nCOMMIT=%s\nAT=%s\n' "${COMMIT_SHA}" "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
  echo "Report committed locally (${COMMIT_SHA}); push skipped (--no-push)."
  echo "Retry with: git push origin 1.1"
  exit 0
fi

if ! git push origin 1.1 2>&1 | tee "${RUN_DIR}/logs/report_push.log"; then
  printf 'STATUS=push-failed\nCOMMIT=%s\nAT=%s\n' "${COMMIT_SHA}" "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
  echo "Report commit ${COMMIT_SHA} is retained locally; push failed and training will not be repeated." >&2
  echo "Retry with: git push origin 1.1" >&2
  exit 1
fi
printf 'STATUS=push-complete\nCOMMIT=%s\nAT=%s\n' "${COMMIT_SHA}" "$(date -Iseconds)" > "${RUN_DIR}/status/publish.env"
printf '%s\n' "$(date -Iseconds)" > "${RUN_DIR}/PUSH_COMPLETE"
echo "Report exported, committed, and pushed: ${REPORT_DIR}"
