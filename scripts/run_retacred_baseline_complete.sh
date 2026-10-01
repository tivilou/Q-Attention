#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3
  else echo "Neither python nor python3 is available; set PYTHON_BIN explicitly." >&2; exit 1
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
RUN_DIR=
CURRENT_STAGE=preflight

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_baseline_complete.sh [options]

Options:
  --seed N              Baseline seed (default: 13)
  --gpu N               Physical GPU index (default: 0)
  --output-dir PATH     New run directory under runs/
  --model-dir PATH      Reuse an existing matching baseline checkpoint; no training
  --skip-preflight      Skip environment/data/test checks
  --write-predictions   Keep private split predictions in the run directory
  --skip-export         Stop after the raw run; do not create or publish a report
  --no-push             Export and commit the report, but do not push
  --dry-run             Print the planned commands without running them
  -h|--help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) [[ $# -ge 2 ]] || { echo "--seed requires a value." >&2; exit 2; }; SEED=$2; shift ;;
    --gpu) [[ $# -ge 2 ]] || { echo "--gpu requires a value." >&2; exit 2; }; GPU_SPEC=$2; shift ;;
    --output-dir) [[ $# -ge 2 ]] || { echo "--output-dir requires a value." >&2; exit 2; }; OUTPUT_DIR=$2; shift ;;
    --model-dir) [[ $# -ge 2 ]] || { echo "--model-dir requires a value." >&2; exit 2; }; MODEL_DIR=$2; shift ;;
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

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be a non-negative integer." >&2; exit 2; }
[[ "$GPU_SPEC" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer." >&2; exit 2; }
cd "$ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -z "$OUTPUT_DIR" ]]; then OUTPUT_DIR="runs/retacred_baseline_complete/$STAMP"_seed"$SEED"; fi
RUN_DIR=$(readlink -m "$OUTPUT_DIR")
case "$RUN_DIR" in "$ROOT"/runs/*) ;; *) echo "Output directory must be inside $ROOT/runs." >&2; exit 2 ;; esac
if [[ "$DRY_RUN" -eq 0 && -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse output directory: $RUN_DIR" >&2
  exit 1
fi
RUN_BASENAME=$(basename "$RUN_DIR")
REPORT_DIR="$ROOT/reports/retacred_baseline_complete/$RUN_BASENAME"
if [[ -n "$MODEL_DIR" ]]; then
  # Keep a diagnostic path for missing inputs; the validator owns existence checks.
  MODEL_DIR=$(readlink -m -- "$MODEL_DIR")
  case "$MODEL_DIR" in "$ROOT"/runs/*) ;; *) echo "--model-dir must be inside $ROOT/runs." >&2; exit 2 ;; esac
  "$PYTHON_BIN" scripts/validate_retacred_baseline_checkpoint.py --root "$ROOT" --model-dir "$MODEL_DIR" --expected-seed "$SEED"
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ "$SKIP_EXPORT" -eq 0 ]]; then
    bash scripts/check_retacred_baseline_publish_preflight.sh --run-dir "$RUN_DIR" --report-dir "$REPORT_DIR"
  fi
  if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
    bash scripts/check_retacred_baseline_complete.sh
  fi
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  nvidia-smi -i "$GPU_SPEC" --query-gpu=name --format=csv,noheader >/dev/null || { echo "GPU $GPU_SPEC is not available." >&2; exit 1; }
fi

if [[ -n "$MODEL_DIR" ]]; then BASELINE_DIR="$MODEL_DIR"; else BASELINE_DIR="$RUN_DIR/baseline"; fi
EVALUATION_DIR="$RUN_DIR/evaluation"
TRAIN_COMMAND=(
  "$PYTHON_BIN" experiments/train_relation_baseline.py
  --train_path data/relation/retacred/train.jsonl
  --valid_path data/relation/retacred/valid.jsonl
  --output_dir "$BASELINE_DIR"
  --epochs 12 --batch_size 128 --lr 0.0005
  --dim 128 --num_layers 4 --num_heads 8 --ff_dim 256
  --dropout 0.1 --max_length 128 --seed "$SEED"
  --selection_metric macro_f1_then_loss --device cuda
)
EVAL_COMMAND=(
  "$PYTHON_BIN" experiments/evaluate_relation_baseline.py
  --model-dir "$BASELINE_DIR"
  --train-path data/relation/retacred/train.jsonl
  --valid-path data/relation/retacred/valid.jsonl
  --test-path data/relation/retacred/test.jsonl
  --output-dir "$EVALUATION_DIR"
  --batch-size 256 --device cuda
)
if [[ "$WRITE_PREDICTIONS" -eq 1 ]]; then EVAL_COMMAND+=(--write-predictions); fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '[dry-run] CUDA_VISIBLE_DEVICES=%q ' "$GPU_SPEC"; printf '%q ' "${TRAIN_COMMAND[@]}"; printf '\n'
  printf '[dry-run] CUDA_VISIBLE_DEVICES=%q ' "$GPU_SPEC"; printf '%q ' "${EVAL_COMMAND[@]}"; printf '\n'
  if [[ "$SKIP_EXPORT" -eq 0 ]]; then
    printf '[dry-run] bash scripts/export_retacred_baseline_report.sh %q %q\n' "$RUN_DIR" "$REPORT_DIR"
    printf '[dry-run] bash scripts/publish_retacred_baseline_report.sh --run-dir %q\n' "$RUN_DIR"
  else
    printf '[dry-run] export and publish skipped (--skip-export)\n'
  fi
  exit 0
fi

record_run_failure() {
  local rc=$1 stage=$2 at
  at=$(date -Iseconds)
  printf 'STATUS=failed\nFAILED_STAGE=%s\nEXIT_STATUS=%s\nFAILED_AT=%s\n' "$stage" "$rc" "$at" >> "$RUN_DIR/status/run.env"
  printf '%s status=failed stage=%s exit=%s\n' "$at" "$stage" "$rc" >> "$RUN_DIR/status/history.log"
  printf '%s\n' "$at" > "$RUN_DIR/RUN_FAILED"
}
on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "$rc" -ne 0 && -n "$RUN_DIR" && -d "$RUN_DIR/status" && ! -e "$RUN_DIR/RUN_COMPLETE" ]]; then
    record_run_failure "$rc" "$CURRENT_STAGE" || true
  fi
  exit "$rc"
}
trap on_exit EXIT

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status"
printf 'STATUS=running\nSEED=%s\nGPU_ID=%s\nSTARTED_AT=%s\n' "$SEED" "$GPU_SPEC" "$(date -Iseconds)" > "$RUN_DIR/status/run.env"
printf '%s status=running stage=preflight\n' "$(date -Iseconds)" > "$RUN_DIR/status/history.log"
if [[ -z "$MODEL_DIR" ]]; then
  CURRENT_STAGE=training
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_SPEC" "${TRAIN_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/logs/baseline_train.log"
else
  CURRENT_STAGE=checkpoint-reuse
  printf 'Reusing validated baseline checkpoint: %s\n' "$MODEL_DIR" | tee "$RUN_DIR/logs/baseline_train.log"
fi
CURRENT_STAGE=evaluation
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_SPEC" "${EVAL_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/logs/baseline_evaluation.log"
CURRENT_STAGE=artifact-copy
cp "$EVALUATION_DIR/metrics.json" "$RUN_DIR/metrics.json"
cp "$EVALUATION_DIR/run_summary.json" "$RUN_DIR/run_summary.json"
cp "$EVALUATION_DIR/run_summary.data" "$RUN_DIR/run_summary.data"
cp "$EVALUATION_DIR/run_summary.md" "$RUN_DIR/run_summary.md"
cp "$EVALUATION_DIR/run_config.json" "$RUN_DIR/run_config.json"
CURRENT_STAGE=provenance-check
"$PYTHON_BIN" - "$RUN_DIR/run_summary.json" "$SEED" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
expected_seed = int(sys.argv[2])
actual_seed = int(payload["seed"])
if actual_seed != expected_seed:
    raise SystemExit(f"checkpoint/report seed mismatch: runner={expected_seed} report={actual_seed}")
if payload.get("status") != "complete":
    raise SystemExit(f"evaluation status is not complete: {payload.get('status')!r}")
print(f"baseline provenance seed={actual_seed}")
PY
printf '%s\n' "$(date -Iseconds)" > "$RUN_DIR/RUN_COMPLETE"
printf 'STATUS=complete\nSEED=%s\nGPU_ID=%s\nCOMPLETED_AT=%s\n' "$SEED" "$GPU_SPEC" "$(date -Iseconds)" >> "$RUN_DIR/status/run.env"
printf '%s status=complete stage=run\n' "$(date -Iseconds)" >> "$RUN_DIR/status/history.log"
echo "RUN_DIR=$RUN_DIR"

if [[ "$SKIP_EXPORT" -eq 1 ]]; then
  printf 'STATUS=skipped\nREASON=skip-export\nAT=%s\n' "$(date -Iseconds)" >> "$RUN_DIR/status/export.env"
  printf '%s status=skipped stage=export reason=skip-export\n' "$(date -Iseconds)" >> "$RUN_DIR/status/history.log"
  echo "Report export skipped (--skip-export)."
  exit 0
fi
CURRENT_STAGE=export
if ! bash scripts/export_retacred_baseline_report.sh "$RUN_DIR" "$REPORT_DIR" 2>&1 | tee "$RUN_DIR/logs/report_export.log"; then
  printf 'STATUS=failed\nREASON=export\nAT=%s\n' "$(date -Iseconds)" >> "$RUN_DIR/status/export.env"
  printf '%s status=failed stage=export\n' "$(date -Iseconds)" >> "$RUN_DIR/status/history.log"
  echo "Report export failed; raw run is complete and can be re-exported without retraining." >&2
  exit 1
fi
printf 'STATUS=complete\nREPORT_DIR=%s\nAT=%s\n' "$REPORT_DIR" "$(date -Iseconds)" >> "$RUN_DIR/status/export.env"
printf '%s status=complete stage=export\n' "$(date -Iseconds)" >> "$RUN_DIR/status/history.log"
printf '%s\n' "$(date -Iseconds)" > "$RUN_DIR/EXPORT_COMPLETE"

CURRENT_STAGE=publish
PUBLISH_ARGS=(--run-dir "$RUN_DIR")
[[ "$NO_PUSH" -eq 1 ]] && PUBLISH_ARGS+=(--no-push)
if ! bash scripts/publish_retacred_baseline_report.sh "${PUBLISH_ARGS[@]}"; then
  echo "Report publication failed; rerun bash scripts/publish_retacred_baseline_report.sh --run-dir $RUN_DIR. Training will not repeat." >&2
  exit 1
fi
echo "Report exported and publication workflow completed: $REPORT_DIR"
