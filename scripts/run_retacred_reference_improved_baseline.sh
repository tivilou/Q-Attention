#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-}
SEED=13
GPU_SPEC=0
OUTPUT_DIR=
UPSTREAM_DIR="${ROOT}/.external/RE_improved_baseline"
DATA_DIR="${ROOT}/data/relation/retacred"
NO_PUSH=0
SKIP_EXPORT=0
DRY_RUN=0
SKIP_PREFLIGHT=0
CURRENT_STAGE=preflight
RUN_DIR=

UPSTREAM_URL="https://github.com/wzhouad/RE_improved_baseline.git"
UPSTREAM_COMMIT="fe00781116ab4c34cbcd9a2ae5b02513afae87ee"

resolve_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null || { echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2; exit 1; }
    return
  fi
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3
  else echo "Neither python nor python3 is available; set PYTHON_BIN explicitly." >&2; exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_retacred_reference_improved_baseline.sh [options]

Options:
  --seed N              Random seed (default: 13)
  --gpu N               Physical GPU index (default: 0)
  --output-dir PATH     New run directory under runs/
  --upstream-dir PATH   Pinned source checkout (default: .external/RE_improved_baseline)
  --data-dir PATH       Canonical Re-TACRED directory (default: data/relation/retacred)
  --skip-preflight      Skip dependency/data checks (diagnostic only)
  --skip-export         Keep raw run only
  --no-push              Export and commit report, but do not push
  --dry-run             Print the plan without cloning, training, or writing a run
  -h|--help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) [[ $# -ge 2 ]] || { echo "--seed requires a value." >&2; exit 2; }; SEED=$2; shift ;;
    --gpu) [[ $# -ge 2 ]] || { echo "--gpu requires a value." >&2; exit 2; }; GPU_SPEC=$2; shift ;;
    --output-dir) [[ $# -ge 2 ]] || { echo "--output-dir requires a value." >&2; exit 2; }; OUTPUT_DIR=$2; shift ;;
    --upstream-dir) [[ $# -ge 2 ]] || { echo "--upstream-dir requires a value." >&2; exit 2; }; UPSTREAM_DIR=$2; shift ;;
    --data-dir) [[ $# -ge 2 ]] || { echo "--data-dir requires a value." >&2; exit 2; }; DATA_DIR=$2; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
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
resolve_python
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -z "$OUTPUT_DIR" ]]; then OUTPUT_DIR="runs/retacred_reference_improved_baseline/${STAMP}_seed${SEED}"; fi
RUN_DIR=$(readlink -m -- "$OUTPUT_DIR")
case "$RUN_DIR" in "$ROOT"/runs/*) ;; *) echo "Output directory must be inside $ROOT/runs." >&2; exit 2 ;; esac
RUN_BASENAME=$(basename "$RUN_DIR")
REPORT_DIR="$ROOT/reports/retacred_reference_improved_baseline/$RUN_BASENAME"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] upstream=$UPSTREAM_URL@$UPSTREAM_COMMIT"
  echo "[dry-run] upstream_dir=$UPSTREAM_DIR"
  echo "[dry-run] data_dir=$DATA_DIR"
  echo "[dry-run] output_dir=$RUN_DIR"
  echo "[dry-run] ${PYTHON_BIN} experiments/run_retacred_reference_improved_baseline.py --seed $SEED --gpu $GPU_SPEC"
  exit 0
fi

[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to reuse output directory: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "Project worktree must be clean before a formal reference run." >&2
  git status --short >&2
  exit 1
}
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Formal reference runner must run on branch 1.1." >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "HEAD must contain origin/main." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "HEAD must contain origin/1.1." >&2; exit 1; }

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone --no-checkout "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi
[[ -z "$(git -C "$UPSTREAM_DIR" status --porcelain --untracked-files=all)" ]] || {
  echo "Upstream checkout is dirty: $UPSTREAM_DIR" >&2
  exit 1
}
git -C "$UPSTREAM_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_COMMIT"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status"
printf 'STATUS=running\nSEED=%s\nGPU_ID=%s\nSTARTED_AT=%s\n' "$SEED" "$GPU_SPEC" "$(date -Iseconds)" > "$RUN_DIR/status/run.env"
printf '%s status=running stage=preflight\n' "$(date -Iseconds)" > "$RUN_DIR/status/history.log"

record_failure() {
  local rc=$1
  printf 'STATUS=failed\nFAILED_STAGE=%s\nEXIT_STATUS=%s\nFAILED_AT=%s\n' "$CURRENT_STAGE" "$rc" "$(date -Iseconds)" >> "$RUN_DIR/status/run.env"
  printf '%s status=failed stage=%s exit=%s\n' "$(date -Iseconds)" "$CURRENT_STAGE" "$rc" >> "$RUN_DIR/status/history.log"
  printf '%s\n' "$(date -Iseconds)" > "$RUN_DIR/RUN_FAILED"
}
on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "$rc" -ne 0 && -n "$RUN_DIR" && -d "$RUN_DIR/status" && ! -e "$RUN_DIR/RUN_COMPLETE" ]]; then
    record_failure "$rc" || true
  fi
  exit "$rc"
}
trap on_exit EXIT

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
  CURRENT_STAGE=preflight
  bash scripts/check_retacred_reference_improved_baseline.sh --upstream-dir "$UPSTREAM_DIR" --data-dir "$DATA_DIR" --gpu "$GPU_SPEC" 2>&1 | tee "$RUN_DIR/logs/preflight.log"
fi

CURRENT_STAGE=data-conversion
"$PYTHON_BIN" experiments/prepare_retacred_reference_improved_baseline.py \
  --train-path "$DATA_DIR/train.jsonl" \
  --valid-path "$DATA_DIR/valid.jsonl" \
  --test-path "$DATA_DIR/test.jsonl" \
  --output-dir "$RUN_DIR/data" 2>&1 | tee "$RUN_DIR/logs/data_conversion.log"

CURRENT_STAGE=training
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_SPEC" "$PYTHON_BIN" experiments/run_retacred_reference_improved_baseline.py \
  --upstream-dir "$UPSTREAM_DIR" \
  --data-dir "$RUN_DIR/data" \
  --output-dir "$RUN_DIR" \
  --seed "$SEED" \
  --device cuda 2>&1 | tee "$RUN_DIR/logs/reference_baseline.log"

CURRENT_STAGE=artifact-validation
for required_artifact in \
  run_summary.json \
  metrics.json \
  run_config.json \
  run_summary.data \
  case_study.json \
  case_study_tensor_manifest.json \
  case_study_tensors.pt \
  checkpoint/model.pt \
  checkpoint/label_to_id.json; do
  [[ -s "$RUN_DIR/$required_artifact" ]] || {
    echo "Required reference-run artifact is missing or empty: $RUN_DIR/$required_artifact" >&2
    exit 1
  }
done

printf '%s\n' "$(date -Iseconds)" > "$RUN_DIR/RUN_COMPLETE"
printf 'STATUS=complete\nSEED=%s\nGPU_ID=%s\nCOMPLETED_AT=%s\n' "$SEED" "$GPU_SPEC" "$(date -Iseconds)" >> "$RUN_DIR/status/run.env"
printf '%s status=complete stage=run\n' "$(date -Iseconds)" >> "$RUN_DIR/status/history.log"
echo "RUN_DIR=$RUN_DIR"

if [[ "$SKIP_EXPORT" -eq 1 ]]; then
  printf 'STATUS=skipped\nREASON=skip-export\nAT=%s\n' "$(date -Iseconds)" >> "$RUN_DIR/status/export.env"
  echo "Report export skipped (--skip-export)."
  exit 0
fi

CURRENT_STAGE=export
bash scripts/export_retacred_reference_improved_baseline_report.sh "$RUN_DIR" "$REPORT_DIR" 2>&1 | tee "$RUN_DIR/logs/report_export.log"
printf 'STATUS=complete\nREPORT_DIR=%s\nAT=%s\n' "$REPORT_DIR" "$(date -Iseconds)" >> "$RUN_DIR/status/export.env"
printf '%s\n' "$(date -Iseconds)" > "$RUN_DIR/EXPORT_COMPLETE"

CURRENT_STAGE=publish
PUBLISH_ARGS=(--run-dir "$RUN_DIR")
[[ "$NO_PUSH" -eq 1 ]] && PUBLISH_ARGS+=(--no-push)
bash scripts/publish_retacred_reference_improved_baseline_report.sh "${PUBLISH_ARGS[@]}"
echo "Reference report exported and publication workflow completed: $REPORT_DIR"
