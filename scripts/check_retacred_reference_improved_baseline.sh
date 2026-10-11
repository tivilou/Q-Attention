#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_DIR="${ROOT}/.external/RE_improved_baseline"
DATA_DIR="${ROOT}/data/relation/retacred"
GPU_SPEC=0
ALLOW_DIRTY=0
SKIP_GPU=0

usage() {
  cat <<'EOF'
Usage: bash scripts/check_retacred_reference_improved_baseline.sh [options]

Options:
  --upstream-dir PATH  Pinned upstream checkout (default: .external/RE_improved_baseline)
  --data-dir PATH      Canonical Re-TACRED directory (default: data/relation/retacred)
  --gpu N              GPU index to verify (default: 0)
  --allow-dirty        Permit a dirty project worktree for bounded diagnostics
  --skip-gpu           Skip nvidia-smi and CUDA checks
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir) [[ $# -ge 2 ]] || { echo "--upstream-dir requires a value." >&2; exit 2; }; UPSTREAM_DIR=$2; shift ;;
    --data-dir) [[ $# -ge 2 ]] || { echo "--data-dir requires a value." >&2; exit 2; }; DATA_DIR=$2; shift ;;
    --gpu) [[ $# -ge 2 ]] || { echo "--gpu requires a value." >&2; exit 2; }; GPU_SPEC=$2; shift ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --skip-gpu) SKIP_GPU=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "$GPU_SPEC" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer." >&2; exit 2; }
cd "$ROOT"
command -v git >/dev/null || { echo "git is required." >&2; exit 1; }
[[ -x "scripts/check_retacred_reference_improved_baseline_source.sh" ]] || { echo "Missing source gate." >&2; exit 1; }

if [[ "$ALLOW_DIRTY" -eq 0 ]]; then
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
    echo "Project worktree is dirty; use --allow-dirty only for diagnostics." >&2
    git status --short >&2
    exit 1
  }
fi

for split in train valid test; do
  [[ -s "$DATA_DIR/${split}.jsonl" ]] || { echo "Missing canonical split: $DATA_DIR/${split}.jsonl" >&2; exit 1; }
done

TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
VALID_COUNT=$(wc -l < "$DATA_DIR/valid.jsonl")
TEST_COUNT=$(wc -l < "$DATA_DIR/test.jsonl")
[[ "$TRAIN_COUNT" -eq 58465 ]] || { echo "Unexpected train count: $TRAIN_COUNT" >&2; exit 1; }
[[ "$VALID_COUNT" -eq 19584 ]] || { echo "Unexpected valid count: $VALID_COUNT" >&2; exit 1; }
[[ "$TEST_COUNT" -eq 13418 ]] || { echo "Unexpected test count: $TEST_COUNT" >&2; exit 1; }

"${PYTHON_BIN:-python3}" - "$DATA_DIR" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "experiments"))
from prepare_retacred_reference_improved_baseline import _load_records, UPSTREAM_LABEL_TO_ID

root = Path(sys.argv[1])
labels = set()
for split in ("train", "valid", "test"):
    records = _load_records(root / f"{split}.jsonl")
    labels.update(record["label"] for record in records)
if len(labels) != 40 or not labels.issubset(UPSTREAM_LABEL_TO_ID):
    raise SystemExit(f"canonical labels are incompatible with upstream Re-TACRED map: {len(labels)}")
print(f"canonical data=OK labels={len(labels)}")
PY

[[ -d "$UPSTREAM_DIR/.git" ]] || { echo "Missing upstream checkout: $UPSTREAM_DIR" >&2; exit 1; }
bash scripts/check_retacred_reference_improved_baseline_source.sh --upstream-dir "$UPSTREAM_DIR"

PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" - <<'PY'
import importlib.util
required = ("torch", "transformers", "numpy", "tqdm", "ujson")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python dependencies: " + ", ".join(missing))
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
print("python dependencies=OK")
PY

if [[ "$SKIP_GPU" -eq 0 ]]; then
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required." >&2; exit 1; }
  nvidia-smi -i "$GPU_SPEC" --query-gpu=name,memory.total --format=csv,noheader
fi

echo "Improved Baseline preflight OK"
echo "train_records=$TRAIN_COUNT"
echo "valid_records=$VALID_COUNT"
echo "test_records=$TEST_COUNT"
