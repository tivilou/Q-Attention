#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
usage() { echo "Usage: bash scripts/export_retacred_reference_improved_baseline_report.sh RUN_DIR [REPORT_DIR]"; }
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
cd "$ROOT"
[[ "$(git branch --show-current)" == "1.1" ]] || { echo "Exporter requires branch 1.1." >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Worktree must be clean before export." >&2; git status --short >&2; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "HEAD must contain origin/main." >&2; exit 1; }
git merge-base --is-ancestor origin/1.1 HEAD || { echo "HEAD must contain origin/1.1." >&2; exit 1; }
[[ -z "$(git diff --cached --name-only)" ]] || { echo "Index must be empty before export." >&2; exit 1; }

RUN_DIR=$(readlink -f -- "$1")
case "$RUN_DIR" in "$ROOT"/runs/*) ;; *) echo "RUN_DIR must be inside $ROOT/runs." >&2; exit 1 ;; esac
for file in RUN_COMPLETE run_summary.json run_summary.data run_summary.md run_config.json metrics.json data_manifest.json case_study.json case_study_tensor_manifest.json case_study_tensors.pt; do
  [[ -f "$RUN_DIR/$file" ]] || { echo "Missing $RUN_DIR/$file" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR/RUN_FAILED" ]] || { echo "Run contains RUN_FAILED." >&2; exit 1; }

REPORT_DIR=${2:-reports/retacred_reference_improved_baseline/$(basename "$RUN_DIR")}
REPORT_DIR=$(readlink -m -- "$REPORT_DIR")
case "$REPORT_DIR" in "$ROOT"/reports/retacred_reference_improved_baseline/*) ;; *) echo "REPORT_DIR is outside the reference report namespace." >&2; exit 1 ;; esac
[[ ! -e "$REPORT_DIR" ]] || { echo "Refusing to overwrite $REPORT_DIR." >&2; exit 1; }

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
assert summary.get("schema") == "q-attention.retacred-reference-improved-baseline.v1"
assert summary.get("status") == "complete"
assert summary.get("test_used_for_training_or_selection") is False
assert summary.get("upstream", {}).get("commit")
assert summary.get("configuration", {}).get("input_format") == "typed_entity_marker_punct"
required = {"micro_precision", "micro_recall", "micro_f1", "accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"}
for split in ("train", "valid", "test"):
    metrics = summary["splits"][split]["metrics"]
    assert required.issubset(metrics), (split, sorted(required - set(metrics)))
    assert "source_non_null_micro_f1" in metrics
    assert summary["splits"][split]["per_class"]
manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
expected = {"train": 58465, "valid": 19584, "test": 13418}
for split, count in expected.items():
    assert manifest["splits"][split]["records"] == count
case_study = json.loads((run / "case_study.json").read_text(encoding="utf-8"))
assert case_study.get("status") == "complete"
assert case_study.get("samples")
tensor_manifest = json.loads((run / "case_study_tensor_manifest.json").read_text(encoding="utf-8"))
assert tensor_manifest.get("sha256") == summary.get("case_study_tensor_sha256")
assert tensor_manifest.get("samples")
data_text = (run / "run_summary.data").read_text(encoding="utf-8")
assert "schema=q-attention.retacred-reference-improved-baseline-run-summary-data.v1" in data_text
assert "status=complete" in data_text
print("reference report identity=OK")
PY

TMP_DIR=$(mktemp -d "$ROOT/reports/.reference-export.XXXXXX")
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT
mkdir -p "$TMP_DIR/logs" "$TMP_DIR/case_study"
for file in RUN_COMPLETE run_summary.json run_summary.data run_summary.md run_config.json metrics.json data_manifest.json case_study.json case_study_tensor_manifest.json; do
  cp "$RUN_DIR/$file" "$TMP_DIR/$file"
done
cp "$RUN_DIR/case_study.json" "$TMP_DIR/case_study/case_study.json"
cp "$RUN_DIR/case_study_tensor_manifest.json" "$TMP_DIR/case_study/tensor_manifest.json"
"$PYTHON_BIN" - "$RUN_DIR" "$TMP_DIR" <<'PY'
import json
import sys
from pathlib import Path

run, stage = map(Path, sys.argv[1:])
manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
(stage / "data_counts.txt").write_text(
    "".join(f"{manifest['splits'][split]['records']} {split}.jsonl\n" for split in ("train", "valid", "test")),
    encoding="utf-8",
)
(stage / "data.sha256").write_text(
    "".join(f"{manifest['splits'][split]['source_sha256']}  {split}.jsonl\n" for split in ("train", "valid", "test")),
    encoding="utf-8",
)
summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
(stage / "upstream_source_commit.txt").write_text(summary["upstream"]["commit"] + "\n", encoding="utf-8")
PY
tail -n 1000 "$RUN_DIR/logs/reference_baseline.log" > "$TMP_DIR/logs/reference_baseline.log.tail.txt" 2>/dev/null || true
tail -n 1000 "$RUN_DIR/logs/preflight.log" > "$TMP_DIR/logs/preflight.log.tail.txt" 2>/dev/null || true

if find "$TMP_DIR" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.jsonl' -o -name 'predictions*' \) | grep -q .; then
  echo "Forbidden private artifact detected in report staging." >&2
  exit 1
fi
mkdir -p "$(dirname "$REPORT_DIR")"
mv "$TMP_DIR" "$REPORT_DIR"
trap - EXIT
echo "Reference report ready: $REPORT_DIR"
