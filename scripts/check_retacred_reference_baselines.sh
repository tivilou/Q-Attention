#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REGISTRY="${ROOT}/configs/retacred_reference_baselines.json"

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

[[ -f "${REGISTRY}" ]] || { echo "Missing ${REGISTRY}" >&2; exit 1; }

"${PYTHON_BIN}" - "${REGISTRY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != "q-attention.retacred-reference-baseline-registry.v2":
    raise SystemExit("unsupported reference-baseline registry schema")
if payload.get("dataset") != "Re-TACRED":
    raise SystemExit("registry dataset must be Re-TACRED")
selection = payload.get("selection", {})
if selection.get("metric") != "micro_f1":
    raise SystemExit("reported ranking metric must be micro_f1")
if selection.get("scores_are_formal_project_results") is not False:
    raise SystemExit("paper scores must not be marked as formal project results")

candidates = payload.get("candidates")
if not isinstance(candidates, list) or len(candidates) != 5:
    raise SystemExit("registry must contain exactly five reported configurations")
ranks = [item.get("rank") for item in candidates]
if ranks != [1, 2, 3, 4, 5]:
    raise SystemExit(f"candidate ranks must be 1..5, got {ranks!r}")
ids = [item.get("id") for item in candidates]
if len(set(ids)) != len(ids):
    raise SystemExit("candidate IDs must be unique")

for item in candidates:
    if item.get("status") != "planned":
        raise SystemExit(f"unreviewed candidate {item.get('id')} must remain planned")
    if item.get("strict_comparison_eligible") is not False:
        raise SystemExit(f"unreviewed candidate {item.get('id')} cannot enter strict comparison")
    score = item.get("reported_in_paper", {}).get("test_micro_f1")
    if not isinstance(score, (int, float)) or not 0.0 < score < 1.0:
        raise SystemExit(f"invalid reported score for {item.get('id')}")
    if not item.get("paper_url"):
        raise SystemExit(f"missing paper URL for {item.get('id')}")
    if not item.get("required_gates"):
        raise SystemExit(f"missing fidelity gates for {item.get('id')}")

families = {item.get("method_family") for item in candidates}
if "ctl_drp" not in families:
    raise SystemExit("CTL-DRP family variant must remain visible in the cutoff")
if not payload.get("excluded_or_non_strict"):
    raise SystemExit("non-strict exclusions must be recorded")

print("reference registry=OK")
for item in candidates:
    print(
        f"rank={item['rank']} id={item['id']} "
        f"reported_micro_f1={item['reported_in_paper']['test_micro_f1']:.3f} "
        f"status={item['status']} source_status={item['source_status']}"
    )
PY
