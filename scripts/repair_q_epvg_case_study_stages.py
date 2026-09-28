#!/usr/bin/env python3
"""Repair Case Study stage projections from an immutable sample trace.

The repair is deterministic: it only copies the producer-owned ``stages``
list for a matching sample ID from ``sample_trace.json`` into the corresponding
case in ``case_study.json``. It never retrains, changes metrics, or accepts a
sample/order mismatch. A pre-existing ``stages`` field is considered complete
only when every required stage is present, ordered, and marked ``observed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_SCHEMA = "q-attention.Q-EPVG-case-study.v2"
TRACE_SCHEMA = "sample-trace.v1"
MANIFEST_SCHEMA = "q-attention.q-epvg-case-study-stage-repair.v2"

REQUIRED_STAGES = (
    "data",
    "preprocess",
    "embedding",
    "encoder",
    "training",
    "attention_baseline",
    "scoring",
    "selection",
    "context",
    "classifier",
    "evaluation",
    "diagnosis",
)
OPTIONAL_STAGES = {"retrieval", "generation", "attention_intervention"}
STAGE_ORDER = (
    "data",
    "preprocess",
    "embedding",
    "encoder",
    "training",
    "retrieval",
    "attention_baseline",
    "scoring",
    "selection",
    "attention_intervention",
    "context",
    "classifier",
    "generation",
    "evaluation",
    "diagnosis",
)
ALLOWED_STATUSES = {"observed", "not_applicable", "failed"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _selector_dirs(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in group_dir.glob("seed_*/selectors/*")
        if path.is_dir()
        and (path / "case_study.json").is_file()
        and (path / "sample_trace.json").is_file()
    )


def _stage_completeness_error(stages: object, *, where: str) -> str | None:
    """Return a precise error when a stage projection is not export-complete.

    This deliberately checks the same structural contract that matters for
    repair. The full exporter remains responsible for tensor references and
    manifest checks; this function prevents the repairer from treating a
    merely present, but failed or incomplete, stage list as complete.
    """

    if not isinstance(stages, list) or not stages:
        return "stages must be a non-empty list"

    names: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or not isinstance(stage.get("stage"), str) or not stage["stage"]:
            return f"stages[{index}] is missing a non-empty stage name"
        status = stage.get("status")
        if status not in ALLOWED_STATUSES:
            return f"stages[{index}] ({stage['stage']!r}) has invalid status {status!r}"
        names.append(stage["stage"])

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return f"duplicate stage names: {', '.join(duplicates)}"

    missing = [name for name in REQUIRED_STAGES if name not in names]
    if missing:
        return f"missing required stages: {', '.join(missing)}"

    unknown = [
        name
        for name in names
        if name not in REQUIRED_STAGES and name not in OPTIONAL_STAGES
    ]
    if unknown:
        return f"unknown stages: {', '.join(unknown)}"

    positions = [STAGE_ORDER.index(name) for name in names]
    if positions != sorted(positions):
        return "stages are not in producer order"

    by_name = {stage["stage"]: stage for stage in stages}
    for name in REQUIRED_STAGES:
        status = by_name[name]["status"]
        if status != "observed":
            index = names.index(name)
            return f"required stage {name!r} at index {index} is not fully observed ({status})"
    return None


def _prepare_one(selector_dir: Path) -> dict[str, Any]:
    case_path = selector_dir / "case_study.json"
    trace_path = selector_dir / "sample_trace.json"
    case_payload = _load(case_path)
    trace_payload = _load(trace_path)
    if not isinstance(case_payload, dict):
        raise ValueError(f"{case_path}: expected JSON object")
    if not isinstance(trace_payload, dict):
        raise ValueError(f"{trace_path}: expected JSON object")
    if case_payload.get("schema_version") != CASE_SCHEMA:
        raise ValueError(f"{case_path}: unsupported Case Study schema")
    if trace_payload.get("schema_version") != TRACE_SCHEMA:
        raise ValueError(f"{trace_path}: unsupported sample trace schema")
    cases = case_payload.get("cases")
    samples = trace_payload.get("samples")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{case_path}: cases must be a non-empty list")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{trace_path}: samples must be a non-empty list")
    sample_by_id: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
            raise ValueError(f"{trace_path}: samples[{index}] lacks sample_id")
        sample_id = sample["sample_id"]
        if sample_id in sample_by_id:
            raise ValueError(f"{trace_path}: duplicate sample_id {sample_id!r}")
        stages = sample.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"{trace_path}: samples[{index}].stages must be a non-empty list")
        stage_error = _stage_completeness_error(
            stages,
            where=f"{trace_path}: samples[{index}] ({sample_id})",
        )
        if stage_error is not None:
            raise ValueError(f"{trace_path}: samples[{index}] ({sample_id}): {stage_error}")
        sample_by_id[sample_id] = sample
    updated = 0
    replaced = 0
    replaced_case_ids: list[str] = []
    already_complete = 0
    repaired_payload = json.loads(json.dumps(case_payload))
    case_ids: set[str] = set()
    repaired_cases = repaired_payload["cases"]
    for index, case in enumerate(repaired_cases):
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError(f"{case_path}: cases[{index}] lacks case_id")
        case_id = case["case_id"]
        if case_id in case_ids:
            raise ValueError(f"{case_path}: duplicate case_id {case_id!r}")
        case_ids.add(case_id)
        sample = sample_by_id.get(case_id)
        if sample is None:
            raise ValueError(f"{case_path}: no matching sample_trace entry for {case_id!r}")
        trace_stages = sample["stages"]
        existing = case.get("stages")
        if existing is None:
            case["stages"] = json.loads(json.dumps(trace_stages))
            updated += 1
        else:
            existing_error = _stage_completeness_error(
                existing,
                where=f"{case_path}: {case_id}",
            )
            if existing_error is None:
                if existing != trace_stages:
                    raise ValueError(
                        f"{case_path}: {case_id}.stages is semantically complete "
                        "but disagrees with sample_trace.json"
                    )
                already_complete += 1
            else:
                case["stages"] = json.loads(json.dumps(trace_stages))
                replaced += 1
                replaced_case_ids.append(case_id)

    if len(cases) != len(samples):
        raise ValueError(f"{selector_dir}: case/sample count mismatch ({len(cases)} != {len(samples)})")

    if case_ids != set(sample_by_id):
        missing = sorted(set(sample_by_id) - case_ids)
        extra = sorted(case_ids - set(sample_by_id))
        raise ValueError(
            f"{selector_dir}: case/sample IDs differ; missing_cases={missing!r}, "
            f"extra_cases={extra!r}"
        )

    return {
        "selector_dir": str(selector_dir),
        "selector": str(case_payload.get("selector", selector_dir.name)),
        "case_path": str(case_path),
        "trace_path": str(trace_path),
        "case_count": len(cases),
        "updated_cases": updated + replaced,
        "added_cases": updated,
        "replaced_cases": replaced,
        "replaced_case_ids": replaced_case_ids,
        "already_complete_cases": already_complete,
        "before_sha256": _sha256(case_path),
        "after_payload": repaired_payload,
    }


def repair_group(group_dir: Path, *, apply: bool, root: Path) -> dict[str, Any]:
    group_dir = group_dir.resolve()
    if not group_dir.is_dir():
        raise ValueError(f"group directory does not exist: {group_dir}")
    state_path = group_dir / "multi_seed_run_summary.json"
    if not state_path.is_file():
        raise ValueError(f"missing completed scheduler state: {state_path}")
    state = _load(state_path)
    if not isinstance(state, dict) or state.get("success") is not True:
        raise ValueError("refusing repair: multi_seed_run_summary.json does not declare success")
    selector_dirs = _selector_dirs(group_dir)
    if not selector_dirs:
        raise ValueError(f"no selector Case Study pairs found under {group_dir}")

    plans = [_prepare_one(path) for path in selector_dirs]
    changes = [plan for plan in plans if plan["updated_cases"]]
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_dir": str(group_dir),
        "source": "sample_trace.json -> case_study.json cases[*].stages",
        "git_revision": _git_revision(root),
        "apply": bool(apply),
        "selector_count": len(plans),
        "case_count": sum(int(item["case_count"]) for item in plans),
        "updated_case_count": sum(int(item["updated_cases"]) for item in plans),
        "replaced_case_count": sum(int(item["replaced_cases"]) for item in plans),
        "already_complete_case_count": sum(int(item["already_complete_cases"]) for item in plans),
        "selectors": [
            {
                key: value
                for key, value in plan.items()
                if key != "after_payload"
            }
            for plan in plans
        ],
    }
    if not apply:
        return manifest

    # Check every backup target before the first write so a later conflict
    # cannot leave an earlier selector partially repaired.
    for plan in changes:
        case_path = Path(plan["case_path"])
        backup = case_path.with_name(case_path.name + ".pre_stage_repair")
        if backup.exists():
            raise ValueError(f"refusing overwrite because backup already exists: {backup}")

    for plan in changes:
        case_path = Path(plan["case_path"])
        backup = case_path.with_name(case_path.name + ".pre_stage_repair")
        shutil.copy2(case_path, backup)
        _write_json_atomic(case_path, plan["after_payload"])
        plan["after_sha256"] = _sha256(case_path)
        plan["backup_path"] = str(backup)
    manifest["selectors"] = [
        {
            **{
                key: value
                for key, value in plan.items()
                if key != "after_payload"
            },
        }
        for plan in plans
    ]
    _write_json_atomic(group_dir / "case_study_stage_repair_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="write backups and repaired Case Study files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        manifest = repair_group(args.group_dir, apply=args.apply, root=root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print("status=" + ("APPLIED" if args.apply else "DRY_RUN"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
