#!/usr/bin/env python3
"""Build and atomically publish the public Q-EPVG report projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any


class ExportError(RuntimeError):
    """Raised when a report cannot be staged or validated."""


_CASE_SPLITS = ("train", "valid", "test")
_CASE_CHECKPOINTS = (
    "initial_or_pre_training",
    "best_valid_or_declared_selection_checkpoint",
    "final",
)
_CASE_REQUIRED_STAGES = (
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
_CASE_OPTIONAL_STAGES = {"retrieval", "generation", "attention_intervention"}
_CASE_SEMANTIC_REFS = {
    "source_sample",
    "token_ids",
    "attention_mask",
    "subject_mask",
    "object_mask",
    "gold_relation",
    "diagnosis",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExportError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExportError(f"invalid {description}: {path}: expected an object")
    return payload


def _manifest_ref(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("ref"), str):
        return value["ref"]
    return None


def _validate_manifest(manifest: object, *, where: str) -> str:
    if not isinstance(manifest, dict):
        raise ExportError(f"{where}: tensor manifest entry must be an object")
    required = ("id", "manifest_id", "path", "shape", "dtype", "axis_semantics", "sha256", "byte_count", "preview")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ExportError(f"{where}: tensor manifest missing {', '.join(missing)}")
    semantic_id = manifest["id"]
    manifest_id = manifest["manifest_id"]
    if not isinstance(semantic_id, str) or not semantic_id:
        raise ExportError(f"{where}: tensor manifest id must be a non-empty string")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ExportError(f"{where}: tensor manifest_id must be a non-empty string")
    path = manifest["path"]
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ExportError(f"{where}: unsafe tensor manifest path: {path!r}")
    if not path.replace("\\", "/").startswith("case_study_tensors/"):
        raise ExportError(f"{where}: tensor manifest path must be under case_study_tensors/: {path!r}")
    shape = manifest["shape"]
    axes = manifest["axis_semantics"]
    if not isinstance(shape, list) or not all(isinstance(item, int) and item >= 0 for item in shape):
        raise ExportError(f"{where}: tensor shape must be a list of non-negative integers")
    if not isinstance(axes, list) or len(axes) != len(shape) or not all(isinstance(item, str) and item for item in axes):
        raise ExportError(f"{where}: tensor axis_semantics must match shape dimensions")
    if not isinstance(manifest["dtype"], str) or not manifest["dtype"]:
        raise ExportError(f"{where}: tensor dtype must be a non-empty string")
    if not isinstance(manifest["sha256"], str) or not _SHA256_RE.fullmatch(manifest["sha256"]):
        raise ExportError(f"{where}: tensor sha256 must be a lowercase 64-character digest")
    if not isinstance(manifest["byte_count"], int) or manifest["byte_count"] <= 0:
        raise ExportError(f"{where}: tensor byte_count must be positive")
    preview = manifest["preview"]
    if not isinstance(preview, dict):
        raise ExportError(f"{where}: tensor preview must be an object")
    for key in ("min", "max", "mean", "l2_norm"):
        value = preview.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ExportError(f"{where}: tensor preview.{key} must be finite")
    producer = manifest.get("producer_stage")
    if not isinstance(producer, str) or not producer:
        raise ExportError(f"{where}: tensor producer_stage is required")
    return manifest_id


def _validate_stage_ref(
    value: object,
    *,
    where: str,
    case_manifest_ids: set[str],
    manifest_by_id: dict[str, dict[str, Any]],
) -> None:
    ref = _manifest_ref(value)
    if ref is None:
        raise ExportError(f"{where}: stage reference must be a string or object with ref")
    if ref in _CASE_SEMANTIC_REFS:
        return
    if ref not in case_manifest_ids:
        raise ExportError(f"{where}: dangling stage reference {ref!r}")
    if isinstance(value, dict) and value.get("kind") == "representation":
        producer = value.get("producer_stage")
        expected = manifest_by_id[ref].get("producer_stage")
        if producer != expected:
            raise ExportError(
                f"{where}: producer stage mismatch for {ref!r}: {producer!r} != {expected!r}"
            )


def _validate_case_study_payload(
    payload: dict[str, Any],
    sample_trace: dict[str, Any],
    *,
    selector: str,
) -> None:
    """Validate one selector's source-first semantic trace.

    The report is public safe projection, but it must still retain enough
    producer-owned lineage to prove that every displayed stage came from the
    same frozen sample/checkpoint and that no representation name is dangling.
    """

    if payload.get("schema_version") != "q-attention.Q-EPVG-case-study.v2":
        raise ExportError(
            f"case_study/{selector}.json: unsupported schema_version; "
            "new reports require q-attention.Q-EPVG-case-study.v2"
        )
    if payload.get("lineage_schema_version") != "q-attention.case-study-lineage.v1":
        raise ExportError(f"case_study/{selector}.json: missing producer-owned lineage schema")
    if payload.get("selector") != selector:
        raise ExportError(f"case_study/{selector}.json: selector identity mismatch")
    if payload.get("status") != "observed":
        raise ExportError(f"case_study/{selector}.json: status must be observed")

    manifests = payload.get("tensor_manifest")
    if not isinstance(manifests, list) or not manifests:
        raise ExportError(f"case_study/{selector}.json: tensor_manifest is empty")
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for index, manifest in enumerate(manifests):
        manifest_id = _validate_manifest(manifest, where=f"case_study/{selector}.tensor_manifest[{index}]")
        if manifest_id in manifest_by_id:
            raise ExportError(f"case_study/{selector}.json: duplicate tensor manifest_id {manifest_id!r}")
        manifest_by_id[manifest_id] = manifest  # type: ignore[assignment]
    inventory = payload.get("manifest_inventory")
    if not isinstance(inventory, list) or set(inventory) != set(manifest_by_id):
        raise ExportError(f"case_study/{selector}.json: manifest_inventory does not match tensor_manifest")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ExportError(f"case_study/{selector}.json: cases is empty")
    case_ids: set[str] = set()
    observed_splits: set[str] = set()
    observed_checkpoints: set[str] = set()
    for index, case in enumerate(cases):
        where = f"case_study/{selector}.cases[{index}]"
        if not isinstance(case, dict):
            raise ExportError(f"{where}: case must be an object")
        for key in ("case_id", "split", "checkpoint", "sentence", "tokens", "token_ids", "subject", "object", "gold_relation", "representations"):
            if key not in case:
                raise ExportError(f"{where}: missing {key}")
        case_id = case["case_id"]
        split = case["split"]
        checkpoint = case["checkpoint"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ExportError(f"{where}: case_id must be unique and non-empty")
        if split not in _CASE_SPLITS:
            raise ExportError(f"{where}: invalid split {split!r}")
        if checkpoint not in _CASE_CHECKPOINTS:
            raise ExportError(f"{where}: invalid checkpoint {checkpoint!r}")
        if not isinstance(case["sentence"], str) or not case["sentence"].strip():
            raise ExportError(f"{where}: sentence is empty")
        if not isinstance(case["tokens"], list) or not case["tokens"] or not all(isinstance(item, str) for item in case["tokens"]):
            raise ExportError(f"{where}: tokens must be a non-empty string list")
        if not isinstance(case["token_ids"], list) or len(case["token_ids"]) != len(case["tokens"]):
            raise ExportError(f"{where}: token_ids must align with tokens")
        for entity_name in ("subject", "object"):
            entity = case[entity_name]
            if not isinstance(entity, dict) or not isinstance(entity.get("span"), list) or not isinstance(entity.get("token_positions"), list):
                raise ExportError(f"{where}: {entity_name} span/token_positions are required")
        representations = case["representations"]
        if not isinstance(representations, dict) or not representations:
            raise ExportError(f"{where}: representations is empty")
        case_manifest_ids: set[str] = set()
        for rep_id, manifest in representations.items():
            if not isinstance(rep_id, str) or not isinstance(manifest, dict):
                raise ExportError(f"{where}: invalid representation {rep_id!r}")
            manifest_id = manifest.get("manifest_id")
            if not isinstance(manifest_id, str) or manifest_id not in manifest_by_id:
                raise ExportError(f"{where}: representation {rep_id!r} has dangling manifest_id")
            if manifest.get("id") != rep_id:
                raise ExportError(f"{where}: representation id mismatch for {rep_id!r}")
            canonical_manifest = manifest_by_id[manifest_id]
            for key in ("path", "shape", "dtype", "axis_semantics", "sha256", "byte_count", "producer_stage"):
                if manifest.get(key) != canonical_manifest.get(key):
                    raise ExportError(f"{where}: representation {rep_id!r} disagrees with tensor_manifest.{key}")
            case_manifest_ids.add(manifest_id)
        stages = case.get("stages")
        if not isinstance(stages, list):
            raise ExportError(f"{where}: stages must be a list")
        stage_names = [stage.get("stage") for stage in stages if isinstance(stage, dict)]
        if len(stage_names) != len(set(stage_names)):
            raise ExportError(f"{where}: duplicate stage names")
        missing_stages = [name for name in _CASE_REQUIRED_STAGES if name not in stage_names]
        if missing_stages:
            raise ExportError(f"{where}: missing stages {', '.join(missing_stages)}")
        stage_order = list(_CASE_REQUIRED_STAGES[:4]) + ["training", "retrieval", "attention_baseline", "scoring", "selection", "attention_intervention", "context", "classifier", "generation", "evaluation", "diagnosis"]
        positions = [stage_order.index(name) for name in stage_names if name in stage_order]
        if positions != sorted(positions):
            raise ExportError(f"{where}: stages are not in producer order")
        for stage_index, stage in enumerate(stages):
            stage_where = f"{where}.stages[{stage_index}]"
            if not isinstance(stage, dict) or not isinstance(stage.get("stage"), str):
                raise ExportError(f"{stage_where}: invalid stage")
            status = stage.get("status")
            if status not in {"observed", "not_applicable", "failed"}:
                raise ExportError(f"{stage_where}: invalid status {status!r}")
            if stage["stage"] in _CASE_REQUIRED_STAGES and status != "observed":
                raise ExportError(f"{stage_where}: required stage is not fully observed ({status})")
            if stage["stage"] not in _CASE_REQUIRED_STAGES and stage["stage"] not in _CASE_OPTIONAL_STAGES:
                raise ExportError(f"{stage_where}: unknown stage name {stage['stage']!r}")
            if status == "not_applicable" and stage["stage"] not in _CASE_OPTIONAL_STAGES:
                raise ExportError(f"{stage_where}: only optional stages may be not_applicable")
            for direction in ("input_refs", "output_refs"):
                refs = stage.get(direction)
                if not isinstance(refs, list):
                    raise ExportError(f"{stage_where}: {direction} must be a list")
                for ref_index, ref in enumerate(refs):
                    _validate_stage_ref(ref, where=f"{stage_where}.{direction}[{ref_index}]", case_manifest_ids=case_manifest_ids, manifest_by_id=manifest_by_id)
            for direction in ("inputs", "outputs"):
                value = stage.get(direction)
                if not isinstance(value, dict):
                    raise ExportError(f"{stage_where}: {direction} must be an object")
                reps = value.get("representations", [])
                if not isinstance(reps, list):
                    raise ExportError(f"{stage_where}.{direction}.representations must be a list")
                for rep_index, rep in enumerate(reps):
                    if not isinstance(rep, dict):
                        raise ExportError(f"{stage_where}.{direction}.representations[{rep_index}] must be an object")
                    ref = rep.get("manifest_id")
                    if ref not in case_manifest_ids:
                        raise ExportError(f"{stage_where}.{direction}.representations[{rep_index}] has dangling manifest_id")
            if status == "observed" and stage["stage"] in _CASE_REQUIRED_STAGES:
                if not stage.get("output_refs") and stage["stage"] not in {"data", "training"}:
                    raise ExportError(f"{stage_where}: observed stage has no output_refs")
        case_ids.add(case_id)
        observed_splits.add(split)
        observed_checkpoints.add(checkpoint)
    if observed_splits != set(_CASE_SPLITS):
        raise ExportError(f"case_study/{selector}.json: split coverage must include train, valid, and test")
    if observed_checkpoints != set(_CASE_CHECKPOINTS):
        raise ExportError(f"case_study/{selector}.json: checkpoint coverage is incomplete")

    if sample_trace.get("schema_version") != "sample-trace.v1":
        raise ExportError(f"case_study/{selector}.sample-trace.json: unsupported schema_version")
    contract = sample_trace.get("semantic_contract")
    if not isinstance(contract, dict) or contract.get("version") != "q-attention.case-study-trace-contract.v3" or contract.get("lineage") != "producer_owned_stage_input_output":
        raise ExportError(f"case_study/{selector}.sample-trace.json: lineage contract is missing")
    trace_samples = sample_trace.get("samples")
    if not isinstance(trace_samples, list) or {item.get("sample_id") for item in trace_samples if isinstance(item, dict)} != case_ids:
        raise ExportError(f"case_study/{selector}.sample-trace.json: sample IDs do not match case study cases")
    coverage = sample_trace.get("coverage")
    if not isinstance(coverage, dict):
        raise ExportError(f"case_study/{selector}.sample-trace.json: coverage is missing")
    for stage in _CASE_REQUIRED_STAGES:
        if coverage.get(stage) not in {"observed", "not_applicable"}:
            raise ExportError(f"case_study/{selector}.sample-trace.json: coverage for {stage} is incomplete")


def _validate_source_tensor_files(run_dir: Path, selectors: list[str]) -> None:
    """Check each safe manifest against its private source tensor before export."""
    for selector in selectors:
        case_path = run_dir / "selectors" / selector / "case_study.json"
        payload = _read_json(case_path, description=f"source case study for {selector}")
        manifests = payload.get("tensor_manifest")
        if not isinstance(manifests, list):
            raise ExportError(f"source case study for {selector}: tensor_manifest is missing")
        for index, manifest in enumerate(manifests):
            manifest_id = _validate_manifest(manifest, where=f"source case_study/{selector}.tensor_manifest[{index}]")
            relative = Path(str(manifest["path"]))
            source = (run_dir / "selectors" / selector / relative).resolve()
            selector_root = (run_dir / "selectors" / selector).resolve()
            try:
                source.relative_to(selector_root)
            except ValueError as exc:
                raise ExportError(f"source case study for {selector}: tensor path escapes selector directory: {manifest_id}") from exc
            if not source.is_file():
                raise ExportError(f"source case study for {selector}: missing tensor artifact {relative}")
            raw = source.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != manifest["byte_count"] or digest != manifest["sha256"]:
                raise ExportError(f"source case study for {selector}: tensor checksum mismatch for {manifest_id}")


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise ExportError(f"missing required source file: {path}")
    return path


def _copy(source: Path, destination: Path) -> None:
    _required_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _selector_names(config_path: Path) -> list[str]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - the preflight owns schema errors
        raise ExportError(f"invalid config: {config_path}: {exc}") from exc
    selectors = config.get("selectors")
    if not isinstance(selectors, list) or not selectors or selectors[0] != "disabled":
        raise ExportError("config selectors must start with disabled")
    result: list[str] = []
    for selector in selectors[1:]:
        if not isinstance(selector, str) or not selector or Path(selector).name != selector:
            raise ExportError(f"unsafe selector name: {selector!r}")
        result.append(selector)
    return result


def _write_data_identity(run_dir: Path, stage_dir: Path) -> None:
    counts: list[str] = []
    hashes: list[str] = []
    for split in ("train", "valid", "test"):
        source = _required_file(run_dir / "data" / f"{split}.jsonl")
        raw = source.read_bytes()
        counts.append(f"{source} {raw.count(bytes((10,)))}")
        hashes.append(f"{hashlib.sha256(raw).hexdigest()}  {source}")
    (stage_dir / "data_counts.txt").write_text("\n".join(counts) + "\n", encoding="utf-8")
    (stage_dir / "data.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")


def _attempt_state_path(report_dir: Path) -> Path:
    key = hashlib.sha256(str(report_dir).encode("utf-8")).hexdigest()
    state_dir = Path(tempfile.gettempdir()) / "q-epvg-report-export-state"
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"cannot create exporter attempt journal directory: {state_dir}") from exc
    return state_dir / f"{key}.json"


def _load_attempts(
    path: Path,
    *,
    run_dir: Path,
    report_dir: Path,
    config_sha256: str,
    reporting_commit: str,
) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExportError(f"invalid exporter attempt journal: {path}") from exc
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ExportError(f"invalid exporter attempt journal shape: {path}")
    expected = {
        "source_run": str(run_dir),
        "report_identity": str(report_dir),
        "config_sha256": config_sha256,
        "exporter_revision": reporting_commit,
    }
    for item in payload:
        for key, value in expected.items():
            if item.get(key) != value:
                raise ExportError(
                    f"exporter attempt journal identity mismatch for {key}: {path}"
                )
    return payload


def _save_attempts(path: Path, attempts: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(attempts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError as exc:
        raise ExportError(f"cannot persist exporter attempt journal: {path}") from exc


def _cleanup_staging(report_dir: Path) -> list[str]:
    cleaned: list[str] = []
    for path in sorted(report_dir.parent.glob(f".{report_dir.name}.staging-*")):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise ExportError(f"cannot remove stale exporter staging path: {path}") from exc
        cleaned.append(str(path))
    return cleaned


def _source_run_revision(run_dir: Path) -> str | None:
    summary = run_dir / "run_summary.json"
    if not summary.is_file():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("git_commit", "code_revision", "implementation_revision"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _write_export_manifest(
    stage_dir: Path,
    *,
    run_dir: Path,
    report_dir: Path,
    reporting_commit: str,
    config_sha256: str,
    attempts: list[dict[str, object]],
    stale_staging_cleaned: list[str],
) -> None:
    failures = [item for item in attempts if item.get("status") == "failed"]
    last_failure = failures[-1].get("failure_reason") if failures else None
    payload = {
        "schema_version": "q-attention.report-export.v1",
        "status": "complete",
        "source_run": str(run_dir),
        "source_run_revision": _source_run_revision(run_dir),
        "exporter_revision": reporting_commit,
        "config_sha256": config_sha256,
        "report_identity": str(report_dir),
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "failure_reason": last_failure,
        "failures": failures,
        "stale_staging_cleaned": stale_staging_cleaned,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (stage_dir / "export_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_stage(stage_dir: Path, selectors: list[str]) -> None:
    required = [
        "RUN_COMPLETE",
        "run_summary.json",
        "run_summary.data",
        "run_summary.md",
        "gpu_assignments.json",
        "run_config.json",
        "reporting_commit.txt",
        "data_counts.txt",
        "data.sha256",
        "export_manifest.json",
        "metrics/baseline.json",
    ]
    for selector in selectors:
        required.extend(
            (
                f"metrics/{selector}.json",
                f"case_study/{selector}.json",
                f"case_study/{selector}.sample-trace.json",
            )
        )
    for relative in required:
        path = stage_dir / relative
        if not path.is_file():
            raise ExportError(f"staged report is missing {relative}")
        if path.stat().st_size == 0 and relative != "RUN_COMPLETE":
            raise ExportError(f"staged report contains an empty file: {relative}")
    forbidden_suffixes = (".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".jsonl")
    forbidden = [
        path.relative_to(stage_dir)
        for path in stage_dir.rglob("*")
        if path.is_file() and path.suffix in forbidden_suffixes
    ]
    if forbidden:
        raise ExportError(f"forbidden private artifacts in staged report: {forbidden}")
    # The public projection is intentionally JSON-only, but the JSON must
    # carry a complete, source-first trace.  Validate this after all files are
    # staged so a malformed selector can never be published as an apparently
    # complete report.
    for selector in selectors:
        case_path = stage_dir / "case_study" / f"{selector}.json"
        trace_path = stage_dir / "case_study" / f"{selector}.sample-trace.json"
        case_payload = _read_json(case_path, description=f"case study for {selector}")
        trace_payload = _read_json(trace_path, description=f"sample trace for {selector}")
        _validate_case_study_payload(case_payload, trace_payload, selector=selector)


def export_report(
    *,
    run_dir: Path,
    report_dir: Path,
    config_path: Path,
    reporting_commit: str,
    inject_failure_after: int | None = None,
    inject_validation_failure: bool = False,
) -> Path:
    """Stage, validate, and atomically publish a report from one completed run."""

    run_dir = run_dir.resolve()
    report_dir = report_dir.resolve()
    config_path = config_path.resolve()
    selectors = _selector_names(config_path)
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    if report_dir.exists():
        raise ExportError(f"refusing to overwrite report directory: {report_dir}")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    _validate_source_tensor_files(run_dir, selectors)
    stale_staging_cleaned = _cleanup_staging(report_dir)

    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{report_dir.name}.staging-", dir=report_dir.parent)
    )
    state_path = _attempt_state_path(report_dir)
    attempts = _load_attempts(
        state_path,
        run_dir=run_dir,
        report_dir=report_dir,
        config_sha256=config_sha256,
        reporting_commit=reporting_commit,
    )
    attempt = {
        "attempt": len(attempts) + 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "report_identity": str(report_dir),
        "config_sha256": config_sha256,
        "exporter_revision": reporting_commit,
    }
    attempts.append(attempt)
    try:
        _save_attempts(state_path, attempts)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    copy_count = 0

    def copy_one(source: Path, destination: Path) -> None:
        nonlocal copy_count
        _copy(source, stage_dir / destination)
        copy_count += 1
        if inject_failure_after is not None and copy_count >= inject_failure_after:
            raise ExportError(f"injected copy failure after {copy_count} files")

    try:
        for name in (
            "RUN_COMPLETE",
            "run_summary.json",
            "run_summary.data",
            "run_summary.md",
            "gpu_assignments.json",
        ):
            copy_one(run_dir / name, Path(name))
        copy_one(config_path, Path("run_config.json"))
        copy_one(run_dir / "baseline" / "metrics.json", Path("metrics/baseline.json"))
        for selector in selectors:
            source = run_dir / "selectors" / selector
            copy_one(source / "metrics.json", Path("metrics") / f"{selector}.json")
            copy_one(source / "case_study.json", Path("case_study") / f"{selector}.json")
            copy_one(
                source / "sample_trace.json",
                Path("case_study") / f"{selector}.sample-trace.json",
            )
        (stage_dir / "reporting_commit.txt").write_text(
            reporting_commit + "\n", encoding="utf-8"
        )
        _write_data_identity(run_dir, stage_dir)
        if inject_validation_failure:
            raise ExportError("injected report-validation failure")
        _write_export_manifest(
            stage_dir,
            run_dir=run_dir,
            report_dir=report_dir,
            reporting_commit=reporting_commit,
            config_sha256=config_sha256,
            attempts=attempts,
            stale_staging_cleaned=stale_staging_cleaned,
        )
        _validate_stage(stage_dir, selectors)
        if report_dir.exists():
            raise ExportError(f"refusing to overwrite report directory: {report_dir}")
        stage_dir.rename(report_dir)
        attempt["status"] = "complete"
        attempt["completed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            state_path.unlink(missing_ok=True)
        except OSError:
            pass
        return report_dir
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["failed_at"] = datetime.now(timezone.utc).isoformat()
        attempt["failure_reason"] = str(exc)
        try:
            _save_attempts(state_path, attempts)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reporting-commit", required=True)
    parser.add_argument("--inject-failure-after", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--inject-validation-failure", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        export_report(
            run_dir=args.run_dir,
            report_dir=args.report_dir,
            config_path=args.config,
            reporting_commit=args.reporting_commit,
            inject_failure_after=args.inject_failure_after,
            inject_validation_failure=args.inject_validation_failure,
        )
    except ExportError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"REPORT_DIR={args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
