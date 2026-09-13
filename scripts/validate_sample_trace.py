#!/usr/bin/env python3
"""Validate the portal-safe sample-trace.v1 contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STAGES = (
    "data", "preprocess", "training", "retrieval", "scoring",
    "selection", "context", "generation", "evaluation", "diagnosis",
)
STATUSES = {"observed", "not_applicable", "unavailable", "failed"}


def validate(value: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["trace must be an object"]
    if value.get("schema_version") != "sample-trace.v1":
        errors.append("schema_version must be sample-trace.v1")
    for field in ("trace_id", "experiment", "sample_selection", "coverage", "samples"):
        if field not in value:
            errors.append(f"missing top-level field: {field}")
    selection = value.get("sample_selection")
    if not isinstance(selection, dict):
        errors.append("sample_selection must be an object")
    elif not isinstance(selection.get("selected_count"), int) or selection["selected_count"] < 1:
        errors.append("sample_selection.selected_count must be positive")
    coverage = value.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("coverage must be an object")
    else:
        missing = sorted(set(STAGES) - set(coverage))
        if missing:
            errors.append(f"coverage missing stages: {missing}")
        for stage in STAGES:
            if stage in coverage and coverage[stage] not in STATUSES:
                errors.append(f"coverage.{stage} has invalid status")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty list")
        return errors
    if isinstance(selection, dict) and selection.get("selected_count") != len(samples):
        errors.append("selected_count does not match samples length")
    ids: list[str] = []
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str) or not sample["sample_id"]:
            errors.append(f"{prefix}.sample_id must be a non-empty string")
            continue
        ids.append(sample["sample_id"])
        stages = sample.get("stages")
        if not isinstance(stages, list) or not stages:
            errors.append(f"{prefix}.stages must be non-empty")
            continue
        seen: set[str] = set()
        for stage in stages:
            if not isinstance(stage, dict) or stage.get("stage") not in STAGES:
                errors.append(f"{prefix} has an invalid stage")
                continue
            name = stage["stage"]
            seen.add(name)
            status = stage.get("status")
            if status not in STATUSES:
                errors.append(f"{prefix}.{name}.status is invalid")
            if status == "observed":
                evidence = stage.get("observed_fields") or stage.get("outputs") or stage.get("summary") or stage.get("artifact_refs")
                if not evidence:
                    errors.append(f"{prefix}.{name} observed stage has no emitted evidence")
            if name == "generation" and status == "observed" and stage.get("target_access") not in {"outside_generation_input", "post_generation_only", "training_only", "not_available"}:
                errors.append(f"{prefix}.generation observed stage requires target_access")
        for name in STAGES:
            if coverage.get(name) == "observed" and name not in seen:
                errors.append(f"{prefix} omits observed coverage stage {name}")
    selected_ids = selection.get("selected_sample_ids") if isinstance(selection, dict) else None
    if selected_ids is not None and selected_ids != ids:
        errors.append("selected_sample_ids does not match sample order")
    if len(set(ids)) != len(ids):
        errors.append("sample IDs must be unique")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    errors = validate(value)
    if args.expected_config_sha256 and isinstance(value, dict):
        experiment = value.get("experiment")
        if not isinstance(experiment, dict) or experiment.get("config_sha256") != args.expected_config_sha256:
            errors.append("experiment.config_sha256 does not match the frozen run config")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
