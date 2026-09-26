#!/usr/bin/env python3
"""Read-only preflight and report-artifact validator for Q-EPVG formal runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_CASE_SCHEMA = "q-attention.Q-EPVG-case-study.v1"


def _load(path: Path, errors: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON object expected: {path}")
        return {}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/retacred_q_epvg_formal_single_seed.json"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--fresh-run", action="store_true", help="validate only the frozen config")
    args = parser.parse_args()
    errors: list[str] = []
    config: dict = {}
    if not args.config.is_file():
        errors.append(f"missing config: {args.config}")
    else:
        config = _load(args.config, errors)
        if config.get("schema_version") != "q-attention.q-epvg-formal-single-seed.v1":
            errors.append("unsupported Q-EPVG config schema")
        selectors = config.get("selectors")
        if not isinstance(selectors, list) or selectors[:1] != ["disabled"] or len(selectors) != 28:
            errors.append("selectors must contain disabled plus the frozen 27 Q-EPVG variants")
        for key, expected in (
            ("candidate", "q_epvg_zz_xx_value_only_quantum"),
            ("matched_control", "q_epvg_zz_xx_value_only_classical"),
            ("structural_control", "q_epvg_zz_xx_value_only_random_parity"),
        ):
            if config.get(key) != expected:
                errors.append(f"{key} must be {expected}")
        case = config.get("case_study", {})
        if set(case.get("records", {})) != {"train", "valid", "test"}:
            errors.append("case_study must freeze train/valid/test records")
        if case.get("checkpoints") != ["initial_or_pre_training", "best_valid_or_declared_selection_checkpoint", "final"]:
            errors.append("case_study must freeze the three declared checkpoints")
        if config.get("expected_records") != {"train": 58465, "valid": 19584, "test": 13418}:
            errors.append("unexpected Re-TACRED record-count contract")
    if args.run_dir is not None and not args.fresh_run:
        run = args.run_dir
        for name in ("RUN_COMPLETE", "run_summary.json", "run_summary.data", "run_summary.md", "run_config.json", "gpu_assignments.json", "baseline/metrics.json"):
            if not (run / name).is_file():
                errors.append(f"missing {name}")
        if (run / "run_summary.json").is_file():
            summary = _load(run / "run_summary.json", errors)
            if summary.get("schema_version") != "q-attention.q-epvg-formal-single-seed.run.v1":
                errors.append("wrong run summary schema")
            selectors = config.get("selectors", [])
            if summary.get("selectors") != selectors:
                errors.append("run summary selector list mismatch")
            if summary.get("seed") != 13 or summary.get("formal_experiment") is not True:
                errors.append("run summary is not the frozen seed-13 formal run")
            if summary.get("test_used_for_training_or_selection") is not False:
                errors.append("test leakage flag is not false")
            data = summary.get("data")
            if not isinstance(data, dict) or set(data) != {"train", "valid", "test"}:
                errors.append("run summary data provenance is incomplete")
        summary_data = _load(run / "run_summary.data", errors)
        if summary_data.get("schema_version") != "q-attention.run-summary-data.v1":
            errors.append("wrong run_summary.data schema")
        for selector in config.get("selectors", [])[1:]:
            directory = run / "selectors" / selector
            for name in ("metrics.json", "case_study.json", "sample_trace.json"):
                if not (directory / name).is_file():
                    errors.append(f"missing {selector}/{name}")
            if (directory / "case_study.json").is_file():
                payload = _load(directory / "case_study.json", errors)
                if payload.get("schema_version") != EXPECTED_CASE_SCHEMA:
                    errors.append(f"{selector}: wrong case-study schema")
                if payload.get("required_splits") != ["train", "valid", "test"]:
                    errors.append(f"{selector}: incomplete split coverage")
                if len(payload.get("cases", [])) < 27:
                    errors.append(f"{selector}: expected at least 27 split/checkpoint cases")
                manifests = payload.get("tensor_manifest", [])
                if not isinstance(manifests, list) or not manifests:
                    errors.append(f"{selector}: tensor manifest is missing")
                else:
                    for item in manifests:
                        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                            errors.append(f"{selector}: tensor manifest entry is incomplete")
            if (directory / "sample_trace.json").is_file():
                trace = _load(directory / "sample_trace.json", errors)
                if trace.get("schema_version") != "sample-trace.v1":
                    errors.append(f"{selector}: wrong sample-trace schema")
    print("Q-EPVG formal single-seed preflight (READ ONLY)")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
