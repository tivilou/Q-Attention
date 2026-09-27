from __future__ import annotations

"""Export an audited Q-EPVG multi-seed report-only package.

The exporter validates the completed task graph and every selector's semantic
Case Study before copying a whitelist of safe JSON/Markdown/metric artifacts.
Tensor binaries and raw data remain outside the report package.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from summarize_retacred_q_epvg_formal_multi_seed import (
    EXPECTED_SEEDS,
    _validate_case_study,
    collect,
    load_json,
    sha256,
)


class ExportError(RuntimeError):
    pass


def _attempt_state_path(report_dir: Path) -> Path:
    key = hashlib.sha256(str(report_dir).encode("utf-8")).hexdigest()
    root = Path(tempfile.gettempdir()) / "q-epvg-multi-seed-export-state"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.json"


def _load_attempts(path: Path, run_dir: Path, report_dir: Path, revision: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExportError(f"invalid export attempt journal: {path}") from exc
    if not isinstance(value, list):
        raise ExportError(f"invalid export attempt journal: {path}")
    for item in value:
        if (
            item.get("source_group") != str(run_dir)
            or item.get("report_identity") != str(report_dir)
            or item.get("exporter_revision") != revision
        ):
            raise ExportError("export attempt journal identity mismatch")
    return value


def _save_attempts(path: Path, attempts: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(attempts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise ExportError(f"missing or empty source artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_data_identity(seed_dir: Path, destination: Path) -> None:
    for name in ("data_counts.txt", "data.sha256"):
        _copy(seed_dir / name, destination / name)


def _validate_stage(stage_dir: Path, selectors: list[str], configs: dict[int, dict[str, Any]]) -> None:
    required = [
        "MULTI_SEED_COMPLETE",
        "multi_seed_manifest.json",
        "multi_seed_status.json",
        "multi_seed_run_summary.json",
        "multi_seed_summary.json",
        "multi_seed_summary.md",
        "reporting_commit.txt",
    ]
    for relative in required:
        if not (stage_dir / relative).is_file() or (stage_dir / relative).stat().st_size == 0:
            raise ExportError(f"report is missing {relative}")
    for seed in EXPECTED_SEEDS:
        seed_dir = stage_dir / "seeds" / f"seed_{seed}"
        for relative in (
            "RUN_COMPLETE",
            "run_summary.json",
            "run_summary.data",
            "run_summary.md",
            "run_config.json",
            "data_counts.txt",
            "data.sha256",
            "metrics/baseline.json",
            "provenance.json",
        ):
            path = seed_dir / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ExportError(f"report is missing seed {seed}/{relative}")
        for selector in selectors[1:]:
            for relative in (
                f"metrics/{selector}.json",
                f"case_study/{selector}.json",
                f"case_study/{selector}.sample-trace.json",
            ):
                path = seed_dir / relative
                if not path.is_file() or path.stat().st_size == 0:
                    raise ExportError(f"report is missing seed {seed}/{relative}")
            with tempfile.TemporaryDirectory(prefix="q-epvg-case-validate-") as temp:
                selector_dir = Path(temp)
                (selector_dir / "case_study.json").write_bytes(
                    (seed_dir / "case_study" / f"{selector}.json").read_bytes()
                )
                (selector_dir / "sample_trace.json").write_bytes(
                    (seed_dir / "case_study" / f"{selector}.sample-trace.json").read_bytes()
                )
                _validate_case_study(
                    selector_dir,
                    selector=selector,
                    config=configs[seed],
                    config_sha=sha256(seed_dir / "run_config.json"),
                    allow_missing_tensor_files=True,
                )
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".jsonl", ".log"}
    forbidden = [
        path.relative_to(stage_dir)
        for path in stage_dir.rglob("*")
        if path.is_file() and path.suffix in forbidden_suffixes
    ]
    if forbidden:
        raise ExportError(f"forbidden private artifacts in report: {forbidden}")


def export_report(
    *,
    group_dir: Path,
    report_dir: Path,
    reporting_commit: str,
    inject_failure_after: int | None = None,
    inject_validation_failure: bool = False,
) -> Path:
    group_dir = group_dir.resolve()
    report_dir = report_dir.resolve()
    if report_dir.exists():
        raise ExportError(f"refusing to overwrite report directory: {report_dir}")
    manifest = load_json(group_dir / "multi_seed_manifest.json")
    selectors = [str(item) for item in manifest.get("selectors", [])]
    if selectors[:1] != ["disabled"] or len(selectors) < 2:
        raise ExportError("invalid selector list in multi_seed_manifest.json")
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.staging-", dir=report_dir.parent))
    state_path = _attempt_state_path(report_dir)
    attempts = _load_attempts(state_path, group_dir, report_dir, reporting_commit)
    attempt: dict[str, Any] = {
        "attempt": len(attempts) + 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_group": str(group_dir),
        "report_identity": str(report_dir),
        "exporter_revision": reporting_commit,
    }
    attempts.append(attempt)
    _save_attempts(state_path, attempts)
    copy_count = 0

    def copy_one(source: Path, destination: Path) -> None:
        nonlocal copy_count
        _copy(source, stage_dir / destination)
        copy_count += 1
        if inject_failure_after is not None and copy_count >= inject_failure_after:
            raise ExportError(f"injected copy failure after {copy_count} files")

    try:
        # Validate the immutable source only after the attempt journal exists,
        # so collection/validation failures are retryable and auditable too.
        payload = collect(group_dir)
        configs = {
            seed: load_json(group_dir / f"seed_{seed}" / "run_config.json")
            for seed in EXPECTED_SEEDS
        }
        for name in (
            "MULTI_SEED_COMPLETE",
            "multi_seed_manifest.json",
            "multi_seed_status.json",
            "multi_seed_run_summary.json",
            "multi_seed_summary.json",
            "multi_seed_summary.md",
        ):
            copy_one(group_dir / name, Path(name))
        (stage_dir / "reporting_commit.txt").write_text(reporting_commit + "\n", encoding="utf-8")
        for seed in EXPECTED_SEEDS:
            source = group_dir / f"seed_{seed}"
            destination = Path("seeds") / f"seed_{seed}"
            for name in ("RUN_COMPLETE", "run_summary.json", "run_summary.data", "run_summary.md", "run_config.json", "data_counts.txt", "data.sha256"):
                copy_one(source / name, destination / name)
            provenance = load_json(source / "run_summary.json").get("provenance")
            if not isinstance(provenance, dict):
                raise ExportError(f"seed {seed} run summary lacks provenance")
            (stage_dir / destination / "provenance.json").parent.mkdir(parents=True, exist_ok=True)
            (stage_dir / destination / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            copy_one(source / "baseline" / "metrics.json", destination / "metrics" / "baseline.json")
            for selector in selectors[1:]:
                copy_one(source / "selectors" / selector / "metrics.json", destination / "metrics" / f"{selector}.json")
                copy_one(source / "selectors" / selector / "case_study.json", destination / "case_study" / f"{selector}.json")
                copy_one(source / "selectors" / selector / "sample_trace.json", destination / "case_study" / f"{selector}.sample-trace.json")
            if (source / "imported_report.json").is_file():
                copy_one(source / "imported_report.json", destination / "imported_report.json")
        if inject_validation_failure:
            raise ExportError("injected report-validation failure")
        manifest_payload = {
            "schema_version": "q-attention.q-epvg.multi-seed-report-export.v1",
            "status": "complete",
            "source_group": str(group_dir),
            "report_identity": str(report_dir),
            "reporting_commit": reporting_commit,
            "summary_schema_version": payload["schema_version"],
            "selectors": selectors,
            "seeds": list(EXPECTED_SEEDS),
            "attempt_count": len(attempts),
            "retry_count": max(0, len(attempts) - 1),
            "failure_reason": next((item.get("failure_reason") for item in reversed(attempts) if item.get("status") == "failed"), None),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (stage_dir / "export_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _validate_stage(stage_dir, selectors, configs)
        stage_dir.rename(report_dir)
        attempt["status"] = "complete"
        attempt["completed_at"] = datetime.now(timezone.utc).isoformat()
        _save_attempts(state_path, attempts)
        state_path.unlink(missing_ok=True)
        return report_dir
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["failed_at"] = datetime.now(timezone.utc).isoformat()
        attempt["failure_reason"] = str(exc)
        _save_attempts(state_path, attempts)
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--reporting-commit", required=True)
    parser.add_argument("--inject-failure-after", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--inject-validation-failure", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        export_report(
            group_dir=args.group_dir,
            report_dir=args.report_dir,
            reporting_commit=args.reporting_commit,
            inject_failure_after=args.inject_failure_after,
            inject_validation_failure=args.inject_validation_failure,
        )
    except (ExportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"REPORT_DIR={args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
