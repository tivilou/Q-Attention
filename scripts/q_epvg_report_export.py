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
from pathlib import Path


class ExportError(RuntimeError):
    """Raised when a report cannot be staged or validated."""


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
