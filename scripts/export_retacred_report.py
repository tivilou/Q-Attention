#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_EXPORTS = (
    ("pipeline_summary.json", "pipeline_summary.json"),
    ("run_summary.json", "run_summary.json"),
    ("run_summary.md", "run_summary.md"),
    ("baseline/metrics.json", "baseline_metrics.json"),
    ("classical_steering_eval/metrics.json", "classical_steering_metrics.json"),
    ("quantum_steering_eval/metrics.json", "quantum_steering_metrics.json"),
    ("spectral_filter_sweep/summary.json", "spectral_filter_summary.json"),
    ("relation_routing_eval/metrics.json", "routing_metrics.json"),
)
CONFIG_NAMES = ("retacred_full_gpu.json", "retacred_low_resource_gpu.json")
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExportError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExportError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_within(path: Path, root: Path, label: str, *, directory: bool) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ExportError(f"{label} does not exist: {path}") from exc
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ExportError(f"{label} must be inside {resolved_root}: {resolved}") from exc
    if directory and not resolved.is_dir():
        raise ExportError(f"{label} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise ExportError(f"{label} is not a file: {resolved}")
    return resolved


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ExportError(f"Git command failed: git {' '.join(args)}: {detail}")
    return result.stdout.strip()


def _validate_pipeline(run_dir: Path, config_path: Path, expected_commit: str, label: str) -> None:
    summary = _load_json(run_dir / "pipeline_summary.json")
    reproducibility = summary.get("reproducibility")
    if not isinstance(reproducibility, dict):
        raise ExportError(f"{label}: missing reproducibility metadata")

    git_info = reproducibility.get("git")
    if not isinstance(git_info, dict):
        raise ExportError(f"{label}: missing Git provenance")
    if git_info.get("dirty") is not False:
        raise ExportError(f"{label}: pipeline_summary.json must record git.dirty=false")
    if git_info.get("commit") != expected_commit:
        raise ExportError(
            f"{label}: run commit {git_info.get('commit')!r} does not match current HEAD {expected_commit}"
        )
    if reproducibility.get("cuda_available") is not True or summary.get("device") != "cuda":
        raise ExportError(f"{label}: only completed CUDA runs can be exported")
    if reproducibility.get("config_sha256") != _sha256(config_path):
        raise ExportError(f"{label}: current config hash does not match the config used by the run")

    recorded_output = summary.get("output_dir")
    if not isinstance(recorded_output, str) or Path(recorded_output).resolve() != run_dir:
        raise ExportError(f"{label}: pipeline output_dir does not match the selected run directory")

    protocol = summary.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ExportError(f"{label}: missing evaluation protocol")
    if protocol.get("test_isolated") is not True or protocol.get("final_split") != "test":
        raise ExportError(f"{label}: held-out test protocol is not valid")

    commands = summary.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ExportError(f"{label}: pipeline command records are missing")
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            raise ExportError(f"{label}: command record {index} is invalid")
        if command.get("dry_run") is not False or command.get("returncode") != 0:
            raise ExportError(f"{label}: pipeline command {index} did not complete successfully")


def _collect_run_files(run_dir: Path, destination: str) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for source_name, destination_name in RUN_EXPORTS:
        source = run_dir / source_name
        if not source.is_file():
            raise ExportError(f"Missing required run artifact: {source}")
        if source.suffix == ".json":
            _load_json(source)
        files.append((source, Path(destination) / destination_name))
    return files


def _write_log_tail(source: Path, destination: Path, tail_lines: int) -> None:
    lines: deque[bytes] = deque(maxlen=tail_lines)
    with source.open("rb") as handle:
        for line in handle:
            lines.append(line)
    if not lines:
        raise ExportError(f"Log file is empty: {source}")
    destination.write_bytes(b"".join(lines))


def export_report(
    repo_root: Path,
    full_run: Path,
    low_resource_run: Path,
    full_log: Path,
    low_resource_log: Path,
    report_tag: str,
    tail_lines: int = 1000,
) -> Path:
    repo_root = repo_root.resolve(strict=True)
    if not TAG_PATTERN.fullmatch(report_tag):
        raise ExportError(
            "report tag must contain only letters, numbers, dots, underscores, and hyphens"
        )
    if tail_lines <= 0:
        raise ExportError("tail-lines must be positive")

    status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ExportError(
            "worktree is not clean; move or restore local files before exporting:\n" + status
        )
    head = _git_output(repo_root, "rev-parse", "HEAD")

    full_run = _require_within(
        full_run, repo_root / "runs" / "retacred_full_gpu", "full run directory", directory=True
    )
    low_resource_run = _require_within(
        low_resource_run,
        repo_root / "runs" / "retacred_low_resource_gpu",
        "low-resource run directory",
        directory=True,
    )
    full_log = _require_within(
        full_log, repo_root / "runs" / "handoff_logs", "full log", directory=False
    )
    low_resource_log = _require_within(
        low_resource_log,
        repo_root / "runs" / "handoff_logs",
        "low-resource log",
        directory=False,
    )

    configs = {name: repo_root / "configs" / name for name in CONFIG_NAMES}
    for config_path in configs.values():
        if not config_path.is_file():
            raise ExportError(f"Missing config: {config_path}")
        _load_json(config_path)

    _validate_pipeline(full_run, configs["retacred_full_gpu.json"], head, "full")
    _validate_pipeline(
        low_resource_run, configs["retacred_low_resource_gpu.json"], head, "low-resource"
    )
    copies = [
        *((path, Path("configs") / path.name) for path in configs.values()),
        *_collect_run_files(full_run, "full"),
        *_collect_run_files(low_resource_run, "low_resource"),
    ]

    output_root = repo_root / "reports" / "retacred"
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / report_tag
    if destination.exists():
        raise ExportError(f"Report directory already exists: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{report_tag}.", dir=output_root))
    try:
        for subdirectory in ("configs", "full", "low_resource", "logs"):
            (staging / subdirectory).mkdir(parents=True, exist_ok=True)
        for source, relative_destination in copies:
            shutil.copy2(source, staging / relative_destination)
        _write_log_tail(full_log, staging / "logs" / "retacred_full_gpu.tail.txt", tail_lines)
        _write_log_tail(
            low_resource_log,
            staging / "logs" / "retacred_low_resource_gpu.tail.txt",
            tail_lines,
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    exported_files = [path for path in destination.rglob("*") if path.is_file()]
    if len(exported_files) != 20:
        raise ExportError(f"Internal error: expected 20 exported files, found {len(exported_files)}")
    return destination


def _resolve_argument(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and export public Re-TACRED report artifacts."
    )
    parser.add_argument("--full-run", required=True, type=Path)
    parser.add_argument("--low-resource-run", required=True, type=Path)
    parser.add_argument("--full-log", required=True, type=Path)
    parser.add_argument("--low-resource-log", required=True, type=Path)
    parser.add_argument("--report-tag", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--tail-lines", type=int, default=1000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    try:
        destination = export_report(
            repo_root=repo_root,
            full_run=_resolve_argument(repo_root, args.full_run),
            low_resource_run=_resolve_argument(repo_root, args.low_resource_run),
            full_log=_resolve_argument(repo_root, args.full_log),
            low_resource_log=_resolve_argument(repo_root, args.low_resource_log),
            report_tag=args.report_tag,
            tail_lines=args.tail_lines,
        )
    except ExportError as exc:
        parser.error(str(exc))

    print(f"Report exported: {destination.relative_to(repo_root)}")
    print("Files exported: 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
