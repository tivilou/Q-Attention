#!/usr/bin/env python3
"""Export only auditable summaries from a completed frozen multi-seed run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SEEDS = (7, 11, 13, 17, 23)
TOP_LEVEL_FILES = (
    "MULTI_SEED_COMPLETE",
    "multi_seed_manifest.json",
    "multi_seed_execution_summary.json",
    "aggregate_summary.json",
    "aggregate_summary.md",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".jsonl"}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_commit(expected_commit: str) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("repository must be clean before report export")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError(f"run commit {expected_commit} does not match HEAD {head}")


def export(group_dir: Path, report_dir: Path) -> dict[str, Any]:
    runs_root = (ROOT / "runs").resolve()
    reports_root = (ROOT / "reports/q_consensus_quantum_estimator").resolve()
    group_dir = group_dir.resolve()
    report_dir = report_dir.resolve()
    if not group_dir.is_relative_to(runs_root):
        raise ValueError("group directory must be inside runs/")
    if not report_dir.is_relative_to(reports_root):
        raise ValueError("report directory must be inside reports/q_consensus_quantum_estimator/")
    if report_dir.exists():
        raise ValueError(f"refusing to overwrite report directory: {report_dir}")
    for name in TOP_LEVEL_FILES:
        if not (group_dir / name).is_file():
            raise ValueError(f"missing required group file: {name}")
    manifest = load_json(group_dir / "multi_seed_manifest.json")
    aggregate = load_json(group_dir / "aggregate_summary.json")
    if manifest.get("seeds") != list(FROZEN_SEEDS):
        raise ValueError("run does not contain the frozen seed set")
    if aggregate.get("status") != "complete":
        raise ValueError("aggregate summary is incomplete")
    preflight = manifest.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("gate_status") != "pass":
        raise ValueError("manifest is missing a passed single-seed multi-GPU preflight")
    preflight_source = (ROOT / str(preflight.get("path", ""))).resolve()
    if not preflight_source.is_relative_to(runs_root) or not preflight_source.is_file():
        raise ValueError("preflight summary is missing or outside runs/")
    if sha256(preflight_source) != preflight.get("sha256"):
        raise ValueError("preflight summary hash differs from the full-run manifest")
    require_clean_commit(str(manifest["git_commit"]))

    report_dir.mkdir(parents=True)
    copied: list[Path] = []
    for name in TOP_LEVEL_FILES:
        destination = report_dir / name
        shutil.copy2(group_dir / name, destination)
        copied.append(destination)
    preflight_destination = report_dir / "preflight" / "run_summary.json"
    preflight_destination.parent.mkdir()
    shutil.copy2(preflight_source, preflight_destination)
    copied.append(preflight_destination)
    for seed in FROZEN_SEEDS:
        source = group_dir / f"seed_{seed}" / "run_summary.json"
        if not source.is_file() or not (source.parent / "SEED_COMPLETE").is_file():
            raise ValueError(f"missing completed seed summary: {seed}")
        destination = report_dir / f"seed_{seed}" / "run_summary.json"
        destination.parent.mkdir()
        shutil.copy2(source, destination)
        copied.append(destination)

    forbidden = [
        path for path in report_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise ValueError(f"forbidden artifact in report: {forbidden[0]}")
    report_manifest = {
        "schema_version": "q-attention.q-consensus-quantum-estimator-report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_group": str(group_dir.relative_to(ROOT)),
        "git_commit": manifest["git_commit"],
        "scientific_gate": aggregate["gate"]["status"],
        "files": {
            str(path.relative_to(report_dir)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(copied)
        },
        "excluded": [
            "raw runs",
            "checkpoints",
            "predictions",
            "datasets",
            "seed configs",
            "full logs",
            "credentials",
        ],
    }
    (report_dir / "report_manifest.json").write_text(
        json.dumps(report_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-frozen-multiseed")
    report_dir = args.report_dir or ROOT / "reports/q_consensus_quantum_estimator" / stamp
    try:
        payload = export(args.group_dir, report_dir)
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "report_dir": str(report_dir.resolve()),
                "git_commit": payload["git_commit"],
                "scientific_gate": payload["scientific_gate"],
                "files": len(payload["files"]) + 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
