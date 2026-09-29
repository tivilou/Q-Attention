#!/usr/bin/env python3
"""Upload the immutable tensors needed to recover Q-EPVG Case Study context.

The uploader intentionally sends only producer-owned JSON manifests, the
completed multi-seed summary, and the seven source tensors required by the
published recovery tool. It never uploads the whole run directory, weights,
predictions, logs, or raw data.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = PROJECT_ROOT / "scripts" / "collab" / "lib"
sys.path.insert(0, str(LIB_DIR))

from exchange_upload import (  # noqa: E402
    DEFAULT_CA_FILE,
    ExchangeError,
    safe_relative,
    upload_declared_files,
)


DEFAULT_EXPERIMENT = "retacred_q_epvg_formal_multi_seed"
DEFAULT_SELECTOR = "q_epvg_zz_value_only_quantum"
DEFAULT_EXCHANGE_URL = "https://117.50.198.37:18084"
MAX_FILE_BYTES = 1024**3
REQUIRED_SOURCE_REPRESENTATIONS = (
    "q_epvg_query",
    "q_epvg_key",
    "q_epvg_query_update",
    "q_epvg_score_adjustment",
    "q_epvg_attention",
    "steered_attention_scores",
    "q_epvg_routed_values",
)
OPTIONAL_IDENTITY_FILES = ("data.sha256", "data_counts.txt", "run_summary.data")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExchangeError(f"cannot read JSON {path}: {exc}") from exc


def _regular_file(path: Path, *, label: str, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ExchangeError(f"{label} is missing or is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ExchangeError(f"{label} escapes the run directory: {path}") from exc
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise ExchangeError(f"{label} exceeds the 1 GiB upload limit: {path}")
    return resolved


def _manifest_tensor(
    selector_dir: Path,
    case: dict[str, Any],
    representation: str,
    *,
    group_dir: Path,
) -> tuple[Path, str]:
    representations = case.get("representations")
    if not isinstance(representations, dict):
        raise ExchangeError(f"{case.get('case_id')}: representations is missing")
    manifest = representations.get(representation)
    if not isinstance(manifest, dict):
        raise ExchangeError(f"{case.get('case_id')}: missing representation {representation}")
    manifest_id = manifest.get("manifest_id")
    relative = manifest.get("path")
    expected_relative = f"case_study_tensors/{manifest_id}.pt"
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ExchangeError(f"{case.get('case_id')}: {representation} has no manifest_id")
    if relative != expected_relative:
        raise ExchangeError(
            f"{case.get('case_id')}: {representation} path does not match its manifest_id"
        )
    tensor_path = _regular_file(
        selector_dir / relative,
        label=f"{case.get('case_id')} {representation} tensor",
        root=group_dir,
    )
    return tensor_path, relative


def _collect_group(
    group_dir: Path,
    selector: str,
    *,
    repository_root: Path = PROJECT_ROOT,
) -> tuple[str, list[tuple[Path, str]]]:
    if group_dir.is_symlink():
        raise ExchangeError(f"group directory must not be a symlink: {group_dir}")
    try:
        resolved_group = group_dir.resolve(strict=True)
    except OSError as exc:
        raise ExchangeError(f"group directory does not exist: {group_dir}") from exc
    if not resolved_group.is_dir():
        raise ExchangeError(f"group path is not a directory: {group_dir}")
    resolved_repo = repository_root.resolve(strict=True)
    try:
        resolved_group.relative_to(resolved_repo)
    except ValueError as exc:
        raise ExchangeError("group directory must be inside the project repository") from exc
    safe_relative(resolved_group.name, label="run id")
    safe_relative(selector, label="selector")

    summary = _regular_file(
        resolved_group / "multi_seed_run_summary.json",
        label="multi-seed summary",
        root=resolved_group,
    )
    summary_payload = _load_json(summary)
    if not isinstance(summary_payload, dict) or summary_payload.get("success") is not True:
        raise ExchangeError("multi_seed_run_summary.json does not declare a successful run")

    files: list[tuple[Path, str]] = [(summary, "multi_seed_run_summary.json")]
    seen_remote: set[str] = {files[0][1]}
    matched_seeds: list[str] = []
    tensor_count = 0

    for seed_dir in sorted(resolved_group.glob("seed_*")):
        if seed_dir.is_symlink() or not seed_dir.is_dir():
            continue
        safe_relative(seed_dir.name, label="seed directory")
        selector_dir = seed_dir / "selectors" / selector
        if not selector_dir.is_dir() or selector_dir.is_symlink():
            continue
        case_path = _regular_file(selector_dir / "case_study.json", label="case study", root=resolved_group)
        trace_path = _regular_file(selector_dir / "sample_trace.json", label="sample trace", root=resolved_group)
        case_payload = _load_json(case_path)
        trace_payload = _load_json(trace_path)
        cases = case_payload.get("cases") if isinstance(case_payload, dict) else None
        samples = trace_payload.get("samples") if isinstance(trace_payload, dict) else None
        if not isinstance(cases, list) or not cases:
            raise ExchangeError(f"{case_path}: cases must be a non-empty list")
        if not isinstance(samples, list) or len(samples) != len(cases):
            raise ExchangeError(f"{selector_dir}: case/sample counts do not match")
        case_ids = {case.get("case_id") for case in cases if isinstance(case, dict)}
        sample_ids = {sample.get("sample_id") for sample in samples if isinstance(sample, dict)}
        if case_ids != sample_ids or len(case_ids) != len(cases):
            raise ExchangeError(f"{selector_dir}: case/sample identities do not match")

        entries = [
            (case_path, f"{seed_dir.name}/selectors/{selector}/case_study.json"),
            (trace_path, f"{seed_dir.name}/selectors/{selector}/sample_trace.json"),
        ]
        for local, remote in entries:
            if remote not in seen_remote:
                files.append((local, remote))
                seen_remote.add(remote)
        for case in cases:
            if not isinstance(case, dict):
                raise ExchangeError(f"{case_path}: case entry is not an object")
            for representation in REQUIRED_SOURCE_REPRESENTATIONS:
                tensor_path, remote_relative = _manifest_tensor(
                    selector_dir, case, representation, group_dir=resolved_group
                )
                remote = f"{seed_dir.name}/selectors/{selector}/{remote_relative}"
                if remote not in seen_remote:
                    files.append((tensor_path, remote))
                    seen_remote.add(remote)
                    tensor_count += 1
        matched_seeds.append(seed_dir.name)

    if not matched_seeds:
        raise ExchangeError(f"no seed_*/selectors/{selector} directory found in {resolved_group}")

    # Optional identity artifacts help later audit work but do not block runs
    # produced before these files were added to the formal runner.
    for filename in OPTIONAL_IDENTITY_FILES:
        optional = resolved_group / filename
        if optional.is_file() and not optional.is_symlink():
            remote = filename
            if remote not in seen_remote:
                files.append((_regular_file(optional, label=filename, root=resolved_group), remote))
                seen_remote.add(remote)

    target_dir = f"q-attention/qepvg-case-study-recovery/{resolved_group.name}"
    safe_relative(target_dir, label="target directory")
    print(
        f"Selected {len(matched_seeds)} seed(s): {', '.join(matched_seeds)}; "
        f"JSON={len(matched_seeds) * 2}; source_tensors={tensor_count}"
    )
    return target_dir, files


def _discover_group(repository_root: Path, experiment: str) -> Path:
    base = repository_root / "runs" / experiment
    candidates = []
    for path in base.glob("*"):
        if path.is_dir() and not path.is_symlink() and (path / "multi_seed_run_summary.json").is_file():
            try:
                summary = _load_json(path / "multi_seed_run_summary.json")
            except ExchangeError:
                continue
            if isinstance(summary, dict) and summary.get("success") is True:
                candidates.append(path)
    if not candidates:
        raise ExchangeError(f"no successful multi-seed run group found under {base}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, help="completed multi-seed run directory")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument("--url", default=os.environ.get("PROJECT_EXCHANGE_URL", DEFAULT_EXCHANGE_URL))
    parser.add_argument("--ca-file", type=Path, default=DEFAULT_CA_FILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("PROJECT_EXCHANGE_TOKEN", "")
    if not token:
        token = getpass.getpass("Q-Attention exchange token: ")
    group_dir = args.group_dir or _discover_group(PROJECT_ROOT, args.experiment)
    if not group_dir.is_absolute():
        group_dir = PROJECT_ROOT / group_dir
    target_dir, files = _collect_group(group_dir, args.selector)
    manifest = upload_declared_files(
        base_url=args.url,
        token=token,
        target_dir=target_dir,
        files=files,
        ca_file=args.ca_file,
    )
    print(f"Upload complete: {len(manifest['files'])} files")
    print(f"Target: {manifest['exchange_url']}/{manifest['target_dir']}")
    print(f"Manifest: {manifest['pending_manifest']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExchangeError as exc:
        print(f"Q-EPVG recovery upload failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
