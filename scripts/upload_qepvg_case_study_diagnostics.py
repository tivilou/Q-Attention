#!/usr/bin/env python3
"""Upload only the selected Q-EPVG Case Study and source-trace JSON files."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = PROJECT_ROOT / "scripts" / "collab" / "lib"
sys.path.insert(0, str(LIB_DIR))

from exchange_upload import (  # noqa: E402
    DEFAULT_CA_FILE,
    ExchangeError,
    safe_relative,
    upload_declared_files,
)


DEFAULT_GROUP_DIR = Path(
    "runs/retacred_q_epvg_formal_multi_seed/20260927T071350Z"
)
DEFAULT_SELECTOR = "q_epvg_zz_value_only_quantum"
DEFAULT_EXCHANGE_URL = "https://117.50.198.37:18084"
MAX_FILE_BYTES = 1024**3


def collect_selector_files(
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
    resolved_repository = repository_root.resolve(strict=True)
    try:
        resolved_group.relative_to(resolved_repository)
    except ValueError as exc:
        raise ExchangeError("group directory must be inside the project repository") from exc

    safe_relative(resolved_group.name, label="run id")
    safe_relative(selector, label="selector")
    found: list[tuple[Path, str]] = []
    matched_seeds: list[str] = []

    for seed_dir in sorted(resolved_group.iterdir()):
        if not seed_dir.name.startswith("seed_"):
            continue
        if seed_dir.is_symlink():
            raise ExchangeError(f"seed directory must not be a symlink: {seed_dir.name}")
        if not seed_dir.is_dir():
            continue
        safe_relative(seed_dir.name, label="seed directory")

        selectors_dir = seed_dir / "selectors"
        if selectors_dir.is_symlink():
            raise ExchangeError(f"selector directory must not be a symlink: {seed_dir.name}/selectors")
        if not selectors_dir.is_dir():
            continue
        selector_dir = selectors_dir / selector
        if not selector_dir.exists():
            continue
        if selector_dir.is_symlink() or not selector_dir.is_dir():
            raise ExchangeError(f"selector path is not a regular directory: {selector_dir}")

        for filename in ("case_study.json", "sample_trace.json"):
            path = selector_dir / filename
            if path.is_symlink() or not path.is_file():
                raise ExchangeError(f"required diagnostic file is missing or unsafe: {path}")
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(resolved_group)
            except ValueError as exc:
                raise ExchangeError(f"diagnostic file escapes the run directory: {path}") from exc
            if resolved.stat().st_size > MAX_FILE_BYTES:
                raise ExchangeError(f"diagnostic file exceeds the 1 GiB upload limit: {path.name}")
            remote = f"{seed_dir.name}/selectors/{selector}/{filename}"
            found.append((resolved, remote))
        matched_seeds.append(seed_dir.name)

    if not found:
        raise ExchangeError(
            f"no complete seed_*/selectors/{selector} diagnostic pairs found in {resolved_group}"
        )
    target_dir = f"q-attention/qepvg-case-study-recovery/{resolved_group.name}"
    safe_relative(target_dir, label="target directory")
    print(
        f"Selected {len(found)} files from {len(matched_seeds)} seed(s): "
        f"{', '.join(matched_seeds)}; selector={selector}"
    )
    return target_dir, found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument(
        "--url",
        default=os.environ.get("PROJECT_EXCHANGE_URL", DEFAULT_EXCHANGE_URL),
    )
    parser.add_argument("--ca-file", type=Path, default=DEFAULT_CA_FILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("PROJECT_EXCHANGE_TOKEN", "")
    if not token:
        token = getpass.getpass("Q-Attention exchange token: ")
    group_dir = args.group_dir
    if not group_dir.is_absolute():
        group_dir = PROJECT_ROOT / group_dir
    target_dir, files = collect_selector_files(group_dir, args.selector)
    manifest = upload_declared_files(
        base_url=args.url,
        token=token,
        target_dir=target_dir,
        files=files,
        ca_file=args.ca_file,
    )
    print(
        f"Upload complete: {len(manifest['files'])} files -> "
        f"{manifest['exchange_url']}/{manifest['target_dir']}"
    )
    print(f"Manifest: {manifest['pending_manifest']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExchangeError as exc:
        print(f"Q-EPVG diagnostics upload failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
