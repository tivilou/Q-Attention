#!/usr/bin/env python3
"""Read-only structural diagnostic for a Q-EPVG multi-seed run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001 - diagnostic must report malformed files
        return None, f"{type(exc).__name__}: {exc}"


def _shape(value: Any, *, depth: int = 0) -> dict[str, Any]:
    """Describe JSON structure without printing sample or tensor values."""
    if isinstance(value, dict):
        result: dict[str, Any] = {
            "type": "object",
            "keys": sorted(str(key) for key in value),
        }
        if depth < 1:
            result["children"] = {
                str(key): _shape(child, depth=depth + 1)
                for key, child in value.items()
                if str(key) in {"stages", "input_refs", "output_refs", "representations"}
            }
        return result
    if isinstance(value, list):
        result = {"type": "array", "length": len(value)}
        if value:
            result["item_types"] = sorted({type(item).__name__ for item in value})
            if depth < 1:
                result["first_item"] = _shape(value[0], depth=depth + 1)
        return result
    if value is None:
        return {"type": "null"}
    return {"type": type(value).__name__}


def _repo_info(root: Path) -> None:
    print(f"repo_root={root}")
    for command in (("git", "rev-parse", "HEAD"), ("git", "status", "--short", "--branch")):
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        label = "git_head" if command[1] == "rev-parse" else "git_status"
        value = result.stdout.strip() or result.stderr.strip() or f"exit={result.returncode}"
        print(f"{label}={value}")


def _default_run(root: Path) -> Path:
    group_root = root / "runs" / "retacred_q_epvg_formal_multi_seed"
    candidates = sorted(path for path in group_root.iterdir() if path.is_dir()) if group_root.is_dir() else []
    if not candidates:
        raise ValueError(f"no multi-seed run directory found under {group_root}")
    return candidates[-1]


def _case_study_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(run_dir.rglob("*.json")):
        if "case_study" in path.parts or path.name == "case_study.json":
            payload, error = _load_json(path)
            if error is None and isinstance(payload, dict) and "cases" in payload:
                files.append(path)
    return files


def diagnose(root: Path, run_dir: Path) -> int:
    if not run_dir.is_dir():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    _repo_info(root)
    print(f"run_dir={run_dir}")
    markers = sorted(path.name for path in run_dir.iterdir() if path.is_file())
    print("run_markers=" + ",".join(markers) if markers else "run_markers=<none>")

    files = _case_study_files(run_dir)
    print(f"case_study_files={len(files)}")
    issues: list[str] = []
    if not files:
        issues.append("no Case Study JSON with a top-level cases field was found")

    for path in files:
        relative = path.relative_to(run_dir)
        payload, error = _load_json(path)
        print(f"\nFILE: {relative}")
        if error is not None:
            print(f"json_error={error}")
            issues.append(f"{relative}: invalid JSON")
            continue
        assert isinstance(payload, dict)
        cases = payload.get("cases")
        print(f"schema_version={payload.get('schema_version')!r}")
        print(f"cases_type={type(cases).__name__}")
        print(f"cases_length={len(cases) if isinstance(cases, list) else 'NA'}")
        if not isinstance(cases, list):
            print("cases_shape=" + json.dumps(_shape(cases), ensure_ascii=True, sort_keys=True))
            issues.append(f"{relative}: cases is not a list")
            continue
        if not cases:
            issues.append(f"{relative}: cases is empty")
            continue

        malformed = 0
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                malformed += 1
                issues.append(f"{relative}: cases[{index}] is not an object")
                continue
            stages = case.get("stages")
            if not isinstance(stages, list):
                malformed += 1
                print(
                    "MALFORMED: "
                    + f"cases[{index}].stages_type={type(stages).__name__} "
                    + "stages_shape="
                    + json.dumps(_shape(stages), ensure_ascii=True, sort_keys=True)
                )
                issues.append(f"{relative}: cases[{index}].stages is not a list")
        first = cases[0]
        if isinstance(first, dict):
            print("case0_keys=" + json.dumps(sorted(first.keys()), ensure_ascii=True))
            print(
                "case0_stages_shape="
                + json.dumps(_shape(first.get("stages")), ensure_ascii=True, sort_keys=True)
            )
        print(f"malformed_case_count={malformed}")

    print("\nstatus=" + ("FAIL" if issues else "PASS"))
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        "--group-dir",
        type=Path,
        default=None,
        help="completed multi-seed run directory; defaults to the newest run",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir is not None else _default_run(root)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    try:
        return diagnose(root, run_dir.resolve())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
