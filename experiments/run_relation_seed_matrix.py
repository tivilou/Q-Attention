from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_seed_csv(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def default_output_root(config: dict[str, Any]) -> Path:
    configured = resolve_project_path(config.get("output_dir", "runs/relation"))
    return configured.with_name(f"{configured.name}_multiseed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a relation pipeline over a fixed random-seed matrix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", default="13,17,23,29,31")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stages", default=None, help="Optional comma-separated pipeline stages")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse seed directories that already have run_summary.json")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = parse_seed_csv(args.seeds)
    output_root = resolve_project_path(args.output_root) if args.output_root else default_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    run_dirs: list[Path] = []
    for seed in seeds:
        run_dir = output_root / f"seed_{seed}"
        run_dirs.append(run_dir)
        if args.skip_existing and (run_dir / "run_summary.json").exists():
            records.append({"seed": seed, "run_dir": str(run_dir), "status": "reused"})
            continue
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "run_relation_smoke_pipeline.py"),
            "--config",
            str(config_path),
            "--output_dir",
            str(run_dir),
            "--device",
            args.device,
            "--seed",
            str(seed),
        ]
        if args.stages:
            cmd.extend(["--stages", args.stages])
        if args.dry_run:
            cmd.append("--dry_run")
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        record = {
            "seed": seed,
            "run_dir": str(run_dir),
            "command": cmd,
            "returncode": int(result.returncode),
            "duration_seconds": time.perf_counter() - started,
            "status": "dry_run" if args.dry_run else ("completed" if result.returncode == 0 else "failed"),
        }
        records.append(record)
        if result.returncode != 0:
            break

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "seeds": seeds,
        "output_root": str(output_root),
        "runs": records,
    }
    manifest_path = output_root / "seed_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    failed = [record for record in records if record.get("status") == "failed"]
    if failed:
        raise SystemExit(int(failed[0]["returncode"]))
    if not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "summarize_relation_seed_matrix.py"),
                "--run_dirs",
                ",".join(str(path) for path in run_dirs),
                "--output_json",
                str(output_root / "seed_summary.json"),
                "--output_markdown",
                str(output_root / "seed_summary.md"),
            ],
            cwd=ROOT,
            check=True,
        )
    print(json.dumps({"manifest": str(manifest_path), "num_runs": len(records), "output_root": str(output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
