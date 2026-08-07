from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_integer_list(value: str, *, label: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item or not item.isdigit():
            raise ValueError(f"{label} must be a comma-separated list of non-negative integers")
        number = int(item)
        if number not in seen:
            values.append(number)
            seen.add(number)
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return values


def detect_gpu_ids(spec: str) -> list[int]:
    if spec != "auto":
        gpu_ids = parse_integer_list(spec, label="gpus")
    else:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_ids = parse_integer_list(
            ",".join(line.strip() for line in result.stdout.splitlines() if line.strip()),
            label="detected gpus",
        )
    for gpu_id in gpu_ids:
        subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=name", "--format=csv,noheader"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    return gpu_ids


def gpu_name(gpu_id: int) -> str:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=name", "--format=csv,noheader"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def resolve_group_dir(output_dir: str | None, *, stamp: str) -> Path:
    group_dir = Path(output_dir) if output_dir else Path(
        f"runs/retacred_dual_projector_multiseed_{stamp}"
    )
    if not group_dir.is_absolute():
        group_dir = ROOT / group_dir
    group_dir = group_dir.resolve()
    if not group_dir.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("output directory must be inside the repository runs directory")
    return group_dir


def build_run_command(
    args: argparse.Namespace,
    *,
    seed: int,
    gpu_id: int,
    run_dir: Path,
) -> list[str]:
    command = [
        "bash",
        "scripts/run_retacred_dual_qres_full.sh",
        "--seed",
        str(seed),
        "--gpus",
        str(gpu_id),
        "--parallel-mode",
        "serial",
        "--progress-format",
        args.progress_format,
        "--log-every-batches",
        str(args.log_every_batches),
        "--stale-timeout-minutes",
        str(args.stale_timeout_minutes),
        "--stage-timeout-hours",
        str(args.stage_timeout_hours),
        "--output-dir",
        str(run_dir),
        "--skip-preflight",
        "--no-latest-pointer",
    ]
    if args.skip_canary:
        command.append("--skip-canary")
    if args.canary_only:
        command.append("--canary-only")
    if args.dry_run:
        command.append("--dry-run")
    return command


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_preflight(log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            ["bash", "scripts/check_retacred_dual_qres_full.sh"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode,
            ["bash", "scripts/check_retacred_dual_qres_full.sh"],
        )


def run_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    gpu_id: int,
    run_dir: Path,
    output_lock: threading.Lock,
) -> dict[str, Any]:
    command = build_run_command(args, seed=seed, gpu_id=gpu_id, run_dir=run_dir)
    started = datetime.now().astimezone()
    with output_lock:
        print(f"[scheduler] seed={seed} gpu={gpu_id} started run_dir={run_dir}", flush=True)
    environment = os.environ.copy()
    environment["PYTHON_BIN"] = sys.executable
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        with output_lock:
            print(f"[seed {seed}][gpu {gpu_id}] {line.rstrip()}", flush=True)
    returncode = process.wait()
    completed = datetime.now().astimezone()
    marker = "CANARY_COMPLETE" if args.canary_only else "RUN_COMPLETE"
    success = returncode == 0 and (args.dry_run or (run_dir / marker).is_file())
    result = {
        "seed": seed,
        "gpu_id": gpu_id,
        "run_dir": str(run_dir),
        "started_at": started.isoformat(timespec="seconds"),
        "completed_at": completed.isoformat(timespec="seconds"),
        "elapsed_seconds": round((completed - started).total_seconds(), 1),
        "returncode": returncode,
        "success": success,
    }
    with output_lock:
        print(
            f"[scheduler] seed={seed} gpu={gpu_id} success={str(success).lower()} "
            f"elapsed={result['elapsed_seconds']}s",
            flush=True,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent Re-TACRED Dual Q-RES seeds across available GPUs."
    )
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--stale-timeout-minutes", type=int, default=45)
    parser.add_argument("--stage-timeout-hours", type=int, default=24)
    parser.add_argument("--progress-format", choices=["json", "both"], default="both")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        seeds = parse_integer_list(args.seeds, label="seeds")
        gpu_ids = detect_gpu_ids(args.gpus)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    if args.skip_canary and args.canary_only:
        parser.error("--skip-canary cannot be combined with --canary-only")
    for name in ("log_every_batches", "stale_timeout_minutes", "stage_timeout_hours"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    try:
        group_dir = resolve_group_dir(args.output_dir, stamp=stamp)
    except ValueError as exc:
        parser.error(str(exc))
    if group_dir.exists() and not args.dry_run:
        parser.error(f"refusing to reuse output directory: {group_dir}")

    commands = [
        build_run_command(
            args,
            seed=seed,
            gpu_id=gpu_ids[index % len(gpu_ids)],
            run_dir=group_dir / f"seed_{seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    print(
        f"[scheduler] seeds={','.join(map(str, seeds))} "
        f"gpus={','.join(map(str, gpu_ids))} workers={min(len(seeds), len(gpu_ids))}",
        flush=True,
    )
    if args.dry_run:
        for seed, command in zip(seeds, commands, strict=True):
            print(f"[scheduler] seed={seed} command={' '.join(command)}")
        return 0

    group_dir.mkdir(parents=True)
    if not args.skip_preflight:
        run_preflight(group_dir / "preflight.log")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "seeds": seeds,
        "gpus": [{"id": gpu_id, "name": gpu_name(gpu_id)} for gpu_id in gpu_ids],
        "worker_count": min(len(seeds), len(gpu_ids)),
        "canary_only": args.canary_only,
        "skip_canary": args.skip_canary,
        "progress_format": args.progress_format,
    }
    write_json(group_dir / "multi_seed_manifest.json", manifest)

    jobs: queue.Queue[int] = queue.Queue()
    for seed in seeds:
        jobs.put(seed)
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    output_lock = threading.Lock()
    stop_event = threading.Event()

    def worker(gpu_id: int) -> None:
        while not stop_event.is_set():
            try:
                seed = jobs.get_nowait()
            except queue.Empty:
                return
            result = run_seed(
                args,
                seed=seed,
                gpu_id=gpu_id,
                run_dir=group_dir / f"seed_{seed}",
                output_lock=output_lock,
            )
            with results_lock:
                results.append(result)
                write_json(group_dir / "multi_seed_summary.json", {"results": results})
            jobs.task_done()
            if not result["success"]:
                stop_event.set()

    started_monotonic = time.monotonic()
    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"gpu-{gpu_id}")
        for gpu_id in gpu_ids[: len(seeds)]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    completed_seeds = {result["seed"] for result in results}
    skipped_seeds = [seed for seed in seeds if seed not in completed_seeds]
    success = len(results) == len(seeds) and all(result["success"] for result in results)
    summary = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 1),
        "success": success,
        "skipped_seeds": skipped_seeds,
        "results": sorted(results, key=lambda item: seeds.index(item["seed"])),
    }
    write_json(group_dir / "multi_seed_summary.json", summary)
    marker = "MULTI_SEED_COMPLETE" if success else "MULTI_SEED_FAILED"
    (group_dir / marker).write_text(summary["completed_at"] + "\n", encoding="utf-8")
    if success:
        (ROOT / "runs/latest_dual_qres_multi_seed.txt").write_text(
            str(group_dir.relative_to(ROOT)) + "\n",
            encoding="utf-8",
        )
        print(f"[scheduler] completed group_dir={group_dir.relative_to(ROOT)}", flush=True)
        return 0
    print(
        f"[scheduler] failed group_dir={group_dir.relative_to(ROOT)} "
        f"skipped={','.join(map(str, skipped_seeds)) or 'none'}",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
