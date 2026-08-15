from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEED_DEFAULT = "7,11,13,17,23"


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
    if spec == "auto":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        spec = ",".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    gpu_ids = parse_integer_list(spec, label="gpus")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_reused_seeds(
    specs: list[str],
    *,
    seeds: list[int],
    expected_commit: str,
    expected_config_sha256: str,
) -> dict[int, Path]:
    reused: dict[int, Path] = {}
    runs_root = (ROOT / "runs").resolve()
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--reuse-seed must use SEED=RUN_DIR syntax")
        raw_seed, raw_path = spec.split("=", 1)
        if not raw_seed.isdigit() or not raw_path:
            raise ValueError("--reuse-seed must use SEED=RUN_DIR syntax")
        seed = int(raw_seed)
        if seed not in seeds:
            raise ValueError(f"reused seed {seed} is not present in --seeds")
        if seed in reused:
            raise ValueError(f"duplicate reused seed: {seed}")
        run_dir = Path(raw_path)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir = run_dir.resolve()
        if not run_dir.is_relative_to(runs_root):
            raise ValueError(f"reused seed directory must be inside runs/: {run_dir}")
        if not (run_dir / "RUN_COMPLETE").is_file():
            raise ValueError(f"reused seed is incomplete: {run_dir}")
        if (run_dir / "RUN_FAILED").exists():
            raise ValueError(f"reused seed also contains RUN_FAILED: {run_dir}")
        summary = load_json(run_dir / "run_summary.json")
        if (
            summary.get("formal_experiment") is not True
            or summary.get("status") != "pass"
            or summary.get("partial_selector_run") is not False
            or summary.get("run_type") != "formal_full_relation_transfer"
        ):
            raise ValueError(f"reused seed is not a passed formal run: {run_dir}")
        if summary.get("selectors") != [
            "disabled",
            "q_causal_transport",
            "classical_causal_transport",
            "q_causal_key_only",
        ]:
            raise ValueError(f"reused seed is missing required selectors: {run_dir}")
        result_seeds = {int(row["seed"]) for row in summary.get("results", [])}
        if result_seeds != {seed}:
            raise ValueError(f"reused run seed mismatch: expected {seed}, found {sorted(result_seeds)}")
        provenance = summary.get("provenance", {})
        if provenance.get("git_commit") != expected_commit:
            raise ValueError(f"reused seed {seed} used a different commit")
        if provenance.get("git_dirty") is not False:
            raise ValueError(f"reused seed {seed} recorded git_dirty=true")
        if provenance.get("config_sha256") != expected_config_sha256:
            raise ValueError(f"reused seed {seed} used a different formal config")
        reused[seed] = run_dir
    return reused


def build_command(args: argparse.Namespace, seed: int, gpu_id: int, run_dir: Path) -> list[str]:
    return [
        "bash",
        "scripts/run_qvres_relation_transfer_full.sh",
        "--seed",
        str(seed),
        "--gpus",
        str(gpu_id),
        "--output-dir",
        str(run_dir),
        "--log-every-batches",
        str(args.log_every_batches),
        "--stale-timeout-minutes",
        str(args.stale_timeout_minutes),
        "--run-timeout-hours",
        str(args.run_timeout_hours),
        "--progress-format",
        args.progress_format,
        "--skip-preflight",
    ]


def run_preflight(args: argparse.Namespace, path: Path) -> None:
    command = ["bash", "scripts/check_qvres_relation_transfer_full.sh"]
    if args.skip_tests:
        command.append("--skip-tests")
    with path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
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
        raise subprocess.CalledProcessError(returncode, command)


def run_seed(
    args: argparse.Namespace,
    seed: int,
    gpu_id: int,
    run_dir: Path,
    output_lock: threading.Lock,
) -> dict[str, Any]:
    command = build_command(args, seed, gpu_id, run_dir)
    started = datetime.now().astimezone()
    with output_lock:
        print(f"[scheduler] seed={seed} gpu={gpu_id} started run_dir={run_dir}", flush=True)
    environment = os.environ.copy()
    environment["PYTHON_BIN"] = sys.executable
    try:
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
    except OSError as exc:
        with output_lock:
            print(f"[seed {seed}][gpu {gpu_id}] launcher error: {exc}", file=sys.stderr, flush=True)
        returncode = 1
    completed = datetime.now().astimezone()
    result = {
        "seed": seed,
        "gpu_id": gpu_id,
        "run_dir": str(run_dir),
        "started_at": started.isoformat(timespec="seconds"),
        "completed_at": completed.isoformat(timespec="seconds"),
        "elapsed_seconds": round((completed - started).total_seconds(), 1),
        "returncode": returncode,
        "success": returncode == 0 and (run_dir / "RUN_COMPLETE").is_file(),
    }
    with output_lock:
        print(
            f"[scheduler] seed={seed} gpu={gpu_id} success={str(result['success']).lower()} "
            f"elapsed={result['elapsed_seconds']}s",
            flush=True,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent Q-VRES full Re-TACRED seeds across available GPUs."
    )
    parser.add_argument("--seeds", default=SEED_DEFAULT)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--stale-timeout-minutes", type=int, default=45)
    parser.add_argument("--run-timeout-hours", type=int, default=48)
    parser.add_argument("--progress-format", choices=["json", "both"], default="both")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--reuse-seed",
        action="append",
        default=[],
        metavar="SEED=RUN_DIR",
        help="reuse a completed formal seed run from the same commit",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        seeds = parse_integer_list(args.seeds, label="seeds")
        gpu_ids = detect_gpu_ids(args.gpus)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    for name in ("log_every_batches", "stale_timeout_minutes", "run_timeout_hours"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    config_path = (ROOT / "configs/q_vres_relation_transfer_full.json").resolve()
    config_sha256 = sha256(config_path)
    try:
        reused_seeds = resolve_reused_seeds(
            args.reuse_seed,
            seeds=seeds,
            expected_commit=commit,
            expected_config_sha256=config_sha256,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    pending_seeds = [seed for seed in seeds if seed not in reused_seeds]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group_dir = Path(args.output_dir) if args.output_dir else Path(
        f"runs/q_vres_relation_transfer_full_multiseed_{stamp}"
    )
    if not group_dir.is_absolute():
        group_dir = ROOT / group_dir
    group_dir = group_dir.resolve()
    if not group_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("output directory must be inside the repository runs directory")
    if group_dir.exists() and not args.dry_run:
        raise SystemExit(f"refusing to reuse output directory: {group_dir}")

    commands = [
        build_command(args, seed, gpu_ids[index % len(gpu_ids)], group_dir / f"seed_{seed}")
        for index, seed in enumerate(pending_seeds)
    ]
    print(
        f"[scheduler] seeds={','.join(map(str, seeds))} gpus={','.join(map(str, gpu_ids))} "
        f"workers={min(len(pending_seeds), len(gpu_ids))} reused={','.join(map(str, reused_seeds)) or 'none'}",
        flush=True,
    )
    if args.dry_run:
        for seed, run_dir in reused_seeds.items():
            print(f"[scheduler] seed={seed} reused_from={run_dir}")
        for seed, command in zip(pending_seeds, commands, strict=True):
            print(f"[scheduler] seed={seed} command={' '.join(command)}")
        return 0

    group_dir.mkdir(parents=True)
    if not args.skip_preflight:
        run_preflight(args, group_dir / "preflight.log")

    manifest = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "seeds": seeds,
        "gpus": [{"id": gpu_id, "name": gpu_name(gpu_id)} for gpu_id in gpu_ids],
        "worker_count": min(len(pending_seeds), len(gpu_ids)),
        "reused_seeds": {str(seed): str(path) for seed, path in reused_seeds.items()},
        "progress_format": args.progress_format,
    }
    write_json(group_dir / "multi_seed_manifest.json", manifest)

    state_lock = threading.Lock()
    output_lock = threading.Lock()
    state: dict[int, dict[str, Any]] = {
        seed: {
            "seed": seed,
            "status": "reused" if seed in reused_seeds else "queued",
            "gpu_id": None,
            **({"run_dir": str(reused_seeds[seed])} if seed in reused_seeds else {}),
        }
        for seed in seeds
    }
    results: list[dict[str, Any]] = [
        {
            "seed": seed,
            "gpu_id": None,
            "run_dir": str(run_dir),
            "started_at": None,
            "completed_at": None,
            "elapsed_seconds": 0.0,
            "returncode": 0,
            "success": True,
            "reused": True,
        }
        for seed, run_dir in reused_seeds.items()
    ]

    def write_state() -> None:
        payload = {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "group_dir": str(group_dir),
            "seeds": seeds,
            "assignments": [state[seed] for seed in seeds],
        }
        write_json(group_dir / "multi_seed_status.json", payload)

    write_state()
    for seed, source_dir in reused_seeds.items():
        os.symlink(source_dir, group_dir / f"seed_{seed}", target_is_directory=True)
    jobs: queue.Queue[int] = queue.Queue()
    for seed in pending_seeds:
        jobs.put(seed)
    stop_event = threading.Event()

    def worker(gpu_id: int) -> None:
        while not stop_event.is_set():
            try:
                seed = jobs.get_nowait()
            except queue.Empty:
                return
            with state_lock:
                state[seed] = {"seed": seed, "status": "running", "gpu_id": gpu_id}
                write_state()
            try:
                result = run_seed(
                    args,
                    seed=seed,
                    gpu_id=gpu_id,
                    run_dir=group_dir / f"seed_{seed}",
                    output_lock=output_lock,
                )
                with state_lock:
                    results.append(result)
                    state[seed] = {
                        "seed": seed,
                        "status": "complete" if result["success"] else "failed",
                        "gpu_id": gpu_id,
                        "returncode": result["returncode"],
                    }
                    write_state()
                if not result["success"]:
                    stop_event.set()
            finally:
                jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"qvres-gpu-{gpu_id}")
        for gpu_id in gpu_ids[: len(pending_seeds)]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    completed_seeds = {result["seed"] for result in results}
    for seed in seeds:
        if seed not in completed_seeds and state[seed]["status"] == "queued":
            state[seed] = {"seed": seed, "status": "skipped", "gpu_id": None}
    success = len(results) == len(seeds) and all(result["success"] for result in results)
    summary = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "success": success,
        "skipped_seeds": [seed for seed in seeds if seed not in completed_seeds],
        "results": sorted(results, key=lambda item: seeds.index(item["seed"])),
    }
    write_json(group_dir / "multi_seed_summary.json", summary)
    with state_lock:
        write_state()
    marker = "MULTI_SEED_COMPLETE" if success else "MULTI_SEED_FAILED"
    (group_dir / marker).write_text(summary["completed_at"] + "\n", encoding="utf-8")
    if success:
        print(f"[scheduler] completed group_dir={group_dir.relative_to(ROOT)}", flush=True)
        return 0
    print(
        f"[scheduler] failed group_dir={group_dir.relative_to(ROOT)} "
        f"skipped={','.join(map(str, summary['skipped_seeds'])) or 'none'}",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
