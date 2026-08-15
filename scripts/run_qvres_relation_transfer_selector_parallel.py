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
import time
from typing import Any

from aggregate_qvres_relation_transfer_selector_parallel import (
    TRAINABLE_SELECTORS,
    aggregate_selector_parallel_run,
)


ROOT = Path(__file__).resolve().parents[1]
_CONSOLE_BROKEN = threading.Event()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_status(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(f"{key.upper()}={value}\n" for key, value in fields.items()),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def console_print(message: str, lock: threading.Lock, *, stderr: bool = False) -> None:
    if _CONSOLE_BROKEN.is_set():
        return
    with lock:
        try:
            print(message, file=sys.stderr if stderr else sys.stdout, flush=True)
        except BrokenPipeError:
            _CONSOLE_BROKEN.set()


def read_status(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    fields: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    return fields


def read_heartbeat(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def format_duration(seconds: Any) -> str:
    try:
        total = max(int(round(float(seconds))), 0)
    except (TypeError, ValueError):
        return "unknown"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def selector_dashboard_snapshot(
    run_dir: Path,
    stage_state: dict[str, dict[str, Any]],
    state_lock: threading.Lock,
) -> dict[str, Any]:
    stage_order = ("baseline", *TRAINABLE_SELECTORS)
    with state_lock:
        stages = {name: dict(stage_state[name]) for name in stage_order}
    groups: dict[str, list[str]] = {
        "complete": [],
        "running": [],
        "pending": [],
        "failed": [],
        "skipped": [],
    }
    rows: list[dict[str, Any]] = []
    for name in stage_order:
        stage = stages[name]
        status_fields = read_status(run_dir / "status" / f"{name}.env")
        raw_status = status_fields.get("STATUS", str(stage.get("status", "pending")))
        status = "pending" if raw_status in {"queued", "waiting_for_baseline"} else raw_status
        if status not in groups:
            status = "failed"
        groups[status].append(name)
        if status != "running":
            continue
        heartbeat = read_heartbeat(run_dir / "status" / f"{name}.heartbeat")
        rows.append(
            {
                "stage": name,
                "gpu_id": status_fields.get("GPU_ID", stage.get("gpu_id")),
                "phase": heartbeat.get("phase") or heartbeat.get("stage"),
                "epoch": heartbeat.get("epoch"),
                "epochs": heartbeat.get("epochs"),
                "batch": heartbeat.get("batch"),
                "batches": heartbeat.get("batches"),
                "eta_seconds": heartbeat.get("eta_seconds"),
                "batches_per_second": heartbeat.get("batches_per_second"),
            }
        )
    rows.sort(key=lambda row: (str(row["gpu_id"]), row["stage"]))
    return {
        "total": len(stage_order),
        "counts": {name: len(values) for name, values in groups.items()},
        "groups": groups,
        "rows": rows,
    }


def render_selector_dashboard(snapshot: dict[str, Any]) -> str:
    counts = snapshot["counts"]
    lines = [
        (
            f"Q-VRES stages: {counts['complete']}/{snapshot['total']} complete | "
            f"{counts['running']} running | {counts['pending']} pending | "
            f"{counts['failed']} failed | {counts['skipped']} skipped"
        )
    ]
    for row in snapshot["rows"]:
        progress: list[str] = []
        if row.get("phase"):
            progress.append(str(row["phase"]))
        if row.get("epoch") is not None and row.get("epochs") is not None:
            progress.append(f"epoch {row['epoch']}/{row['epochs']}")
        if row.get("batch") is not None and row.get("batches") is not None:
            progress.append(f"batch {row['batch']}/{row['batches']}")
        if not progress:
            progress.append("starting")
        progress.append(f"ETA {format_duration(row.get('eta_seconds'))}")
        if row.get("batches_per_second") is not None:
            progress.append(f"{row['batches_per_second']} batch/s")
        lines.append(
            f"GPU {row.get('gpu_id', '?')} | {row['stage']} | " + " | ".join(progress)
        )
    for label, key in (
        ("Completed", "complete"),
        ("Pending", "pending"),
        ("Failed", "failed"),
        ("Skipped", "skipped"),
    ):
        values = snapshot["groups"][key]
        lines.append(f"{label}: {', '.join(values) if values else 'none'}")
    return "\n".join(lines)


def build_transfer_command(
    args: argparse.Namespace,
    *,
    selector: str,
    output_dir: Path,
    model_dir: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "experiments/run_q_causal_value_evidence_relation_transfer.py",
        "--config",
        args.config,
        "--formal-experiment",
        "--allow-partial-selectors",
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--selectors",
        selector,
        "--output_dir",
        str(output_dir),
        "--log_every_batches",
        str(args.log_every_batches),
    ]
    if model_dir is not None:
        command.extend(["--model_dir", str(model_dir)])
    return command


def run_stage(
    args: argparse.Namespace,
    *,
    name: str,
    gpu_id: int,
    command: list[str],
    output_dir: Path,
    run_dir: Path,
    output_lock: threading.Lock,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    status_path = run_dir / "status" / f"{name}.env"
    heartbeat_path = run_dir / "status" / f"{name}.heartbeat"
    log_path = run_dir / "logs" / f"{name}.log"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.touch()
    write_status(
        status_path,
        status="running",
        stage=name,
        gpu_id=gpu_id,
        started_at=started.isoformat(timespec="seconds"),
        heartbeat_file=heartbeat_path,
        log_file=log_path,
        output_dir=output_dir,
    )
    console_print(f"[selector-scheduler] stage={name} gpu={gpu_id} started", output_lock)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTHONUNBUFFERED": "1",
            "Q_ATTENTION_PROGRESS_FORMAT": args.progress_format,
            "Q_ATTENTION_HEARTBEAT_FILE": str(heartbeat_path),
        }
    )
    wrapped = [
        sys.executable,
        "scripts/run_with_health_watchdog.py",
        "--heartbeat-file",
        str(heartbeat_path),
        "--stale-seconds",
        str(args.stale_timeout_minutes * 60),
        "--timeout-seconds",
        str(args.run_timeout_hours * 3600),
        "--",
        *command,
    ]
    returncode = 1
    launcher_error: str | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                wrapped,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            write_status(
                status_path,
                status="running",
                stage=name,
                gpu_id=gpu_id,
                pid=process.pid,
                started_at=started.isoformat(timespec="seconds"),
                heartbeat_file=heartbeat_path,
                log_file=log_path,
                output_dir=output_dir,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                console_print(f"[{name}][gpu {gpu_id}] {line.rstrip()}", output_lock)
            returncode = process.wait()
    except OSError as exc:
        launcher_error = str(exc).replace("\n", " ")
        console_print(
            f"[{name}][gpu {gpu_id}] launcher error: {launcher_error}",
            output_lock,
            stderr=True,
        )
    completed = datetime.now().astimezone()
    elapsed_seconds = round(max(time.monotonic() - started_monotonic, 0.0), 1)
    selector_name = "disabled" if name == "baseline" else name
    expected_outputs = (
        output_dir / "run_summary.json",
        output_dir / "run_summary.md",
        output_dir / "run_config.json",
        output_dir / "selectors" / selector_name / "metrics.json",
    )
    if name == "baseline":
        expected_outputs += (output_dir / "baseline" / "metrics.json",)
    missing_outputs = [str(path) for path in expected_outputs if not path.is_file()]
    success = returncode == 0 and not missing_outputs
    stage_error = launcher_error
    if stage_error is None and missing_outputs:
        stage_error = "missing outputs: " + ", ".join(missing_outputs)
    if stage_error is None and returncode != 0:
        stage_error = f"exit code {returncode}"
    write_status(
        status_path,
        status="complete" if success else "failed",
        stage=name,
        gpu_id=gpu_id,
        started_at=started.isoformat(timespec="seconds"),
        completed_at=completed.isoformat(timespec="seconds"),
        elapsed_seconds=elapsed_seconds,
        exit_code=returncode,
        heartbeat_file=heartbeat_path,
        log_file=log_path,
        output_dir=output_dir,
        error=stage_error or "",
    )
    result = {
        "stage": name,
        "gpu_id": gpu_id,
        "output_dir": str(output_dir),
        "started_at": started.isoformat(timespec="seconds"),
        "completed_at": completed.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "returncode": returncode,
        "success": success,
        "missing_outputs": missing_outputs,
        "error": stage_error,
    }
    console_print(
        f"[selector-scheduler] stage={name} gpu={gpu_id} "
        f"success={str(success).lower()} elapsed={result['elapsed_seconds']}s",
        output_lock,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Q-VRES seed with selector-level GPU parallelism."
    )
    parser.add_argument("--config", default="configs/q_vres_relation_transfer_full.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--stale-timeout-minutes", type=int, default=45)
    parser.add_argument("--run-timeout-hours", type=int, default=48)
    parser.add_argument("--dashboard-interval-seconds", type=float, default=10.0)
    parser.add_argument("--progress-format", choices=["json", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    for name in ("log_every_batches", "stale_timeout_minutes", "run_timeout_hours"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.dashboard_interval_seconds <= 0:
        raise SystemExit("--dashboard-interval-seconds must be positive")
    try:
        gpu_ids = detect_gpu_ids(args.gpus)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    run_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("output directory must be inside the repository runs directory")
    if run_dir.exists() and not args.dry_run:
        raise SystemExit(f"refusing to reuse output directory: {run_dir}")

    baseline_stage_dir = run_dir / "stages" / "baseline"
    baseline_model_dir = baseline_stage_dir / "baseline"
    selector_stage_dirs = {
        selector: run_dir / "stages" / selector for selector in TRAINABLE_SELECTORS
    }
    print(
        f"[selector-scheduler] seed={args.seed} gpus={','.join(map(str, gpu_ids))} "
        f"baseline_gpu={gpu_ids[0]} selector_workers={min(len(gpu_ids), len(TRAINABLE_SELECTORS))} "
        f"run_dir={run_dir.relative_to(ROOT)}",
        flush=True,
    )
    print(
        "[selector-scheduler] pending selectors=" + ",".join(TRAINABLE_SELECTORS),
        flush=True,
    )
    if args.dry_run:
        baseline_command = build_transfer_command(
            args,
            selector="disabled",
            output_dir=baseline_stage_dir,
            model_dir=None,
        )
        print(f"[dry-run] baseline gpu={gpu_ids[0]} command={' '.join(baseline_command)}")
        for index, selector in enumerate(TRAINABLE_SELECTORS):
            gpu_hint = gpu_ids[index % min(len(gpu_ids), len(TRAINABLE_SELECTORS))]
            command = build_transfer_command(
                args,
                selector=selector,
                output_dir=selector_stage_dirs[selector],
                model_dir=baseline_model_dir,
            )
            print(
                f"[dry-run] selector={selector} initial_gpu_hint={gpu_hint} "
                f"command={' '.join(command)}"
            )
        return 0

    run_dir.mkdir(parents=True)
    output_lock = threading.Lock()
    state_lock = threading.Lock()
    stage_state: dict[str, dict[str, Any]] = {
        "baseline": {"stage": "baseline", "status": "queued", "gpu_id": gpu_ids[0]},
        **{
            selector: {"stage": selector, "status": "waiting_for_baseline", "gpu_id": None}
            for selector in TRAINABLE_SELECTORS
        },
    }

    def write_scheduler_state() -> None:
        write_json(
            run_dir / "status" / "selector_parallel_status.json",
            {
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "seed": args.seed,
                "run_dir": str(run_dir),
                "stages": [stage_state[name] for name in ("baseline", *TRAINABLE_SELECTORS)],
            },
        )

    write_scheduler_state()
    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    manifest = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path.resolve()),
        "seed": args.seed,
        "parallel_mode": "selectors",
        "assignment_policy": "first_available_gpu",
        "baseline_gpu": gpu_ids[0],
        "gpus": [{"id": gpu_id, "name": gpu_name(gpu_id)} for gpu_id in gpu_ids],
        "dashboard_interval_seconds": args.dashboard_interval_seconds,
    }
    write_json(run_dir / "selector_parallel_manifest.json", manifest)

    baseline_command = build_transfer_command(
        args,
        selector="disabled",
        output_dir=baseline_stage_dir,
        model_dir=None,
    )
    with state_lock:
        stage_state["baseline"]["status"] = "running"
        write_scheduler_state()
    console_print(
        render_selector_dashboard(selector_dashboard_snapshot(run_dir, stage_state, state_lock)),
        output_lock,
    )
    baseline_result = run_stage(
        args,
        name="baseline",
        gpu_id=gpu_ids[0],
        command=baseline_command,
        output_dir=baseline_stage_dir,
        run_dir=run_dir,
        output_lock=output_lock,
    )
    with state_lock:
        stage_state["baseline"].update(
            status="complete" if baseline_result["success"] else "failed",
            returncode=baseline_result["returncode"],
        )
        if baseline_result["success"]:
            for selector in TRAINABLE_SELECTORS:
                stage_state[selector]["status"] = "queued"
        else:
            for selector in TRAINABLE_SELECTORS:
                stage_state[selector]["status"] = "skipped"
        write_scheduler_state()
    console_print(
        render_selector_dashboard(selector_dashboard_snapshot(run_dir, stage_state, state_lock)),
        output_lock,
    )
    if not baseline_result["success"]:
        summary = {"success": False, "results": [baseline_result], "failed_stage": "baseline"}
        write_json(run_dir / "selector_parallel_summary.json", summary)
        (run_dir / "RUN_FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        console_print("[selector-scheduler] baseline failed; selector workers were not started", output_lock, stderr=True)
        return 1

    console_print(
        "[selector-scheduler] baseline complete; queued=" + ",".join(TRAINABLE_SELECTORS),
        output_lock,
    )
    jobs: queue.Queue[str] = queue.Queue()
    for selector in TRAINABLE_SELECTORS:
        jobs.put(selector)
    selector_results: list[dict[str, Any]] = []
    actual_assignments: dict[str, int] = {"baseline": gpu_ids[0]}
    stop_event = threading.Event()
    workers_complete = threading.Event()
    workers_lock = threading.Lock()
    worker_gpu_ids = gpu_ids[: len(TRAINABLE_SELECTORS)]
    remaining_workers = len(worker_gpu_ids)

    def worker(gpu_id: int) -> None:
        nonlocal remaining_workers
        try:
            while not stop_event.is_set():
                try:
                    selector = jobs.get_nowait()
                except queue.Empty:
                    return
                with state_lock:
                    stage_state[selector].update(status="running", gpu_id=gpu_id)
                    actual_assignments[selector] = gpu_id
                    write_scheduler_state()
                console_print(
                    f"[selector-scheduler] assigned selector={selector} gpu={gpu_id}",
                    output_lock,
                )
                try:
                    command = build_transfer_command(
                        args,
                        selector=selector,
                        output_dir=selector_stage_dirs[selector],
                        model_dir=baseline_model_dir,
                    )
                    result = run_stage(
                        args,
                        name=selector,
                        gpu_id=gpu_id,
                        command=command,
                        output_dir=selector_stage_dirs[selector],
                        run_dir=run_dir,
                        output_lock=output_lock,
                    )
                except Exception as exc:
                    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
                    result = {
                        "stage": selector,
                        "gpu_id": gpu_id,
                        "output_dir": str(selector_stage_dirs[selector]),
                        "completed_at": completed_at,
                        "returncode": 1,
                        "success": False,
                        "error": str(exc),
                    }
                    write_status(
                        run_dir / "status" / f"{selector}.env",
                        status="failed",
                        stage=selector,
                        gpu_id=gpu_id,
                        completed_at=completed_at,
                        exit_code=1,
                        error=str(exc).replace("\n", " "),
                    )
                with state_lock:
                    selector_results.append(result)
                    stage_state[selector].update(
                        status="complete" if result["success"] else "failed",
                        returncode=result["returncode"],
                    )
                    write_scheduler_state()
                if not result["success"]:
                    stop_event.set()
                jobs.task_done()
        finally:
            with workers_lock:
                remaining_workers -= 1
                if remaining_workers == 0:
                    workers_complete.set()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"qvres-selector-gpu-{gpu_id}")
        for gpu_id in worker_gpu_ids
    ]
    for thread in threads:
        thread.start()
    while True:
        snapshot = selector_dashboard_snapshot(run_dir, stage_state, state_lock)
        console_print(render_selector_dashboard(snapshot), output_lock)
        if workers_complete.is_set():
            break
        workers_complete.wait(args.dashboard_interval_seconds)
    for thread in threads:
        thread.join()

    completed = {result["stage"] for result in selector_results}
    with state_lock:
        for selector in TRAINABLE_SELECTORS:
            if selector not in completed and stage_state[selector]["status"] == "queued":
                stage_state[selector]["status"] = "skipped"
        write_scheduler_state()
    success = len(selector_results) == len(TRAINABLE_SELECTORS) and all(
        result["success"] for result in selector_results
    )
    scheduler_summary = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "success": success,
        "seed": args.seed,
        "parallel_mode": "selectors",
        "gpu_assignments": actual_assignments,
        "results": [
            baseline_result,
            *sorted(selector_results, key=lambda item: TRAINABLE_SELECTORS.index(item["stage"])),
        ],
    }
    write_json(run_dir / "selector_parallel_summary.json", scheduler_summary)
    if not success:
        (run_dir / "RUN_FAILED").write_text(scheduler_summary["completed_at"] + "\n", encoding="utf-8")
        console_print(
            "[selector-scheduler] failed; all active workers have exited",
            output_lock,
            stderr=True,
        )
        return 1

    try:
        aggregate_selector_parallel_run(
            run_dir,
            baseline_stage_dir,
            selector_stage_dirs,
            actual_assignments,
        )
    except Exception as exc:
        scheduler_summary.update(success=False, aggregation_error=str(exc))
        write_json(run_dir / "selector_parallel_summary.json", scheduler_summary)
        (run_dir / "RUN_FAILED").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        console_print(f"[selector-scheduler] aggregation failed: {exc}", output_lock, stderr=True)
        return 1
    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    scheduler_summary.update(aggregation_status="complete", completed_at=completed_at)
    write_json(run_dir / "selector_parallel_summary.json", scheduler_summary)
    (run_dir / "RUN_COMPLETE").write_text(completed_at + "\n", encoding="utf-8")
    console_print(f"[selector-scheduler] completed run_dir={run_dir.relative_to(ROOT)}", output_lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
