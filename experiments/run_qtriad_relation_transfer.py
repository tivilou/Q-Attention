#!/usr/bin/env python3
"""Run one complete Re-TACRED Q-TRIAD seed under the frozen contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from q_attention.experiments.relation_steering import (  # noqa: E402
    choose_device,
    load_relation_run,
    make_relation_loader,
)
from q_attention.plugins.q_triad import QTriadAttentionScoreKernel  # noqa: E402
from run_q_causal_value_evidence_relation_smoke import (  # noqa: E402
    materialize_subset,
    resolve_path,
)
from run_q_causal_value_evidence_relation_transfer import (  # noqa: E402
    evaluate,
    metric_delta,
    train_kernel,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "retacred_qtriad_formal_single_seed.json"

AUTO_MIN_FREE_MIB = 8 * 1024
HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "low_memory": {
        "pair_chunk_size": 64,
        "activation_checkpointing": True,
        "reason": "at least one selected GPU has less than 16 GiB total or 12 GiB free",
    },
    "balanced": {
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
        "reason": "at least one selected GPU has less than 40 GiB total or 28 GiB free",
    },
    "high_memory": {
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
        "reason": "all selected GPUs have at least 40 GiB total and 28 GiB free; retain streamed backward safety",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated physical GPU IDs or auto; selectors run as independent workers",
    )
    parser.add_argument(
        "--model-parallel-gpus",
        default=None,
        help="comma-separated physical GPU IDs for layer-sharded model parallelism",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=("config", "auto", "low_memory", "balanced", "high_memory"),
        default="config",
        help="execution-memory profile; auto derives it from selected GPU VRAM",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--started-at-utc", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_model_parallel_gpu_ids(spec: str | None) -> list[int]:
    if spec is None:
        return []
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) < 2 or any(not field.isdigit() for field in fields):
        raise ValueError("--model-parallel-gpus must contain at least two GPU IDs")
    ids = [int(field) for field in fields]
    if len(set(ids)) != len(ids):
        raise ValueError("--model-parallel-gpus must not contain duplicate GPU IDs")
    return ids


def local_model_parallel_devices(physical_gpu_ids: list[int]) -> tuple[torch.device, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        fields = [field.strip() for field in visible.split(",")]
        if all(field.isdigit() for field in fields):
            mapping = {int(field): index for index, field in enumerate(fields)}
            missing = [gpu_id for gpu_id in physical_gpu_ids if gpu_id not in mapping]
            if missing:
                raise ValueError(
                    f"model-parallel GPUs {missing} are not visible in CUDA_VISIBLE_DEVICES={visible}"
                )
            return tuple(torch.device(f"cuda:{mapping[gpu_id]}") for gpu_id in physical_gpu_ids)
    available = torch.cuda.device_count()
    if any(gpu_id >= available for gpu_id in physical_gpu_ids):
        raise ValueError(
            f"model-parallel GPU IDs {physical_gpu_ids} exceed visible device count {available}"
        )
    return tuple(torch.device(f"cuda:{gpu_id}") for gpu_id in physical_gpu_ids)


def query_gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed while discovering GPUs: {result.stderr.strip()}")
    inventory: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        index, name, total, free, used = fields
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
                "memory_used_mib": int(used),
            }
        )
    if not inventory:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return inventory


def query_compute_apps() -> list[dict[str, Any]]:
    """Return a compact list of processes currently using CUDA compute."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    apps: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            continue
        pid, process_name, used_memory = fields
        try:
            apps.append(
                {
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_memory_mib": int(used_memory),
                }
            )
        except ValueError:
            continue
    return apps


def validate_gpu_capacity(
    gpu_ids: list[int],
    inventory: list[dict[str, Any]],
    *,
    phase: str,
) -> None:
    """Stop before a run if any selected GPU is already too busy."""
    selected = {
        int(item["index"]): item
        for item in inventory
        if int(item["index"]) in set(gpu_ids)
    }
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in selected]
    if missing:
        raise RuntimeError(f"{phase}: nvidia-smi did not report selected GPU IDs {missing}")
    insufficient = [
        item
        for item in selected.values()
        if int(item["memory_free_mib"]) < AUTO_MIN_FREE_MIB
    ]
    if not insufficient:
        return
    apps = query_compute_apps()
    app_text = ", ".join(
        f"pid={app['pid']} {app['process_name']} ({app['used_memory_mib']} MiB)"
        for app in apps
    ) or "no process details available"
    details = "; ".join(
        f"GPU {item['index']} free={item['memory_free_mib']} MiB/{item['memory_total_mib']} MiB"
        for item in insufficient
    )
    raise RuntimeError(
        f"{phase}: selected GPU capacity is unsafe ({details}; required at least "
        f"{AUTO_MIN_FREE_MIB} MiB free). Compute apps: {app_text}. "
        "Stop or wait for the competing process, then rerun the unchanged contract."
    )


def resolve_gpu_ids(
    spec: str | None,
    device_name: str,
    inventory: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Validate the requested visible GPUs before creating a run directory."""
    if device_name == "cpu":
        if spec:
            raise ValueError("--gpus cannot be used with --device cpu")
        return []
    auto_mode = bool(spec and spec.strip().lower() == "auto")
    if not auto_mode and not torch.cuda.is_available():
        if device_name == "auto" and spec is None:
            return []
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if spec is None:
        ids = [0]
    elif spec.strip().lower() == "auto":
        inventory = inventory or query_gpu_inventory()
        visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = None
        if visible_spec:
            visible_fields = [field.strip() for field in visible_spec.split(",")]
            if all(field.isdigit() for field in visible_fields):
                allowed = {int(field) for field in visible_fields}
        candidates = [
            item
            for item in inventory
            if allowed is None or int(item["index"]) in allowed
        ]
        ids = [
            int(item["index"])
            for item in candidates
            if int(item["memory_free_mib"]) >= AUTO_MIN_FREE_MIB
        ]
        if not ids:
            raise RuntimeError(
                f"--gpus auto found no GPU with at least {AUTO_MIN_FREE_MIB // 1024} GiB free"
            )
    else:
        fields = [field.strip() for field in spec.split(",")]
        if not fields or any(not field.isdigit() for field in fields):
            raise ValueError("--gpus must be a comma-separated list of non-negative integers")
        ids = [int(field) for field in fields]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("--gpus must contain at least one unique GPU ID")
    if auto_mode:
        return ids
    available = torch.cuda.device_count()
    visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = None
    if visible_spec:
        visible_fields = [field.strip() for field in visible_spec.split(",")]
        if all(field.isdigit() for field in visible_fields):
            visible_ids = {int(field) for field in visible_fields}
    if visible_ids is not None and not set(ids).issubset(visible_ids):
        raise ValueError(f"requested GPU IDs {ids} are not in CUDA_VISIBLE_DEVICES={visible_spec}")
    if visible_ids is None and any(index >= available for index in ids):
        raise ValueError(f"requested GPU IDs {ids} exceed visible device count {available}")
    return ids


def choose_hardware_profile(
    requested: str,
    config: dict[str, Any],
    gpu_ids: list[int],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    if requested == "config":
        return {
            "name": "config",
            "pair_chunk_size": int(config["kernel"].get("pair_chunk_size", 256)),
            "activation_checkpointing": True,
            "selection_reason": "frozen config profile",
        }
    if requested == "auto":
        selected = [item for item in inventory if int(item["index"]) in set(gpu_ids)]
        if not selected:
            raise RuntimeError("no GPU inventory entries match the selected GPU IDs")
        min_total = min(int(item["memory_total_mib"]) for item in selected)
        min_free = min(int(item["memory_free_mib"]) for item in selected)
        if min_total < 16 * 1024 or min_free < 12 * 1024:
            requested = "low_memory"
        elif min_total < 40 * 1024 or min_free < 28 * 1024:
            requested = "balanced"
        else:
            requested = "high_memory"
        profile = dict(HARDWARE_PROFILES[requested])
        profile.update(
            {
                "name": requested,
                "selection_reason": profile.pop("reason"),
                "minimum_memory_total_mib": min_total,
                "minimum_memory_free_mib": min_free,
            }
        )
        return profile
    profile = dict(HARDWARE_PROFILES[requested])
    profile.update({"name": requested, "selection_reason": "explicit profile override"})
    profile.pop("reason", None)
    return profile


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_scheduler_event(run_dir: Path, payload: dict[str, Any]) -> None:
    """Keep machine-readable scheduler events separate from the human console UI."""
    event_path = run_dir / "scheduler_events.jsonl"
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a heartbeat snapshot, tolerating startup races and stale files."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _format_duration(seconds: Any) -> str:
    if seconds is None:
        return "--:--"
    try:
        total = max(int(round(float(seconds))), 0)
    except (TypeError, ValueError):
        return "--:--"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _render_selector_dashboard(
    statuses: dict[str, dict[str, Any]],
    active: dict[str, dict[str, Any]],
) -> str:
    """Render a readable, append-only snapshot for serial and multi-GPU runs."""
    total = len(statuses)
    counts = {
        state: sum(item.get("status") == state for item in statuses.values())
        for state in ("complete", "running", "pending", "failed", "not_started")
    }
    lines = [
        (
            f"Q-TRIAD selectors: {counts['complete']}/{total} complete | "
            f"{counts['running']} running | {counts['pending']} queued | "
            f"{counts['not_started']} not started | {counts['failed']} failed"
        )
    ]
    for selector, item in sorted(statuses.items(), key=lambda pair: (str(pair[1].get("gpu", "?")), pair[0])):
        status = str(item.get("status", "pending")).upper()
        gpu = item.get("gpu") if item.get("gpu") is not None else "-"
        if status != "RUNNING":
            lines.append(f"GPU {gpu} | {selector:<28} | {status.lower()}")
            continue
        heartbeat_value = item.get("heartbeat_file")
        heartbeat = _read_json(Path(str(heartbeat_value))) if heartbeat_value else {}
        progress: list[str] = []
        phase = heartbeat.get("phase") or heartbeat.get("stage")
        if phase:
            progress.append(str(phase))
        epoch, epochs = heartbeat.get("epoch"), heartbeat.get("epochs")
        if epoch is not None and epochs is not None:
            progress.append(f"epoch {epoch}/{epochs}")
        batch, batches = heartbeat.get("batch"), heartbeat.get("batches")
        percent = heartbeat.get("percent")
        if batch is not None and batches is not None:
            try:
                suffix = f" {float(percent):.1f}%" if percent is not None else ""
            except (TypeError, ValueError):
                suffix = ""
            progress.append(f"batch {batch}/{batches}{suffix}")
        if not progress:
            progress.append("starting")
        eta = _format_duration(heartbeat.get("eta_seconds"))
        rate = heartbeat.get("batches_per_second")
        try:
            rate_text = f" | {float(rate):.2f} batch/s" if rate is not None else ""
        except (TypeError, ValueError):
            rate_text = ""
        progress.append(f"ETA {eta}{rate_text}")
        lines.append(f"GPU {gpu} | {selector:<28} | " + " | ".join(progress))

    for label, state in (
        ("Completed", "complete"),
        ("Queued", "pending"),
        ("Not started", "not_started"),
        ("Failed", "failed"),
    ):
        names = [name for name, item in statuses.items() if item.get("status") == state]
        lines.append(f"{label}: {', '.join(names) if names else 'none'}")
    if active:
        updated = []
        for selector, entry in active.items():
            elapsed = time.monotonic() - float(entry["started_monotonic"])
            updated.append(f"{selector} {_format_duration(elapsed)}")
        lines.append("Active time: " + ", ".join(sorted(updated)))
    return "\n".join(lines)


def _render_baseline_line(line: str, *, epochs: int) -> str | None:
    """Render one baseline JSON event without losing the original log line."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return f"[baseline] {stripped}"
    if not isinstance(payload, dict):
        return f"[baseline] {stripped}"
    event = payload.get("event")
    phase = payload.get("phase")
    epoch = payload.get("epoch")
    epoch_text = f" epoch {epoch}/{epochs}" if epoch is not None else ""
    label = f"[baseline]" + (f"[{phase}]" if phase else "")
    if event == "phase_start":
        return f"{label}{epoch_text} started | batches={payload.get('batches', '?')}"
    if event == "batch_progress":
        try:
            percent = float(payload.get("percent", 0.0))
        except (TypeError, ValueError):
            percent = 0.0
        width = 24
        filled = min(max(int(round(width * percent / 100.0)), 0), width)
        bar = "#" * filled + "-" * (width - filled)
        try:
            rate = f" | {float(payload['batches_per_second']):.2f} batch/s"
        except (KeyError, TypeError, ValueError):
            rate = ""
        return (
            f"{label}{epoch_text} [{bar}] {percent:5.1f}% "
            f"batch {payload.get('batch', '?')}/{payload.get('batches', '?')} | "
            f"elapsed {_format_duration(payload.get('elapsed_seconds'))} | "
            f"ETA {_format_duration(payload.get('eta_seconds'))}{rate}"
        )
    if event == "phase_complete":
        return (
            f"{label}{epoch_text} complete | "
            f"batches={payload.get('completed_batches', '?')} | "
            f"elapsed {_format_duration(payload.get('elapsed_seconds'))}"
        )
    if event == "health_warning":
        warning = payload.get("warning") or payload.get("message") or "health warning"
        return f"[baseline] warning | {warning}"
    if "epoch" in payload and isinstance(payload.get("valid"), dict):
        valid = payload["valid"]
        return (
            f"[baseline] epoch {payload['epoch']}/{epochs} complete | "
            f"train_loss={float(payload.get('train_loss', 0.0)):.4f} | "
            f"valid_loss={float(valid.get('loss', 0.0)):.4f} | "
            f"valid_macro_f1={float(valid.get('macro_f1', 0.0)):.4f}"
        )
    if "output_dir" in payload and "best_valid" in payload:
        best_valid = payload.get("best_valid") or {}
        return (
            "[baseline] complete | "
            f"best_valid_macro_f1={float(best_valid.get('macro_f1', 0.0)):.4f}"
        )
    return f"[baseline] event={event or 'json'}"


def _run_baseline_logged_command(
    command: list[str],
    log_path: Path,
    heartbeat_path: Path,
    *,
    epochs: int,
) -> dict[str, Any]:
    """Run baseline with a readable console while retaining its raw JSON log."""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "Q_ATTENTION_PROGRESS_FORMAT": "json",
            "Q_ATTENTION_HEARTBEAT_FILE": str(heartbeat_path),
        }
    )
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[baseline] starting | epochs={epochs}", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
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
            log_handle.write(line)
            log_handle.flush()
            rendered = _render_baseline_line(line, epochs=epochs)
            if rendered is not None:
                print(rendered, flush=True)
        return_code = process.wait()
    result = {
        "command": command,
        "returncode": return_code,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "log_path": str(log_path),
    }
    if return_code != 0:
        print(f"[baseline] failed | exit_code={return_code}", file=sys.stderr, flush=True)
        raise RuntimeError(f"command failed with return code {return_code}: {command}")
    print(f"[baseline] process complete | elapsed {_format_duration(result['duration_seconds'])}", flush=True)
    return result


def _terminate_worker(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def run_selector_workers(
    *,
    selectors: list[str],
    gpu_ids: list[int],
    args: argparse.Namespace,
    config_path: Path,
    baseline_dir: Path,
    data_dir: Path,
    run_dir: Path,
    seed: int,
    hardware_profile: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run independent selectors with one dynamically scheduled worker per GPU."""
    hardware_profile = hardware_profile or {
        "name": "config",
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    pending = list(selectors)
    available = list(gpu_ids)
    statuses: dict[str, dict[str, Any]] = {
        selector: {"selector": selector, "status": "pending", "gpu": None}
        for selector in selectors
    }
    active: dict[str, dict[str, Any]] = {}
    assignments_path = run_dir / "gpu_assignments.json"
    _write_json_atomic(
        assignments_path,
        {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
    )
    dashboard_at = 0.0

    def fail_run(reason: str) -> None:
        for entry in active.values():
            _terminate_worker(entry["process"])
            entry["handle"].close()
        now = datetime.now(timezone.utc).isoformat()
        for selector in pending:
            statuses[selector].update({"status": "not_started", "finished_at": now})
        _write_json_atomic(
            assignments_path,
            {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
        )
        (run_dir / "RUN_FAILED").write_text(
            json.dumps(
                {
                    "failed_at_utc": now,
                    "reason": reason,
                    "workers": statuses,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    try:
        while pending or active:
            while pending and available:
                selector = pending.pop(0)
                gpu_id = available.pop(0)
                selector_dir = run_dir / "selectors" / selector
                selector_dir.mkdir(parents=True, exist_ok=True)
                log_handle = (selector_dir / "worker.log").open("w", encoding="utf-8")
                heartbeat_path = selector_dir / "heartbeat.json"
                heartbeat_path.touch()
                command = [
                    args.python_bin,
                    str(ROOT / "experiments" / "run_qtriad_selector_worker.py"),
                    "--config", str(config_path),
                    "--baseline-dir", str(baseline_dir),
                    "--data-dir", str(data_dir),
                    "--output-dir", str(selector_dir),
                    "--selector", selector,
                    "--device", "cuda" if gpu_id >= 0 else "cpu",
                    "--seed", str(seed),
                    "--log-every-batches", str(args.log_every_batches),
                    "--pair-chunk-size", str(hardware_profile["pair_chunk_size"]),
                    "--activation-checkpointing", str(int(hardware_profile["activation_checkpointing"])),
                ]
                environment = os.environ.copy()
                if gpu_id >= 0:
                    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                environment["PYTHONUNBUFFERED"] = "1"
                environment["Q_ATTENTION_PROGRESS_FORMAT"] = "json"
                environment["Q_ATTENTION_HEARTBEAT_FILE"] = str(heartbeat_path)
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                started_at = datetime.now(timezone.utc).isoformat()
                active[selector] = {
                    "process": process,
                    "handle": log_handle,
                    "gpu": gpu_id,
                    "started_at": started_at,
                    "started_monotonic": time.monotonic(),
                    "heartbeat_file": heartbeat_path,
                }
                statuses[selector].update(
                    {
                        "status": "running",
                        "gpu": gpu_id,
                        "pid": process.pid,
                        "started_at": started_at,
                        "heartbeat_file": str(heartbeat_path),
                        "log_file": str(selector_dir / "worker.log"),
                    }
                )
                _write_json_atomic(
                    assignments_path,
                    {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
                )
                _append_scheduler_event(
                    run_dir,
                    {
                        "event": "selector_started",
                        "selector": selector,
                        "gpu": gpu_id,
                        "pid": process.pid,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                )
                print(f"[selector-scheduler] started {selector} on GPU {gpu_id}", flush=True)

            for selector, entry in list(active.items()):
                process = entry["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                entry["handle"].close()
                finished_at = datetime.now(timezone.utc).isoformat()
                duration = round(time.monotonic() - entry["started_monotonic"], 3)
                if return_code != 0:
                    statuses[selector].update(
                        {"status": "failed", "return_code": return_code, "finished_at": finished_at, "duration_seconds": duration}
                    )
                    _write_json_atomic(assignments_path, {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses})
                    fail_run(f"selector worker {selector} failed with exit code {return_code}")
                    raise RuntimeError(f"selector worker {selector} failed; inspect {run_dir / 'selectors' / selector / 'worker.log'}")
                metrics_path = run_dir / "selectors" / selector / "metrics.json"
                if not metrics_path.exists():
                    statuses[selector].update(
                        {"status": "failed", "return_code": return_code, "finished_at": finished_at, "duration_seconds": duration}
                    )
                    fail_run(f"selector worker {selector} exited without metrics.json")
                    raise RuntimeError(f"selector worker {selector} produced no metrics.json")
                statuses[selector].update(
                    {"status": "complete", "return_code": 0, "finished_at": finished_at, "duration_seconds": duration}
                )
                available.append(int(entry["gpu"]))
                del active[selector]
                _write_json_atomic(assignments_path, {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses})
                _append_scheduler_event(
                    run_dir,
                    {
                        "event": "selector_complete",
                        "selector": selector,
                        "gpu": entry["gpu"],
                        "duration_seconds": duration,
                        "timestamp": finished_at,
                    },
                )
                print(
                    f"[selector-scheduler] complete {selector} on GPU {entry['gpu']} "
                    f"in {_format_duration(duration)}",
                    flush=True,
                )

            now = time.monotonic()
            if now - dashboard_at >= 30.0:
                dashboard_at = now
                counts = {state: sum(item["status"] == state for item in statuses.values()) for state in ("pending", "running", "complete", "failed", "not_started")}
                _append_scheduler_event(
                    run_dir,
                    {
                        "event": "selector_dashboard",
                        **counts,
                        "active_gpus": {
                            selector: item["gpu"]
                            for selector, item in statuses.items()
                            if item["status"] == "running"
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                )
                print(_render_selector_dashboard(statuses, active), flush=True)
            if active:
                time.sleep(0.5)
    except BaseException as exc:
        if not (run_dir / "RUN_FAILED").exists():
            fail_run(f"selector scheduler aborted: {type(exc).__name__}: {exc}")
        raise
    return statuses


def build_kernel(
    mode: str,
    model: torch.nn.Module,
    seed: int,
    config: dict[str, Any],
    *,
    pair_chunk_size: int | None = None,
    activation_checkpointing: bool | None = None,
    model_parallel_devices: tuple[torch.device, ...] = (),
) -> QTriadAttentionScoreKernel:
    kernel_config = config["kernel"]
    kernel = QTriadAttentionScoreKernel(
        num_layers=model.config.num_layers,
        num_heads=model.config.num_heads,
        head_dim=model.config.dim // model.config.num_heads,
        num_qubits=int(kernel_config["num_qubits"]),
        circuit_depth=int(kernel_config["circuit_depth"]),
        angle_scale=float(kernel_config["angle_scale"]),
        max_gain=float(kernel_config["max_gain"]),
        initial_gain=float(kernel_config["initial_gain"]),
        seed=seed + 307,
        control_mode=mode,
        pair_chunk_size=int(pair_chunk_size if pair_chunk_size is not None else kernel_config.get("pair_chunk_size", 256)),
        activation_checkpointing=(
            bool(activation_checkpointing)
            if activation_checkpointing is not None
            else True
        ),
    )
    if model_parallel_devices:
        kernel.configure_model_parallel(model_parallel_devices)
    return kernel


def evaluate_selector(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    label_count: int,
    kernel: QTriadAttentionScoreKernel | None,
    stage: str,
) -> dict[str, Any]:
    return evaluate(
        model,
        loader,
        device,
        label_count,
        kernel=kernel,
        stage=stage,
        log_every_batches=50,
        collect_geometry=True,
    )


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.qtriad-formal-single-seed.v1":
        raise ValueError("unsupported Q-TRIAD formal config")
    seed = int(config["seed"] if args.seed is None else args.seed)
    if seed != 13:
        raise ValueError("the formal handoff contract is frozen to seed 13")
    selectors = list(config["selectors"])
    if selectors[0] != "disabled" or config["candidate"] not in selectors:
        raise ValueError("config must include disabled and the candidate selector")
    if config["matched_control"] not in selectors:
        raise ValueError("config must include the matched classical control")
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive")
    model_parallel_gpu_ids = parse_model_parallel_gpu_ids(args.model_parallel_gpus)
    if model_parallel_gpu_ids and args.gpus:
        raise ValueError("--gpus/selector-parallel cannot be combined with --model-parallel-gpus")
    if model_parallel_gpu_ids:
        if args.device == "cpu":
            raise ValueError("model parallelism requires CUDA")
        inventory = query_gpu_inventory()
        validate_gpu_capacity(model_parallel_gpu_ids, inventory, phase="before model-parallel baseline")
        model_parallel_devices = local_model_parallel_devices(model_parallel_gpu_ids)
        gpu_ids: list[int] = []
        profile_gpu_ids = model_parallel_gpu_ids
    else:
        model_parallel_devices = ()
        inventory = query_gpu_inventory() if args.gpus and args.gpus.strip().lower() == "auto" else []
        if args.gpus and args.gpus.strip().lower() != "auto" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()
        gpu_ids = resolve_gpu_ids(args.gpus, args.device, inventory)
        if gpu_ids and not inventory:
            inventory = query_gpu_inventory()
        if gpu_ids and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        if gpu_ids:
            inventory = query_gpu_inventory()
            validate_gpu_capacity(gpu_ids, inventory, phase="before baseline")
        profile_gpu_ids = gpu_ids
    profile_request = "auto" if args.gpus and args.gpus.strip().lower() == "auto" and args.hardware_profile == "config" else args.hardware_profile
    hardware_profile = choose_hardware_profile(profile_request, config, profile_gpu_ids, inventory)
    hardware_profile.update(
        {
            "requested_gpu_spec": args.model_parallel_gpus or args.gpus or "default",
            "selected_gpu_ids": profile_gpu_ids,
            "gpu_inventory": inventory,
        }
    )
    device = model_parallel_devices[0] if model_parallel_devices else (torch.device("cuda:0") if gpu_ids else choose_device(args.device))
    stamp = args.started_at_utc or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if len(stamp) != 16 or not stamp.endswith("Z"):
        raise ValueError("--started-at-utc must use UTC format YYYYMMDDTHHMMSSZ")
    run_dir = args.output_dir or ROOT / "runs" / "retacred_qtriad_formal_single_seed" / f"{stamp}_seed13"
    run_dir = resolve_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_source = resolve_path(config["train_path"])
    valid_source = resolve_path(config["valid_path"])
    test_source = resolve_path(config["test_path"])
    valid_info = materialize_subset(valid_source, data_dir / "valid.jsonl", 0, seed=seed + 101, split="valid")
    valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
    required_labels = {record.label for record in valid_records}
    train_info = materialize_subset(train_source, data_dir / "train.jsonl", 0, seed=seed, split="train", required_labels=required_labels)
    test_info = materialize_subset(test_source, data_dir / "test.jsonl", 0, seed=seed + 211, split="test")
    train_records = load_relation_jsonl(data_dir / "train.jsonl")
    test_records = load_relation_jsonl(data_dir / "test.jsonl")
    expected = config["expected_records"]
    for name, info in (("train", train_info), ("valid", valid_info), ("test", test_info)):
        if int(info["records"]) != int(expected[name]):
            raise ValueError(f"{name} record count differs from frozen contract")
    max_length = max(len(record.tokens) for record in train_records + valid_records + test_records) + 4
    model_config = config["model"]
    baseline_dir = run_dir / "baseline"
    baseline_command = [
        args.python_bin,
        str(ROOT / "experiments" / "train_relation_baseline.py"),
        "--train_path", str(data_dir / "train.jsonl"),
        "--valid_path", str(data_dir / "valid.jsonl"),
        "--output_dir", str(baseline_dir),
        "--device", "cuda" if gpu_ids else str(device),
        "--epochs", str(config["baseline"]["epochs"]),
        "--batch_size", str(config["baseline"]["batch_size"]),
        "--lr", str(config["baseline"]["lr"]),
        "--dim", str(model_config["dim"]),
        "--num_layers", str(model_config["num_layers"]),
        "--num_heads", str(model_config["num_heads"]),
        "--ff_dim", str(model_config["ff_dim"]),
        "--dropout", str(model_config["dropout"]),
        "--max_length", str(max_length),
        "--seed", str(seed),
    ]
    if model_parallel_gpu_ids:
        baseline_command.extend(
            ["--model-parallel-gpus", ",".join(str(gpu_id) for gpu_id in model_parallel_gpu_ids)]
        )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _run_baseline_logged_command(
        baseline_command,
        run_dir / "baseline_train.log",
        baseline_dir / "heartbeat.json",
        epochs=int(config["baseline"]["epochs"]),
    )
    artifacts = load_relation_run(
        baseline_dir,
        device,
        model_parallel_devices=model_parallel_devices,
    )
    valid_loader = make_relation_loader(valid_records, artifacts.vocab, artifacts.label_to_id, batch_size=int(config["kernel"]["batch_size"]))
    test_loader = make_relation_loader(test_records, artifacts.vocab, artifacts.label_to_id, batch_size=int(config["kernel"]["batch_size"]))
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    baseline_valid = evaluate_selector(artifacts.model, valid_loader, device, len(artifacts.label_to_id), None, "baseline_valid")
    baseline_test = evaluate_selector(artifacts.model, test_loader, device, len(artifacts.label_to_id), None, "baseline_test")
    (run_dir / "baseline_eval.json").write_text(
        json.dumps({"valid": baseline_valid, "test": baseline_test}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    disabled_dir = run_dir / "selectors" / "disabled"
    disabled_dir.mkdir(parents=True, exist_ok=True)
    disabled_row = {
        "selector": "disabled",
        "seed": seed,
        "valid": baseline_valid,
        "test": {**baseline_test, "delta_vs_baseline": metric_delta(baseline_test["metrics"], baseline_test["metrics"])},
        "train": {"history": [], "best_epoch": 0, "runtime_seconds": 0.0},
        "metadata": {"type": "disabled"},
        "trainable_parameters": 0,
        "finite": True,
    }
    (disabled_dir / "metrics.json").write_text(
        json.dumps(disabled_row, indent=2, sort_keys=True), encoding="utf-8"
    )
    if model_parallel_devices:
        # Model-parallel mode keeps one sharded model alive and runs selectors
        # serially; independent selector workers would each duplicate the
        # sharded model and defeat the memory purpose of this option.
        worker_statuses = {}
        train_args = argparse.Namespace(
            batch_size=int(config["kernel"]["batch_size"]),
            epochs=int(config["kernel"]["epochs"]),
            kernel_lr=float(config["kernel"]["lr"]),
            log_every_batches=args.log_every_batches,
        )
        pending_selectors = [selector for selector in selectors if selector != "disabled"]
        for selector in pending_selectors:
            worker_statuses[selector] = {
                "selector": selector,
                "status": "pending",
                "physical_gpu_ids": model_parallel_gpu_ids,
            }
        _write_json_atomic(
            run_dir / "gpu_assignments.json",
            {
                "parallel_mode": "model_parallel",
                "requested_gpu_ids": model_parallel_gpu_ids,
                "hardware_profile": hardware_profile,
                "workers": worker_statuses,
            },
        )
        current_selector: str | None = None
        try:
            for selector in pending_selectors:
                current_selector = selector
                selector_dir = run_dir / "selectors" / selector
                selector_dir.mkdir(parents=True, exist_ok=True)
                started_at = datetime.now(timezone.utc).isoformat()
                started = time.perf_counter()
                worker_statuses[selector].update({"status": "running", "started_at": started_at})
                _write_json_atomic(
                    run_dir / "gpu_assignments.json",
                    {
                        "parallel_mode": "model_parallel",
                        "requested_gpu_ids": model_parallel_gpu_ids,
                        "hardware_profile": hardware_profile,
                        "workers": worker_statuses,
                    },
                )
                kernel = build_kernel(
                    selector,
                    artifacts.model,
                    seed,
                    config,
                    pair_chunk_size=int(hardware_profile["pair_chunk_size"]),
                    activation_checkpointing=bool(hardware_profile["activation_checkpointing"]),
                    model_parallel_devices=model_parallel_devices,
                )
                train_result = train_kernel(
                    artifacts.model,
                    kernel,
                    train_records,
                    valid_loader,
                    artifacts,
                    device,
                    selector,
                    seed,
                    train_args,
                    selector_dir,
                )
                valid_result = evaluate_selector(
                    artifacts.model,
                    valid_loader,
                    device,
                    len(artifacts.label_to_id),
                    kernel,
                    f"{selector}_valid_final",
                )
                test_result = evaluate_selector(
                    artifacts.model,
                    test_loader,
                    device,
                    len(artifacts.label_to_id),
                    kernel,
                    f"{selector}_test",
                )
                metadata = kernel.metadata()
                trainable_parameters = sum(parameter.numel() for parameter in kernel.parameters())
                row = {
                    "selector": selector,
                    "seed": seed,
                    "valid": valid_result,
                    "test": {
                        **test_result,
                        "delta_vs_baseline": metric_delta(test_result["metrics"], baseline_test["metrics"]),
                    },
                    "train": train_result,
                    "metadata": metadata,
                    "trainable_parameters": trainable_parameters,
                    "finite": all(
                        torch.isfinite(torch.tensor(value))
                        for value in list(valid_result["metrics"].values())
                        + list(test_result["metrics"].values())
                    ),
                }
                torch.save(
                    {"state_dict": kernel.state_dict(), "metadata": metadata},
                    selector_dir / "best_kernel_with_metadata.pt",
                )
                (selector_dir / "metrics.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
                )
                worker_statuses[selector].update(
                    {
                        "status": "complete",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                _write_json_atomic(
                    run_dir / "gpu_assignments.json",
                    {
                        "parallel_mode": "model_parallel",
                        "requested_gpu_ids": model_parallel_gpu_ids,
                        "hardware_profile": hardware_profile,
                        "workers": worker_statuses,
                    },
                )
                del kernel
                gc.collect()
                torch.cuda.empty_cache()
                current_selector = None
        except BaseException as exc:
            now = datetime.now(timezone.utc).isoformat()
            if current_selector is not None and worker_statuses[current_selector]["status"] == "running":
                worker_statuses[current_selector].update(
                    {
                        "status": "failed",
                        "finished_at": now,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            for selector, status in worker_statuses.items():
                if status["status"] == "pending":
                    status.update({"status": "not_started", "finished_at": now})
            _write_json_atomic(
                run_dir / "gpu_assignments.json",
                {
                    "parallel_mode": "model_parallel",
                    "requested_gpu_ids": model_parallel_gpu_ids,
                    "hardware_profile": hardware_profile,
                    "workers": worker_statuses,
                },
            )
            (run_dir / "RUN_FAILED").write_text(
                json.dumps(
                    {
                        "failed_at_utc": now,
                        "stage": "model_parallel_selectors",
                        "failed_selector": current_selector,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "workers": worker_statuses,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
        _write_json_atomic(
            run_dir / "gpu_assignments.json",
            {
                "parallel_mode": "model_parallel",
                "requested_gpu_ids": model_parallel_gpu_ids,
                "hardware_profile": hardware_profile,
                "workers": worker_statuses,
            },
        )
    else:
        # Release the parent's model copy before selector workers load the shared
        # checkpoint. This is important when the first worker uses GPU 0.
        artifacts.model.to("cpu")
        del valid_loader, test_loader
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        del artifacts
        del train_records
        if gpu_ids:
            gc.collect()
            torch.cuda.empty_cache()
            inventory = query_gpu_inventory()
            validate_gpu_capacity(gpu_ids, inventory, phase="before selector workers")
        worker_selectors = [selector for selector in selectors if selector != "disabled"]
        worker_statuses = run_selector_workers(
            selectors=worker_selectors,
            gpu_ids=gpu_ids or [-1],
            args=args,
            config_path=config_path,
            baseline_dir=baseline_dir,
            data_dir=data_dir,
            run_dir=run_dir,
            seed=seed,
            hardware_profile=hardware_profile,
        )
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        metrics_path = run_dir / "selectors" / selector / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"missing selector metrics: {metrics_path}")
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    by_name = {row["selector"]: row for row in rows}
    candidate = by_name[config["candidate"]]
    matched = by_name[config["matched_control"]]
    disabled = by_name["disabled"]
    candidate_minus_disabled = metric_delta(candidate["test"]["metrics"], disabled["test"]["metrics"])
    candidate_minus_matched = metric_delta(candidate["test"]["metrics"], matched["test"]["metrics"])
    gates = {
        "candidate_minus_disabled_macro_f1": candidate_minus_disabled["delta_macro_f1"],
        "candidate_minus_matched_macro_f1": candidate_minus_matched["delta_macro_f1"],
        "practical_gain_gate": candidate_minus_disabled["delta_macro_f1"] >= float(config["gates"]["minimum_candidate_minus_disabled_macro_f1"]),
        "matched_comparator_gate": candidate_minus_matched["delta_macro_f1"] >= float(config["gates"]["minimum_candidate_minus_matched_macro_f1"]),
        "finite_metrics": all(row["finite"] for row in rows),
        "test_used_for_training_or_selection": False,
    }
    summary = {
        "schema_version": "q-attention.qtriad-formal-single-seed.run.v1",
        "name": config["name"],
        "formal_experiment": True,
        "stage": "formal_single_seed",
        "seed": seed,
        "run_dir": str(run_dir),
        "device": str(device),
        "parallel_mode": "model_parallel" if model_parallel_devices else ("selector_parallel" if gpu_ids else "serial"),
        "model_parallel": {
            "enabled": bool(model_parallel_devices),
            "physical_gpu_ids": model_parallel_gpu_ids,
            "device_map": artifacts.model.model_parallel_metadata() if model_parallel_devices else {"enabled": False},
        },
        "multi_gpu": {
            "requested_gpu_ids": gpu_ids,
            "selector_parallelism": len(gpu_ids),
            "worker_statuses": worker_statuses,
        },
        "hardware_profile": hardware_profile,
        "selectors": selectors,
        "candidate": config["candidate"],
        "matched_control": config["matched_control"],
        "data": {"train": train_info, "valid": valid_info, "test": test_info},
        "baseline": {"valid": baseline_valid, "test": baseline_test, "command": baseline_command},
        "rows": rows,
        "candidate_minus_disabled": candidate_minus_disabled,
        "candidate_minus_matched": candidate_minus_matched,
        "gates": gates,
        "test_used_for_training_or_selection": False,
        "claim_limits": config["claim_limits"],
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "git_revision": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "visible_cuda_devices": gpu_ids,
            "model_parallel_gpu_ids": model_parallel_gpu_ids,
            "model_parallel_device_map": artifacts.model.model_parallel_metadata() if model_parallel_devices else {"enabled": False},
            "pair_chunk_size": int(hardware_profile["pair_chunk_size"]),
            "activation_checkpointing": bool(hardware_profile["activation_checkpointing"]),
            "started_at_utc": stamp,
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Q-TRIAD Re-TACRED Formal Single Seed",
        "",
        "This is one complete seed-13 run under the frozen natural-task contract.",
        "",
        f"- candidate: `{config['candidate']}`",
        f"- matched control: `{config['matched_control']}`",
        f"- parallel mode: `{'model_parallel' if model_parallel_devices else ('selector_parallel' if gpu_ids else 'serial')}`",
        f"- model-parallel physical GPUs: `{model_parallel_gpu_ids}`",
        f"- hardware profile: `{hardware_profile['name']}` (pair_chunk_size={hardware_profile['pair_chunk_size']}, activation_checkpointing={str(hardware_profile['activation_checkpointing']).lower()})",
        f"- selected physical GPUs: `{gpu_ids}`",
        f"- candidate minus disabled test macro-F1: `{candidate_minus_disabled['delta_macro_f1']:.6f}`",
        f"- candidate minus matched test macro-F1: `{candidate_minus_matched['delta_macro_f1']:.6f}`",
        f"- practical gain gate: `{str(gates['practical_gain_gate']).lower()}`",
        f"- matched comparator gate: `{str(gates['matched_comparator_gate']).lower()}`",
        "",
        "The test split is evaluated only after training and validation selection. This single seed does not authorize multi-seed replication.",
    ]
    (run_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "RUN_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
