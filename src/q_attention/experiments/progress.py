from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from datetime import datetime, timedelta
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, TypeVar


T = TypeVar("T")

_GPU_MEMORY_SAMPLE_INTERVAL_SECONDS = 5.0
_GPU_MEMORY_CACHE: list[dict[str, Any]] | None = None
_GPU_MEMORY_CACHE_AT: float | None = None
_GPU_MEMORY_CACHE_KEY: tuple[str | None, int] | None = None


def _gpu_memory_snapshot(*, force: bool = False) -> list[dict[str, Any]] | None:
    """Collect process and device memory without making CPU runs depend on CUDA."""
    global _GPU_MEMORY_CACHE, _GPU_MEMORY_CACHE_AT, _GPU_MEMORY_CACHE_KEY
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device_count = torch.cuda.device_count()
    except (ImportError, RuntimeError, OSError):
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cache_key = (visible, device_count)
    now = time.monotonic()
    if (
        not force
        and _GPU_MEMORY_CACHE_KEY == cache_key
        and _GPU_MEMORY_CACHE_AT is not None
        and now - _GPU_MEMORY_CACHE_AT < _GPU_MEMORY_SAMPLE_INTERVAL_SECONDS
    ):
        return _GPU_MEMORY_CACHE

    physical_ids: list[int | None]
    if visible:
        fields = [field.strip() for field in visible.split(",")]
        physical_ids = [int(field) if field.isdigit() else None for field in fields]
    else:
        physical_ids = list(range(device_count))
    if len(physical_ids) < device_count:
        physical_ids.extend([None] * (device_count - len(physical_ids)))

    nvidia_rows: dict[int, dict[str, int]] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) != 4:
                continue
            try:
                index, used, free, total = (int(field) for field in fields)
            except ValueError:
                continue
            nvidia_rows[index] = {
                "nvidia_used_mib": used,
                "nvidia_free_mib": free,
                "nvidia_total_mib": total,
            }

    snapshots: list[dict[str, Any]] = []
    for local_index in range(device_count):
        try:
            allocated = torch.cuda.memory_allocated(local_index)
            reserved = torch.cuda.memory_reserved(local_index)
            peak_allocated = torch.cuda.max_memory_allocated(local_index)
            peak_reserved = torch.cuda.max_memory_reserved(local_index)
        except (RuntimeError, OSError):
            continue
        physical_index = physical_ids[local_index]
        item: dict[str, Any] = {
            "local_index": local_index,
            "physical_index": physical_index,
            "allocated_mib": round(allocated / (1024 * 1024), 1),
            "reserved_mib": round(reserved / (1024 * 1024), 1),
            "peak_allocated_mib": round(peak_allocated / (1024 * 1024), 1),
            "peak_reserved_mib": round(peak_reserved / (1024 * 1024), 1),
        }
        if physical_index is not None and physical_index in nvidia_rows:
            item.update(nvidia_rows[physical_index])
        snapshots.append(item)
    _GPU_MEMORY_CACHE = snapshots or None
    _GPU_MEMORY_CACHE_AT = now
    _GPU_MEMORY_CACHE_KEY = cache_key
    return _GPU_MEMORY_CACHE


def _format_gib(value: Any) -> str | None:
    try:
        return f"{float(value) / 1024:.1f}"
    except (TypeError, ValueError):
        return None


def format_gpu_memory(value: Any) -> str:
    """Format a compact memory summary for the human-readable console UI."""
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = item.get("physical_index")
        if label is None:
            label = f"local{item.get('local_index', '?')}"
        used = _format_gib(item.get("nvidia_used_mib"))
        total = _format_gib(item.get("nvidia_total_mib"))
        free = _format_gib(item.get("nvidia_free_mib"))
        allocated = _format_gib(item.get("allocated_mib"))
        reserved = _format_gib(item.get("reserved_mib"))
        peak = _format_gib(item.get("peak_reserved_mib"))
        device_text = (
            f"GPU {label} VRAM {used}/{total} GiB used, free {free} GiB"
            if used is not None and total is not None and free is not None
            else f"GPU {label} VRAM unavailable"
        )
        process_text = (
            f"proc {allocated}/{reserved} GiB"
            f" alloc/reserved, peak {peak} GiB"
            if allocated is not None and reserved is not None and peak is not None
            else "proc memory unavailable"
        )
        parts.append(f"{device_text} | {process_text}")
    return " ; ".join(parts)


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_text(payload: dict[str, Any]) -> str | None:
    event = payload["event"]
    stage = payload.get("stage", "run")
    phase = payload.get("phase")
    label = f"[{stage}]" + (f"[{phase}]" if phase else "")
    epoch = payload.get("epoch")
    epochs = payload.get("epochs")
    epoch_text = f" epoch {epoch}/{epochs}" if epoch is not None and epochs is not None else ""

    if event == "phase_start":
        return f"{label}{epoch_text} started | batches={payload.get('batches', '?')}"
    if event == "batch_start":
        memory = format_gpu_memory(payload.get("gpu_memory"))
        memory_text = f" | {memory}" if memory else ""
        return (
            f"{label}{epoch_text} IN PROGRESS | "
            f"batch {payload.get('batch', '?')}/{payload.get('batches', '?')} | "
            "awaiting first completed update"
            f"{memory_text}"
        )
    if event == "batch_progress":
        percent = float(payload.get("percent", 0.0))
        width = 20
        filled = min(max(int(round(width * percent / 100.0)), 0), width)
        bar = "#" * filled + "-" * (width - filled)
        finish = payload.get("estimated_completion_time", "unknown")
        memory = format_gpu_memory(payload.get("gpu_memory"))
        memory_text = f" | {memory}" if memory else ""
        return (
            f"{label}{epoch_text} [{bar}] {percent:6.2f}% "
            f"batch {payload.get('batch', '?')}/{payload.get('batches', '?')} | "
            f"elapsed {_format_duration(payload.get('elapsed_seconds'))} | "
            f"ETA {_format_duration(payload.get('eta_seconds'))} | "
            f"finish {finish} | {payload.get('batches_per_second', 0.0)} batch/s"
            f"{memory_text}"
        )
    if event == "phase_complete":
        return (
            f"{label}{epoch_text} complete | batches={payload.get('completed_batches', '?')} | "
            f"elapsed {_format_duration(payload.get('elapsed_seconds'))}"
        )
    return None


def log_event(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **fields,
    }
    memory = _gpu_memory_snapshot(force=event != "batch_progress")
    if memory is not None:
        payload["gpu_memory"] = memory
    heartbeat_value = os.environ.get("Q_ATTENTION_HEARTBEAT_FILE")
    if heartbeat_value:
        heartbeat = Path(heartbeat_value)
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        temporary = heartbeat.with_name(f".{heartbeat.name}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, heartbeat)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if os.environ.get("Q_ATTENTION_PROGRESS_FORMAT", "json") == "both":
        progress_text = _progress_text(payload)
        if progress_text is not None:
            print(progress_text, flush=True)


def limit_batches(loader: Iterable[T] | Sized, max_batches: int) -> tuple[Iterable[T], int]:
    if max_batches < 0:
        raise ValueError("max_batches must be non-negative")
    total_batches = len(loader)  # type: ignore[arg-type]
    if max_batches == 0:
        return loader, total_batches  # type: ignore[return-value]
    return islice(loader, max_batches), min(total_batches, max_batches)  # type: ignore[arg-type]


def tracked_batches(
    batches: Iterable[T],
    *,
    total_batches: int,
    stage: str,
    phase: str,
    log_every_batches: int,
    epoch: int | None = None,
    epochs: int | None = None,
    completed_batches: int = 0,
) -> Iterator[T]:
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if log_every_batches <= 0:
        raise ValueError("log_every_batches must be positive")
    if not 0 <= completed_batches <= total_batches:
        raise ValueError("completed_batches must be within the total batch range")

    context: dict[str, Any] = {
        "stage": stage,
        "phase": phase,
        "batches": total_batches,
    }
    if epoch is not None:
        context["epoch"] = epoch
    if epochs is not None:
        context["epochs"] = epochs

    started_at = time.monotonic()
    log_event("phase_start", **context)
    completed = completed_batches
    for completed, batch in enumerate(batches, start=completed_batches + 1):
        # Publish an explicit pre-update heartbeat.  A slow first batch is
        # therefore shown as in progress rather than as completed progress
        # with a fabricated rate or ETA.
        log_event(
            "batch_start",
            **context,
            batch=completed,
            completed_batches=completed - 1,
            percent=round(100.0 * (completed - 1) / total_batches, 2),
            current_batch_elapsed_seconds=0.0,
            batches_per_second=None,
            eta_seconds=None,
            estimated_completion_time=None,
        )
        yield batch
        if (
            completed == 1
            or completed % log_every_batches == 0
            or completed == total_batches
        ):
            elapsed = max(time.monotonic() - started_at, 0.0)
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = max(total_batches - completed, 0)
            eta = remaining / rate if rate > 0.0 else None
            estimated_completion = (
                datetime.now().astimezone() + timedelta(seconds=eta)
                if eta is not None
                else None
            )
            log_event(
                "batch_progress",
                **context,
                batch=completed,
                percent=round(100.0 * completed / total_batches, 2),
                elapsed_seconds=round(elapsed, 1),
                eta_seconds=round(eta, 1) if eta is not None else None,
                estimated_completion_time=(
                    estimated_completion.isoformat(timespec="seconds")
                    if estimated_completion is not None
                    else None
                ),
                batches_per_second=round(rate, 4),
            )

    log_event(
        "phase_complete",
        **context,
        completed_batches=completed,
        elapsed_seconds=round(max(time.monotonic() - started_at, 0.0), 1),
    )


def tracked_limited_batches(
    loader: Iterable[T] | Sized,
    max_batches: int,
    *,
    stage: str,
    phase: str,
    log_every_batches: int,
    epoch: int | None = None,
    epochs: int | None = None,
) -> Iterator[T]:
    batches, total_batches = limit_batches(loader, max_batches)
    return tracked_batches(
        batches,
        total_batches=total_batches,
        stage=stage,
        phase=phase,
        log_every_batches=log_every_batches,
        epoch=epoch,
        epochs=epochs,
    )
