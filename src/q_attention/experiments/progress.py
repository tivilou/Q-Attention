from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from datetime import datetime, timedelta
from itertools import islice
import json
import os
from pathlib import Path
import time
from typing import Any, TypeVar


T = TypeVar("T")


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
    if event == "batch_progress":
        percent = float(payload.get("percent", 0.0))
        width = 20
        filled = min(max(int(round(width * percent / 100.0)), 0), width)
        bar = "#" * filled + "-" * (width - filled)
        finish = payload.get("estimated_completion_time", "unknown")
        return (
            f"{label}{epoch_text} [{bar}] {percent:6.2f}% "
            f"batch {payload.get('batch', '?')}/{payload.get('batches', '?')} | "
            f"elapsed {_format_duration(payload.get('elapsed_seconds'))} | "
            f"ETA {_format_duration(payload.get('eta_seconds'))} | "
            f"finish {finish} | {payload.get('batches_per_second', 0.0)} batch/s"
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
) -> Iterator[T]:
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if log_every_batches <= 0:
        raise ValueError("log_every_batches must be positive")

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
    completed = 0
    for completed, batch in enumerate(batches, start=1):
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
