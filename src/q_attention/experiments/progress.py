from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from datetime import datetime
from itertools import islice
import json
import os
from pathlib import Path
import time
from typing import Any, TypeVar


T = TypeVar("T")


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
            log_event(
                "batch_progress",
                **context,
                batch=completed,
                percent=round(100.0 * completed / total_batches, 2),
                elapsed_seconds=round(elapsed, 1),
                eta_seconds=round(eta, 1) if eta is not None else None,
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
