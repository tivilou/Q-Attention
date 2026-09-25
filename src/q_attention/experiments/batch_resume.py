"""Atomic batch-level checkpointing primitives for deterministic training resume."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import tempfile
from typing import Any

import torch
from torch.utils.data import Sampler


CHECKPOINT_SCHEMA = "q-attention.batch-resume.v1"
PAUSED_EXIT_CODE = 75


class ResumeCompatibilityError(RuntimeError):
    """A requested resume would change declared training semantics."""


class TrainingPaused(RuntimeError):
    """Raised only after the current optimizer update has been checkpointed."""


class TrainingMemoryPressure(RuntimeError):
    """Raised after a post-update checkpoint when safe CUDA reclaim was insufficient."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


ResumeContractCompatibility = Callable[
    [Mapping[str, Any], Mapping[str, Any]], bool
]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def execution_contract_compatible(
    persisted: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    ignored_source_files: Sequence[str] = (),
) -> bool:
    """Compare a checkpoint contract while allowing an explicit execution migration.

    Scientific/data fields remain exact. Only the source revision and named
    orchestration files may differ, and callers must gate this check behind an
    explicit migration option.
    """
    try:
        left = json.loads(json.dumps(dict(persisted)))
        right = json.loads(json.dumps(dict(current)))
    except (TypeError, ValueError):
        return False
    for contract in (left, right):
        source = contract.get("source")
        if not isinstance(source, dict):
            continue
        source.pop("git_revision", None)
        files = source.get("files")
        if isinstance(files, dict):
            for name in ignored_source_files:
                files.pop(name, None)
    return left == right


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _torch_load(path: Path) -> Any:
    # Checkpoints are trusted local artifacts written by this runner and include
    # optimizer/RNG objects, which cannot be read with weights_only=True.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError as exc:
            raise ResumeCompatibilityError("checkpoint requires NumPy RNG restoration but NumPy is unavailable") from exc


def epoch_permutation(size: int, *, seed: int, epoch: int) -> list[int]:
    if size <= 0:
        raise ValueError("dataset must contain at least one record")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) * 1_000_003 + int(epoch)) % (2**63 - 1))
    return torch.randperm(size, generator=generator).tolist()


def batch_count(size: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (size + batch_size - 1) // batch_size


class RemainingBatchSampler(Sampler[list[int]]):
    """Yield a stored epoch permutation from its next uncommitted batch."""

    def __init__(self, permutation: Sequence[int], batch_size: int, next_batch_index: int) -> None:
        self.permutation = list(permutation)
        self.batch_size = int(batch_size)
        self.next_batch_index = int(next_batch_index)
        total = batch_count(len(self.permutation), self.batch_size)
        if not 0 <= self.next_batch_index <= total:
            raise ValueError("next_batch_index is outside the epoch batch range")

    def __iter__(self):
        for batch_index in range(self.next_batch_index, batch_count(len(self.permutation), self.batch_size)):
            start = batch_index * self.batch_size
            yield self.permutation[start : start + self.batch_size]

    def __len__(self) -> int:
        return batch_count(len(self.permutation), self.batch_size) - self.next_batch_index


@dataclass
class BatchCursor:
    epoch: int
    permutation: list[int]
    next_batch_index: int
    global_step: int
    total_loss: float
    total_items: int

    @classmethod
    def fresh(cls, *, dataset_size: int, seed: int) -> "BatchCursor":
        return cls(
            epoch=1,
            permutation=epoch_permutation(dataset_size, seed=seed, epoch=1),
            next_batch_index=0,
            global_step=0,
            total_loss=0.0,
            total_items=0,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BatchCursor":
        return cls(
            epoch=int(payload["epoch"]),
            permutation=[int(index) for index in payload["permutation"]],
            next_batch_index=int(payload["next_batch_index"]),
            global_step=int(payload["global_step"]),
            total_loss=float(payload.get("total_loss", 0.0)),
            total_items=int(payload.get("total_items", 0)),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "permutation": self.permutation,
            "next_batch_index": self.next_batch_index,
            "global_step": self.global_step,
            "total_loss": self.total_loss,
            "total_items": self.total_items,
        }

    def advance_epoch(self, *, dataset_size: int, seed: int) -> None:
        self.epoch += 1
        self.permutation = epoch_permutation(dataset_size, seed=seed, epoch=self.epoch)
        self.next_batch_index = 0
        self.total_loss = 0.0
        self.total_items = 0


class PauseController:
    """Signal handler that records intent; training loops checkpoint after an update."""

    def __init__(self) -> None:
        self.requested = False
        self.reason: str | None = None
        self._previous: dict[int, Any] = {}

    def request(self, signum: int | None = None) -> None:
        self.requested = True
        self.reason = signal.Signals(signum).name if signum is not None else "requested"

    def install(self) -> None:
        if os.name != "posix":
            return
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            value = getattr(signal, name, None)
            if value is None:
                continue
            self._previous[int(value)] = signal.getsignal(value)
            signal.signal(value, lambda signum, _frame: self.request(signum))

    def close(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()


class BatchCheckpointManager:
    """Own one atomic checkpoint stream and its immutable compatibility contract."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        contract: Mapping[str, Any],
        resume: bool,
        resume_contract_compatible: ResumeContractCompatibility | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.contract = dict(contract)
        self.contract_fingerprint = fingerprint(self.contract)
        self._accepted_resume_fingerprints = {self.contract_fingerprint}
        self.checkpoint_path = self.output_dir / "checkpoints" / "latest.pt"
        self.contract_path = self.output_dir / "checkpoints" / "contract.json"
        if resume:
            if not self.checkpoint_path.exists():
                raise ResumeCompatibilityError(
                    f"no compatible batch-resume checkpoint exists at {self.checkpoint_path}; baseline-level restart is the only safe option"
                )
            persisted = json.loads(self.contract_path.read_text(encoding="utf-8")) if self.contract_path.exists() else None
            persisted_fingerprint = (
                persisted.get("fingerprint") if isinstance(persisted, dict) else None
            )
            accepted_legacy = (
                persisted.get("accepted_legacy_fingerprints", [])
                if isinstance(persisted, dict)
                else []
            )
            if isinstance(accepted_legacy, list):
                self._accepted_resume_fingerprints.update(
                    value for value in accepted_legacy if isinstance(value, str)
                )
            if (
                not isinstance(persisted, dict)
                or persisted_fingerprint not in self._accepted_resume_fingerprints
            ):
                persisted_contract = (
                    persisted.get("contract") if isinstance(persisted, dict) else None
                )
                if (
                    not isinstance(persisted_contract, Mapping)
                    or not isinstance(resume_contract_compatible, Callable)
                    or not resume_contract_compatible(persisted_contract, self.contract)
                ):
                    raise ResumeCompatibilityError("resume contract differs from the checkpointed training contract")
                if isinstance(persisted_fingerprint, str):
                    self._accepted_resume_fingerprints.add(persisted_fingerprint)
            if not isinstance(persisted, dict):
                raise ResumeCompatibilityError("resume contract differs from the checkpointed training contract")
        else:
            if self.checkpoint_path.exists() or self.contract_path.exists():
                raise FileExistsError(f"checkpoint state already exists in {self.output_dir}; use --resume")
            atomic_write_json(self.contract_path, {
                "schema_version": CHECKPOINT_SCHEMA,
                "fingerprint": self.contract_fingerprint,
                "contract": self.contract,
            })

    def save(self, payload: Mapping[str, Any]) -> None:
        snapshot = {
            "schema_version": CHECKPOINT_SCHEMA,
            "contract_fingerprint": self.contract_fingerprint,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            **dict(payload),
        }
        atomic_torch_save(self.checkpoint_path, snapshot)

    def load(self) -> dict[str, Any]:
        payload = _torch_load(self.checkpoint_path)
        if not isinstance(payload, dict):
            raise ResumeCompatibilityError("checkpoint payload is not a dictionary")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ResumeCompatibilityError("unsupported batch-resume checkpoint schema")
        if payload.get("contract_fingerprint") not in self._accepted_resume_fingerprints:
            raise ResumeCompatibilityError("checkpoint contract fingerprint mismatch")
        return payload

    def write_paused_marker(self, *, stage: str, cursor: BatchCursor, reason: str | None) -> None:
        atomic_write_json(
            self.output_dir / "RUN_PAUSED",
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "stage": stage,
                "reason": reason or "requested",
                "checkpoint": str(self.checkpoint_path),
                "epoch": cursor.epoch,
                "next_batch_index": cursor.next_batch_index,
                "global_step": cursor.global_step,
                "paused_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def clear_pause_marker(self) -> None:
        (self.output_dir / "RUN_PAUSED").unlink(missing_ok=True)
