from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .progress import log_event


def require_finite_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    stage: str,
    epoch: int,
    batch_index: int,
) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"non-finite {name} at stage={stage} epoch={epoch} batch={batch_index}"
        )


def require_finite_gradients(
    module: torch.nn.Module,
    *,
    stage: str,
    epoch: int,
    batch_index: int,
) -> None:
    invalid = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if invalid:
        names = ", ".join(invalid)
        raise FloatingPointError(
            f"non-finite gradients at stage={stage} epoch={epoch} "
            f"batch={batch_index}: {names}"
        )


def require_finite_parameters(
    module: torch.nn.Module,
    *,
    stage: str,
    epoch: int,
    batch_index: int,
) -> None:
    invalid = [
        name
        for name, parameter in module.named_parameters()
        if not torch.isfinite(parameter).all()
    ]
    if invalid:
        names = ", ".join(invalid)
        raise FloatingPointError(
            f"non-finite parameters at stage={stage} epoch={epoch} "
            f"batch={batch_index}: {names}"
        )


def require_finite_values(value: Any, name: str) -> None:
    """Reject NaN/Inf in nested metric and diagnostic payloads."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite metric value: {name}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_values(item, f"{name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            require_finite_values(item, f"{name}[{index}]")
        return
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"non-finite metric value: {name}")


@dataclass
class EpochHealthMonitor:
    """Warn about stalled task metrics without changing checkpoint selection."""

    stage: str
    patience: int = 3
    min_delta: float = 1e-6
    best_loss: float | None = None
    no_improvement_epochs: int = 0
    consecutive_mechanism_failures: int = 0
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("health warning patience must be positive")
        if self.min_delta < 0:
            raise ValueError("health warning min_delta must be non-negative")

    def observe(
        self,
        *,
        epoch: int,
        valid_loss: float,
        mechanism_pass: bool | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(float(valid_loss)):
            raise FloatingPointError(
                f"non-finite validation loss at stage={self.stage} epoch={epoch}"
            )
        if self.best_loss is None or valid_loss < self.best_loss - self.min_delta:
            self.best_loss = float(valid_loss)
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1

        if mechanism_pass is None:
            self.consecutive_mechanism_failures = 0
        elif mechanism_pass:
            self.consecutive_mechanism_failures = 0
        else:
            self.consecutive_mechanism_failures += 1

        checks = (
            (
                "no_task_improvement",
                self.no_improvement_epochs,
                "validation loss did not improve",
            ),
            (
                "mechanism_selectivity_failure",
                self.consecutive_mechanism_failures,
                "mechanism selectivity did not pass",
            ),
        )
        for warning, count, message in checks:
            if count >= self.patience and count % self.patience == 0:
                payload = {
                    "warning": warning,
                    "message": message,
                    "stage": self.stage,
                    "epoch": epoch,
                    "consecutive_epochs": count,
                }
                self.warnings.append(payload)
                log_event("health_warning", **payload)

        return {
            "best_loss": self.best_loss,
            "no_improvement_epochs": self.no_improvement_epochs,
            "consecutive_mechanism_failures": self.consecutive_mechanism_failures,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_loss": self.best_loss,
            "no_improvement_epochs": self.no_improvement_epochs,
            "consecutive_mechanism_failures": self.consecutive_mechanism_failures,
            "warnings": self.warnings,
        }
