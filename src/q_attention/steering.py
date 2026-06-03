"""Tensor-level key steering."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SteeringResult:
    """Result returned by ``apply_key_steering``."""

    keys: torch.Tensor
    delta: torch.Tensor
    changed_count: int


def apply_key_steering(
    keys: torch.Tensor,
    projector: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    gain: float = 1.0,
    return_delta: bool = False,
) -> torch.Tensor | SteeringResult:
    """Apply ``k' = k + gain * Pk`` to selected key vectors.

    Args:
        keys: Tensor with shape ``(..., dim)``.
        projector: Tensor with shape ``(dim, dim)``.
        mask: Optional boolean tensor with shape ``keys.shape[:-1]``.
        gain: Steering strength.
        return_delta: If true, return ``SteeringResult`` with diagnostics.
    """
    if keys.ndim < 2:
        raise ValueError("keys must have at least 2 dimensions")
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ValueError("projector must have shape (dim, dim)")
    if keys.shape[-1] != projector.shape[0]:
        raise ValueError(f"key dim {keys.shape[-1]} does not match projector dim {projector.shape[0]}")

    projector = projector.to(device=keys.device, dtype=keys.dtype)
    output = keys.clone()
    delta = torch.zeros_like(keys)

    if mask is None:
        selected_delta = torch.matmul(keys, projector.transpose(0, 1))
        delta = gain * selected_delta
        output = output + delta
        changed_count = int(keys.reshape(-1, keys.shape[-1]).shape[0])
    else:
        if mask.shape != keys.shape[:-1]:
            raise ValueError(f"mask shape {mask.shape} must match keys leading shape {keys.shape[:-1]}")
        mask = mask.to(device=keys.device, dtype=torch.bool)
        selected = keys[mask]
        if selected.numel() > 0:
            selected_delta = gain * torch.matmul(selected, projector.transpose(0, 1))
            output[mask] = selected + selected_delta
            delta[mask] = selected_delta
        changed_count = int(mask.sum().item())

    if return_delta:
        return SteeringResult(output, delta, changed_count)
    return output
