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


def _project_keys(keys: torch.Tensor, projector: torch.Tensor) -> torch.Tensor:
    """Apply a shared or batch-wise projector to key vectors."""
    if projector.ndim == 2:
        if projector.shape[0] != projector.shape[1]:
            raise ValueError("projector must have shape (dim, dim)")
        if keys.shape[-1] != projector.shape[0]:
            raise ValueError(f"key dim {keys.shape[-1]} does not match projector dim {projector.shape[0]}")
        return torch.matmul(keys, projector.transpose(0, 1))

    if projector.ndim == 3:
        if keys.ndim != 3:
            raise ValueError("batch-wise projectors require keys with shape (batch, tokens, dim)")
        if projector.shape[1] != projector.shape[2]:
            raise ValueError("batch-wise projector must have shape (batch, dim, dim)")
        if keys.shape[0] != projector.shape[0]:
            raise ValueError(f"key batch {keys.shape[0]} does not match projector batch {projector.shape[0]}")
        if keys.shape[-1] != projector.shape[1]:
            raise ValueError(f"key dim {keys.shape[-1]} does not match projector dim {projector.shape[1]}")
        return torch.einsum("btd,bdh->bth", keys, projector.transpose(-1, -2))

    raise ValueError("projector must have shape (dim, dim) or (batch, dim, dim)")


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
        keys: Tensor with shape ``(..., dim)`` for a shared projector, or
            ``(batch, tokens, dim)`` for a batch-wise projector.
        projector: Tensor with shape ``(dim, dim)`` or ``(batch, dim, dim)``.
        mask: Optional boolean tensor with shape ``keys.shape[:-1]``.
        gain: Steering strength.
        return_delta: If true, return ``SteeringResult`` with diagnostics.
    """
    if keys.ndim < 2:
        raise ValueError("keys must have at least 2 dimensions")

    projector = projector.to(device=keys.device, dtype=keys.dtype)
    output = keys.clone()
    delta = torch.zeros_like(keys)
    projected = _project_keys(keys, projector)

    if mask is None:
        delta = gain * projected
        output = output + delta
        changed_count = int(keys.reshape(-1, keys.shape[-1]).shape[0])
    else:
        if mask.shape != keys.shape[:-1]:
            raise ValueError(f"mask shape {mask.shape} must match keys leading shape {keys.shape[:-1]}")
        mask = mask.to(device=keys.device, dtype=torch.bool)
        selected = keys[mask]
        if selected.numel() > 0:
            selected_delta = gain * projected[mask]
            output[mask] = selected + selected_delta
            delta[mask] = selected_delta
        changed_count = int(mask.sum().item())

    if return_delta:
        return SteeringResult(output, delta, changed_count)
    return output