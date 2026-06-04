"""Spectral projector construction for key-space steering.

Phase 1 implements the classical tensor-level mechanism that later quantum
modules will extend. The main output is a projector ``P`` used by
``k' = k + gPk`` during inference-time key steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SpectralProjectorConfig:
    """Configuration for projector construction.

    Args:
        mode: Singular-value filtering mode.
        rank: Explicit number of singular directions. If ``None``, ``energy`` is used.
        energy: Cumulative singular-value energy used to infer rank in ``hard_topk`` mode.
        threshold: Threshold for smooth filters.
        sharpness: Slope for sigmoid-based filters.
        eps: Numerical stability constant.
    """

    mode: str = "hard_topk"
    rank: int | None = None
    energy: float = 0.9
    threshold: float = 0.5
    sharpness: float = 8.0
    eps: float = 1e-8


def _as_matrix(keys: torch.Tensor, name: str) -> torch.Tensor:
    """Flatten leading dimensions and keep the final feature dimension."""
    if keys.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions")
    return keys.reshape(-1, keys.shape[-1]).float()


def cross_covariance(source_keys: torch.Tensor, target_keys: torch.Tensor) -> torch.Tensor:
    """Compute ``source.T @ target / n`` after flattening key tensors."""
    source = _as_matrix(source_keys, "source_keys")
    target = _as_matrix(target_keys, "target_keys")
    if source.shape != target.shape:
        raise ValueError(f"source and target shapes must match after flattening: {source.shape} != {target.shape}")
    if source.shape[0] == 0:
        raise ValueError("at least one key vector is required")
    return torch.matmul(source.transpose(0, 1), target) / source.shape[0]


def _rank_from_energy(singular_values: torch.Tensor, energy: float, eps: float) -> int:
    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0, 1]")
    total = singular_values.sum().clamp_min(eps)
    cumulative = singular_values.cumsum(dim=0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(energy, device=singular_values.device)).item()) + 1


def singular_weights(singular_values: torch.Tensor, config: SpectralProjectorConfig) -> torch.Tensor:
    """Return filter weights for singular directions."""
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a 1D tensor")
    if singular_values.numel() == 0:
        raise ValueError("singular_values cannot be empty")

    s = singular_values.float()
    s_norm = s / s.max().clamp_min(config.eps)

    if config.mode == "hard_topk":
        rank = config.rank or _rank_from_energy(s, config.energy, config.eps)
        rank = min(rank, s.numel())
        weights = torch.zeros_like(s_norm)
        weights[:rank] = 1.0
        return weights

    if config.mode == "high_pass":
        return torch.sigmoid(config.sharpness * (s_norm - config.threshold))

    if config.mode == "band_pass":
        lower = torch.sigmoid(config.sharpness * (s_norm - config.threshold))
        upper = torch.sigmoid(config.sharpness * ((1.0 - config.threshold) - s_norm))
        return lower * upper

    if config.mode == "soft_energy":
        return s_norm / s_norm.sum().clamp_min(config.eps)

    raise ValueError(f"unknown projector filter mode: {config.mode}")


def spectral_effective_rank(singular_values: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute entropy-based effective rank from non-negative singular values."""
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a 1D tensor")
    if singular_values.numel() == 0:
        raise ValueError("singular_values cannot be empty")
    values = singular_values.float().clamp_min(0.0)
    probs = values / values.sum().clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())


def spectral_filter_diagnostics(
    singular_values: torch.Tensor,
    config: SpectralProjectorConfig,
    *,
    active_eps: float = 1e-6,
    head: int = 8,
) -> dict[str, Any]:
    """Summarize how a spectral filter weights singular directions."""
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a 1D tensor")
    if singular_values.numel() == 0:
        raise ValueError("singular_values cannot be empty")
    values = singular_values.float()
    weights = singular_weights(values, config)
    return {
        "mode": config.mode,
        "rank": config.rank,
        "energy": config.energy,
        "threshold": config.threshold,
        "sharpness": config.sharpness,
        "num_singular_values": int(values.numel()),
        "active_directions": int((weights > active_eps).sum().item()),
        "weight_sum": float(weights.sum().item()),
        "weight_max": float(weights.max().item()),
        "weight_min": float(weights.min().item()),
        "effective_rank": spectral_effective_rank(values, eps=config.eps),
        "max_singular_value": float(values.max().item()),
        "min_singular_value": float(values.min().item()),
        "top_singular_values": [float(value) for value in values[: min(head, values.numel())].tolist()],
        "top_filter_weights": [float(value) for value in weights[: min(head, weights.numel())].tolist()],
    }


def build_projector(
    source_keys: torch.Tensor | None = None,
    target_keys: torch.Tensor | None = None,
    *,
    basis: torch.Tensor | None = None,
    singular_values: torch.Tensor | None = None,
    config: SpectralProjectorConfig | None = None,
) -> torch.Tensor:
    """Build a spectral key-space projector.

    Either provide ``source_keys`` and ``target_keys`` to compute an SVD of their
    cross-covariance, or provide an existing ``basis`` and ``singular_values``.
    """
    config = config or SpectralProjectorConfig()

    if basis is None:
        if source_keys is None or target_keys is None:
            raise ValueError("provide either source/target keys or an explicit basis")
        omega = cross_covariance(source_keys, target_keys)
        basis, singular_values, _ = torch.linalg.svd(omega, full_matrices=False)
    else:
        if basis.ndim != 2:
            raise ValueError("basis must have shape (dim, rank)")
        basis = basis.float()
        if singular_values is None:
            singular_values = torch.ones(basis.shape[1], device=basis.device, dtype=basis.dtype)

    if singular_values is None:
        raise ValueError("singular_values could not be inferred")
    if singular_values.ndim != 1 or singular_values.shape[0] != basis.shape[1]:
        raise ValueError("singular_values must be 1D and match basis columns")

    weights = singular_weights(singular_values.to(basis.device), config).to(dtype=basis.dtype)
    weighted_basis = basis * weights.unsqueeze(0)
    projector = torch.matmul(weighted_basis, basis.transpose(0, 1))
    return projector.to(dtype=basis.dtype)