"""Adaptive projector routing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

ROUTER_SCORE_MODES = ("prototype", "energy", "hybrid")


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for softmax projector routing."""

    temperature: float = 1.0
    eps: float = 1e-8
    score_mode: str = "hybrid"
    prototype_weight: float = 1.0
    energy_weight: float = 1.0
    normalize_scores: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.score_mode not in ROUTER_SCORE_MODES:
            raise ValueError(f"unknown router score mode: {self.score_mode}")


@dataclass(frozen=True)
class ProjectorBank:
    """A bank of projector experts and their routing prototypes."""

    names: tuple[str, ...]
    projectors: torch.Tensor
    prototypes: torch.Tensor
    gain_scales: torch.Tensor | None = None
    logit_biases: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.projectors.ndim != 3:
            raise ValueError("projectors must have shape (experts, dim, dim)")
        if self.projectors.shape[1] != self.projectors.shape[2]:
            raise ValueError("each projector must have shape (dim, dim)")
        if self.prototypes.ndim != 2:
            raise ValueError("prototypes must have shape (experts, dim)")
        if self.projectors.shape[0] != len(self.names):
            raise ValueError("number of names must match number of projectors")
        if self.prototypes.shape[0] != len(self.names):
            raise ValueError("number of prototypes must match number of projectors")
        if self.prototypes.shape[1] != self.projectors.shape[1]:
            raise ValueError("prototype dim must match projector dim")
        if self.gain_scales is None:
            object.__setattr__(self, "gain_scales", torch.ones(len(self.names), dtype=self.projectors.dtype))
        elif self.gain_scales.ndim != 1 or self.gain_scales.shape[0] != len(self.names):
            raise ValueError("gain_scales must have shape (experts,)")
        if self.logit_biases is None:
            object.__setattr__(self, "logit_biases", torch.zeros(len(self.names), dtype=self.projectors.dtype))
        elif self.logit_biases.ndim != 1 or self.logit_biases.shape[0] != len(self.names):
            raise ValueError("logit_biases must have shape (experts,)")


@dataclass(frozen=True)
class RoutingResult:
    """Router output for one batch."""

    weights: torch.Tensor
    projectors: torch.Tensor
    scores: torch.Tensor
    entropy: torch.Tensor


def _optional_float_vector(values: Sequence[float] | torch.Tensor | None) -> torch.Tensor | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.float()
    return torch.tensor([float(value) for value in values], dtype=torch.float32)


def stack_projector_bank(
    names: Sequence[str],
    projectors: Sequence[torch.Tensor],
    prototypes: Sequence[torch.Tensor],
    *,
    gain_scales: Sequence[float] | torch.Tensor | None = None,
    logit_biases: Sequence[float] | torch.Tensor | None = None,
) -> ProjectorBank:
    """Create a validated projector bank from Python sequences."""
    if not names:
        raise ValueError("at least one projector expert is required")
    if len(projectors) != len(names) or len(prototypes) != len(names):
        raise ValueError("names, projectors, and prototypes must have the same length")
    stacked_projectors = torch.stack([projector.float() for projector in projectors], dim=0)
    stacked_prototypes = torch.stack([prototype.float() for prototype in prototypes], dim=0)
    return ProjectorBank(
        names=tuple(str(name) for name in names),
        projectors=stacked_projectors,
        prototypes=stacked_prototypes,
        gain_scales=_optional_float_vector(gain_scales),
        logit_biases=_optional_float_vector(logit_biases),
    )


def projector_prototype(keys: torch.Tensor, projector: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Build a simple expert prototype from steered anchor keys."""
    if keys.ndim != 2:
        raise ValueError("keys must have shape (num_vectors, dim)")
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ValueError("projector must have shape (dim, dim)")
    if keys.shape[-1] != projector.shape[0]:
        raise ValueError("key dim must match projector dim")
    projected = torch.matmul(keys.float(), projector.float().transpose(0, 1))
    prototype = projected.mean(dim=0)
    if prototype.norm().item() <= eps:
        prototype = keys.float().mean(dim=0)
    return F.normalize(prototype, p=2, dim=0, eps=eps)


def projector_energy_scores(anchors: torch.Tensor, projectors: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Score how strongly each anchor lies in each projector's active subspace."""
    projected = torch.einsum("bd,edh->beh", anchors, projectors)
    energy = projected.pow(2).sum(dim=-1)
    scale = projectors.flatten(1).pow(2).sum(dim=-1).clamp_min(eps)
    return energy / scale.unsqueeze(0)


def _normalize_router_scores(scores: torch.Tensor, eps: float) -> torch.Tensor:
    if scores.shape[-1] <= 1:
        return scores
    centered = scores - scores.mean(dim=-1, keepdim=True)
    spread = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return centered / spread


def route_projectors(anchor_representations: torch.Tensor, bank: ProjectorBank, config: RouterConfig | None = None) -> RoutingResult:
    """Route each example to a soft mixture of projector experts."""
    config = config or RouterConfig()
    if anchor_representations.ndim != 2:
        raise ValueError("anchor_representations must have shape (batch, dim)")
    if anchor_representations.shape[-1] != bank.prototypes.shape[-1]:
        raise ValueError("anchor dim must match bank prototype dim")

    anchors = F.normalize(anchor_representations.float(), p=2, dim=-1, eps=config.eps)
    prototypes = F.normalize(bank.prototypes.to(device=anchors.device), p=2, dim=-1, eps=config.eps)
    bank_projectors = bank.projectors.to(device=anchors.device, dtype=anchors.dtype)

    prototype_scores = torch.matmul(anchors, prototypes.transpose(0, 1))
    energy_scores = projector_energy_scores(anchors, bank_projectors, eps=config.eps)
    if config.score_mode == "prototype":
        raw_scores = prototype_scores
    elif config.score_mode == "energy":
        raw_scores = energy_scores
    else:
        raw_scores = config.prototype_weight * prototype_scores + config.energy_weight * energy_scores

    if config.normalize_scores:
        raw_scores = _normalize_router_scores(raw_scores, config.eps)
    logit_biases = bank.logit_biases.to(device=anchors.device, dtype=anchors.dtype)
    scores = raw_scores / config.temperature + logit_biases.unsqueeze(0)
    weights = torch.softmax(scores, dim=-1)
    gain_scales = bank.gain_scales.to(device=anchors.device, dtype=anchors.dtype)
    scaled_projectors = bank_projectors * gain_scales.view(-1, 1, 1)
    dynamic_projectors = torch.einsum("be,edh->bdh", weights, scaled_projectors)
    entropy = -(weights * torch.log(weights.clamp_min(config.eps))).sum(dim=-1)
    return RoutingResult(weights=weights, projectors=dynamic_projectors, scores=scores, entropy=entropy)
