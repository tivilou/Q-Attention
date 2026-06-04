"""Toy adaptive projector routing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for softmax projector routing."""

    temperature: float = 1.0
    eps: float = 1e-8


@dataclass(frozen=True)
class ProjectorBank:
    """A bank of projector experts and their routing prototypes."""

    names: tuple[str, ...]
    projectors: torch.Tensor
    prototypes: torch.Tensor

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


@dataclass(frozen=True)
class RoutingResult:
    """Router output for one batch."""

    weights: torch.Tensor
    projectors: torch.Tensor
    scores: torch.Tensor
    entropy: torch.Tensor


def stack_projector_bank(names: Sequence[str], projectors: Sequence[torch.Tensor], prototypes: Sequence[torch.Tensor]) -> ProjectorBank:
    """Create a validated projector bank from Python sequences."""
    if not names:
        raise ValueError("at least one projector expert is required")
    if len(projectors) != len(names) or len(prototypes) != len(names):
        raise ValueError("names, projectors, and prototypes must have the same length")
    stacked_projectors = torch.stack([projector.float() for projector in projectors], dim=0)
    stacked_prototypes = torch.stack([prototype.float() for prototype in prototypes], dim=0)
    return ProjectorBank(names=tuple(str(name) for name in names), projectors=stacked_projectors, prototypes=stacked_prototypes)


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


def route_projectors(anchor_representations: torch.Tensor, bank: ProjectorBank, config: RouterConfig | None = None) -> RoutingResult:
    """Route each example to a soft mixture of projector experts."""
    config = config or RouterConfig()
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if anchor_representations.ndim != 2:
        raise ValueError("anchor_representations must have shape (batch, dim)")
    if anchor_representations.shape[-1] != bank.prototypes.shape[-1]:
        raise ValueError("anchor dim must match bank prototype dim")

    anchors = F.normalize(anchor_representations.float(), p=2, dim=-1, eps=config.eps)
    prototypes = F.normalize(bank.prototypes.to(device=anchors.device), p=2, dim=-1, eps=config.eps)
    scores = torch.matmul(anchors, prototypes.transpose(0, 1)) / config.temperature
    weights = torch.softmax(scores, dim=-1)
    bank_projectors = bank.projectors.to(device=anchors.device, dtype=anchors.dtype)
    dynamic_projectors = torch.einsum("be,edh->bdh", weights, bank_projectors)
    entropy = -(weights * torch.log(weights.clamp_min(config.eps))).sum(dim=-1)
    return RoutingResult(weights=weights, projectors=dynamic_projectors, scores=scores, entropy=entropy)