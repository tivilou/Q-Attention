"""Q-Attention: spectral key steering utilities for attention models."""

from .projectors import SpectralProjectorConfig, build_projector, cross_covariance
from .spans import batched_span_mask, span_mask
from .steering import SteeringResult, apply_key_steering

__all__ = [
    "SpectralProjectorConfig",
    "SteeringResult",
    "apply_key_steering",
    "batched_span_mask",
    "build_projector",
    "cross_covariance",
    "span_mask",
]
