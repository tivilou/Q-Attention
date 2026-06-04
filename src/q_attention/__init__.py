"""Q-Attention: spectral key steering utilities for attention models."""

from .projectors import (
    SpectralProjectorConfig,
    build_projector,
    cross_covariance,
    singular_weights,
    spectral_effective_rank,
    spectral_filter_diagnostics,
)
from .quantum import (
    QuantumFeatureMapConfig,
    QuantumProjectorResult,
    angle_feature_map,
    build_quantum_projector,
    deterministic_projection,
    fidelity_kernel,
    quantum_weighted_covariance,
)
from .routing import (
    ProjectorBank,
    RouterConfig,
    RoutingResult,
    projector_prototype,
    route_projectors,
    stack_projector_bank,
)
from .spans import batched_span_mask, span_mask
from .steering import SteeringResult, apply_key_steering

__all__ = [
    "ProjectorBank",
    "QuantumFeatureMapConfig",
    "QuantumProjectorResult",
    "RouterConfig",
    "RoutingResult",
    "SpectralProjectorConfig",
    "SteeringResult",
    "angle_feature_map",
    "apply_key_steering",
    "batched_span_mask",
    "build_projector",
    "build_quantum_projector",
    "cross_covariance",
    "deterministic_projection",
    "fidelity_kernel",
    "projector_prototype",
    "quantum_weighted_covariance",
    "route_projectors",
    "singular_weights",
    "span_mask",
    "spectral_effective_rank",
    "spectral_filter_diagnostics",
    "stack_projector_bank",
]