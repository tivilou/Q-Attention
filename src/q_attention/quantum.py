"""Torch-only quantum-inspired projector utilities.

The functions in this module are intentionally lightweight. They do not require
quantum simulators; instead, they provide a toy angle-encoding feature map and a
fidelity-style kernel that can build a key-space projector compatible with
``k' = k + gPk``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from q_attention.projectors import SpectralProjectorConfig, build_projector


@dataclass(frozen=True)
class QuantumFeatureMapConfig:
    """Configuration for the toy quantum-inspired feature map."""

    num_qubits: int = 4
    angle_scale: float = 1.0
    seed: int = 17
    max_state_dim: int = 1024
    eps: float = 1e-8


@dataclass(frozen=True)
class QuantumProjectorResult:
    """Projector result plus diagnostics useful for toy ablations."""

    projector: torch.Tensor
    features: torch.Tensor
    kernel: torch.Tensor
    singular_values: torch.Tensor
    metadata: dict[str, Any]


def _as_key_matrix(keys: torch.Tensor) -> torch.Tensor:
    if keys.ndim != 2:
        raise ValueError("keys must have shape (num_vectors, dim)")
    if keys.shape[0] == 0:
        raise ValueError("at least one key vector is required")
    return keys.float()


def deterministic_projection(input_dim: int, num_qubits: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """Create a deterministic random projection from key space to qubit angles."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(input_dim, num_qubits, generator=generator, dtype=torch.float32)
    projection = projection / float(input_dim) ** 0.5
    return projection.to(device=device)


def angle_feature_map(keys: torch.Tensor, config: QuantumFeatureMapConfig | None = None) -> torch.Tensor:
    """Map key vectors to a toy tensor-product angle-encoded state."""
    config = config or QuantumFeatureMapConfig()
    state_dim = 2 ** config.num_qubits
    if state_dim > config.max_state_dim:
        raise ValueError(f"state dimension {state_dim} exceeds max_state_dim={config.max_state_dim}")

    x = _as_key_matrix(keys)
    x = F.normalize(x, p=2, dim=-1, eps=config.eps)
    projection = deterministic_projection(x.shape[-1], config.num_qubits, seed=config.seed, device=x.device)
    angles = config.angle_scale * torch.matmul(x, projection)

    state = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for qubit in range(config.num_qubits):
        local = torch.stack((torch.cos(angles[:, qubit]), torch.sin(angles[:, qubit])), dim=-1)
        state = (state.unsqueeze(-1) * local.unsqueeze(1)).reshape(x.shape[0], -1)
    return F.normalize(state, p=2, dim=-1, eps=config.eps)


def fidelity_kernel(features_a: torch.Tensor, features_b: torch.Tensor | None = None, *, eps: float = 1e-8) -> torch.Tensor:
    """Compute a fidelity-style kernel ``|<Phi(x), Phi(y)>|^2``."""
    if features_a.ndim != 2:
        raise ValueError("features_a must have shape (num_vectors, feature_dim)")
    features_b = features_a if features_b is None else features_b
    if features_b.ndim != 2 or features_b.shape[-1] != features_a.shape[-1]:
        raise ValueError("features_b must be 2D and share the feature dimension")
    a = F.normalize(features_a.float(), p=2, dim=-1, eps=eps)
    b = F.normalize(features_b.float(), p=2, dim=-1, eps=eps)
    return torch.matmul(a, b.transpose(0, 1)).pow(2).clamp_min(0.0)


def quantum_weighted_covariance(
    keys: torch.Tensor,
    kernel: torch.Tensor,
    *,
    center: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Lift a quantum kernel back into key space as ``X.T @ K_q @ X``."""
    x = _as_key_matrix(keys)
    if kernel.shape != (x.shape[0], x.shape[0]):
        raise ValueError(f"kernel shape {kernel.shape} does not match num_vectors={x.shape[0]}")
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    k = kernel.to(device=x.device, dtype=x.dtype)
    normalizer = k.sum().clamp_min(eps)
    omega = torch.matmul(x.transpose(0, 1), torch.matmul(k, x)) / normalizer
    return 0.5 * (omega + omega.transpose(0, 1))


def build_quantum_projector(
    keys: torch.Tensor,
    *,
    quantum_config: QuantumFeatureMapConfig | None = None,
    projector_config: SpectralProjectorConfig | None = None,
    center: bool = False,
) -> QuantumProjectorResult:
    """Build a key-space projector using a toy quantum-inspired kernel."""
    quantum_config = quantum_config or QuantumFeatureMapConfig()
    projector_config = projector_config or SpectralProjectorConfig()
    x = _as_key_matrix(keys)
    features = angle_feature_map(x, quantum_config)
    kernel = fidelity_kernel(features, eps=quantum_config.eps)
    omega = quantum_weighted_covariance(x, kernel, center=center, eps=quantum_config.eps)
    basis, singular_values, _ = torch.linalg.svd(omega, full_matrices=False)
    projector = build_projector(basis=basis, singular_values=singular_values, config=projector_config)
    metadata: dict[str, Any] = {
        "projector_family": "toy_quantum",
        "num_vectors": int(x.shape[0]),
        "key_dim": int(x.shape[1]),
        "state_dim": int(features.shape[1]),
        "center": center,
        "quantum_config": asdict(quantum_config),
        "projector_config": asdict(projector_config),
        "kernel_trace": float(torch.trace(kernel).item()),
        "kernel_mean": float(kernel.mean().item()),
        "top_singular_values": [float(value) for value in singular_values[: min(8, singular_values.numel())].tolist()],
    }
    return QuantumProjectorResult(
        projector=projector,
        features=features,
        kernel=kernel,
        singular_values=singular_values,
        metadata=metadata,
    )