from __future__ import annotations

import pytest
import torch

from q_attention.projectors import SpectralProjectorConfig
from q_attention.quantum import (
    QuantumFeatureMapConfig,
    angle_feature_map,
    build_quantum_projector,
    deterministic_projection,
    fidelity_kernel,
    quantum_weighted_covariance,
)


def test_angle_feature_map_has_tensor_product_state_shape_and_norm() -> None:
    torch.manual_seed(41)
    keys = torch.randn(5, 6)
    features = angle_feature_map(keys, QuantumFeatureMapConfig(num_qubits=3, seed=7))

    assert features.shape == (5, 8)
    assert torch.allclose(features.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_deterministic_projection_is_seed_reproducible() -> None:
    first = deterministic_projection(6, 4, seed=11, device=torch.device("cpu"))
    second = deterministic_projection(6, 4, seed=11, device=torch.device("cpu"))
    third = deterministic_projection(6, 4, seed=12, device=torch.device("cpu"))

    assert torch.allclose(first, second)
    assert not torch.allclose(first, third)


def test_fidelity_kernel_is_symmetric_with_unit_diagonal() -> None:
    torch.manual_seed(43)
    features = angle_feature_map(torch.randn(4, 5), QuantumFeatureMapConfig(num_qubits=2, seed=3))
    kernel = fidelity_kernel(features)

    assert kernel.shape == (4, 4)
    assert torch.allclose(kernel, kernel.transpose(0, 1), atol=1e-5)
    assert torch.allclose(torch.diag(kernel), torch.ones(4), atol=1e-5)


def test_quantum_weighted_covariance_is_key_space_symmetric() -> None:
    torch.manual_seed(47)
    keys = torch.randn(6, 5)
    features = angle_feature_map(keys, QuantumFeatureMapConfig(num_qubits=3, seed=9))
    kernel = fidelity_kernel(features)
    omega = quantum_weighted_covariance(keys, kernel, center=True)

    assert omega.shape == (5, 5)
    assert torch.allclose(omega, omega.transpose(0, 1), atol=1e-5)


def test_build_quantum_projector_returns_steering_compatible_projector() -> None:
    torch.manual_seed(53)
    keys = torch.randn(10, 7)
    result = build_quantum_projector(
        keys,
        quantum_config=QuantumFeatureMapConfig(num_qubits=3, angle_scale=1.2, seed=2),
        projector_config=SpectralProjectorConfig(rank=2),
    )

    assert result.projector.shape == (7, 7)
    assert torch.allclose(result.projector, result.projector.transpose(0, 1), atol=1e-5)
    assert result.metadata["projector_family"] == "toy_quantum"
    assert result.metadata["state_dim"] == 8


def test_angle_feature_map_rejects_overlarge_state_dimension() -> None:
    with pytest.raises(ValueError):
        angle_feature_map(torch.randn(2, 3), QuantumFeatureMapConfig(num_qubits=6, max_state_dim=16))