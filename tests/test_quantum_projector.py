from __future__ import annotations

import pytest
import torch

from q_attention.projectors import SpectralProjectorConfig
from q_attention.quantum import (
    QuantumFeatureMapConfig,
    SupervisedQuantumProjectorConfig,
    angle_feature_map,
    build_quantum_projector,
    build_quantum_residual_projector,
    build_supervised_quantum_projector,
    deterministic_projection,
    fidelity_kernel,
    fit_supervised_quantum_feature_map,
    parameterized_quantum_feature_map,
    quantum_weighted_covariance,
    transform_quantum_kernel,
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
    assert result.metadata["projector_family"] == "quantum_contrastive"
    assert result.metadata["kernel_mode"] == "centered_fidelity"
    assert result.metadata["state_dim"] == 8


def test_angle_feature_map_rejects_overlarge_state_dimension() -> None:
    with pytest.raises(ValueError):
        angle_feature_map(torch.randn(2, 3), QuantumFeatureMapConfig(num_qubits=6, max_state_dim=16))

def test_centered_quantum_kernel_removes_uniform_component() -> None:
    torch.manual_seed(59)
    features = angle_feature_map(torch.randn(6, 5), QuantumFeatureMapConfig(num_qubits=3, seed=5))
    raw_kernel = fidelity_kernel(features)
    centered = transform_quantum_kernel(raw_kernel, mode="centered_fidelity")

    assert torch.allclose(centered, centered.transpose(0, 1), atol=1e-6)
    assert abs(float(centered.mean().item())) < 1e-6


def test_quantum_feature_map_rejects_unknown_kernel_mode() -> None:
    with pytest.raises(ValueError):
        QuantumFeatureMapConfig(kernel_mode="unknown")


def test_parameterized_quantum_feature_map_is_normalized_and_entangled() -> None:
    features = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
    config = SupervisedQuantumProjectorConfig(num_qubits=2, depth=2, training_steps=2, max_train_samples=2)
    parameters = {
        "ry_scale": torch.ones(2, 2),
        "ry_bias": torch.zeros(2, 2),
        "rz_scale": 0.5 * torch.ones(2, 2),
        "rz_bias": torch.zeros(2, 2),
    }

    states = parameterized_quantum_feature_map(features, config, parameters=parameters)

    assert states.shape == (2, 4)
    assert states.is_complex()
    assert torch.allclose(states.abs().pow(2).sum(dim=-1), torch.ones(2), atol=1e-5)


def test_supervised_quantum_training_preserves_best_kernel_alignment() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.0],
            [1.1, -0.1, 0.1],
            [-1.0, 0.0, -0.1],
            [-0.9, -0.1, 0.0],
            [-1.1, 0.1, -0.1],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    config = SupervisedQuantumProjectorConfig(
        num_qubits=2,
        depth=2,
        training_steps=20,
        learning_rate=0.08,
        max_train_samples=6,
        seed=5,
    )

    parameters, diagnostics = fit_supervised_quantum_feature_map(features, labels, config)

    assert set(parameters) == {"ry_scale", "ry_bias", "rz_scale", "rz_bias"}
    assert diagnostics["final_alignment"] >= diagnostics["initial_alignment"] - 1e-6
    assert diagnostics["num_train_samples"] == 6


def test_build_supervised_quantum_projector_is_standalone_and_symmetric() -> None:
    torch.manual_seed(71)
    keys = torch.randn(12, 5)
    relation_features = torch.cat((keys, keys.square()), dim=-1)
    labels = torch.tensor([0] * 6 + [1] * 6)
    result = build_supervised_quantum_projector(
        keys,
        relation_features,
        labels,
        quantum_config=SupervisedQuantumProjectorConfig(
            num_qubits=2,
            depth=1,
            training_steps=10,
            max_train_samples=12,
            seed=7,
        ),
        projector_config=SpectralProjectorConfig(rank=2),
        max_vectors=12,
    )

    assert result.projector.shape == (5, 5)
    assert torch.allclose(result.projector, result.projector.transpose(0, 1), atol=1e-5)
    assert result.metadata["projector_family"] == "quantum_label_aligned"
    assert result.metadata["standalone"] is True
    assert result.metadata["circuit"]["entanglement"] == "ring_cnot"
    assert result.metadata["training"]["final_alignment"] >= result.metadata["training"]["initial_alignment"] - 1e-6


def test_quantum_residual_projector_only_adds_complementary_directions() -> None:
    classical = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    quantum = torch.diag(torch.tensor([1.0, 1.0, 0.0]))

    hybrid = build_quantum_residual_projector(classical, quantum, alpha=0.5)

    assert torch.allclose(hybrid, torch.diag(torch.tensor([1.0, 0.5, 0.0])))
