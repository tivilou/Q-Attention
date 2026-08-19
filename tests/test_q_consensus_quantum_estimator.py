from __future__ import annotations

import torch

from q_attention.plugins.q_consensus_quantum_estimator import (
    ConsensusQuantumEstimatorConfig,
    build_consensus_estimator,
)


def frames() -> torch.Tensor:
    result = torch.eye(4).repeat(3, 1, 1)
    result[1, 0, 0] = -1.0
    result[1, 1, 1] = -1.0
    result[2, 2, 2] = -1.0
    result[2, 3, 3] = -1.0
    return result


def config() -> ConsensusQuantumEstimatorConfig:
    return ConsensusQuantumEstimatorConfig(
        num_candidates=3,
        head_dim=4,
        register_qubits=3,
        depth=2,
        seed=7,
    )


def test_quantum_and_classical_fields_are_parameter_matched_and_finite() -> None:
    torch.manual_seed(7)
    query = torch.randn(5, 2, 4)
    key = torch.randn(5, 2, 6, 4)
    quantum = build_consensus_estimator("quantum", config(), frames())
    classical = build_consensus_estimator("classical", config(), frames())
    quantum_parameters = sum(parameter.numel() for parameter in quantum.parameters())
    classical_parameters = sum(parameter.numel() for parameter in classical.parameters())
    assert quantum_parameters == classical_parameters
    quantum_field = quantum.field(query, key)
    classical_field = classical.field(query, key)
    assert quantum_field.shape == classical_field.shape == (5, 2, 3, 6)
    assert torch.isfinite(quantum_field).all()
    assert torch.isfinite(classical_field).all()
    assert not torch.allclose(quantum_field, classical_field)


def test_field_depends_on_query_and_candidate_frame() -> None:
    torch.manual_seed(11)
    query = torch.randn(4, 2, 4)
    key = torch.randn(4, 2, 6, 4)
    estimator = build_consensus_estimator("quantum", config(), frames())
    field = estimator.field(query, key)
    shuffled_query = estimator.field(torch.roll(query, 1, 0), key)
    assert not torch.allclose(field, shuffled_query)
    permuted_frames = frames()[torch.tensor([1, 2, 0])]
    permuted = build_consensus_estimator("quantum", config(), permuted_frames)
    permuted.load_state_dict(
        {
            name: value
            for name, value in estimator.state_dict().items()
            if name != "candidate_frames"
        },
        strict=False,
    )
    permuted_field = permuted.field(query, key)
    assert torch.allclose(permuted_field, field[:, :, [1, 2, 0], :], atol=1e-6)


def test_signed_candidate_loss_has_nonzero_finite_gradients() -> None:
    torch.manual_seed(13)
    query = torch.randn(6, 2, 4)
    key = torch.randn(6, 2, 6, 4)
    labels = torch.randint(0, 3, (6, 2))
    estimator = build_consensus_estimator("quantum", config(), frames())
    scores = estimator.candidate_scores(query, key)
    loss = torch.nn.functional.cross_entropy(scores.reshape(-1, 3), labels.reshape(-1))
    loss.backward()
    gradients = [parameter.grad for parameter in estimator.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_metadata_declares_bounded_experimental_scope() -> None:
    estimator = build_consensus_estimator("quantum", config(), frames())
    metadata = estimator.metadata()["plugin"]
    assert metadata["type"] == "standalone_query_local_candidate_field"
    assert metadata["resource_estimate"]["data_qubits"] == 3
    assert "not modeled" in metadata["resource_estimate"]["shots"]
    assert "matched classical product-state control wins" in metadata["failure_signatures"]
