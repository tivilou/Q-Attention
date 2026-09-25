from __future__ import annotations

import inspect

import torch

from q_attention.plugins.q_coherence_destruction_differential import (
    CoherenceDifferentialConfig,
    build_coherence_differential,
    explicit_dephased_connected_yyyy,
)
from q_attention.plugins.q_connected_consensus_witness import (
    build_connected_consensus_witness,
)


def frames() -> torch.Tensor:
    result = torch.eye(4).repeat(3, 1, 1)
    angles = torch.tensor([0.0, 2.0 * torch.pi / 3.0, 4.0 * torch.pi / 3.0])
    result[:, 0, 0] = torch.cos(angles)
    result[:, 0, 1] = -torch.sin(angles)
    result[:, 1, 0] = torch.sin(angles)
    result[:, 1, 1] = torch.cos(angles)
    result[:, 2, 2] = torch.cos(angles)
    result[:, 2, 3] = -torch.sin(angles)
    result[:, 3, 2] = torch.sin(angles)
    result[:, 3, 3] = torch.cos(angles)
    return result


def config() -> CoherenceDifferentialConfig:
    return CoherenceDifferentialConfig(num_candidates=3, head_dim=4, seed=7)


def sample() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return torch.randn(3, 2, 4), torch.randn(3, 2, 6, 4)


def test_dephased_analytic_null_matches_explicit_density_matrix() -> None:
    query, key = sample()
    model = build_coherence_differential("quantum", config(), frames())
    features_a, features_b = model._local_features(query[:1], key[:1])
    state = model._joint_state(features_a, features_b)
    explicit = explicit_dephased_connected_yyyy(state)
    components = model.pair_score_components(query[:1], key[:1])

    assert float(explicit.abs().max()) <= 1e-6
    assert torch.equal(
        components["dephased_yyyy"],
        torch.zeros_like(components["dephased_yyyy"]),
    )


def test_quantum_differential_is_nontrivial_and_product_is_null() -> None:
    query, key = sample()
    quantum = build_coherence_differential("quantum", config(), frames())
    product = build_coherence_differential("product", config(), frames())
    product.load_state_dict(quantum.state_dict())

    quantum_score = quantum.pair_scores(query, key)
    product_score = product.pair_scores(query, key)
    assert quantum_score.shape == product_score.shape == (3, 2, 3, 15)
    assert torch.isfinite(quantum_score).all()
    assert float(quantum_score.abs().max()) > 1e-5
    assert float(product_score.abs().max()) <= 1e-6


def test_four_body_differential_is_swap_symmetric_for_every_pair() -> None:
    query, key = sample()
    model = build_coherence_differential("quantum", config(), frames())
    scores = model.pair_scores(query, key)
    for pair_index, pair in enumerate(model.pair_indices.tolist()):
        left, right = pair
        swapped_key = key.clone()
        swapped_key[:, :, left], swapped_key[:, :, right] = (
            key[:, :, right].clone(),
            key[:, :, left].clone(),
        )
        swapped = model.pair_scores(query, swapped_key)
        assert torch.allclose(scores[..., pair_index], swapped[..., pair_index], atol=1e-6)


def test_raw_xx_component_matches_qccw_with_identical_parameters() -> None:
    query, key = sample()
    qcdd = build_coherence_differential("quantum", config(), frames())
    qccw = build_connected_consensus_witness("quantum", config(), frames())
    qccw.load_state_dict(qcdd.state_dict())

    raw = qcdd.pair_score_components(query, key)["raw_qccw_xx"]
    expected = qccw.pair_scores(query, key)
    assert torch.allclose(raw, expected, atol=1e-6)


def test_quantum_and_sincos_controls_have_four_finite_trainable_parameters() -> None:
    query, key = sample()
    for kind in ("quantum", "sincos"):
        model = build_coherence_differential(kind, config(), frames())
        assert sum(parameter.numel() for parameter in model.parameters()) == 4
        scores, _ = model.candidate_scores(query, key)
        loss = scores.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_plugin_inference_signature_excludes_labels_and_evidence_slots() -> None:
    parameters = set(
        inspect.signature(
            build_coherence_differential("quantum", config(), frames()).pair_scores
        ).parameters
    )
    assert parameters == {"query", "key", "key_second"}
