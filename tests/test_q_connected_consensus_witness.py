from __future__ import annotations

import torch

from q_attention.plugins.q_connected_consensus_witness import (
    ConnectedConsensusWitnessConfig,
    build_connected_consensus_witness,
    unordered_pair_index,
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


def config() -> ConnectedConsensusWitnessConfig:
    return ConnectedConsensusWitnessConfig(num_candidates=3, head_dim=4, seed=7)


def test_product_connected_null_is_numerically_zero() -> None:
    torch.manual_seed(7)
    query = torch.randn(4, 2, 4)
    key = torch.randn(4, 2, 6, 4)
    product = build_connected_consensus_witness("product", config(), frames())
    score = product.pair_scores(query, key)
    assert score.shape == (4, 2, 3, 15)
    assert torch.isfinite(score).all()
    assert float(score.abs().max()) <= 1e-6


def test_entangled_pair_scores_are_swap_symmetric_and_nontrivial() -> None:
    torch.manual_seed(11)
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 2, 6, 4)
    model = build_connected_consensus_witness("quantum", config(), frames())
    score = model.pair_scores(query, key)
    swapped = model.pair_scores(query, key[:, :, [1, 0, 2, 3, 4, 5]])
    assert torch.isfinite(score).all()
    assert float(score.abs().max()) > 1e-5
    assert torch.allclose(score[..., 0], swapped[..., 0], atol=1e-6)


def test_pair_scores_are_invariant_under_each_pair_exchange() -> None:
    torch.manual_seed(13)
    query = torch.randn(2, 2, 4)
    key = torch.randn(2, 2, 6, 4)
    model = build_connected_consensus_witness("quantum", config(), frames())
    pairs = model.pair_indices
    score = model.pair_scores(query, key)
    for pair_index, pair in enumerate(pairs.tolist()):
        swapped_key = key.clone()
        left, right = pair
        swapped_key[:, :, left], swapped_key[:, :, right] = (
            key[:, :, right].clone(),
            key[:, :, left].clone(),
        )
        swapped = model.pair_scores(query, swapped_key)
        assert torch.allclose(score[..., pair_index], swapped[..., pair_index], atol=1e-6)


def test_bilinear_has_equal_four_parameter_budget_and_finite_gradients() -> None:
    torch.manual_seed(17)
    query = torch.randn(5, 2, 4)
    key = torch.randn(5, 2, 6, 4)
    model = build_connected_consensus_witness("bilinear", config(), frames())
    assert sum(parameter.numel() for parameter in model.parameters()) == 4
    scores, _ = model.candidate_scores(query, key)
    loss = scores.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_unordered_pair_index_uses_all_fifteen_pairs() -> None:
    model = build_connected_consensus_witness("quantum", config(), frames())
    indices = unordered_pair_index(model.pair_indices, model.pair_indices)
    assert indices.tolist() == list(range(15))


def test_forward_signature_has_no_label_or_evidence_input() -> None:
    model = build_connected_consensus_witness("quantum", config(), frames())
    assert not hasattr(model, "labels")
    assert not hasattr(model, "evidence_slot")
