from __future__ import annotations

import torch

from q_attention.plugins.q_ceasc import QCEASCConfig
from q_attention.plugins.q_ceasc_grouped_counterfactual import (
    QCEASCGroupedCounterfactualConfig,
    GroupedCounterfactualQCEASC,
    build_fixed_group_manifest,
    build_seeded_random_group_manifest,
)


def _make(
    mode: str = "q_ceasc_grouped_counterfactual",
    *,
    group_intervention_chunk_size: int = 64,
) -> GroupedCounterfactualQCEASC:
    base = QCEASCConfig(
        query_dim=6,
        key_dim=6,
        auxiliary_qubits=3,
        depth=2,
        support_width=2,
        action_rank=2,
        seed=13091,
    )
    return GroupedCounterfactualQCEASC(
        QCEASCGroupedCounterfactualConfig(
            base=base,
            group_size=2,
            group_intervention_chunk_size=group_intervention_chunk_size,
        ),
        mode,
    )


def _inputs(seed: int = 13):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(2, 6, generator=generator)
    key = torch.randn(2, 6, 6, generator=generator)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[1, -1] = False
    entity = torch.zeros(2, 6, dtype=torch.bool)
    entity[:, 0] = True
    entity[:, 1] = True
    return query, key, valid, entity


def test_fixed_manifest_is_label_free_and_masks_non_active_positions() -> None:
    valid = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    entity = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    manifest = build_fixed_group_manifest(valid, entity, group_size=2)
    assert manifest.tolist() == [[-1, 0, 1, 1, -1]]


def test_seeded_random_manifest_is_frozen_label_free_and_size_matched() -> None:
    valid = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    entity = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    first = build_seeded_random_group_manifest(valid, entity, group_size=2, seed=29)
    second = build_seeded_random_group_manifest(valid, entity, group_size=2, seed=29)
    positional = build_fixed_group_manifest(valid, entity, group_size=2)
    assert torch.equal(first, second)
    assert first[0, 0].item() == -1
    assert torch.bincount(first[first >= 0]).tolist() == [2, 2, 1]
    assert not torch.equal(first, positional)


def test_grouped_influence_is_finite_masked_and_zero_sum() -> None:
    kernel = _make()
    query, key, valid, entity = _inputs()
    result = kernel.evaluate(query, key, valid, entity)
    assert torch.isfinite(result.residual).all()
    assert torch.isfinite(result.group_influence_vectors).all()
    masked_values = result.residual.masked_select(~(valid & ~entity))
    assert torch.allclose(masked_values, torch.zeros_like(masked_values))
    assert torch.allclose(result.residual.sum(dim=-1), torch.zeros(2), atol=1e-6)
    assert result.diagnostics["group_evaluation_count"] == 2
    assert result.diagnostics["context_only"] is True
    assert result.diagnostics["target_free"] is True


def test_group_permutation_is_equivariant() -> None:
    kernel = _make()
    query, key, valid, entity = _inputs()
    group_ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, -1]])
    original = kernel.evaluate(query, key, valid, entity, group_ids)
    permutation = torch.tensor([2, 3, 0, 1, 4, 5])
    permuted_key = key[:, permutation]
    permuted_valid = valid[:, permutation]
    permuted_entity = entity[:, permutation]
    permuted_groups = group_ids[:, permutation]
    permuted = kernel.evaluate(
        query, permuted_key, permuted_valid, permuted_entity, permuted_groups
    )
    inverse = torch.argsort(permutation)
    assert torch.allclose(permuted.residual[:, inverse], original.residual, atol=1e-5)
    assert torch.allclose(
        permuted.member_scores[:, inverse], original.member_scores, atol=1e-5
    )


def test_unique_group_is_distinguishable_from_symmetric_groups() -> None:
    kernel = _make()
    query = torch.zeros(1, 6)
    key = torch.zeros(1, 6, 6)
    key[:, 2] = 1.0
    key[:, 3] = 1.0
    key[:, 4] = -1.0
    key[:, 5] = -1.0
    valid = torch.ones(1, 6, dtype=torch.bool)
    entity = torch.zeros_like(valid)
    group_ids = torch.tensor([[0, 0, 1, 2, 3, 3]])
    result = kernel.evaluate(query, key, valid, entity, group_ids)
    assert torch.isfinite(result.group_scores).all()
    assert torch.allclose(result.group_scores[:, 1], result.group_scores[:, 2], atol=1e-5)


def test_grouped_quantum_and_classical_controls_are_not_exact_replays() -> None:
    query, key, valid, entity = _inputs()
    group_ids = build_fixed_group_manifest(valid, entity, group_size=2)
    quantum = _make("q_ceasc_grouped_counterfactual").evaluate(
        query, key, valid, entity, group_ids
    )
    classical = _make("classical_grouped_counterfactual").evaluate(
        query, key, valid, entity, group_ids
    )
    assert not torch.allclose(quantum.residual, classical.residual, atol=1e-7, rtol=1e-6)


def test_random_grouping_control_is_not_a_fixed_replay() -> None:
    query, key, valid, entity = _inputs(seed=71)
    positional = build_fixed_group_manifest(valid, entity, group_size=2)
    random = build_seeded_random_group_manifest(valid, entity, group_size=2, seed=53)
    kernel = _make("q_ceasc_grouped_counterfactual")
    positional_result = kernel.evaluate(query, key, valid, entity, positional)
    random_result = kernel.evaluate(query, key, valid, entity, random)
    assert not torch.allclose(
        positional_result.residual, random_result.residual, atol=1e-7, rtol=1e-6
    )


def test_random_grouping_score_kernel_is_supported() -> None:
    from q_attention.plugins.q_ceasc_grouped_counterfactual import (
        QCEASCGroupedCounterfactualScoreKernel,
        QCEASCGroupedCounterfactualScoreKernelConfig,
    )

    config = QCEASCGroupedCounterfactualScoreKernelConfig(
        num_layers=1,
        num_heads=1,
        head_dim=6,
        group_size=2,
        query_chunk_size=4,
        max_context_size=6,
    )
    kernel = QCEASCGroupedCounterfactualScoreKernel(
        config, "random_grouped_counterfactual"
    )
    query, key, valid, entity = _inputs()
    captures = []
    kernel.capture_callback = captures.append
    output = kernel(
        query[:, None, None, :].expand(-1, 1, 6, -1),
        key[:, None, :, :],
        layer_index=0,
        attention_mask=valid,
        subject_mask=entity,
        object_mask=torch.zeros_like(entity),
        random_seed=53,
    )
    assert output.shape == (2, 1, 6, 6)
    assert torch.isfinite(output).all()
    manifests = torch.cat([item["group_manifest"] for item in captures], dim=1)
    assert torch.equal(manifests[:, :1].expand_as(manifests), manifests)
    assert all("auxiliary_state" in item and "coefficients" in item for item in captures)


def test_grouped_counterfactual_has_finite_gradients() -> None:
    kernel = _make()
    query, key, valid, entity = _inputs()
    query.requires_grad_()
    key.requires_grad_()
    result = kernel.evaluate(query, key, valid, entity)
    loss = result.residual.square().mean() + result.projected_support.square().mean()
    loss.backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in kernel.parameters())


def test_group_intervention_batching_preserves_exact_result() -> None:
    query, key, valid, entity = _inputs(seed=89)
    manifest = build_fixed_group_manifest(valid, entity, group_size=2)
    sequential = _make(group_intervention_chunk_size=1)
    batched = _make(group_intervention_chunk_size=64)
    batched.load_state_dict(sequential.state_dict())
    sequential_result = sequential.evaluate(query, key, valid, entity, manifest)
    batched_result = batched.evaluate(query, key, valid, entity, manifest)
    assert torch.allclose(sequential_result.residual, batched_result.residual, atol=1e-6)
    assert torch.allclose(
        sequential_result.group_influence_vectors,
        batched_result.group_influence_vectors,
        atol=1e-6,
    )
    assert batched_result.diagnostics["group_kernel_call_count"] == 1
    assert sequential_result.diagnostics["group_kernel_call_count"] == 2
