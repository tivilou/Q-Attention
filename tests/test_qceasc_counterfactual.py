from __future__ import annotations

import torch

from q_attention.plugins.q_ceasc import QCEASCConfig
from q_attention.plugins.q_ceasc_counterfactual import (
    QCEASC_COUNTERFACTUAL_CONTROL_MODES,
    QCEASCCounterfactualConfig,
    QCEASCCounterfactualScoreKernelConfig,
    build_qceasc_counterfactual,
    build_qceasc_counterfactual_score_kernel,
)


def _inputs(batch: int = 3, context: int = 6, dim: int = 6):
    generator = torch.Generator(device="cpu").manual_seed(19101)
    query = torch.randn(batch, dim, generator=generator)
    key = torch.randn(batch, context, dim, generator=generator)
    valid = torch.ones(batch, context, dtype=torch.bool)
    valid[0, -1] = False
    entity = torch.zeros(batch, context, dtype=torch.bool)
    if batch > 1:
        entity[1, 0] = True
    return query, key, valid, entity


def _config(seed: int = 41) -> QCEASCCounterfactualConfig:
    return QCEASCCounterfactualConfig(
        base=QCEASCConfig(
            query_dim=6,
            key_dim=6,
            auxiliary_qubits=3,
            depth=2,
            support_width=2,
            action_rank=2,
            seed=seed,
        )
    )


def test_counterfactual_controls_have_equal_budget():
    kernels = [build_qceasc_counterfactual(mode, _config()) for mode in QCEASC_COUNTERFACTUAL_CONTROL_MODES]
    assert {kernel.parameter_count for kernel in kernels} == {25}


def test_leave_one_out_is_finite_masked_and_zero_sum():
    query, key, valid, entity = _inputs()
    kernel = build_qceasc_counterfactual("q_ceasc_counterfactual", _config())
    result = kernel.evaluate(query, key, valid, entity)
    active = valid & ~entity
    assert result.influence_vectors.shape == (3, 6, 6)
    assert torch.isfinite(result.residual).all()
    assert torch.isfinite(result.influence_vectors).all()
    assert torch.allclose(result.residual[~active], torch.zeros_like(result.residual[~active]), atol=1e-7)
    assert torch.allclose(result.residual.sum(dim=-1), torch.zeros(3), atol=1e-6)


def test_counterfactual_permutation_equivariance_and_empty_rows():
    query, key, valid, entity = _inputs(batch=2)
    kernel = build_qceasc_counterfactual("q_ceasc_counterfactual", _config())
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    original = kernel.evaluate(query, key, valid, entity)
    permuted = kernel.evaluate(query, key[:, permutation], valid[:, permutation], entity[:, permutation])
    restored = torch.zeros_like(permuted.residual)
    restored[:, permutation] = permuted.residual
    assert torch.allclose(original.residual, restored, atol=4e-5, rtol=4e-5)
    empty = kernel.evaluate(query, key, torch.zeros_like(valid), entity)
    assert torch.allclose(empty.residual, torch.zeros_like(empty.residual))
    assert torch.allclose(empty.influence_vectors, torch.zeros_like(empty.influence_vectors))


def test_counterfactual_gradients_are_finite():
    query, key, valid, entity = _inputs(batch=2)
    kernel = build_qceasc_counterfactual("q_ceasc_counterfactual", _config())
    query = query.requires_grad_()
    key = key.requires_grad_()
    result = kernel.evaluate(query, key, valid, entity)
    loss = result.residual.square().mean() + result.projected_support.square().mean()
    loss.backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in kernel.parameters())


def test_zero_norm_key_is_finite_and_has_no_probe_contribution():
    query, key, valid, entity = _inputs(batch=1)
    key[:, 2] = 0.0
    kernel = build_qceasc_counterfactual("q_ceasc_counterfactual", _config())
    result = kernel.evaluate(query, key, valid, entity)
    assert torch.isfinite(result.residual).all()
    assert torch.isfinite(result.influence_scores).all()
    assert abs(float(result.influence_scores[0, 2])) < 1e-7


def test_counterfactual_score_wrapper_shape_and_masks():
    config = QCEASCCounterfactualScoreKernelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        auxiliary_qubits=2,
        support_width=2,
        action_rank=2,
        query_chunk_size=2,
    )
    kernel = build_qceasc_counterfactual_score_kernel("q_ceasc_counterfactual", config)
    query = torch.randn(2, 2, 5, 4)
    key = torch.randn(2, 2, 5, 4)
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    subject = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    object_ = torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.bool)
    residual = kernel(query, key, layer_index=0, attention_mask=attention, subject_mask=subject, object_mask=object_)
    assert residual.shape == (2, 2, 5, 5)
    assert torch.isfinite(residual).all()
    assert torch.allclose(residual[:, :, :, -1], torch.zeros(2, 2, 5), atol=1e-7)
    assert torch.allclose(residual.sum(dim=-1), torch.zeros(2, 2, 5), atol=1e-6)
    assert torch.allclose(residual[:, :, -1, :], torch.zeros(2, 2, 5), atol=1e-7)
