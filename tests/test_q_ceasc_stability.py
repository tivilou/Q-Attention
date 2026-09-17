from __future__ import annotations

import torch

from q_attention.plugins.q_ceasc import QCEASCConfig, build_qceasc
from q_attention.plugins.q_ceasc_score import QCEASCScoreKernelConfig
from q_attention.plugins.q_ceasc_stability import (
    QCEASCStabilityConfig,
    QCEASCStabilityScoreKernel,
    QCEASC_STABILITY_CONTROL_MODES,
    build_qceasc_stability,
    build_qceasc_stability_score_kernel,
)


def _stage_inputs(batch: int = 4, context: int = 6, dim: int = 6):
    generator = torch.Generator(device="cpu").manual_seed(18031)
    query = torch.randn(batch, dim, generator=generator)
    key = torch.randn(batch, context, dim, generator=generator)
    valid = torch.ones(batch, context, dtype=torch.bool)
    valid[0, -1] = False
    entity = torch.zeros(batch, context, dtype=torch.bool)
    entity[1, 0] = True
    return query, key, valid, entity


def _stage_config(seed: int = 41) -> QCEASCStabilityConfig:
    return QCEASCStabilityConfig(
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


def test_stability_controls_have_equal_double_view_budget():
    kernels = [build_qceasc_stability(mode, _stage_config()) for mode in QCEASC_STABILITY_CONTROL_MODES]
    assert {kernel.parameter_count for kernel in kernels} == {50}
    assert {kernel.metadata()["single_view_parameter_count"] for kernel in kernels} == {25}


def test_stability_gate_preserves_mask_projection_and_zero_sum():
    query, key, valid, entity = _stage_inputs()
    active = valid & ~entity
    kernel = build_qceasc_stability("q_ceasc_stability", _stage_config())
    result = kernel.evaluate(query, key, valid, entity)
    assert torch.isfinite(result.residual).all()
    assert torch.isfinite(result.diagnostics["agreement"]).all()
    assert torch.isfinite(result.diagnostics["agreement_gate"]).all()
    assert torch.all(result.diagnostics["agreement_gate"] >= 0.0)
    assert torch.all(result.diagnostics["agreement_gate"] <= 1.0)
    assert torch.allclose(
        result.residual[~active], torch.zeros_like(result.residual[~active]), atol=1e-7
    )
    assert torch.allclose(
        result.residual.sum(dim=-1), torch.zeros(query.shape[0]), atol=1e-6
    )
    assert result.diagnostics["projection_idempotence_error"] < 2e-5
    assert torch.isfinite(result.diagnostics["out_of_span_support_norm"]).all()


def test_stability_is_permutation_equivariant_and_empty_safe():
    query, key, valid, entity = _stage_inputs(batch=2)
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    kernel = build_qceasc_stability("q_ceasc_stability", _stage_config())
    result = kernel.evaluate(query, key, valid, entity)
    permuted = kernel.evaluate(
        query, key[:, permutation], valid[:, permutation], entity[:, permutation]
    )
    restored = torch.zeros_like(permuted.residual)
    restored[:, permutation] = permuted.residual
    assert torch.allclose(result.residual, restored, atol=4e-5, rtol=4e-5)
    empty_valid = torch.zeros_like(valid)
    empty = kernel.evaluate(query, key, empty_valid, entity)
    assert torch.allclose(empty.residual, torch.zeros_like(empty.residual))
    assert torch.allclose(
        empty.diagnostics["agreement_gate"], torch.zeros(query.shape[0])
    )


def test_stability_gradients_are_finite_for_quantum_and_classical_views():
    query, key, valid, entity = _stage_inputs(batch=2)
    for mode in ("q_ceasc_stability", "classical_stability"):
        local_query = query.clone().requires_grad_()
        local_key = key.clone().requires_grad_()
        kernel = build_qceasc_stability(mode, _stage_config())
        result = kernel.evaluate(local_query, local_key, valid, entity)
        loss = result.residual.square().mean() + result.projected_support.square().mean()
        loss.backward()
        assert local_query.grad is not None and torch.isfinite(local_query.grad).all()
        assert local_key.grad is not None and torch.isfinite(local_key.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in kernel.parameters()
        )


def test_two_views_are_not_accidentally_identical():
    query, key, valid, entity = _stage_inputs(batch=2)
    kernel = build_qceasc_stability("q_ceasc_stability", _stage_config())
    result = kernel.evaluate(query, key, valid, entity)
    assert float(
        (result.view_one.residual - result.view_two.residual).abs().max()
    ) > 1e-8
    assert torch.isfinite(result.diagnostics["agreement"]).all()


def _score_inputs():
    query = torch.randn(2, 2, 5, 4)
    key = torch.randn(2, 2, 5, 4)
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    subject = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    object_ = torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.bool)
    return query, key, attention, subject, object_


def test_score_wrapper_shape_masks_and_budget():
    config = QCEASCScoreKernelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        auxiliary_qubits=2,
        support_width=2,
        action_rank=2,
        query_chunk_size=2,
    )
    kernel = build_qceasc_stability_score_kernel("q_ceasc_stability", config)
    query, key, attention, subject, object_ = _score_inputs()
    residual = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    assert residual.shape == (2, 2, 5, 5)
    assert torch.isfinite(residual).all()
    assert torch.allclose(residual[:, :, :, -1], torch.zeros(2, 2, 5), atol=1e-7)
    assert torch.allclose(residual.sum(dim=-1), torch.zeros(2, 2, 5), atol=1e-6)
    assert torch.allclose(residual[:, :, -1, :], torch.zeros(2, 2, 5), atol=1e-7)
    assert kernel.parameter_count == 2 * kernel.view_one.parameter_count
