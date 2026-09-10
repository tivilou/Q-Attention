from __future__ import annotations

import pytest
import torch

from q_attention.plugins.q_ceasc import (
    QCEASC_CONTROL_MODES,
    QCEASCConfig,
    build_qceasc,
)


def _inputs(batch: int = 4, context: int = 6, dim: int = 6):
    generator = torch.Generator(device="cpu").manual_seed(90211)
    query = torch.randn(batch, dim, generator=generator)
    key = torch.randn(batch, context, dim, generator=generator)
    valid = torch.ones(batch, context, dtype=torch.bool)
    valid[0, -1] = False
    entity = torch.zeros(batch, context, dtype=torch.bool)
    if batch > 1:
        entity[1, 0] = True
    return query, key, valid, entity


def _config() -> QCEASCConfig:
    return QCEASCConfig(
        query_dim=6,
        key_dim=6,
        auxiliary_qubits=3,
        depth=2,
        support_width=2,
        action_rank=2,
        seed=41,
    )


def test_controls_have_equal_trainable_budget_and_signed_readout_metadata():
    kernels = [build_qceasc(mode, _config()) for mode in QCEASC_CONTROL_MODES]
    assert {kernel.parameter_count for kernel in kernels} == {25}
    assert {kernel.metadata()["parameter_count"] for kernel in kernels} == {25}
    assert kernels[0].metadata()["readout"] == "signed_bipolar_observable_expectations"


def test_projection_and_attention_invariants():
    query, key, valid, entity = _inputs()
    active = valid & ~entity
    for mode in QCEASC_CONTROL_MODES:
        result = build_qceasc(mode, _config()).evaluate(query, key, valid, entity)
        assert torch.isfinite(result.residual).all()
        assert torch.isfinite(result.projected_support).all()
        assert torch.allclose(
            result.residual[~active],
            torch.zeros_like(result.residual[~active]),
            atol=1e-7,
        )
        assert torch.allclose(
            result.residual.sum(dim=-1),
            torch.zeros(query.shape[0]),
            atol=1e-6,
        )
        assert result.diagnostics["projection_idempotence_error"] < 2e-5
    quantum = build_qceasc("q_ceasc", _config()).evaluate(query, key, valid, entity)
    assert float(quantum.diagnostics["out_of_span_residual_norm"].max()) > 1e-8
    assert float(quantum.diagnostics["span_leakage_norm"].max()) < 2e-5


def test_context_permutation_equivariance():
    query, key, valid, entity = _inputs(batch=2)
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    kernel = build_qceasc("q_ceasc", _config())
    result = kernel.evaluate(query, key, valid, entity)
    permuted = kernel.evaluate(
        query,
        key[:, permutation],
        valid[:, permutation],
        entity[:, permutation],
    )
    restored = torch.zeros_like(permuted.residual)
    restored[:, permutation] = permuted.residual
    assert torch.allclose(result.residual, restored, atol=3e-5, rtol=3e-5)


def test_gradients_are_finite_for_quantum_and_classical_controls():
    query, key, valid, entity = _inputs(batch=2)
    for mode in ("q_ceasc", "classical_span"):
        local_query = query.clone().requires_grad_()
        local_key = key.clone().requires_grad_()
        kernel = build_qceasc(mode, _config())
        result = kernel.evaluate(local_query, local_key, valid, entity)
        loss = result.residual.square().mean() + result.projected_support.square().mean()
        loss.backward()
        assert local_query.grad is not None and torch.isfinite(local_query.grad).all()
        assert local_key.grad is not None and torch.isfinite(local_key.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in kernel.parameters()
        )


def test_product_null_has_finite_active_gradients():
    query, key, valid, entity = _inputs(batch=2)
    kernel = build_qceasc("quantum_product", _config())
    result = kernel.evaluate(query, key, valid, entity)
    result.residual.square().mean().backward()
    used = [parameter for parameter in kernel.parameters() if parameter.grad is not None]
    assert used
    assert all(torch.isfinite(parameter.grad).all() for parameter in used)
    assert any(parameter.grad is None for parameter in kernel.parameters())


def test_signed_readout_and_entangling_diagnostics_are_observable():
    query, key, valid, entity = _inputs(batch=2)
    kernel = build_qceasc("q_ceasc", _config())
    basis_states = torch.eye(kernel.state_dim, dtype=torch.complex64)
    coefficients = kernel._signed_coefficients(basis_states)
    assert float(coefficients.min()) < 0.0
    assert float(coefficients.max()) > 0.0
    quantum = kernel.evaluate(query, key, valid, entity)
    product = build_qceasc("quantum_product", _config()).evaluate(
        query, key, valid, entity
    )
    assert "entangling_covariance_norm" in quantum.diagnostics
    assert torch.isfinite(quantum.diagnostics["entangling_covariance_norm"]).all()
    assert float((quantum.residual - product.residual).abs().max()) > 1e-8


def test_span_rank_cutoff_and_frozen_buffers_are_recorded():
    action_span = torch.zeros(6, 2)
    action_span[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_span[:, 1] = torch.tensor([1.0, 1e-8, 0.0, 0.0, 0.0, 0.0])
    config = _config()
    kernel = build_qceasc("q_ceasc", config, action_span=action_span)
    assert int(kernel.span_basis.shape[1]) == 1
    assert not kernel.action_span.requires_grad
    assert not kernel.support_dictionary.requires_grad
    assert kernel.metadata()["span_rcond"] == config.span_rcond


def test_float64_path_is_finite():
    query, key, valid, entity = _inputs(batch=1, context=4)
    result = build_qceasc("q_ceasc", _config()).double().evaluate(
        query.double(), key.double(), valid, entity
    )
    assert result.residual.dtype == torch.float64
    assert torch.isfinite(result.residual).all()


def test_invalid_rows_are_rejected():
    query, key, valid, entity = _inputs(batch=2)
    valid[:] = False
    with pytest.raises(ValueError, match="at least one"):
        build_qceasc("q_ceasc", _config()).evaluate(query, key, valid, entity)
