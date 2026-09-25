from __future__ import annotations

from dataclasses import replace

import torch

from q_attention.plugins.q_ceasc import QCEASCConfig
from q_attention.plugins.q_ceasc_counterfactual import (
    QCEASCCounterfactualConfig,
    build_qceasc_counterfactual,
)


def _inputs(batch: int = 2, context: int = 5, dim: int = 6):
    generator = torch.Generator(device="cpu").manual_seed(19101)
    query = torch.randn(batch, dim, generator=generator)
    key = torch.randn(batch, context, dim, generator=generator)
    valid = torch.ones(batch, context, dtype=torch.bool)
    valid[0, -1] = False
    entity = torch.zeros(batch, context, dtype=torch.bool)
    entity[1, 0] = True
    return query, key, valid, entity


def _config(seed: int = 53, leave_one_out_chunk_size: int = 64):
    return QCEASCCounterfactualConfig(
        base=QCEASCConfig(
            query_dim=6,
            key_dim=6,
            auxiliary_qubits=3,
            depth=2,
            support_width=2,
            action_rank=2,
            seed=seed,
        ),
        leave_one_out_chunk_size=leave_one_out_chunk_size,
    )


def test_leave_one_out_chunking_preserves_forward_and_gradients() -> None:
    reference_config = replace(_config(seed=53), leave_one_out_chunk_size=1)
    vectorized_config = replace(_config(seed=53), leave_one_out_chunk_size=3)
    reference = build_qceasc_counterfactual(
        "q_ceasc_counterfactual", reference_config
    )
    vectorized = build_qceasc_counterfactual(
        "q_ceasc_counterfactual", vectorized_config
    )
    vectorized.load_state_dict(reference.state_dict())

    query, key, valid, entity = _inputs(batch=2, context=5, dim=6)
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    vectorized_query = query.detach().clone().requires_grad_()
    vectorized_key = key.detach().clone().requires_grad_()
    reference_result = reference.evaluate(
        reference_query, reference_key, valid, entity
    )
    vectorized_result = vectorized.evaluate(
        vectorized_query, vectorized_key, valid, entity
    )

    for name in (
        "residual",
        "support_vector",
        "projected_support",
        "coefficients",
        "auxiliary_state",
        "influence_scores",
        "influence_vectors",
    ):
        reference_value = getattr(reference_result, name)
        vectorized_value = getattr(vectorized_result, name)
        assert torch.allclose(
            reference_value,
            vectorized_value,
            atol=2e-6,
            rtol=2e-5,
        ), name

    reference_loss = (
        reference_result.residual.square().mean()
        + reference_result.projected_support.square().mean()
    )
    vectorized_loss = (
        vectorized_result.residual.square().mean()
        + vectorized_result.projected_support.square().mean()
    )
    reference_grads = torch.autograd.grad(
        reference_loss,
        (reference_query, reference_key, *reference.parameters()),
    )
    vectorized_grads = torch.autograd.grad(
        vectorized_loss,
        (vectorized_query, vectorized_key, *vectorized.parameters()),
    )
    assert len(reference_grads) == len(vectorized_grads)
    for reference_grad, vectorized_grad in zip(reference_grads, vectorized_grads):
        assert torch.allclose(
            reference_grad,
            vectorized_grad,
            atol=2e-5,
            rtol=2e-4,
        )
