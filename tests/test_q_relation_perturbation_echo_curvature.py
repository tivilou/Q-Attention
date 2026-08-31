from __future__ import annotations

import torch

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.models.relation_transformer import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_relation_perturbation_echo_curvature import (
    LocalRelationEchoCurvatureControl,
    RelationPerturbationEchoConfig,
    RelationPerturbationEchoCurvatureKernel,
)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    attention = torch.ones(2, 5, dtype=torch.bool)
    attention[1, -1] = False
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 1] = True
    object_[:, 3] = True
    object_[1, 3] = False
    object_[1, 2] = True
    return {
        "query": torch.randn(2, 2, 5, 3, requires_grad=True),
        "key": torch.randn(2, 2, 5, 3, requires_grad=True),
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
    }


def _config() -> RelationPerturbationEchoConfig:
    return RelationPerturbationEchoConfig(num_layers=1, num_heads=2, head_dim=3, num_qubits=2, seed=19)


def test_q_rpec_residual_invariants_and_gradient() -> None:
    batch = _batch()
    kernel = RelationPerturbationEchoCurvatureKernel(_config())
    residual = kernel(**batch, layer_index=0)
    assert residual.shape == (2, 2, 5, 5)
    assert torch.isfinite(residual).all()
    entity = batch["subject_mask"] | batch["object_mask"]
    context = batch["attention_mask"] & ~entity
    assert (residual * entity[:, None, None, :]).abs().max() <= 1e-7
    assert (residual * (~batch["attention_mask"])[:, None, None, :]).abs().max() <= 1e-7
    assert (residual * context[:, None, None, :]).sum(dim=-1).abs().max() <= 1e-5
    assert residual.abs().sum() > 1e-8
    residual.square().mean().backward()
    grads = [p.grad for p in kernel.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.any(g.abs() > 0) for g in grads if g is not None)


def test_q_rpec_quantum_differs_from_parameter_matched_local_control() -> None:
    batch = _batch()
    quantum = RelationPerturbationEchoCurvatureKernel(_config())
    control = LocalRelationEchoCurvatureControl(_config())
    control.load_state_dict(quantum.state_dict())
    q = quantum(**batch, layer_index=0)
    c = control(**batch, layer_index=0)
    assert quantum.parameter_count == control.parameter_count
    assert torch.isfinite(q).all() and torch.isfinite(c).all()
    assert (q - c).abs().max() > 1e-7


def test_q_rpec_relation_perturbation_changes_action() -> None:
    batch = _batch()
    kernel = RelationPerturbationEchoCurvatureKernel(_config())
    base = kernel(**batch, layer_index=0)
    moved = dict(batch)
    moved["key"] = batch["key"].clone()
    moved["key"][:, :, 1, :] += 0.25
    changed = kernel(**moved, layer_index=0)
    assert (base - changed).abs().mean() > 1e-7


def test_q_rpec_analytic_observable_matches_statevector_reference_with_gradients() -> None:
    torch.manual_seed(41)
    config = _config()
    reference = RelationPerturbationEchoCurvatureKernel(config, pair_chunk_size=2)
    analytic = RelationPerturbationEchoCurvatureKernel(config, pair_chunk_size=7)
    analytic.load_state_dict(reference.state_dict())
    reference_inputs = [torch.randn(6, 3, requires_grad=True) for _ in range(3)]
    analytic_inputs = [item.detach().clone().requires_grad_() for item in reference_inputs]
    reference_output = reference._observable_statevector_reference(
        *reference_inputs, layer_index=0, head_index=1
    )
    analytic_output = analytic._observable(
        *analytic_inputs, layer_index=0, head_index=1
    )
    assert torch.allclose(analytic_output, reference_output, atol=2e-6, rtol=2e-5)
    reference_grads = torch.autograd.grad(
        reference_output.sum(),
        tuple(reference_inputs) + tuple(reference.parameters()),
        allow_unused=True,
    )
    analytic_grads = torch.autograd.grad(
        analytic_output.sum(),
        tuple(analytic_inputs) + tuple(analytic.parameters()),
        allow_unused=True,
    )
    for expected, actual in zip(reference_grads, analytic_grads):
        if expected is None or actual is None:
            assert expected is None and actual is None
        else:
            assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)


def test_q_rpec_pair_chunk_size_does_not_change_forward_or_gradient() -> None:
    torch.manual_seed(43)
    config = _config()
    small = RelationPerturbationEchoCurvatureKernel(config, pair_chunk_size=1)
    large = RelationPerturbationEchoCurvatureKernel(config, pair_chunk_size=11)
    large.load_state_dict(small.state_dict())
    batch_small = _batch()
    batch_large = {name: value.detach().clone().requires_grad_(value.requires_grad) for name, value in batch_small.items()}
    output_small = small(**batch_small, layer_index=0)
    output_large = large(**batch_large, layer_index=0)
    assert torch.allclose(output_small, output_large, atol=2e-6, rtol=2e-5)
    grads_small = torch.autograd.grad(output_small.square().mean(), tuple(small.parameters()))
    grads_large = torch.autograd.grad(output_large.square().mean(), tuple(large.parameters()))
    for expected, actual in zip(grads_small, grads_large):
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)


def test_q_rpec_score_hook_integrates_with_relation_transformer() -> None:
    torch.manual_seed(17)
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=32,
            num_labels=3,
            dim=12,
            num_layers=1,
            num_heads=3,
            ff_dim=24,
            dropout=0.0,
            max_length=6,
        )
    )
    attention = torch.ones(2, 6, dtype=torch.bool)
    attention[1, -1] = False
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 1] = True
    object_[:, 4] = True
    object_[1, 4] = False
    object_[1, 3] = True
    kernel = RelationPerturbationEchoCurvatureKernel(
        RelationPerturbationEchoConfig(num_layers=1, num_heads=3, head_dim=4, num_qubits=2, seed=31)
    )
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    batch = {
        "input_ids": torch.randint(1, 32, (2, 6)),
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
    }
    with adapter.steering(
        AttentionScoreHookConfig(
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
    ):
        logits = model(**batch)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in kernel.parameters()
    )
