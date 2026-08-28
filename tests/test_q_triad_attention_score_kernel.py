from __future__ import annotations

import torch

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.models.relation_transformer import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_triad import QTriadAttentionScoreKernel


def _batch() -> tuple[RelationExtractionModel, dict[str, torch.Tensor]]:
    torch.manual_seed(7)
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=32,
            num_labels=4,
            dim=16,
            num_layers=2,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
            max_length=8,
        )
    )
    attention_mask = torch.ones(3, 8, dtype=torch.bool)
    attention_mask[1, -2:] = False
    attention_mask[2, -1:] = False
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, 1] = True
    object_mask[:, 5] = True
    object_mask[1, 5] = False
    object_mask[1, 4] = True
    object_mask[2, 5] = False
    object_mask[2, 3] = True
    return model, {
        "input_ids": torch.randint(1, 32, (3, 8)),
        "attention_mask": attention_mask,
        "subject_mask": subject_mask,
        "object_mask": object_mask,
    }


def _kernel(mode: str) -> QTriadAttentionScoreKernel:
    return QTriadAttentionScoreKernel(
        num_layers=2,
        num_heads=4,
        head_dim=4,
        num_qubits=2,
        circuit_depth=1,
        control_mode=mode,
        seed=19,
    )


def _hook_config(batch: dict[str, torch.Tensor]) -> AttentionScoreHookConfig:
    return AttentionScoreHookConfig(
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
    )


def test_qtriad_score_hook_is_finite_and_label_free() -> None:
    model, batch = _batch()
    kernel = _kernel("q_triad")
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)  # type: ignore[arg-type]
    with adapter.steering(_hook_config(batch)):
        logits = model(**batch)
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert kernel.metadata()["inference_mode"] == "label_free"
    assert "labels" not in kernel.metadata()["target_input"]


def test_qtriad_kernel_receives_gradients_through_attention_action() -> None:
    model, batch = _batch()
    kernel = _kernel("q_triad")
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)  # type: ignore[arg-type]
    model.zero_grad(set_to_none=True)
    kernel.zero_grad(set_to_none=True)
    with adapter.steering(_hook_config(batch)):
        loss = model(**batch).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in kernel.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(torch.any(gradient.abs() > 0).item()) for gradient in gradients if gradient is not None)


def test_classical_density_control_matches_qtriad_action() -> None:
    model, batch = _batch()
    query = torch.randn(3, 4, 8, 4)
    key = torch.randn(3, 4, 8, 4)
    scores = torch.randn(3, 4, 8, 8)
    common = {
        "scores": scores,
        "layer_index": 0,
        "attention_mask": batch["attention_mask"],
        "subject_mask": batch["subject_mask"],
        "object_mask": batch["object_mask"],
    }
    quantum = _kernel("q_triad")
    classical = _kernel("classical_density_tensor")
    classical.load_state_dict(quantum.state_dict())
    q_residual = quantum(query, key, **common)
    c_residual = classical(query, key, **common)
    assert torch.allclose(q_residual, c_residual, atol=3e-5, rtol=3e-5)
    assert model is not None


def test_qtriad_action_preserves_entity_keys_and_context_mass() -> None:
    _model, batch = _batch()
    kernel = _kernel("q_triad")
    query = torch.randn(3, 4, 8, 4)
    key = torch.randn(3, 4, 8, 4)
    scores = torch.randn(3, 4, 8, 8)
    residual = kernel(
        query,
        key,
        scores=scores,
        layer_index=0,
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
    )
    entity = batch["subject_mask"] | batch["object_mask"]
    assert (residual * entity[:, None, None, :].to(residual.dtype)).abs().max() <= 1e-7
    context = batch["attention_mask"] & ~entity
    context_sum = (residual * context[:, None, None, :].to(residual.dtype)).sum(dim=-1)
    assert context_sum.abs().max() <= 1e-6
