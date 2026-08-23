from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from experiments import train_relation_attention_score_kernel
from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.experiments import (
    GradientNormTracker,
    counterfactual_evidence_objective,
    diagnose_relation_attention_score_kernel,
    diagnose_relation_attention_score_task_alignment,
    diagnose_relation_counterfactual_evidence,
    diagnose_relation_evidence_measurement_frames,
    diagnose_relation_evidence_task_alignment,
    diagnose_relation_expert_direction_alignment,
    diagnose_relation_expert_routing,
    diagnose_relation_routing_task_alignment,
    expert_routing_objective,
    relation_selection_score,
)
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins import (
    ClassicalRelationAttentionScoreKernel,
    ClassicalRelationEvidenceSelector,
    ClassicalRelationObservableExpertRouter,
    QuantumRelationAttentionScoreKernel,
    QuantumRelationEvidenceSelector,
    QuantumRelationObservableExpertRouter,
    RelationExpertRouterConfig,
    RelationEvidenceSelectorConfig,
    RelationScoreKernelConfig,
    load_relation_attention_score_kernel_checkpoint,
    save_relation_attention_score_kernel_checkpoint,
    score_residual_to_query_aligned_key_delta,
)


def relation_masks(
    batch: int,
    tokens: int,
    *,
    padded: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = torch.ones(batch, tokens, dtype=torch.bool)
    if padded:
        attention[:, -1] = False
    subject = torch.zeros(batch, tokens, dtype=torch.bool)
    object_ = torch.zeros(batch, tokens, dtype=torch.bool)
    subject[:, 0] = True
    object_[:, 2] = True
    return attention, subject, object_


def kernel_config(
    num_layers: int = 1,
    *,
    score_readout: str = "fidelity",
    input_encoding: str = "joint",
    query_scope: str = "all",
    relation_anchor_mode: str = "entity_pair",
    normalize_readout_energy: bool = False,
) -> RelationScoreKernelConfig:
    return RelationScoreKernelConfig(
        num_layers=num_layers,
        num_heads=2,
        head_dim=4,
        num_qubits=3,
        depth=2,
        initial_gain=0.05,
        normalize_readout_energy=normalize_readout_energy,
        score_readout=score_readout,
        input_encoding=input_encoding,
        query_scope=query_scope,
        relation_anchor_mode=relation_anchor_mode,
        seed=13,
    )


def evidence_config(
    num_layers: int = 1,
    *,
    evidence_readout: str = "factorized_observable",
    relation_anchor_mode: str = "entity_pair",
) -> RelationEvidenceSelectorConfig:
    return RelationEvidenceSelectorConfig(
        num_layers=num_layers,
        num_heads=2,
        head_dim=4,
        num_qubits=3,
        depth=2,
        mask_floor=0.1,
        evidence_readout=evidence_readout,
        relation_anchor_mode=relation_anchor_mode,
        seed=41,
    )


def router_config(num_layers: int = 1) -> RelationExpertRouterConfig:
    return RelationExpertRouterConfig(
        num_layers=num_layers,
        num_heads=2,
        head_dim=4,
        num_observables=6,
        num_experts=4,
        router_qubits=2,
        depth=2,
        initial_gain=0.05,
        seed=73,
    )


def standalone_router_config(
    num_layers: int = 1,
    *,
    routing_conditioning: str = "relation",
    direction_mode: str = "fixed",
    trainable_projection: bool = False,
) -> RelationExpertRouterConfig:
    return RelationExpertRouterConfig(
        num_layers=num_layers,
        num_heads=2,
        head_dim=4,
        num_observables=6,
        num_experts=4,
        router_qubits=2,
        depth=2,
        initial_gain=0.05,
        residual_reference="baseline",
        normalize_routed_energy=True,
        routing_conditioning=routing_conditioning,
        trainable_projection=trainable_projection,
        direction_mode=direction_mode,
        seed=73,
    )


def test_quantum_kernel_is_centered_masked_and_functionally_headwise() -> None:
    torch.manual_seed(7)
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    attention, subject, object_ = relation_masks(3, 5, padded=True)
    quantum = QuantumRelationAttentionScoreKernel(kernel_config())
    classical = ClassicalRelationAttentionScoreKernel(kernel_config())

    centered = quantum.centered_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    assert centered.shape == (3, 2, 5, 5)
    assert torch.isfinite(centered).all()
    assert torch.equal(centered[:, :, -1], torch.zeros_like(centered[:, :, -1]))
    assert torch.equal(centered[:, :, :, -1], torch.zeros_like(centered[:, :, :, -1]))
    assert torch.allclose(centered[:, :, :-1, :-1].sum(dim=-1), torch.zeros(3, 2, 4), atol=1e-6)
    flat = centered.transpose(0, 1).reshape(2, -1)
    assert F.cosine_similarity(flat[0], flat[1], dim=0).abs() < 0.99
    assert sum(p.numel() for p in quantum.parameters()) == sum(
        p.numel() for p in classical.parameters()
    )


def test_score_residual_has_exact_query_aligned_key_steering_realization() -> None:
    torch.manual_seed(11)
    query = torch.randn(2, 3, 4, 5)
    residual = torch.randn(2, 3, 4, 6)

    delta = score_residual_to_query_aligned_key_delta(query, residual)
    recovered = torch.einsum("bhid,bhijd->bhij", query, delta) / math.sqrt(query.shape[-1])

    assert delta.shape == (2, 3, 4, 6, 5)
    assert torch.allclose(recovered, residual, atol=1e-5)


def test_interference_readout_is_signed_centered_and_checkpointable(tmp_path) -> None:
    torch.manual_seed(13)
    query = torch.randn(2, 2, 4, 4)
    key = torch.randn(2, 2, 4, 4)
    attention, subject, object_ = relation_masks(2, 4)
    fidelity = QuantumRelationAttentionScoreKernel(kernel_config())
    interference = QuantumRelationAttentionScoreKernel(
        kernel_config(score_readout="interference")
    )
    interference.load_state_dict(fidelity.state_dict())

    fidelity_scores = fidelity.centered_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    interference_scores = interference.centered_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    checkpoint = tmp_path / "interference.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, interference)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)

    assert not torch.allclose(interference_scores, fidelity_scores)
    assert torch.allclose(interference_scores.sum(dim=-1), torch.zeros(2, 2, 4), atol=1e-6)
    assert (interference_scores < 0).any() and (interference_scores > 0).any()
    assert restored.config.score_readout == "interference"


def test_factorized_encoding_uses_shared_local_coordinates() -> None:
    torch.manual_seed(19)
    query = torch.randn(2, 2, 4, 4)
    key = torch.randn(2, 2, 4, 4)
    attention, subject, object_ = relation_masks(2, 4)
    config = kernel_config(input_encoding="factorized_shared")
    quantum = QuantumRelationAttentionScoreKernel(config)
    classical = ClassicalRelationAttentionScoreKernel(config)

    centered = quantum.centered_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    assert torch.equal(quantum.query_projections, quantum.key_projections)
    assert quantum.relation_projections.shape == (2, 16, 3)
    assert torch.isfinite(centered).all()
    assert torch.allclose(centered.sum(dim=-1), torch.zeros(2, 2, 4), atol=1e-6)
    assert sum(p.numel() for p in quantum.parameters()) == sum(
        p.numel() for p in classical.parameters()
    )


def test_entity_query_scope_only_changes_relation_anchor_rows() -> None:
    torch.manual_seed(21)
    query = torch.randn(2, 2, 5, 4)
    key = torch.randn(2, 2, 5, 4)
    attention, subject, object_ = relation_masks(2, 5)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(input_encoding="factorized_shared", query_scope="entities")
    )

    residual = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    active_queries = (subject | object_)[:, None, :, None].expand_as(residual)
    assert torch.equal(
        residual.masked_select(~active_queries),
        torch.zeros_like(residual.masked_select(~active_queries)),
    )
    assert residual.masked_select(active_queries).abs().max() > 0.0


def test_global_context_kernel_anchor_ignores_entity_masks() -> None:
    torch.manual_seed(22)
    query = torch.randn(2, 2, 5, 4)
    key = torch.randn(2, 2, 5, 4)
    attention, subject, object_ = relation_masks(2, 5)
    alternate_subject = subject.roll(shifts=1, dims=-1)
    alternate_object = object_.roll(shifts=-1, dims=-1)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(relation_anchor_mode="global_context")
    )

    original = kernel.centered_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    alternate = kernel.centered_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=alternate_subject,
        object_mask=alternate_object,
    )

    torch.testing.assert_close(original, alternate)


def test_global_context_requires_all_query_scope() -> None:
    with pytest.raises(
        ValueError,
        match="label-free global_context action requires query_scope='all'",
    ):
        kernel_config(
            relation_anchor_mode="global_context",
            query_scope="entities",
        )


def test_soft_role_pair_is_label_free_mask_invariant_and_noncollapsed() -> None:
    torch.manual_seed(91)
    query = torch.randn(2, 2, 6, 4)
    key = torch.randn(2, 2, 6, 4)
    attention, subject, object_ = relation_masks(2, 6, padded=True)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(relation_anchor_mode="soft_role_pair", input_encoding="factorized_shared")
    )
    residual = kernel(
        query, key, layer_index=0, attention_mask=attention,
        subject_mask=subject, object_mask=object_,
    )
    swapped = kernel(
        query, key, layer_index=0, attention_mask=attention,
        subject_mask=object_, object_mask=subject,
    )
    torch.testing.assert_close(residual, swapped)
    diagnostics = kernel.relation_role_diagnostics(key, attention)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(diagnostics["weights"]).all()
    assert diagnostics["effective_tokens"].min() > 1.0
    assert diagnostics["normalized_entropy"].min() > 0.0
    assert sum(p.numel() for p in kernel.parameters()) > 0


def test_score_kernel_training_defaults_to_label_free_global_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_relation_attention_score_kernel.py",
            "--model_dir",
            "baseline",
            "--output_dir",
            "output",
        ],
    )

    args = train_relation_attention_score_kernel.parse_args()

    assert args.relation_anchor_mode == "global_context"
    assert args.query_scope == "all"
    train_relation_attention_score_kernel.validate_args(args)


def test_score_kernel_training_rejects_span_query_action_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_relation_attention_score_kernel.py",
            "--model_dir",
            "baseline",
            "--output_dir",
            "output",
            "--query_scope",
            "entities",
        ],
    )
    monkeypatch.setattr(
        train_relation_attention_score_kernel,
        "load_relation_run",
        lambda *args, **kwargs: pytest.fail("baseline loading must not run"),
    )

    with pytest.raises(
        ValueError,
        match="label-free global_context action requires query_scope='all'",
    ):
        train_relation_attention_score_kernel.main()


def test_observable_bank_is_parameter_matched_and_supports_diversity_loss(tmp_path) -> None:
    torch.manual_seed(27)
    query = torch.randn(4, 2, 5, 4)
    key = torch.randn(4, 2, 5, 4)
    attention, subject, object_ = relation_masks(4, 5, padded=True)
    config = kernel_config(
        score_readout="observable",
        input_encoding="factorized_shared",
    )
    quantum = QuantumRelationAttentionScoreKernel(config)
    classical = ClassicalRelationAttentionScoreKernel(config)

    with quantum.capture_centered_kernels():
        residual = quantum(
            query,
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        diversity = quantum.functional_diversity_loss()
        (residual.square().mean() + 0.1 * diversity).backward()
    classical_residual = classical(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    checkpoint = tmp_path / "observable.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, quantum)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_residual = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    weights = quantum.observable_weights(0)
    assert weights is not None
    assert weights.shape == (2, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert not torch.equal(weights[0], weights[1])
    assert diversity >= 0.0
    assert quantum.observable_logits is not None
    assert quantum.observable_logits.grad is not None
    assert quantum.observable_logits.grad.abs().max() > 0.0
    assert classical_residual.shape == residual.shape
    assert torch.isfinite(classical_residual).all()
    assert torch.equal(restored_residual, residual.detach())
    assert sum(p.numel() for p in quantum.parameters()) == sum(
        p.numel() for p in classical.parameters()
    )


def test_score_adapter_is_observational_at_zero_gain_and_restores_model() -> None:
    torch.manual_seed(17)
    config = RelationTransformerConfig(
        vocab_size=24,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    kernel = QuantumRelationAttentionScoreKernel(kernel_config())
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    input_ids = torch.randint(0, config.vocab_size, (4, 5))
    attention, subject, object_ = relation_masks(4, 5)
    hook_config = AttentionScoreHookConfig(attention, subject, object_)

    baseline = model(input_ids, attention, subject, object_)
    with adapter.steering(hook_config):
        steered = model(input_ids, attention, subject, object_)
    restored = model(input_ids, attention, subject, object_)
    with torch.no_grad():
        kernel.raw_gains.zero_()
    with adapter.steering(hook_config):
        zero_gain = model(input_ids, attention, subject, object_)

    assert not adapter.attached
    assert not torch.allclose(steered, baseline)
    assert torch.equal(restored, baseline)
    assert torch.equal(zero_gain, baseline)


def test_task_loss_reaches_every_quantum_parameter() -> None:
    torch.manual_seed(23)
    config = RelationTransformerConfig(
        vocab_size=20,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel = QuantumRelationAttentionScoreKernel(kernel_config())
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    input_ids = torch.randint(0, config.vocab_size, (6, 5))
    attention, subject, object_ = relation_masks(6, 5)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    hook_config = AttentionScoreHookConfig(attention, subject, object_)

    with adapter.steering(hook_config):
        logits = model(input_ids, attention, subject, object_)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    gradients = {name: parameter.grad for name, parameter in kernel.named_parameters()}
    assert gradients
    assert all(gradient is not None for gradient in gradients.values())
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
    assert all(float(gradient.abs().max().item()) > 1e-10 for gradient in gradients.values())


def test_score_kernel_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(29)
    query = torch.randn(2, 2, 4, 4)
    key = torch.randn(2, 2, 4, 4)
    attention, subject, object_ = relation_masks(2, 4)
    kernel = QuantumRelationAttentionScoreKernel(kernel_config())
    expected = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    checkpoint = tmp_path / "score_kernel.pt"

    save_relation_attention_score_kernel_checkpoint(
        checkpoint,
        kernel,
        extra_metadata={"base_model": "frozen"},
    )
    restored, metadata = load_relation_attention_score_kernel_checkpoint(checkpoint)
    actual = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    assert restored.kernel_type == "quantum"
    assert metadata == {"base_model": "frozen"}
    assert torch.equal(actual, expected)


def test_score_kernel_diagnostics_measure_attention_intervention() -> None:
    torch.manual_seed(31)
    config = RelationTransformerConfig(
        vocab_size=24,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    kernel = QuantumRelationAttentionScoreKernel(kernel_config())
    attention, subject, object_ = relation_masks(4, 5, padded=True)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (4, 5)),
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
        "labels": torch.tensor([0, 1, 2, 0]),
    }

    diagnostics = diagnose_relation_attention_score_kernel(
        model,
        [batch],
        torch.device("cpu"),
        score_module_paths=model.score_module_paths,
        score_kernel=kernel,
    )

    layer = diagnostics["layers"][0]
    assert diagnostics["num_batches"] == 1
    assert layer["base_scores"]["rms"] > 0.0
    assert layer["centered_kernel"]["std"] > 0.0
    assert layer["score_residual"]["rms"] > 0.0
    assert layer["residual_to_base_rms_ratio"] > 0.0
    assert layer["attention_total_variation"]["mean"] > 0.0
    assert layer["cross_head_centered_kernel"]["cosine"]["count"] == 4


def test_selection_score_and_gradient_tracker_are_deterministic() -> None:
    lower_loss = {"macro_f1": 0.4, "loss": 0.8}
    higher_f1 = {"macro_f1": 0.5, "loss": 1.2}
    assert relation_selection_score(higher_f1, "macro_f1_then_loss") > relation_selection_score(
        lower_loss, "macro_f1_then_loss"
    )
    assert relation_selection_score(lower_loss, "valid_loss") > relation_selection_score(
        higher_f1, "valid_loss"
    )

    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    tracker = GradientNormTracker([("weight", parameter)])
    parameter.square().sum().backward()
    tracker.update()
    summary = tracker.summary()

    assert summary["optimizer_steps"] == 1
    assert summary["parameters"]["weight"]["steps_with_gradient"] == 1
    assert summary["parameters"]["weight"]["l2_norm"]["mean"] > 0.0


def test_task_alignment_diagnostics_compare_residual_with_loss_gradient() -> None:
    torch.manual_seed(37)
    config = RelationTransformerConfig(
        vocab_size=24,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(input_encoding="factorized_shared")
    )
    attention, subject, object_ = relation_masks(4, 5, padded=True)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (4, 5)),
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
        "labels": torch.tensor([0, 1, 2, 0]),
    }

    diagnostics = diagnose_relation_attention_score_task_alignment(
        model,
        [batch],
        torch.device("cpu"),
        score_module_paths=model.score_module_paths,
        score_kernel=kernel,
    )

    assert diagnostics["num_batches"] == 1
    assert diagnostics["actual_loss_change"]["count"] == 4
    assert diagnostics["first_order_loss_change"]["count"] == 1
    assert diagnostics["layers"][0]["residual_descent_cosine"]["count"] == 4
    assert diagnostics["layers"][0]["task_gradient"]["rms"] > 0.0


def test_evidence_selectors_are_bounded_parameter_matched_and_size_matched() -> None:
    torch.manual_seed(43)
    key = torch.randn(3, 2, 6, 4)
    attention, subject, object_ = relation_masks(3, 6, padded=True)
    quantum = QuantumRelationEvidenceSelector(evidence_config())
    classical = ClassicalRelationEvidenceSelector(evidence_config())
    joint_quantum = QuantumRelationEvidenceSelector(
        evidence_config(evidence_readout="joint_observable")
    )
    joint_classical = ClassicalRelationEvidenceSelector(
        evidence_config(evidence_readout="joint_observable")
    )

    scores = quantum.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    shuffled = quantum.permuted_context_scores(
        scores,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        seed=47,
    )
    context = attention & ~(subject | object_)

    assert scores.shape == (3, 2, 6)
    assert torch.equal(scores[:, :, -1], torch.zeros_like(scores[:, :, -1]))
    assert scores.masked_select(attention[:, None, :]).min() > 0.0
    assert scores.max() < 1.0
    for batch_index in range(scores.shape[0]):
        indices = context[batch_index]
        for head_index in range(scores.shape[1]):
            expected = scores[batch_index, head_index, indices].sort().values
            actual = shuffled[batch_index, head_index, indices].sort().values
            assert torch.equal(actual, expected)
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in classical.parameters()
    )
    assert sum(parameter.numel() for parameter in joint_quantum.parameters()) == sum(
        parameter.numel() for parameter in joint_classical.parameters()
    )
    assert sum(parameter.numel() for parameter in quantum.parameters()) > sum(
        parameter.numel() for parameter in joint_quantum.parameters()
    )


def test_global_context_evidence_selector_ignores_entity_masks() -> None:
    torch.manual_seed(44)
    key = torch.randn(2, 2, 6, 4)
    attention, subject, object_ = relation_masks(2, 6)
    alternate_subject = subject.roll(shifts=2, dims=-1)
    alternate_object = object_.roll(shifts=-2, dims=-1)
    selector = QuantumRelationEvidenceSelector(
        evidence_config(relation_anchor_mode="global_context")
    )

    original = selector.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    alternate = selector.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=alternate_subject,
        object_mask=alternate_object,
    )

    torch.testing.assert_close(original, alternate)


def test_evidence_selector_composes_with_kernel_views_and_checkpoint(tmp_path) -> None:
    torch.manual_seed(53)
    query = torch.randn(2, 2, 6, 4)
    key = torch.randn(2, 2, 6, 4)
    attention, subject, object_ = relation_masks(2, 6, padded=True)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(
            score_readout="observable",
            input_encoding="factorized_shared",
        )
    )
    kernel.attach_evidence_selector(QuantumRelationEvidenceSelector(evidence_config()))

    full = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    keep = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        evidence_view="keep",
    )
    drop = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        evidence_view="drop",
    )
    entity_columns = (subject | object_)[:, None, None, :].expand_as(full)
    checkpoint = tmp_path / "evidence_kernel.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_full = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    assert keep.shape == full.shape == drop.shape
    assert torch.isfinite(keep).all() and torch.isfinite(drop).all()
    assert torch.equal((keep - full).masked_select(entity_columns), torch.zeros_like(
        (keep - full).masked_select(entity_columns)
    ))
    assert not torch.allclose(keep, drop)
    assert restored.evidence_selector is not None
    assert restored.evidence_selector.selector_type == "quantum"
    assert torch.equal(restored_full, full)


def test_counterfactual_objective_reaches_selector_and_exports_diagnostics() -> None:
    torch.manual_seed(59)
    config = RelationTransformerConfig(
        vocab_size=24,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(
            score_readout="observable",
            input_encoding="factorized_shared",
        )
    )
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    selector = QuantumRelationEvidenceSelector(evidence_config())
    kernel.attach_evidence_selector(selector)
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    attention, subject, object_ = relation_masks(6, 6, padded=True)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (6, 6)),
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
        "labels": torch.tensor([0, 1, 2, 0, 1, 2]),
    }

    objective, components = counterfactual_evidence_objective(
        model,
        batch,
        adapter,
        counterfactual_weight=1.0,
        keep_weight=1.0,
        drop_weight=1.0,
        budget_weight=0.2,
        evidence_budget=0.35,
        rank_margin=0.01,
        random_seed=61,
    )
    objective.backward()
    selector.zero_grad(set_to_none=True)
    paired_objective, paired_components = counterfactual_evidence_objective(
        model,
        batch,
        adapter,
        counterfactual_weight=1.0,
        keep_weight=1.0,
        drop_weight=1.0,
        budget_weight=0.2,
        evidence_budget=0.35,
        rank_margin=0.01,
        random_seed=61,
        objective_mode="paired_contrast",
        task_alignment_weight=0.2,
    )
    paired_objective.backward()
    selector.zero_grad(set_to_none=True)
    hinge_objective, hinge_components = counterfactual_evidence_objective(
        model,
        batch,
        adapter,
        counterfactual_weight=1.0,
        keep_weight=1.0,
        drop_weight=1.0,
        budget_weight=0.2,
        evidence_budget=0.35,
        rank_margin=0.01,
        random_seed=61,
        objective_mode="paired_hinge",
        task_alignment_weight=0.2,
    )
    hinge_objective.backward()
    diagnostics = diagnose_relation_counterfactual_evidence(
        model,
        [batch],
        torch.device("cpu"),
        adapter=adapter,
        random_repeats=2,
        random_seed=67,
    )
    strict_diagnostics = diagnose_relation_counterfactual_evidence(
        model,
        [batch],
        torch.device("cpu"),
        adapter=adapter,
        random_repeats=2,
        random_seed=67,
        minimum_advantage=1.0,
    )
    alignment = diagnose_relation_evidence_task_alignment(
        model,
        [batch],
        torch.device("cpu"),
        adapter=adapter,
    )

    assert torch.isfinite(objective)
    assert torch.isfinite(paired_objective)
    assert torch.isfinite(hinge_objective)
    assert all(torch.isfinite(value) for value in components.values())
    assert all(torch.isfinite(value) for value in paired_components.values())
    assert all(torch.isfinite(value) for value in hinge_components.values())
    assert hinge_components["keep_rank_loss"] >= 0.0
    assert hinge_components["drop_rank_loss"] >= 0.0
    assert paired_components["task_alignment_loss"].requires_grad
    assert -1.0 <= paired_components["task_alignment"].item() <= 1.0
    gradients = [parameter.grad for parameter in selector.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert diagnostics["num_batches"] == 1
    assert diagnostics["minimum_advantage"] == 1e-6
    assert isinstance(diagnostics["selectivity_pass"], bool)
    assert not strict_diagnostics["selectivity_pass"]
    assert diagnostics["metrics"]["keep_advantage"]["count"] == 6
    assert diagnostics["metrics"]["drop_advantage"]["count"] == 6
    assert diagnostics["layers"][0]["context_evidence"]["std"] > 0.0
    assert diagnostics["metrics"]["context_steering"]["count"] == diagnostics[
        "metrics"
    ]["context_evidence"]["count"]
    assert diagnostics["metrics"]["steering_sufficiency_delta"]["max"] == 0.0
    assert diagnostics["metrics"]["steering_sufficiency_cosine"]["mean"] == pytest.approx(
        1.0
    )
    assert diagnostics["layers"][0]["sufficiency_observable_weights"] is None
    assert diagnostics["layers"][0]["sufficiency_sharpness"] is None
    assert alignment["layers"][0]["evidence_gradient"]["count"] > 0


def test_relation_frame_bank_exports_frame_contribution_gradients() -> None:
    torch.manual_seed(69)
    config = RelationTransformerConfig(
        vocab_size=24,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(
            score_readout="observable",
            input_encoding="factorized_shared",
        )
    )
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            depth=2,
            evidence_readout="connected_relation_token",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="relation_frame_bank",
            intervention_mode="direct_bias",
            seed=70,
        )
    )
    kernel.attach_evidence_selector(selector)
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    attention, subject, object_ = relation_masks(6, 6, padded=True)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (6, 6)),
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
        "labels": torch.tensor([0, 1, 2, 0, 1, 2]),
    }

    diagnostics = diagnose_relation_evidence_measurement_frames(
        model,
        [batch],
        torch.device("cpu"),
        adapter=adapter,
    )

    assert diagnostics["num_batches"] == 1
    assert diagnostics["measurement_frame_view"] == "full"
    assert len(diagnostics["layers"]) == 1
    assert len(diagnostics["layers"][0]["heads"]) == 2
    for head in diagnostics["layers"][0]["heads"]:
        assert set(head["frames"]) == {"z", "x"}
        for frame in head["frames"].values():
            assert frame["contribution"]["count"] == 18
            assert frame["absolute_contribution"]["mean"] > 0.0
            assert frame["absolute_task_gradient"]["mean"] > 0.0
            assert frame["task_gradient_l2"]["count"] == 1
            assert frame["task_descent_effect"]["count"] == 18


def test_expert_router_is_balanced_zero_mean_and_parameter_matched() -> None:
    torch.manual_seed(71)
    relation = torch.randn(5, 16)
    quantum = QuantumRelationObservableExpertRouter(router_config())
    classical = ClassicalRelationObservableExpertRouter(router_config())
    query_quantum = QuantumRelationObservableExpertRouter(
        standalone_router_config(routing_conditioning="query")
    )
    query_classical = ClassicalRelationObservableExpertRouter(
        standalone_router_config(routing_conditioning="query")
    )
    probabilities = quantum.head_probabilities(
        relation,
        layer_index=0,
        head_index=0,
        routing_mode="learned",
    )
    uniform = quantum.head_probabilities(
        relation,
        layer_index=0,
        head_index=0,
        routing_mode="uniform",
    )
    codes = quantum.expert_codes[0]

    assert probabilities.shape == (5, 4)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.equal(uniform, torch.full_like(uniform, 0.25))
    assert torch.allclose(codes.sum(dim=0), torch.zeros(6), atol=1e-7)
    normalized = F.normalize(codes, p=2, dim=-1)
    cosine = torch.matmul(normalized, normalized.transpose(0, 1))
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    assert torch.allclose(
        cosine.masked_select(off_diagonal).abs(),
        torch.full((12,), 1.0 / 3.0),
        atol=1e-6,
    )
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in classical.parameters()
    )
    assert sum(parameter.numel() for parameter in query_quantum.parameters()) == sum(
        parameter.numel() for parameter in query_classical.parameters()
    )


def test_query_conditioned_router_varies_by_token_and_preserves_uniformity() -> None:
    torch.manual_seed(73)
    relation = torch.randn(3, 16)
    query_context = torch.randn(3, 5, 20)
    router = QuantumRelationObservableExpertRouter(
        standalone_router_config(routing_conditioning="query")
    )

    learned = router.head_probabilities(
        relation,
        layer_index=0,
        head_index=0,
        routing_mode="learned",
        query_context=query_context,
    )
    uniform = router.head_probabilities(
        relation,
        layer_index=0,
        head_index=0,
        routing_mode="uniform",
        query_context=query_context,
    )

    assert learned.shape == (3, 5, 4)
    assert torch.allclose(learned.sum(dim=-1), torch.ones(3, 5), atol=1e-6)
    assert not torch.allclose(learned[:, 0], learned[:, 1])
    assert torch.equal(uniform, torch.full_like(uniform, 0.25))


def test_task_aligned_observable_directions_are_centered_and_trainable() -> None:
    torch.manual_seed(77)
    config = standalone_router_config(direction_mode="task_aligned")
    quantum = QuantumRelationObservableExpertRouter(config)
    classical = ClassicalRelationObservableExpertRouter(config)
    components = torch.randn(3, 5, 5, config.num_observables)
    relation = torch.randn(3, 4 * config.head_dim)
    base_weights = torch.softmax(torch.randn(config.num_observables), dim=-1)

    codes = quantum.direction_codes(0, 0)
    uniform = quantum.route_components(
        components,
        base_weights,
        relation,
        layer_index=0,
        head_index=0,
        routing_mode="uniform",
    )
    learned = quantum.route_components(
        components,
        base_weights,
        relation,
        layer_index=0,
        head_index=0,
        routing_mode="learned",
    )
    objective = learned.square().mean() + 0.1 * quantum.direction_diversity_loss()
    objective.backward()

    assert torch.allclose(codes.sum(dim=0), torch.zeros_like(codes[0]), atol=1e-6)
    assert torch.allclose(codes.norm(), torch.tensor(2.0), atol=1e-6)
    assert torch.equal(uniform, torch.zeros_like(uniform))
    assert quantum.expert_direction_parameters is not None
    assert quantum.expert_direction_parameters.grad is not None
    assert torch.isfinite(quantum.expert_direction_parameters.grad).all()
    assert quantum.expert_direction_parameters.grad.abs().sum() > 0.0
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in classical.parameters()
    )


def test_measurement_aligned_quantum_directions_are_trainable_and_identifiable(
    tmp_path,
) -> None:
    torch.manual_seed(78)
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    attention, subject, object_ = relation_masks(3, 5, padded=True)
    quantum_kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )
    classical_kernel = ClassicalRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )
    router_config_ = standalone_router_config(
        routing_conditioning="query",
        direction_mode="measurement_aligned",
    )
    quantum_router = QuantumRelationObservableExpertRouter(router_config_)
    classical_router = ClassicalRelationObservableExpertRouter(router_config_)
    quantum_kernel.attach_expert_router(quantum_router)
    classical_kernel.attach_expert_router(classical_router)

    uniform = quantum_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="uniform",
    )
    learned = quantum_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )
    objective = learned.square().mean() + 0.1 * quantum_router.direction_diversity_loss()
    objective.backward()

    assert torch.equal(uniform, torch.zeros_like(uniform))
    assert not torch.equal(learned, uniform)
    assert quantum_router.expert_measurement_angles is not None
    assert quantum_router.expert_measurement_angles.grad is not None
    assert torch.isfinite(quantum_router.expert_measurement_angles.grad).all()
    assert quantum_router.expert_measurement_angles.grad.abs().sum() > 0.0
    assert sum(parameter.numel() for parameter in quantum_router.parameters()) == sum(
        parameter.numel() for parameter in classical_router.parameters()
    )
    checkpoint = tmp_path / "measurement_aligned_router.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, quantum_kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_learned = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )

    assert restored.expert_router is not None
    assert restored.expert_router.config.direction_mode == "measurement_aligned"
    assert torch.equal(restored_learned, learned)


def test_quantum_measurement_features_match_explicit_pauli_matrices() -> None:
    torch.manual_seed(79)
    num_qubits = 3
    states = F.normalize(torch.randn(2, 3, 2**num_qubits), p=2, dim=-1)
    angles = torch.randn(4, num_qubits)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )

    actual = kernel._measurement_observable_features(states, angles)
    basis = torch.arange(2**num_qubits)
    expected_experts = []
    for expert_angles in angles:
        local_operators = []
        for qubit, angle in enumerate(expert_angles):
            mask = 1 << qubit
            operator = torch.zeros(2**num_qubits, 2**num_qubits)
            operator[basis, basis] = torch.cos(angle) * (
                1.0 - 2.0 * basis.bitwise_right_shift(qubit).bitwise_and(1)
            )
            operator[basis, basis.bitwise_xor(mask)] = torch.sin(angle)
            local_operators.append(operator)
        observables = local_operators + [
            local_operators[qubit]
            @ local_operators[(qubit + 1) % num_qubits]
            for qubit in range(num_qubits)
        ]
        expected_experts.append(
            torch.stack(
                [
                    torch.einsum("...i,ij,...j->...", states, observable, states)
                    for observable in observables
                ],
                dim=-1,
            )
        )
    expected = torch.stack(expected_experts, dim=-2)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_connected_correlations_separate_entangled_and_product_states() -> None:
    num_qubits = 3
    angles = torch.zeros(1, num_qubits)
    quantum = QuantumRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )
    classical = ClassicalRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )
    ghz = torch.zeros(1, 1, 2**num_qubits)
    ghz[..., 0] = 1.0 / math.sqrt(2.0)
    ghz[..., -1] = 1.0 / math.sqrt(2.0)
    product = torch.zeros_like(ghz)
    product[..., 0] = 1.0
    separable = F.normalize(torch.randn(1, 1, 2 * num_qubits), p=2, dim=-1)

    ghz_connected = quantum._connected_correlation_features(
        quantum._measurement_observable_features(ghz, angles)
    )
    product_connected = quantum._connected_correlation_features(
        quantum._measurement_observable_features(product, angles)
    )
    separable_connected = classical._connected_correlation_features(
        classical._measurement_observable_features(separable, angles)
    )

    torch.testing.assert_close(ghz_connected, torch.ones_like(ghz_connected))
    torch.testing.assert_close(
        product_connected,
        torch.zeros_like(product_connected),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        separable_connected,
        torch.zeros_like(separable_connected),
        atol=1e-7,
        rtol=0.0,
    )


def test_connected_quantum_directions_are_standalone_and_trainable(tmp_path) -> None:
    torch.manual_seed(80)
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    attention, subject, object_ = relation_masks(3, 5, padded=True)
    quantum_kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )
    separable_kernel = ClassicalRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )
    connected_config = standalone_router_config(
        routing_conditioning="query_expert",
        direction_mode="connected_aligned",
        trainable_projection=True,
    )
    matched_config = standalone_router_config(
        routing_conditioning="query_expert",
        direction_mode="measurement_aligned",
        trainable_projection=True,
    )
    quantum_router = QuantumRelationObservableExpertRouter(connected_config)
    separable_router = ClassicalRelationObservableExpertRouter(connected_config)
    matched_router = ClassicalRelationObservableExpertRouter(matched_config)
    quantum_kernel.attach_expert_router(quantum_router)
    separable_kernel.attach_expert_router(separable_router)

    uniform = quantum_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="uniform",
    )
    learned = quantum_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )
    separable = separable_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )
    separable_objective = separable.square().mean()
    separable_objective.backward()
    objective = learned.square().mean() + 0.1 * quantum_router.direction_diversity_loss()
    objective.backward()

    assert torch.equal(uniform, torch.zeros_like(uniform))
    assert not torch.equal(learned, uniform)
    torch.testing.assert_close(separable, torch.zeros_like(separable), atol=1e-7, rtol=0.0)
    for parameter in separable_router.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert quantum_router.expert_measurement_angles is not None
    assert quantum_router.expert_measurement_angles.grad is not None
    assert torch.isfinite(quantum_router.expert_measurement_angles.grad).all()
    assert quantum_router.expert_measurement_angles.grad.abs().sum() > 0.0
    assert isinstance(quantum_router.relation_projections, torch.nn.Parameter)
    assert quantum_router.relation_projections.grad is not None
    assert torch.isfinite(quantum_router.relation_projections.grad).all()
    assert sum(parameter.numel() for parameter in quantum_router.parameters()) == sum(
        parameter.numel() for parameter in separable_router.parameters()
    )
    assert sum(parameter.numel() for parameter in quantum_router.parameters()) == sum(
        parameter.numel() for parameter in matched_router.parameters()
    )

    checkpoint = tmp_path / "connected_aligned_router.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, quantum_kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_learned = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )

    assert restored.expert_router is not None
    assert restored.expert_router.config.direction_mode == "connected_aligned"
    assert torch.equal(restored_learned, learned)


def test_continuous_connected_readout_is_resource_selective_and_matched(
    tmp_path,
) -> None:
    torch.manual_seed(82)
    query = torch.randn(3, 2, 5, 4)
    key = torch.randn(3, 2, 5, 4)
    attention, subject, object_ = relation_masks(3, 5, padded=True)
    connected_config = kernel_config(
        score_readout="continuous_connected",
        input_encoding="factorized_shared",
        normalize_readout_energy=True,
    )
    matched_config = kernel_config(
        score_readout="continuous_measurement",
        input_encoding="factorized_shared",
        normalize_readout_energy=True,
    )
    quantum = QuantumRelationAttentionScoreKernel(connected_config)
    separable = ClassicalRelationAttentionScoreKernel(connected_config)
    matched = ClassicalRelationAttentionScoreKernel(matched_config)

    def residual(kernel):
        return kernel(
            query,
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )

    quantum_residual = residual(quantum)
    separable_residual = residual(separable)
    matched_residual = residual(matched)

    assert quantum.expert_router is None
    assert not torch.equal(quantum_residual, torch.zeros_like(quantum_residual))
    torch.testing.assert_close(
        separable_residual,
        torch.zeros_like(separable_residual),
        atol=1e-7,
        rtol=0.0,
    )
    assert not torch.equal(matched_residual, torch.zeros_like(matched_residual))
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in separable.parameters()
    )
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in matched.parameters()
    )

    valid = (
        attention[:, None, :, None] & attention[:, None, None, :]
    ).to(quantum_residual.dtype)
    count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
    residual_rms = torch.sqrt(
        (quantum_residual.square() * valid).sum(dim=(-1, -2)) / count
    )
    baseline = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    key_mask = attention[:, None, None, :].to(baseline.dtype)
    key_count = key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    baseline = baseline - (baseline * key_mask).sum(dim=-1, keepdim=True) / key_count
    baseline_rms = torch.sqrt(
        (baseline.square() * valid).sum(dim=(-1, -2)) / count
    )
    torch.testing.assert_close(
        residual_rms / baseline_rms,
        quantum.gains(0).abs().unsqueeze(0).expand_as(residual_rms),
        atol=1e-5,
        rtol=1e-5,
    )

    quantum_residual.square().mean().backward()
    separable_residual.square().mean().backward()
    matched_residual.square().mean().backward()
    for kernel in (quantum, separable, matched):
        assert kernel.measurement_angles is not None
        assert kernel.measurement_angles.grad is not None
        assert torch.isfinite(kernel.measurement_angles.grad).all()
    assert quantum.measurement_angles.grad.abs().sum() > 0.0
    assert matched.measurement_angles.grad.abs().sum() > 0.0

    checkpoint = tmp_path / "continuous_connected.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, quantum)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_residual = residual(restored)

    assert restored.config.score_readout == "continuous_connected"
    assert restored.expert_router is None
    assert torch.equal(restored_residual, quantum_residual)


def test_continuous_connected_bank_is_directly_optimized_and_matched(tmp_path) -> None:
    torch.manual_seed(84)
    query = torch.randn(2, 2, 5, 4)
    key = torch.randn(2, 2, 5, 4)
    attention, subject, object_ = relation_masks(2, 5, padded=True)
    quantum = QuantumRelationAttentionScoreKernel(
        kernel_config(
            score_readout="continuous_connected_bank",
            input_encoding="factorized_shared",
            normalize_readout_energy=True,
        )
    )
    separable = ClassicalRelationAttentionScoreKernel(
        kernel_config(
            score_readout="continuous_connected_bank",
            input_encoding="factorized_shared",
            normalize_readout_energy=True,
        )
    )
    matched = ClassicalRelationAttentionScoreKernel(
        kernel_config(
            score_readout="continuous_measurement_bank",
            input_encoding="factorized_shared",
            normalize_readout_energy=True,
        )
    )

    def residual(kernel):
        return kernel(
            query,
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )

    quantum_residual = residual(quantum)
    separable_residual = residual(separable)
    matched_residual = residual(matched)
    assert quantum.measurement_angles is not None
    assert quantum.measurement_angles.ndim == 4
    assert quantum.readout_logits is not None
    assert quantum.readout_logits.shape[-1] == 4
    assert not torch.equal(quantum_residual, torch.zeros_like(quantum_residual))
    torch.testing.assert_close(
        separable_residual,
        torch.zeros_like(separable_residual),
        atol=1e-7,
        rtol=0.0,
    )
    assert not torch.equal(matched_residual, torch.zeros_like(matched_residual))
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in separable.parameters()
    )
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in matched.parameters()
    )

    quantum_residual.square().mean().backward()
    separable_residual.square().mean().backward()
    matched_residual.square().mean().backward()
    for kernel in (quantum, separable, matched):
        assert kernel.readout_logits is not None
        assert kernel.readout_logits.grad is not None
        assert torch.isfinite(kernel.readout_logits.grad).all()
        assert kernel.measurement_angles.grad is not None
        assert torch.isfinite(kernel.measurement_angles.grad).all()
    assert quantum.readout_logits.grad.abs().sum() > 0.0

    checkpoint = tmp_path / "continuous_connected_bank.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, quantum)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_residual = residual(restored)
    assert restored.config.score_readout == "continuous_connected_bank"
    assert torch.equal(restored_residual, quantum_residual)


def test_legacy_expert_router_checkpoint_defaults_to_fixed(tmp_path) -> None:
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(score_readout="observable", input_encoding="factorized_shared")
    )
    kernel.attach_expert_router(QuantumRelationObservableExpertRouter(router_config()))
    checkpoint = tmp_path / "legacy_fixed_router.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["kernel_metadata"]["expert_router"]["config"]["direction_mode"]
    del payload["kernel_metadata"]["expert_router"]["config"]["trainable_projection"]
    torch.save(payload, checkpoint)

    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)

    assert restored.expert_router is not None
    assert restored.expert_router.config.direction_mode == "fixed"
    assert not restored.expert_router.config.trainable_projection


def test_uniform_routing_restores_core_and_composes_with_evidence_checkpoint(tmp_path) -> None:
    torch.manual_seed(79)
    query = torch.randn(2, 2, 6, 4)
    key = torch.randn(2, 2, 6, 4)
    attention, subject, object_ = relation_masks(2, 6, padded=True)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(
            score_readout="observable",
            input_encoding="factorized_shared",
        )
    )
    core = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    kernel.attach_evidence_selector(QuantumRelationEvidenceSelector(evidence_config()))
    evidence_only = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    kernel.attach_expert_router(QuantumRelationObservableExpertRouter(router_config()))
    uniform = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="uniform",
    )
    learned = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        evidence_view="keep",
        routing_mode="learned",
    )
    checkpoint = tmp_path / "routed_evidence_kernel.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_learned = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        evidence_view="keep",
        routing_mode="learned",
    )

    assert not torch.equal(core, evidence_only)
    assert torch.equal(uniform, evidence_only)
    assert not torch.equal(learned, uniform)
    assert restored.evidence_selector is not None
    assert restored.expert_router is not None
    assert torch.equal(restored_learned, learned)


def test_standalone_routing_is_zero_at_uniform_and_energy_calibrated(tmp_path) -> None:
    torch.manual_seed(81)
    query = torch.randn(3, 2, 6, 4)
    key = torch.randn(3, 2, 6, 4)
    attention, subject, object_ = relation_masks(3, 6, padded=True)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(
            score_readout="observable",
            input_encoding="factorized_shared",
        )
    )
    router = QuantumRelationObservableExpertRouter(
        standalone_router_config(direction_mode="task_aligned")
    )
    kernel.attach_expert_router(router)
    uniform = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="uniform",
    )
    learned = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )
    with torch.no_grad():
        kernel.raw_gains.fill_(-3.0)
    unchanged = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )
    checkpoint = tmp_path / "standalone_router.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    restored_learned = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        routing_mode="learned",
    )

    valid = (
        attention[:, None, :, None] & attention[:, None, None, :]
    ).to(learned.dtype)
    count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
    rms = torch.sqrt((learned.square() * valid).sum(dim=(-1, -2)) / count)
    baseline = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    key_mask = attention[:, None, None, :].to(baseline.dtype)
    key_count = key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    baseline = baseline - (baseline * key_mask).sum(dim=-1, keepdim=True) / key_count
    baseline_rms = torch.sqrt(
        (baseline.square() * valid).sum(dim=(-1, -2)) / count
    )

    assert torch.equal(uniform, torch.zeros_like(uniform))
    assert torch.allclose(
        rms / baseline_rms,
        router.gains(0).abs().unsqueeze(0),
        atol=1e-5,
    )
    assert torch.equal(unchanged, learned)
    assert restored.expert_router is not None
    assert restored.expert_router.config.residual_reference == "baseline"
    assert restored.expert_router.config.normalize_routed_energy
    assert restored.expert_router.config.direction_mode == "task_aligned"
    assert torch.equal(restored_learned, unchanged)


def test_information_loss_penalizes_dead_experts_per_head() -> None:
    router = QuantumRelationObservableExpertRouter(router_config())
    complementary_collapsed = (
        torch.tensor([[0.5, 0.5, 0.0, 0.0]]).expand(8, -1),
        torch.tensor([[0.0, 0.0, 0.5, 0.5]]).expand(8, -1),
    )

    with router.capture_routing():
        router._captured_probabilities.extend(
            (0, head_index, probabilities)
            for head_index, probabilities in enumerate(complementary_collapsed)
        )
        router._captured_probability_masks.extend((None, None))
        information = router.information_components()

    assert information["dead_expert_barrier"] > 1.0
    assert torch.allclose(
        information["mutual_information"],
        torch.zeros_like(information["mutual_information"]),
        atol=1e-7,
    )


def test_routing_objective_reaches_router_and_exports_diagnostics() -> None:
    torch.manual_seed(83)
    config = RelationTransformerConfig(
        vocab_size=24,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel = QuantumRelationAttentionScoreKernel(
        kernel_config(
            score_readout="observable",
            input_encoding="factorized_shared",
        )
    )
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    router = QuantumRelationObservableExpertRouter(
        standalone_router_config(
            routing_conditioning="query_expert",
            trainable_projection=True,
        )
    )
    kernel.attach_expert_router(router)
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    attention, subject, object_ = relation_masks(6, 6, padded=True)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (6, 6)),
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
        "labels": torch.tensor([0, 1, 2, 0, 1, 2]),
    }

    objective, components = expert_routing_objective(
        model,
        batch,
        adapter,
        information_weight=0.1,
        utility_alignment_weight=0.2,
    )
    objective.backward()
    diagnostics = diagnose_relation_expert_routing(
        model,
        [batch],
        torch.device("cpu"),
        adapter=adapter,
    )
    alignment = diagnose_relation_routing_task_alignment(
        model,
        [batch],
        torch.device("cpu"),
        adapter=adapter,
    )
    direction_alignment = diagnose_relation_expert_direction_alignment(
        model,
        [batch],
        torch.device("cpu"),
        adapter=adapter,
    )

    assert torch.isfinite(objective)
    assert all(torch.isfinite(value) for value in components.values())
    assert components["utility_alignment_loss"].requires_grad
    assert -1.0 <= components["utility_alignment"].item() <= 1.0
    assert isinstance(router.relation_projections, torch.nn.Parameter)
    assert router.relation_projections.grad is not None
    gradients = [parameter.grad for parameter in router.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert diagnostics["num_batches"] == 1
    assert isinstance(diagnostics["mechanism_pass"], bool)
    assert len(diagnostics["layers"][0]["usage"]) == 4
    assert len(diagnostics["layers"][0]["heads"]) == 2
    assert diagnostics["layers"][0]["conditional_entropy"]["count"] == 60
    assert all(
        head["conditional_entropy"]["count"] == 30
        for head in diagnostics["layers"][0]["heads"]
    )
    assert diagnostics["layers"][0]["expert_cross_cosine"]["count"] == 72
    assert all(
        head["expert_cross_cosine"]["count"] == 36
        for head in diagnostics["layers"][0]["heads"]
    )
    assert alignment["layers"][0]["routing_gradient"]["count"] > 0
    assert direction_alignment["num_batches"] == 1
    assert direction_alignment["routing_conditioning"] == "query_expert"
    direction_head = direction_alignment["layers"][0]["heads"][0]
    assert len(direction_head["experts"]) == 4
    assert direction_head["query_roles"]["all"]["routed_alignment"]["count"] == 30
    assert direction_head["query_roles"]["subject"]["routed_alignment"]["count"] == 6
    assert direction_head["query_roles"]["object"]["routed_alignment"]["count"] == 6
    assert direction_head["query_roles"]["context"]["routed_alignment"]["count"] == 18
