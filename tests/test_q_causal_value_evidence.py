from __future__ import annotations

import pytest
import torch

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_causal_value_evidence import (
    CausalValueTransportConfig,
    build_causal_value_transport_kernel,
)
from experiments.run_q_causal_value_evidence_toy import (
    _summarize_group_metrics,
    build_parameter_efficiency_manifests,
    make_balanced_split,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(29)
    scores = torch.randn(3, 1, 6, 6)
    query = torch.randn(3, 1, 6, 4)
    key = torch.randn(3, 1, 6, 4)
    value = torch.randn(3, 1, 6, 4)
    attention_mask = torch.ones(3, 6, dtype=torch.bool)
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, 0] = True
    object_mask[:, 1] = True
    return scores, query, key, value, attention_mask, subject_mask, object_mask


def _build(kind: str, mode: str = "leave_one_out"):
    return build_causal_value_transport_kernel(
        kind,
        CausalValueTransportConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            register_qubits=2,
            depth=2,
            value_feature_mode=mode,
            seed=41,
        ),
    )


def _run(kernel, inputs):
    return kernel(
        inputs[1],
        inputs[2],
        inputs[3],
        scores=inputs[0],
        layer_index=0,
        attention_mask=inputs[4],
        subject_mask=inputs[5],
        object_mask=inputs[6],
    )


def test_quantum_and_classical_transport_are_parameter_matched() -> None:
    quantum = _build("quantum")
    classical = _build("classical")
    assert sum(p.numel() for p in quantum.parameters()) == sum(
        p.numel() for p in classical.parameters()
    )


def test_transport_fraction_is_nonnegative_and_bounded() -> None:
    kernel = _build("quantum")
    with torch.no_grad():
        kernel.raw_transport.fill_(-100.0)
    near_zero = kernel.transport_fractions(0)
    assert torch.all(near_zero >= 0.0)
    assert torch.all(near_zero <= kernel.config.max_transport)
    with torch.no_grad():
        kernel.raw_transport.fill_(100.0)
    near_max = kernel.transport_fractions(0)
    assert torch.all(near_max >= 0.0)
    assert torch.all(near_max <= kernel.config.max_transport)


def test_signed_initial_transport_is_rejected() -> None:
    with pytest.raises(ValueError, match="initial_transport"):
        CausalValueTransportConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            initial_transport=-0.1,
        )


def test_transport_preserves_mass_for_extreme_evidence() -> None:
    inputs = list(_inputs())
    scores = inputs[0].clone()
    scores[..., 4] = 12.0
    scores[..., 5] = -12.0
    kernel = _build("quantum")
    evidence = torch.zeros_like(scores)
    evidence[..., 4] = 1e-12
    evidence[..., 5] = 1.0
    strength = torch.ones_like(scores)
    residual = kernel._transport_residual(
        torch.softmax(scores, dim=-1),
        evidence,
        strength,
        attention_mask=inputs[4],
        subject_mask=inputs[5],
        object_mask=inputs[6],
        layer_index=0,
    )
    base = torch.softmax(scores, dim=-1)
    steered = torch.softmax(scores + residual, dim=-1)
    context = inputs[4] & ~(inputs[5] | inputs[6])
    context_mask = context[:, None, None, :]
    entity_mask = (inputs[5] | inputs[6])[:, None, None, :]
    assert torch.isfinite(residual).all()
    assert torch.allclose(
        steered.sum(dim=-1), torch.ones_like(steered.sum(dim=-1)), atol=1e-6
    )
    assert torch.allclose(
        (steered * context_mask).sum(dim=-1),
        (base * context_mask).sum(dim=-1),
        atol=1e-6,
    )
    assert torch.allclose(
        steered * entity_mask,
        base * entity_mask,
        atol=1e-6,
    )


def test_transport_preserves_rows_and_entity_attention() -> None:
    inputs = _inputs()
    residual = _run(_build("quantum"), inputs)
    base = torch.softmax(inputs[0], dim=-1)
    steered = torch.softmax(inputs[0] + residual, dim=-1)
    context = inputs[4] & ~(inputs[5] | inputs[6])
    context_mask = context[:, None, None, :]
    entity_mask = (inputs[5] | inputs[6])[:, None, None, :]
    assert torch.allclose(steered.sum(dim=-1), torch.ones_like(steered.sum(dim=-1)), atol=1e-6)
    assert torch.allclose(
        (steered * context_mask).sum(dim=-1),
        (base * context_mask).sum(dim=-1),
        atol=1e-5,
    )
    assert torch.allclose(steered * entity_mask, base * entity_mask, atol=1e-5)


def test_value_change_changes_quantum_transport() -> None:
    inputs = list(_inputs())
    first = _run(_build("quantum"), inputs)
    inputs[3] = inputs[3].clone()
    inputs[3][:, :, 4, 0] += 2.0
    second = _run(_build("quantum"), inputs)
    assert not torch.allclose(first, second)


def test_key_only_ablation_ignores_value_change() -> None:
    inputs = list(_inputs())
    kernel = _build("quantum", mode="key_only")
    first = _run(kernel, inputs)
    inputs[3] = inputs[3].clone()
    inputs[3][:, :, 4, 0] += 2.0
    second = _run(kernel, inputs)
    assert torch.allclose(first, second, atol=1e-6)


def test_transport_requires_pre_softmax_scores_and_value() -> None:
    inputs = _inputs()
    kernel = _build("quantum")
    try:
        kernel(
            inputs[1],
            inputs[2],
            inputs[3],
            layer_index=0,
            attention_mask=inputs[4],
            subject_mask=inputs[5],
            object_mask=inputs[6],
        )
    except ValueError as exc:
        assert "requires" in str(exc)
    else:
        raise AssertionError("missing scores must be rejected")


def test_transport_kernel_runs_through_score_hook_adapter() -> None:
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=24,
            num_labels=3,
            dim=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    ).eval()
    kernel = build_causal_value_transport_kernel(
        "quantum",
        CausalValueTransportConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            register_qubits=2,
            depth=1,
            seed=43,
        ),
    )
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    input_ids = torch.randint(0, 24, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, 0] = True
    object_mask[:, 1] = True
    with adapter.steering(
        AttentionScoreHookConfig(attention_mask, subject_mask, object_mask)
    ):
        logits = model(input_ids, attention_mask, subject_mask, object_mask)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert not adapter.attached


def test_group_metric_summary_reports_counts_and_means() -> None:
    values = {
        "query_accuracy": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "context_target_mass_gain": torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
    }
    groups = {
        "query_2": torch.tensor([[True, False], [True, False]]),
        "target_5": torch.tensor([[False, True], [True, False]]),
    }
    summary = _summarize_group_metrics(values, groups)
    assert summary["query_2"]["count"] == 2
    assert summary["query_2"]["query_accuracy"] == pytest.approx(0.5)
    assert summary["query_2"]["context_target_mass_gain"] == pytest.approx(0.4)
    assert summary["target_5"]["count"] == 2
    assert summary["target_5"]["query_accuracy"] == 0.0


def test_balanced_split_breaks_fixed_role_couplings() -> None:
    split = make_balanced_split(7, 8, torch.device("cpu"))
    assert split["relation_sign"].sum().item() == 0.0
    assert torch.equal(
        split["query_type"].sum(dim=0),
        torch.zeros(2),
    )
    target_positions = set(split["target_key"].reshape(-1).tolist())
    assert target_positions == {4, 5}


def test_qvres_parameter_manifests_are_budget_matched() -> None:
    manifests = build_parameter_efficiency_manifests("test-revision", "balanced")
    by_selector = {item["mechanism"]: item for item in manifests}
    assert by_selector["disabled"]["components"] == {"classifier": 6}
    for selector in (
        "q_causal_transport",
        "classical_causal_transport",
        "q_causal_key_only",
    ):
        assert by_selector[selector]["components"] == {
            "classifier": 6,
            "intervention": 17,
        }
        assert by_selector[selector]["total_trainable_parameters"] == 23
        assert by_selector[selector]["metadata"]["protocol"] == "balanced"
