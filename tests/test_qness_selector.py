import torch
import torch.nn.functional as F

from q_attention.plugins import (
    ClassicalNESSRelationEvidenceSelector,
    QuantumNESSRelationEvidenceSelector,
    QuantumRelationAttentionScoreKernel,
    QuantumRelationEvidenceSelector,
    RelationEvidenceSelectorConfig,
    RelationScoreKernelConfig,
    build_relation_evidence_selector,
    load_relation_attention_score_kernel_checkpoint,
    save_relation_attention_score_kernel_checkpoint,
)


def qness_config(**overrides) -> RelationEvidenceSelectorConfig:
    values = {
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 4,
        "num_qubits": 4,
        "depth": 2,
        "evidence_readout": "connected_relation_token",
        "evidence_task_readout": "dual",
        "evidence_weight_mode": "signed_centered_l1",
        "evidence_measurement_mode": "fixed",
        "evidence_gate_calibration": "none",
        "seed": 101,
    }
    values.update(overrides)
    return RelationEvidenceSelectorConfig(**values)


def relation_masks(batch: int = 2, tokens: int = 6):
    attention = torch.ones(batch, tokens, dtype=torch.bool)
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 0] = True
    object_[:, 1] = True
    return attention, subject, object_


def test_qness_registry_builds_all_controls_with_matched_parameter_counts() -> None:
    selectors = {
        name: build_relation_evidence_selector(name, qness_config())
        for name in (
            "qness",
            "qness_commuting",
            "qness_separable",
            "qness_phase_scrambled",
            "qness_dephased",
            "qness_classical",
        )
    }
    assert isinstance(selectors["qness"], QuantumNESSRelationEvidenceSelector)
    assert isinstance(
        selectors["qness_classical"], ClassicalNESSRelationEvidenceSelector
    )
    assert selectors["qness_commuting"].config.qness_control == "commuting"
    assert selectors["qness_separable"].config.qness_control == "separable"
    assert selectors["qness_phase_scrambled"].config.qness_control == "phase_scrambled"
    assert selectors["qness_dephased"].config.qness_control == "dephased"
    counts = {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in selectors.values()
    }
    assert len(counts) == 1


def test_qness_observables_are_noncommuting_and_controls_remove_resources() -> None:
    torch.manual_seed(103)
    token = F.normalize(torch.randn(12, 4), dim=-1)
    relation = F.normalize(torch.randn(12, 4), dim=-1)
    qness = build_relation_evidence_selector("qness", qness_config())
    commuting = build_relation_evidence_selector("qness_commuting", qness_config())
    separable = build_relation_evidence_selector("qness_separable", qness_config())
    scrambled = build_relation_evidence_selector(
        "qness_phase_scrambled", qness_config()
    )
    dephased = build_relation_evidence_selector("qness_dephased", qness_config())
    for control in (commuting, separable, scrambled, dephased):
        control.load_state_dict(qness.state_dict())

    necessity, sufficiency, resources = qness._qness_observable_features(
        token, relation
    )
    commuting_n, commuting_s, commuting_resources = (
        commuting._qness_observable_features(token, relation)
    )
    separable_n, _separable_s, separable_resources = (
        separable._qness_observable_features(token, relation)
    )
    scrambled_n, scrambled_s, _scrambled_resources = (
        scrambled._qness_observable_features(token, relation)
    )
    _dephased_n, dephased_s, dephased_resources = (
        dephased._qness_observable_features(token, relation)
    )

    assert not torch.allclose(necessity, sufficiency, atol=1e-6)
    torch.testing.assert_close(commuting_n, necessity)
    torch.testing.assert_close(commuting_s, commuting_n)
    torch.testing.assert_close(scrambled_n, necessity)
    assert not torch.allclose(scrambled_s, sufficiency, atol=1e-6)
    torch.testing.assert_close(dephased_s, torch.zeros_like(dephased_s))
    assert torch.all(resources["observable_commutator_norm"] > 0.0)
    assert torch.equal(
        commuting_resources["observable_commutator_norm"],
        torch.zeros_like(commuting_resources["observable_commutator_norm"]),
    )
    assert separable_n.abs().max() < 1e-5
    assert separable_resources["mutual_information"].abs().max() < 1e-5
    assert torch.equal(
        dephased_resources["off_diagonal_density_norm"],
        torch.zeros_like(dephased_resources["off_diagonal_density_norm"]),
    )


def test_qness_views_use_independent_sufficiency_and_necessity_signals() -> None:
    selector = build_relation_evidence_selector("qness", qness_config())
    attention, subject, object_ = relation_masks(batch=1, tokens=6)
    necessity = torch.tensor(
        [[[0.5, 0.5, 0.8, 0.2, 0.7, 0.3], [0.5, 0.5, 0.6, 0.4, 0.9, 0.1]]]
    )
    sufficiency = torch.tensor(
        [[[0.5, 0.5, 0.1, 0.9, 0.4, 0.6], [0.5, 0.5, 0.3, 0.7, 0.2, 0.8]]]
    )
    altered_sufficiency = sufficiency.roll(shifts=1, dims=-1)
    keep = selector.view_weights(
        sufficiency,
        view="keep",
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        steering_scores=necessity,
    )
    altered_keep = selector.view_weights(
        altered_sufficiency,
        view="keep",
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        steering_scores=necessity,
    )
    drop = selector.view_weights(
        sufficiency,
        view="drop",
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        steering_scores=necessity,
    )
    altered_drop = selector.view_weights(
        altered_sufficiency,
        view="drop",
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        steering_scores=necessity,
    )
    context = attention & ~(subject | object_)
    assert not torch.equal(keep.masked_select(context[:, None, :]), altered_keep.masked_select(context[:, None, :]))
    torch.testing.assert_close(drop, altered_drop)
    assert not torch.allclose(
        drop.masked_select(context[:, None, :]),
        1.0 - keep.masked_select(context[:, None, :]),
    )


def test_qness_delta_is_attention_weighted_and_centered() -> None:
    torch.manual_seed(107)
    selector = build_relation_evidence_selector("qness", qness_config())
    base_scores = torch.randn(2, 2, 3, 6)
    necessity = torch.sigmoid(torch.randn(2, 2, 6))
    sufficiency = torch.sigmoid(torch.randn(2, 2, 6))
    attention, _subject, _object = relation_masks(batch=2, tokens=6)
    delta = selector._qness_delta_from_readouts(
        base_scores,
        necessity,
        sufficiency,
        attention,
    )
    weights = torch.softmax(
        base_scores.masked_fill(~attention[:, None, None, :], -torch.inf),
        dim=-1,
    )
    torch.testing.assert_close(
        (weights * delta).sum(dim=-1),
        torch.zeros(2, 2, 3),
        atol=1e-6,
        rtol=1e-6,
    )


def test_qness_kernel_is_trainable_and_checkpointable(tmp_path) -> None:
    torch.manual_seed(109)
    kernel = QuantumRelationAttentionScoreKernel(
        RelationScoreKernelConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            score_readout="interference",
            query_scope="entities",
            seed=109,
        )
    )
    selector = build_relation_evidence_selector("qness", qness_config(seed=109))
    kernel.attach_evidence_selector(selector)
    query = torch.randn(2, 2, 6, 4)
    key = torch.randn(2, 2, 6, 4)
    attention, subject, object_ = relation_masks()
    with selector.capture_token_scores():
        residual = kernel(
            query,
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        resources = selector.captured_quantum_diagnostics()
        necessity = selector.captured_steering_scores()[0]
        sufficiency = selector.captured_token_scores()[0]
        complement_error = (necessity + sufficiency - 1.0).abs()
        assert complement_error[:, :, 2:].mean() > 1e-3
        assert resources
    assert torch.isfinite(residual).all()
    residual.square().mean().backward()
    assert selector.observable_logits.grad is not None
    assert selector.sufficiency_observable_logits is not None
    assert selector.sufficiency_observable_logits.grad is not None
    assert torch.isfinite(selector.observable_logits.grad).all()
    assert torch.isfinite(selector.sufficiency_observable_logits.grad).all()

    checkpoint = tmp_path / "qness.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, metadata = load_relation_attention_score_kernel_checkpoint(checkpoint)
    assert isinstance(restored.evidence_selector, QuantumNESSRelationEvidenceSelector)
    assert restored.metadata()["evidence_selector"]["qness"][
        "non_complementary_readouts"
    ]
    restored_residual = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    torch.testing.assert_close(restored_residual, residual.detach())


def test_legacy_selector_regression_fixture() -> None:
    torch.manual_seed(113)
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            seed=113,
        )
    )
    centered = torch.randn(2, 2, 6, 6)
    evidence = torch.rand(2, 2, 6)
    attention, subject, object_ = relation_masks()
    actual = selector.steering_residual(
        centered,
        evidence,
        evidence,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    key_mask = attention[:, None, None, :].to(centered.dtype)
    expected = centered * (2.0 * evidence[:, :, None, :])
    expected = expected - (
        (expected * key_mask).sum(dim=-1, keepdim=True)
        / key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    )
    expected = expected * attention[:, None, :, None] * key_mask
    torch.testing.assert_close(actual, expected)
