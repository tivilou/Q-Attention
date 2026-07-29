from __future__ import annotations

import pytest
import torch

from q_attention.plugins import (
    ClassicalKeySteeringPlugin,
    EVIDENCE_CORRELATION_CHANNELS,
    HeadwiseQuantumProjectorConfig,
    QuantumKeySteeringConfig,
    QuantumKeySteeringPlugin,
    QuantumRelationAttentionScoreKernel,
    QuantumRelationEvidenceSelector,
    RelationEvidenceSelectorConfig,
    RelationScoreKernelConfig,
    StrongClassicalRelationEvidenceSelector,
    load_relation_attention_score_kernel_checkpoint,
    save_relation_attention_score_kernel_checkpoint,
)
from q_attention.plugins.attention_evidence import (
    _apply_ry,
    _connected_z_correlations,
    _cross_register_entangle,
    _join_register_states,
    _relation_token_z_correlations,
    _total_z_correlations,
    _two_qubit_local_bloch_angles,
    _two_qubit_pauli_features,
)
from q_attention.plugins.quantum_steering import QuantumSteeringContext


def _context(keys: torch.Tensor) -> QuantumSteeringContext:
    attention = torch.ones(keys.shape[:2], dtype=torch.bool)
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 0] = True
    object_[:, 2] = True
    return QuantumSteeringContext(
        keys=keys,
        layer_index=0,
        attention_mask=attention,
        steering_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )


def test_quantum_key_steer_is_rank_constrained_and_zero_gain_identity() -> None:
    config = QuantumKeySteeringConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        depth=2,
        rank=2,
        initial_gain=0.0,
    )
    plugin = QuantumKeySteeringPlugin(config)
    projectors = plugin.projectors(0)
    assert projectors.shape == (2, 4, 4)
    assert torch.allclose(projectors, projectors.transpose(-1, -2), atol=1e-5)
    assert torch.allclose(projectors @ projectors, projectors, atol=1e-5)
    assert torch.allclose(
        torch.diagonal(projectors, dim1=-2, dim2=-1).sum(dim=-1),
        torch.full((2,), 2.0),
        atol=1e-5,
    )

    keys = torch.randn(2, 5, 8)
    output = plugin(_context(keys)).delta
    assert output is not None
    assert torch.equal(output, torch.zeros_like(keys))


def test_product_circuit_is_a_distinct_no_cross_entanglement_ablation() -> None:
    entangled = QuantumKeySteeringPlugin(
        QuantumKeySteeringConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            depth=2,
            rank=2,
            seed=17,
            circuit_type="entangled",
        )
    )
    product = QuantumKeySteeringPlugin(
        QuantumKeySteeringConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            depth=2,
            rank=2,
            seed=17,
            circuit_type="product",
        )
    )
    assert not torch.allclose(entangled.projectors(0), product.projectors(0))


def test_classical_key_control_matches_quantum_trainable_parameter_count() -> None:
    config = HeadwiseQuantumProjectorConfig(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        depth=2,
        rank=3,
    )
    quantum = QuantumKeySteeringPlugin(QuantumKeySteeringConfig(**config.__dict__))
    classical = ClassicalKeySteeringPlugin(config)
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in classical.parameters()
    )
    keys = torch.randn(2, 5, 16, requires_grad=True)
    delta = classical(_context(keys)).delta
    assert delta is not None
    delta.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in classical.parameters()
    )


def _evidence_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = torch.randn(2, 2, 6, 4)
    attention = torch.ones(2, 6, dtype=torch.bool)
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 0] = True
    object_[:, 2] = True
    return key, attention, subject, object_


def test_qres_has_nonzero_cross_connected_correlation_and_separable_is_zero() -> None:
    torch.manual_seed(23)
    relation = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    token = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    product_state = _join_register_states(relation, token)
    qres = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            seed=29,
            cross_entanglement=True,
        )
    )
    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            seed=29,
            cross_entanglement=False,
        )
    )
    connected = qres._connected_observable_features(token, relation)
    separated = separable._connected_observable_features(token, relation)
    joint, marginal_product, product_connected = _relation_token_z_correlations(
        product_state,
        2,
    )
    assert connected.abs().mean() > 1e-5
    assert torch.allclose(separated, torch.zeros_like(separated), atol=1e-6)
    assert torch.allclose(joint, marginal_product + product_connected, atol=1e-6)
    assert torch.allclose(_total_z_correlations(product_state, 2), joint, atol=1e-6)
    assert torch.allclose(
        _connected_z_correlations(product_state, 2),
        separated,
        atol=1e-6,
    )


def test_qres_two_qubit_relation_measurements_match_known_states() -> None:
    product = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    bell = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])

    assert torch.allclose(
        _two_qubit_pauli_features(product),
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        atol=1e-6,
    )
    assert torch.allclose(
        _two_qubit_pauli_features(bell),
        torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        atol=1e-6,
    )


def test_qres_relation_frame_angles_match_known_bloch_directions() -> None:
    product = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    plus = torch.full((1, 4), 0.5)
    bell = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])

    assert torch.allclose(
        _two_qubit_local_bloch_angles(product),
        torch.zeros(1, 2),
        atol=1e-6,
    )
    assert torch.allclose(
        _two_qubit_local_bloch_angles(plus),
        torch.full((1, 2), torch.pi / 2),
        atol=1e-6,
    )
    assert torch.allclose(
        _two_qubit_local_bloch_angles(bell),
        torch.zeros(1, 2),
        atol=1e-6,
    )


def test_qres_relation_conditioned_measurement_changes_polarity_and_is_matched() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_conditioned",
        intervention_mode="direct_bias",
        initial_conditioning_gain=1.0,
        seed=30,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    assert sum(parameter.numel() for parameter in quantum.parameters()) == 48
    assert sum(parameter.numel() for parameter in classical.parameters()) == 48

    with torch.no_grad():
        quantum.observable_logits.zero_()
    relation_states = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        requires_grad=True,
    )
    weights = quantum.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
    )
    assert torch.allclose(weights.sum(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(weights.abs().sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.any(weights[0] * weights[1] < 0.0)
    weights.square().sum().backward()
    assert relation_states.grad is not None
    assert quantum.raw_conditioning_gains is not None
    assert quantum.raw_conditioning_gains.grad is not None


def test_qres_relation_directional_axes_are_matched_and_entanglement_dependent() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_correlation_mode="phase_selective",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_directional",
        intervention_mode="direct_bias",
        seed=30,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    assert sum(parameter.numel() for parameter in quantum.parameters()) == 58
    assert sum(parameter.numel() for parameter in classical.parameters()) == 58
    assert sum(parameter.numel() for parameter in strong_classical.parameters()) == 58
    assert torch.allclose(
        quantum.conditioning_gains(0),
        torch.full((2,), 0.5),
        atol=1e-6,
    )
    assert torch.allclose(
        quantum.frame_fusion_gains(0),
        torch.full((2,), 1.0),
        atol=1e-6,
    )

    relation_states = torch.nn.functional.normalize(
        torch.randn(12, 4, generator=torch.Generator().manual_seed(31)),
        dim=-1,
    ).requires_grad_()
    weights = quantum.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
    )
    frame_weights = weights.reshape(12, 2, -1)
    assert weights.shape == (12, 8)
    assert torch.allclose(
        frame_weights.sum(dim=-1),
        torch.zeros(12, 2),
        atol=1e-6,
    )
    assert torch.allclose(
        frame_weights.abs().sum(dim=-1),
        torch.ones(12, 2),
        atol=1e-6,
    )
    assert torch.isfinite(weights).all()
    weights.square().sum().backward()
    assert relation_states.grad is not None
    assert torch.isfinite(relation_states.grad).all()

    token_states = torch.nn.functional.normalize(
        torch.randn(12, 4, generator=torch.Generator().manual_seed(32)),
        dim=-1,
    )
    quantum_features = quantum._connected_observable_features(
        token_states,
        relation_states.detach(),
    )
    classical_features = classical._connected_observable_features(
        token_states,
        relation_states.detach(),
    )
    strong_features = strong_classical._connected_observable_features(
        token_states,
        relation_states.detach(),
    )
    assert quantum_features.shape == classical_features.shape == (12, 8)
    assert torch.isfinite(quantum_features).all()
    assert torch.isfinite(classical_features).all()
    assert torch.isfinite(strong_features).all()

    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )
    _separable_features, separable_channels = separable._relation_token_observable_features(
        token_states,
        relation_states.detach(),
    )
    assert torch.allclose(
        separable_channels["phase_selective"],
        torch.zeros_like(separable_channels["phase_selective"]),
        atol=1e-6,
    )


def test_qres_entanglement_directional_axes_follow_phase_and_match_controls() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_correlation_mode="phase_selective",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="entanglement_directional",
        intervention_mode="direct_bias",
        seed=33,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    parameter_counts = {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in (quantum, classical, strong_classical)
    }
    assert parameter_counts == {58}

    token_states = torch.nn.functional.normalize(
        torch.randn(12, 4, generator=torch.Generator().manual_seed(34)),
        dim=-1,
    )
    relation_states = torch.nn.functional.normalize(
        torch.randn(12, 4, generator=torch.Generator().manual_seed(35)),
        dim=-1,
    )
    quantum_features, quantum_channels = (
        quantum._relation_token_observable_features(token_states, relation_states)
    )
    quantum_direction = quantum._entanglement_direction_features(quantum_channels)
    direction_frames = quantum_direction.reshape(12, 2, -1)
    assert quantum_direction.shape == (12, 8)
    assert torch.isfinite(quantum_direction).all()
    assert quantum_direction.abs().max() <= 1.0
    assert quantum_direction.abs().mean() > 1e-5
    assert torch.allclose(
        direction_frames[:, 0] + direction_frames[:, 1],
        torch.zeros(12, 4),
        atol=1e-6,
    )

    quantum_weights = quantum.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
        correlation_channels=quantum_channels,
    )
    weight_frames = quantum_weights.reshape(12, 2, -1)
    assert torch.allclose(
        weight_frames.sum(dim=-1),
        torch.zeros(12, 2),
        atol=1e-6,
    )
    assert torch.allclose(
        weight_frames.abs().sum(dim=-1),
        torch.ones(12, 2),
        atol=1e-6,
    )
    assert not torch.allclose(
        quantum_weights,
        quantum.observable_weights(0)[0].expand_as(quantum_weights),
        atol=1e-6,
    )
    quantum_weights.square().sum().backward()
    assert quantum.raw_conditioning_gains is not None
    assert quantum.raw_conditioning_gains.grad is not None

    _classical_features, classical_channels = (
        classical._relation_token_observable_features(token_states, relation_states)
    )
    classical_weights = classical.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
        correlation_channels=classical_channels,
    )
    assert not torch.allclose(
        classical_weights,
        classical.observable_weights(0)[0].expand_as(classical_weights),
        atol=1e-6,
    )
    _strong_features, strong_channels = (
        strong_classical._relation_token_observable_features(
            token_states,
            relation_states,
        )
    )
    strong_weights = strong_classical.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
        correlation_channels=strong_channels,
    )
    assert not torch.allclose(
        strong_weights,
        strong_classical.observable_weights(0)[0].expand_as(strong_weights),
        atol=1e-6,
    )

    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )
    separable_features, separable_channels = (
        separable._relation_token_observable_features(token_states, relation_states)
    )
    separable_direction = separable._entanglement_direction_features(
        separable_channels
    )
    separable_weights = separable.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
        correlation_channels=separable_channels,
    )
    assert torch.allclose(
        separable_features,
        torch.zeros_like(separable_features),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_direction,
        torch.zeros_like(separable_direction),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_weights,
        separable.observable_weights(0)[0].expand_as(separable_weights),
        atol=1e-6,
    )

    metadata = quantum.metadata()["measurement_resources"]["conditioning"]
    assert metadata["type"] == "entanglement_phase_observable_axis"
    assert metadata["zero_without_cross_register_entanglement"] is True

    with pytest.raises(ValueError, match="requires.*phase_selective"):
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "evidence_correlation_mode": "connected"}
        )


def test_qres_entanglement_phase_offset_is_bounded_matched_and_trainable() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_correlation_mode="phase_selective",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="entanglement_phase_offset",
        intervention_mode="direct_bias",
        seed=36,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    assert {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in (quantum, classical, strong_classical)
    } == {58}

    token_states = torch.nn.functional.normalize(
        torch.randn(12, 4, generator=torch.Generator().manual_seed(37)),
        dim=-1,
    )
    relation_states = torch.nn.functional.normalize(
        torch.randn(12, 4, generator=torch.Generator().manual_seed(38)),
        dim=-1,
    )
    _quantum_features, quantum_channels = (
        quantum._relation_token_observable_features(token_states, relation_states)
    )
    phase_offset = quantum.conditioning_gains(0)[0]
    phase_sine, phase_cosine, reliability = (
        quantum._entanglement_phase_components(quantum_channels)
    )
    phase_features = quantum._entanglement_phase_offset_features(
        quantum_channels,
        phase_offset,
    )
    expected_direction = reliability * torch.sin(phase_offset) * (
        torch.cos(phase_offset) * phase_sine
        + torch.sin(phase_offset) * phase_cosine
    )
    phase_frames = phase_features.reshape(12, 2, -1)
    assert torch.allclose(phase_frames[:, 0], -expected_direction, atol=1e-6)
    assert torch.allclose(phase_frames[:, 1], expected_direction, atol=1e-6)
    assert phase_features.abs().max() <= 1.0
    assert phase_features.abs().mean() > 1e-5

    weights = quantum.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
        correlation_channels=quantum_channels,
    )
    weight_frames = weights.reshape(12, 2, -1)
    assert torch.allclose(
        weight_frames.sum(dim=-1),
        torch.zeros(12, 2),
        atol=1e-6,
    )
    assert torch.allclose(
        weight_frames.abs().sum(dim=-1),
        torch.ones(12, 2),
        atol=1e-6,
    )
    weights.square().sum().backward()
    assert quantum.raw_conditioning_gains is not None
    assert quantum.raw_conditioning_gains.grad is not None

    zero_offset = QuantumRelationEvidenceSelector(config)
    with torch.no_grad():
        assert zero_offset.raw_conditioning_gains is not None
        zero_offset.raw_conditioning_gains.zero_()
    _zero_features, zero_channels = zero_offset._relation_token_observable_features(
        token_states,
        relation_states,
    )
    zero_weights = zero_offset.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
        correlation_channels=zero_channels,
    )
    assert torch.allclose(
        zero_weights,
        zero_offset.observable_weights(0)[0].expand_as(zero_weights),
        atol=1e-6,
    )

    _classical_features, classical_channels = (
        classical._relation_token_observable_features(token_states, relation_states)
    )
    classical_weights = classical.conditioned_observable_weights(
        relation_states,
        layer_index=0,
        head_index=0,
        correlation_channels=classical_channels,
    )
    assert not torch.allclose(
        classical_weights,
        classical.observable_weights(0)[0].expand_as(classical_weights),
        atol=1e-6,
    )

    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )
    separable_features, separable_channels = (
        separable._relation_token_observable_features(token_states, relation_states)
    )
    separable_phase_features = separable._entanglement_phase_offset_features(
        separable_channels,
        separable.conditioning_gains(0)[0],
    )
    assert torch.allclose(
        separable_features,
        torch.zeros_like(separable_features),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_phase_features,
        torch.zeros_like(separable_phase_features),
        atol=1e-6,
    )

    metadata = quantum.metadata()["measurement_resources"]["conditioning"]
    assert metadata["type"] == "phase_offset_entanglement_axis"
    assert metadata["zero_at_parameter_origin"] is True


def test_qres_relation_frame_is_unavoidable_matched_and_separable_zero() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_frame",
        intervention_mode="direct_bias",
        seed=31,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    assert quantum.raw_conditioning_gains is None
    assert sum(parameter.numel() for parameter in quantum.parameters()) == 46
    assert sum(parameter.numel() for parameter in classical.parameters()) == 46

    torch.manual_seed(32)
    relation = torch.nn.functional.normalize(
        torch.randn(12, 4), dim=-1
    ).requires_grad_()
    token = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    connected = quantum._connected_observable_features(token, relation)
    fixed = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "evidence_measurement_mode": "fixed"}
        )
    )._connected_observable_features(token, relation)
    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )._connected_observable_features(token, relation)
    assert connected.abs().mean() > 1e-5
    assert not torch.allclose(connected, fixed, atol=1e-6)
    assert torch.allclose(separable, torch.zeros_like(separable), atol=1e-6)
    connected.square().mean().backward()
    assert relation.grad is not None
    assert torch.isfinite(relation.grad).all()

    key, attention, subject, object_ = _evidence_inputs()
    with quantum.capture_token_scores():
        quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        captured = quantum.captured_relation_frame_angles()
        assert len(captured) == config.num_heads
        assert captured[0][2].shape == (key.shape[0], key.shape[2], 2)


def test_qres_relation_frame_bank_is_noncommuting_matched_and_separable_zero() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_frame_bank",
        intervention_mode="direct_bias",
        seed=33,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    assert quantum.observable_logits.shape == (1, 2, 8)
    assert classical.observable_logits.shape == (1, 2, 8)
    assert sum(parameter.numel() for parameter in quantum.parameters()) == 54
    assert sum(parameter.numel() for parameter in classical.parameters()) == 54
    assert quantum.metadata()["measurement_resources"] == {
        "measurement_frames": 2,
        "observable_count": 8,
        "relative_to_relation_frame": {
            "additional_qubits": 0,
            "additional_cross_register_cnots": 0,
            "measurement_circuit_multiplier": 2.0,
        },
    }

    torch.manual_seed(34)
    relation = torch.nn.functional.normalize(
        torch.randn(12, 4), dim=-1
    ).requires_grad_()
    token = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    bank = quantum._connected_observable_features(token, relation)
    single_frame = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "evidence_measurement_mode": "relation_frame"}
        )
    )._connected_observable_features(token, relation)
    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )._connected_observable_features(token, relation)
    classical_bank = classical._connected_observable_features(token, relation)

    assert bank.shape == (12, 8)
    assert classical_bank.shape == (12, 8)
    assert torch.allclose(bank[:, :4], single_frame, atol=1e-6)
    assert not torch.allclose(bank[:, :4], bank[:, 4:], atol=1e-6)
    assert bank[:, 4:].abs().mean() > 1e-5
    assert torch.allclose(separable, torch.zeros_like(separable), atol=1e-6)
    bank.square().mean().backward()
    assert relation.grad is not None
    assert torch.isfinite(relation.grad).all()

    key, attention, subject, object_ = _evidence_inputs()
    with quantum.capture_token_scores():
        scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        captured = quantum.captured_relation_frame_angles()
        frame_contributions = (
            quantum.captured_measurement_frame_contributions()
        )
        assert scores.shape == (2, 2, 6)
        assert len(captured) == config.num_heads
        assert captured[0][2].shape == (key.shape[0], key.shape[2], 2)
        assert len(frame_contributions) == config.num_heads
        assert frame_contributions[0][2].shape == (key.shape[0], key.shape[2], 2)
    with quantum.use_measurement_frame_view("z"):
        z_scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
    with quantum.use_measurement_frame_view("x"):
        x_scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
    restored_full_scores = quantum.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    offsets = quantum.offsets[0].view(1, -1, 1)
    assert torch.allclose(restored_full_scores, scores, atol=1e-6)
    assert not torch.allclose(z_scores, x_scores, atol=1e-6)
    assert torch.allclose(
        torch.logit(scores),
        torch.logit(z_scores) + torch.logit(x_scores) - offsets,
        atol=1e-5,
    )
    direct_bias = quantum.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    (scores.square().mean() + direct_bias.square().mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in quantum.parameters()
    )


def test_qres_coherent_frames_are_independent_gated_and_strongly_controlled() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_frame_coherent",
        intervention_mode="direct_bias",
        initial_frame_fusion_gain=1.0,
        seed=35,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )
    parameter_counts = {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in (quantum, classical, strong_classical, separable)
    }
    assert parameter_counts == {56}
    weights = quantum.observable_weights(0).reshape(2, 2, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(weights.abs().sum(dim=-1), torch.ones(2, 2), atol=1e-6)
    assert torch.allclose(
        quantum.frame_fusion_gains(0),
        torch.ones(2),
        atol=1e-6,
    )

    torch.manual_seed(36)
    relation = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    token = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    quantum_features = quantum._connected_observable_features(token, relation)
    classical_features = classical._connected_observable_features(token, relation)
    strong_features = strong_classical._connected_observable_features(token, relation)
    separated_features = separable._connected_observable_features(token, relation)
    cross_frame_state = _cross_register_entangle(
        _join_register_states(relation, token),
        quantum.register_qubits,
    )
    frame_angles = quantum.relation_frame_angles(relation)
    for qubit in range(quantum.register_qubits):
        cross_frame_state = _apply_ry(
            cross_frame_state,
            -frame_angles[:, qubit],
            qubit,
            2 * quantum.register_qubits,
        )
        cross_frame_state = _apply_ry(
            cross_frame_state,
            -(frame_angles[:, qubit] + torch.pi / 2),
            quantum.register_qubits + qubit,
            2 * quantum.register_qubits,
        )
    expected_cross_frame = _connected_z_correlations(
        cross_frame_state,
        quantum.register_qubits,
    )
    assert torch.allclose(quantum_features[:, 4:], expected_cross_frame, atol=1e-6)
    assert quantum_features[:, 4:].abs().mean() > 1e-5
    assert torch.allclose(
        classical_features[:, 4:],
        torch.zeros_like(classical_features[:, 4:]),
        atol=1e-6,
    )
    assert strong_features[:, 4:].abs().mean() > 1e-5
    assert torch.allclose(
        separated_features,
        torch.zeros_like(separated_features),
        atol=1e-6,
    )

    key, attention, subject, object_ = _evidence_inputs()
    with quantum.capture_token_scores():
        full_scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        captured_gates = quantum.captured_coherence_gates()
        assert len(captured_gates) == config.num_heads
        ratios = captured_gates[0][2][..., 0]
        effective_gates = captured_gates[0][2][..., 1]
        assert torch.all((ratios >= 0.0) & (ratios <= 1.0))
        assert torch.allclose(ratios, effective_gates, atol=1e-6)
    with quantum.use_measurement_frame_view("z"):
        z_scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
    with quantum.use_measurement_frame_view("x"):
        x_scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
    offsets = quantum.offsets[0].view(1, -1, 1)
    assert torch.allclose(
        torch.logit(full_scores),
        torch.logit(z_scores) + torch.logit(x_scores) - offsets,
        atol=1e-5,
    )
    direct_bias = quantum.direct_key_bias(
        full_scores,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    (full_scores.square().mean() + direct_bias.square().mean()).backward()
    assert quantum.raw_frame_fusion_gains is not None
    assert quantum.raw_frame_fusion_gains.grad is not None
    assert torch.isfinite(quantum.raw_frame_fusion_gains.grad).all()


def test_qres_total_correlation_preserves_product_and_connected_channels() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_correlation_mode="total",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_frame_coherent",
        intervention_mode="direct_bias",
        seed=37,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    connected = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "evidence_correlation_mode": "connected"}
        )
    )
    connected.load_state_dict(quantum.state_dict())
    multiscale = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "evidence_correlation_mode": "multiscale"}
        )
    )
    multiscale.load_state_dict(quantum.state_dict())
    correlation_gated = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{
                **config.__dict__,
                "evidence_correlation_mode": "correlation_gated",
            }
        )
    )
    correlation_gated.load_state_dict(quantum.state_dict())
    signed_gated = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "evidence_correlation_mode": "signed_gated"}
        )
    )
    signed_gated.load_state_dict(quantum.state_dict())
    standardized = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{
                **config.__dict__,
                "evidence_correlation_mode": "standardized_connected",
            }
        )
    )
    standardized.load_state_dict(quantum.state_dict())
    standardized_signed = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{
                **config.__dict__,
                "evidence_correlation_mode": "standardized_signed_gated",
            }
        )
    )
    standardized_signed.load_state_dict(quantum.state_dict())
    phase_selective = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{
                **config.__dict__,
                "evidence_correlation_mode": "phase_selective",
            }
        )
    )
    phase_selective.load_state_dict(quantum.state_dict())
    phase_rotated = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{
                **config.__dict__,
                "evidence_correlation_mode": "phase_rotated",
            }
        )
    )
    phase_rotated.load_state_dict(quantum.state_dict())
    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    parameter_counts = {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in (
            quantum,
            connected,
            multiscale,
            correlation_gated,
            signed_gated,
            standardized,
            standardized_signed,
            phase_selective,
            phase_rotated,
            separable,
            classical,
            strong_classical,
        )
    }
    assert parameter_counts == {56}

    torch.manual_seed(38)
    relation = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    token = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    total_features, channels = quantum._relation_token_observable_features(
        token,
        relation,
    )
    connected_features = connected._connected_observable_features(token, relation)
    multiscale_features = multiscale._connected_observable_features(token, relation)
    correlation_gated_features = correlation_gated._connected_observable_features(
        token, relation
    )
    signed_gated_features = signed_gated._connected_observable_features(
        token, relation
    )
    standardized_features = standardized._connected_observable_features(
        token, relation
    )
    standardized_signed_features = (
        standardized_signed._connected_observable_features(token, relation)
    )
    phase_selective_features = phase_selective._connected_observable_features(
        token, relation
    )
    phase_rotated_features = phase_rotated._connected_observable_features(
        token, relation
    )
    separable_features, separable_channels = (
        separable._relation_token_observable_features(token, relation)
    )
    assert torch.allclose(total_features, channels["total"], atol=1e-6)
    assert torch.allclose(
        channels["total"],
        channels["post_entanglement_product"] + channels["connected"],
        atol=1e-6,
    )
    assert torch.allclose(
        channels["multiscale"],
        channels["pre_entanglement_product"] + channels["connected"],
        atol=1e-6,
    )
    assert torch.allclose(connected_features, channels["connected"], atol=1e-6)
    assert torch.allclose(multiscale_features, channels["multiscale"], atol=1e-6)
    expected_gate = channels["connected"].square() / (
        channels["connected"].square()
        + channels["pre_entanglement_product"].square()
        + config.eps
    )
    assert torch.all((0.0 <= expected_gate) & (expected_gate <= 1.0))
    assert torch.allclose(
        channels["correlation_gated"],
        channels["connected"]
        + expected_gate * channels["pre_entanglement_product"],
        atol=1e-6,
    )
    assert torch.allclose(
        correlation_gated_features,
        channels["correlation_gated"],
        atol=1e-6,
    )
    expected_signed_gate = (
        channels["connected"] * channels["pre_entanglement_product"]
    ) / (
        channels["connected"].square()
        + channels["pre_entanglement_product"].square()
        + config.eps
    )
    signed_residual = (
        expected_signed_gate * channels["pre_entanglement_product"]
    )
    assert torch.all(
        signed_residual * channels["connected"] >= -config.eps
    )
    assert torch.allclose(
        channels["signed_gated"],
        channels["connected"] + signed_residual,
        atol=1e-6,
    )
    assert torch.allclose(
        signed_gated_features,
        channels["signed_gated"],
        atol=1e-6,
    )
    assert torch.all(channels["standardized_connected"].abs() <= 1.0)
    assert torch.allclose(
        standardized_features,
        channels["standardized_connected"],
        atol=1e-6,
    )
    standardized_residual = (
        channels["standardized_signed_gated"]
        - channels["standardized_connected"]
    )
    assert torch.all(
        standardized_residual * channels["standardized_connected"]
        >= -config.eps
    )
    assert torch.allclose(
        standardized_signed_features,
        channels["standardized_signed_gated"],
        atol=1e-6,
    )
    assert torch.isfinite(channels["phase_selective"]).all()
    assert torch.allclose(
        phase_selective_features,
        channels["phase_selective"],
        atol=1e-6,
    )
    assert torch.isfinite(channels["phase_rotated"]).all()
    rotated_norm = torch.linalg.vector_norm(
        channels["phase_rotated"].reshape(12, 2, -1),
        dim=1,
    )
    base_norm = torch.linalg.vector_norm(
        channels["standardized_signed_gated"].reshape(12, 2, -1),
        dim=1,
    )
    assert torch.allclose(rotated_norm, base_norm, atol=1e-5)
    assert torch.allclose(
        phase_rotated_features,
        channels["phase_rotated"],
        atol=1e-6,
    )
    assert not torch.allclose(total_features, connected_features, atol=1e-6)
    assert separable_features.abs().mean() > 1e-5
    assert torch.allclose(
        separable_channels["connected"],
        torch.zeros_like(separable_channels["connected"]),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_features,
        separable_channels["pre_entanglement_product"],
        atol=1e-6,
    )
    assert torch.allclose(
        separable_channels["multiscale"],
        separable_channels["pre_entanglement_product"],
        atol=1e-6,
    )
    assert torch.allclose(
        separable_channels["correlation_gated"],
        torch.zeros_like(separable_channels["correlation_gated"]),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_channels["signed_gated"],
        torch.zeros_like(separable_channels["signed_gated"]),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_channels["standardized_connected"],
        torch.zeros_like(separable_channels["standardized_connected"]),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_channels["standardized_signed_gated"],
        torch.zeros_like(separable_channels["standardized_signed_gated"]),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_channels["phase_selective"],
        torch.zeros_like(separable_channels["phase_selective"]),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_channels["phase_rotated"],
        torch.zeros_like(separable_channels["phase_rotated"]),
        atol=1e-6,
    )

    key, attention, subject, object_ = _evidence_inputs()
    with quantum.capture_token_scores():
        scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        captured_channels = quantum.captured_correlation_channel_contributions()
        assert len(captured_channels) == config.num_heads
        contributions = captured_channels[0][2]
        channel_index = {
            name: index for index, name in enumerate(EVIDENCE_CORRELATION_CHANNELS)
        }
        assert torch.allclose(
            contributions[..., channel_index["total"], :],
            contributions[..., channel_index["post_entanglement_product"], :]
            + contributions[..., channel_index["connected"], :],
            atol=1e-6,
        )
    direct_bias = quantum.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    (scores.square().mean() + direct_bias.square().mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in quantum.parameters()
    )
    metadata = quantum.metadata()["correlation_readout"]
    assert metadata["mode"] == "total"
    assert metadata["additional_measurement_circuits_per_frame"] == 0
    assert (
        multiscale.metadata()["correlation_readout"][
            "additional_measurement_circuits_per_frame"
        ]
        == 2
    )
    gated_metadata = correlation_gated.metadata()["correlation_readout"]
    assert gated_metadata["additional_measurement_circuits_per_frame"] == 2
    assert gated_metadata["fusion"] == {
        "type": "connected_energy_gate",
        "trainable_parameters": 0,
    }
    signed_metadata = signed_gated.metadata()["correlation_readout"]
    assert signed_metadata["additional_measurement_circuits_per_frame"] == 2
    assert signed_metadata["fusion"] == {
        "type": "signed_connected_energy_gate",
        "trainable_parameters": 0,
    }


def test_qres_scores_are_masked_neutral_at_zero_connected_signal_and_differentiable() -> None:
    key, attention, subject, object_ = _evidence_inputs()
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            cross_entanglement=False,
            seed=31,
        )
    )
    scores = selector.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    assert scores.shape == (2, 2, 6)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, torch.full_like(scores, 0.5), atol=1e-5)
    scores.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in selector.parameters()
    )


def test_qres_dual_channel_uses_independent_parameter_matched_banks() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_correlation_mode="dual_channel",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_frame_coherent",
        intervention_mode="direct_bias",
        seed=41,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    parameter_counts = {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in (quantum, classical, strong_classical)
    }
    assert parameter_counts == {72}

    torch.manual_seed(42)
    relation = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    token = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)
    features, channels = quantum._relation_token_observable_features(
        token, relation
    )
    assert features.shape == (12, 16)
    assert torch.allclose(features, channels["dual_channel"], atol=1e-6)
    assert torch.allclose(
        channels["dual_channel"],
        channels["connected"] + channels["pre_entanglement_product"],
        atol=1e-6,
    )
    assert torch.allclose(
        channels["connected"] * channels["pre_entanglement_product"],
        torch.zeros_like(features),
        atol=1e-6,
    )
    for selector in (classical, strong_classical):
        control_features, control_channels = (
            selector._relation_token_observable_features(token, relation)
        )
        assert control_features.shape == features.shape
        assert torch.allclose(
            control_features,
            control_channels["connected"]
            + control_channels["pre_entanglement_product"],
            atol=1e-6,
        )

    weights = quantum.observable_weights(0).reshape(2, 2, -1)
    assert torch.allclose(
        weights.abs().sum(dim=-1),
        torch.ones_like(weights[..., 0]),
        atol=1e-6,
    )
    key, attention, subject, object_ = _evidence_inputs()
    scores = quantum.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    direct_bias = quantum.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    (scores.square().mean() + direct_bias.square().mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in quantum.parameters()
    )
    metadata = quantum.metadata()["correlation_readout"]
    assert metadata["additional_measurement_circuits_per_frame"] == 2
    assert metadata["fusion"] == {
        "type": "independent_signed_observable_banks",
        "classical_secondary_bank": "signed_square",
        "observable_parameter_multiplier": 2.0,
    }

    with pytest.raises(ValueError, match="two-frame"):
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            evidence_correlation_mode="dual_channel",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="fixed",
        )


def test_qres_born_reliability_is_entanglement_dependent_and_matched() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_correlation_mode="born_reliability",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_frame_coherent",
        intervention_mode="direct_bias",
        seed=43,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    parameter_counts = {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in (quantum, classical, strong_classical)
    }
    assert parameter_counts == {58}
    assert torch.allclose(
        quantum.reliability_exponents(0),
        torch.ones(2),
        atol=1e-6,
    )

    key, attention, subject, object_ = _evidence_inputs()
    with quantum.capture_token_scores():
        scores = quantum.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        captured = quantum.captured_reliability_gates()
        assert len(captured) == config.num_heads
        for _layer_index, _head_index, gates in captured:
            assert gates.shape == (2, 6, 2, 2)
            quality = gates[..., 0]
            effective = gates[..., 1]
            assert torch.all((0.0 <= quality) & (quality <= 1.0))
            assert torch.all((0.0 <= effective) & (effective <= 1.0))
            assert torch.allclose(quality, effective, atol=1e-6)
    direct_bias = quantum.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    (scores.square().mean() + direct_bias.square().mean()).backward()
    assert quantum.raw_reliability_exponents is not None
    assert quantum.raw_reliability_exponents.grad is not None
    assert torch.isfinite(quantum.raw_reliability_exponents.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in quantum.parameters()
    )

    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )
    separable.load_state_dict(quantum.state_dict())
    separable_scores = separable.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    assert torch.allclose(
        separable_scores,
        torch.full_like(separable_scores, 0.5),
        atol=1e-5,
    )
    for selector in (classical, strong_classical):
        control_scores = selector.token_scores(
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        assert torch.isfinite(control_scores).all()

    metadata = quantum.metadata()["correlation_readout"]
    assert metadata["additional_measurement_circuits_per_frame"] == 2
    assert metadata["fusion"] == {
        "type": "born_energy_reliability_exponent",
        "quality": "connected_over_connected_plus_post_product",
        "trainable_parameters_per_layer_head": 1,
        "free_amplitude": False,
    }


def test_qres_leave_one_out_context_anchor_excludes_the_candidate_token() -> None:
    torch.manual_seed(33)
    key, attention, subject, object_ = _evidence_inputs()
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            relation_anchor_mode="leave_one_out_context",
            seed=35,
        )
    )
    _, anchor = selector._encoding_inputs(
        key,
        attention,
        subject,
        object_,
        0,
    )
    changed = key.clone()
    changed[:, 0, 1] += 5.0
    _, changed_anchor = selector._encoding_inputs(
        changed,
        attention,
        subject,
        object_,
        0,
    )

    assert torch.allclose(anchor[:, 1], changed_anchor[:, 1], atol=1e-6)
    assert not torch.allclose(anchor[:, 3], changed_anchor[:, 3])


def test_qres_quantum_and_classical_controls_match_parameter_count_and_masks() -> None:
    key, attention, subject, object_ = _evidence_inputs()
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        seed=37,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in classical.parameters()
    )
    scores = quantum.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    assert torch.all(scores[:, :, 0] > 0.0)
    assert torch.all(scores[:, :, 2] > 0.0)
    assert torch.isfinite(classical.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )).all()


def test_qres_signed_readout_is_balanced_parameter_matched_and_differentiable() -> None:
    key, attention, subject, object_ = _evidence_inputs()
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_weight_mode="signed_centered_l1",
        seed=39,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical = ClassicalRelationEvidenceSelector(config)
    weights = quantum.observable_weights(0)
    assert torch.allclose(weights.sum(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(weights.abs().sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all((weights > 0.0).any(dim=-1))
    assert torch.all((weights < 0.0).any(dim=-1))
    assert sum(parameter.numel() for parameter in quantum.parameters()) == sum(
        parameter.numel() for parameter in classical.parameters()
    )
    scores = quantum.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    scores.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in quantum.parameters()
    )


def test_qres_direct_bias_is_standalone_centered_and_resource_selective() -> None:
    torch.manual_seed(40)
    key, attention, subject, object_ = _evidence_inputs()
    query = torch.randn_like(key)
    kernel_config = RelationScoreKernelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        initial_gain=0.0,
        seed=40,
    )
    selector_config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        evidence_readout="connected_relation_token",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="relation_conditioned",
        intervention_mode="direct_bias",
        relation_anchor_mode="leave_one_out_context",
        initial_direct_gain=0.2,
        seed=41,
    )
    kernel = QuantumRelationAttentionScoreKernel(kernel_config)
    selector = QuantumRelationEvidenceSelector(selector_config)
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    classical_selector = ClassicalRelationEvidenceSelector(selector_config)
    assert sum(parameter.numel() for parameter in selector.parameters()) == sum(
        parameter.numel() for parameter in classical_selector.parameters()
    )
    kernel.attach_evidence_selector(selector)
    residual = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    scores = selector.token_scores(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    key_bias = selector.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    context = attention & ~(subject | object_)
    assert residual.abs().mean() > 1e-6
    assert torch.equal(key_bias[:, :, 0], torch.zeros_like(key_bias[:, :, 0]))
    assert torch.equal(key_bias[:, :, 2], torch.zeros_like(key_bias[:, :, 2]))
    assert torch.allclose(
        (key_bias * context[:, None, :]).sum(dim=-1),
        torch.zeros(2, 2),
        atol=1e-6,
    )

    separable_kernel = QuantumRelationAttentionScoreKernel(kernel_config)
    separable_selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{
                **selector_config.__dict__,
                "cross_entanglement": False,
            }
        )
    )
    separable_kernel.attach_evidence_selector(separable_selector)
    separable_residual = separable_kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    assert torch.allclose(
        separable_residual,
        torch.zeros_like(separable_residual),
        atol=1e-6,
    )
    residual.square().mean().backward()
    assert selector.raw_direct_gains is not None
    assert selector.raw_direct_gains.grad is not None
    assert torch.isfinite(selector.raw_direct_gains.grad).all()


def test_qres_checkpoint_round_trip_preserves_connected_readout(tmp_path) -> None:
    kernel = QuantumRelationAttentionScoreKernel(
        RelationScoreKernelConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            seed=41,
        )
    )
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="relation_conditioned",
            evidence_gate_calibration="context_budget",
            evidence_budget=0.4,
            evidence_view_score_mode="polarity_magnitude",
            intervention_mode="direct_bias",
            direct_bias_mode="positive_excess",
            relation_anchor_mode="leave_one_out_context",
            seed=43,
        )
    )
    kernel.attach_evidence_selector(selector)
    checkpoint = tmp_path / "qres.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    assert restored.evidence_selector is not None
    assert (
        restored.evidence_selector.config.evidence_readout
        == "connected_relation_token"
    )
    assert (
        restored.evidence_selector.config.relation_anchor_mode
        == "leave_one_out_context"
    )
    assert (
        restored.evidence_selector.config.evidence_weight_mode
        == "signed_centered_l1"
    )
    assert restored.evidence_selector.config.intervention_mode == "direct_bias"
    assert restored.evidence_selector.config.direct_bias_mode == "positive_excess"
    assert (
        restored.evidence_selector.config.evidence_gate_calibration
        == "context_budget"
    )
    assert restored.evidence_selector.config.evidence_budget == 0.4
    assert (
        restored.evidence_selector.config.evidence_view_score_mode
        == "polarity_magnitude"
    )
    assert (
        restored.evidence_selector.config.evidence_measurement_mode
        == "relation_conditioned"
    )
    assert restored.evidence_selector.raw_conditioning_gains is not None
    assert torch.equal(restored.state_dict()["evidence_selector.observable_logits"],
                       kernel.state_dict()["evidence_selector.observable_logits"])


def test_qres_context_budget_calibration_fixes_mass_and_preserves_gradients() -> None:
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_gate_calibration="context_budget",
            evidence_budget=0.35,
            seed=44,
        )
    )
    logits = torch.tensor(
        [
            [[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [1.5, -0.5, 2.5, 0.5, -1.5, 3.5]],
            [[0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8]],
        ],
        requires_grad=True,
    )
    attention_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, True, False]]
    )
    subject_mask = torch.tensor(
        [[True, False, False, False, False, False], [True, False, False, False, False, False]]
    )
    object_mask = torch.tensor(
        [[False, True, False, False, False, False], [False, True, False, False, False, False]]
    )
    scores = selector._evidence_probabilities(
        logits,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )
    shifted_scores = selector._evidence_probabilities(
        logits + 4.0,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )
    context = attention_mask & ~(subject_mask | object_mask)
    weights = context[:, None, :].expand_as(scores)
    count = weights.sum(dim=-1)
    context_mean = (scores * weights).sum(dim=-1) / count

    assert torch.allclose(
        context_mean,
        torch.full_like(context_mean, 0.35),
        atol=1e-6,
    )
    assert torch.allclose(scores[weights], shifted_scores[weights], atol=1e-6)
    assert torch.equal(scores[1, :, -1], torch.zeros(2))

    token_weights = torch.arange(scores.shape[-1], dtype=scores.dtype)
    (scores * token_weights).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0


def test_qres_polarity_magnitude_view_keeps_both_evidence_extremes() -> None:
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            num_qubits=4,
            evidence_view_score_mode="polarity_magnitude",
            seed=45,
        )
    )
    scores = torch.tensor([[[0.5, 0.5, 0.05, 0.35, 0.5, 0.9]]])
    attention_mask = torch.ones(1, 6, dtype=torch.bool)
    subject_mask = torch.tensor([[True, False, False, False, False, False]])
    object_mask = torch.tensor([[False, True, False, False, False, False]])

    magnitude = selector._view_scores(
        scores,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )
    keep = selector.view_weights(
        scores,
        view="keep",
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )
    random_keep = selector.view_weights(
        scores,
        view="random_keep",
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
        random_seed=17,
    )

    assert magnitude[0, 0, 2] > magnitude[0, 0, 3]
    assert magnitude[0, 0, 5] > magnitude[0, 0, 4]
    assert torch.equal(keep[0, 0, :2], torch.ones(2))
    assert torch.allclose(
        keep[0, 0, 2:].sort().values,
        random_keep[0, 0, 2:].sort().values,
    )


def test_qres_positive_excess_bias_only_promotes_high_evidence() -> None:
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=1,
            head_dim=4,
            num_qubits=4,
            intervention_mode="direct_bias",
            direct_bias_mode="positive_excess",
            initial_direct_gain=0.2,
            seed=46,
        )
    )
    with torch.no_grad():
        selector.raw_direct_gains.fill_(-1.0)
    scores = torch.tensor([[[0.5, 0.5, 0.1, 0.4, 0.8]]])
    attention_mask = torch.ones(1, 5, dtype=torch.bool)
    subject_mask = torch.tensor([[True, False, False, False, False]])
    object_mask = torch.tensor([[False, True, False, False, False]])

    bias = selector.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )

    assert torch.equal(bias[0, 0, :4], torch.zeros(4))
    assert bias[0, 0, 4] > 0.0
    assert torch.all(bias >= 0.0)


def test_qres_evidence_weighted_bias_attenuates_low_evidence_suppression() -> None:
    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=1,
        head_dim=4,
        num_qubits=4,
        intervention_mode="direct_bias",
        direct_bias_mode="evidence_weighted_centered",
        initial_direct_gain=0.2,
        seed=47,
    )
    selector = QuantumRelationEvidenceSelector(config)
    centered_selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "direct_bias_mode": "centered"}
        )
    )
    centered_selector.load_state_dict(selector.state_dict())
    scores = torch.tensor([[[0.5, 0.5, 0.1, 0.4, 0.8]]])
    attention_mask = torch.ones(1, 5, dtype=torch.bool)
    subject_mask = torch.tensor([[True, False, False, False, False]])
    object_mask = torch.tensor([[False, True, False, False, False]])

    weighted = selector.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )
    centered = centered_selector.direct_key_bias(
        scores,
        layer_index=0,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )

    assert weighted[0, 0, 2] < 0.0
    assert weighted[0, 0, 4] > 0.0
    assert weighted[0, 0, 2].abs() < centered[0, 0, 2].abs()
    assert weighted[0, 0, 4].abs() < centered[0, 0, 4].abs()


def test_qres_checkpoint_round_trip_preserves_relation_frame(tmp_path) -> None:
    kernel = QuantumRelationAttentionScoreKernel(
        RelationScoreKernelConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            seed=45,
        )
    )
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="relation_frame",
            relation_frame_scale=0.75,
            intervention_mode="direct_bias",
            seed=46,
        )
    )
    kernel.attach_evidence_selector(selector)
    checkpoint = tmp_path / "qres-relation-frame.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)

    assert restored.evidence_selector is not None
    assert (
        restored.evidence_selector.config.evidence_measurement_mode
        == "relation_frame"
    )
    assert restored.evidence_selector.config.relation_frame_scale == 0.75
    assert restored.evidence_selector.raw_conditioning_gains is None


def test_qres_checkpoint_round_trip_preserves_relation_frame_bank(tmp_path) -> None:
    kernel = QuantumRelationAttentionScoreKernel(
        RelationScoreKernelConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            seed=47,
        )
    )
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="relation_frame_bank",
            relation_frame_scale=0.8,
            intervention_mode="direct_bias",
            seed=48,
        )
    )
    kernel.attach_evidence_selector(selector)
    checkpoint = tmp_path / "qres-relation-frame-bank.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)

    assert restored.evidence_selector is not None
    assert (
        restored.evidence_selector.config.evidence_measurement_mode
        == "relation_frame_bank"
    )
    assert restored.evidence_selector.config.relation_frame_scale == 0.8
    assert restored.evidence_selector.observable_logits.shape[-1] == 8
    assert torch.equal(
        restored.state_dict()["evidence_selector.observable_logits"],
        kernel.state_dict()["evidence_selector.observable_logits"],
    )


def test_qres_checkpoint_round_trip_preserves_coherent_strong_control(tmp_path) -> None:
    kernel = QuantumRelationAttentionScoreKernel(
        RelationScoreKernelConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            seed=49,
        )
    )
    selector = StrongClassicalRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_readout="connected_relation_token",
            evidence_correlation_mode="total",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="relation_frame_coherent",
            max_frame_fusion_gain=1.5,
            initial_frame_fusion_gain=0.75,
            intervention_mode="direct_bias",
            seed=50,
        )
    )
    kernel.attach_evidence_selector(selector)
    checkpoint = tmp_path / "qres-coherent-strong-classical.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)

    assert isinstance(
        restored.evidence_selector,
        StrongClassicalRelationEvidenceSelector,
    )
    assert (
        restored.evidence_selector.config.evidence_measurement_mode
        == "relation_frame_coherent"
    )
    assert restored.evidence_selector.config.evidence_correlation_mode == "total"
    assert restored.evidence_selector.config.max_frame_fusion_gain == 1.5
    assert torch.allclose(
        restored.evidence_selector.frame_fusion_gains(0),
        torch.full((2,), 0.75),
        atol=1e-6,
    )


def test_qres_dual_task_readout_is_positive_matched_and_resource_neutral() -> None:
    from q_attention.plugins import ClassicalRelationEvidenceSelector

    config = RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        depth=2,
        evidence_gate_calibration="context_budget",
        evidence_budget=0.35,
        evidence_task_readout="dual",
        evidence_readout="connected_relation_token",
        evidence_correlation_mode="phase_selective",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="entanglement_phase_offset",
        intervention_mode="direct_bias",
        seed=61,
    )
    quantum = QuantumRelationEvidenceSelector(config)
    classical = ClassicalRelationEvidenceSelector(config)
    strong_classical = StrongClassicalRelationEvidenceSelector(config)
    assert {
        sum(parameter.numel() for parameter in selector.parameters())
        for selector in (quantum, classical, strong_classical)
    } == {92}

    weights = quantum.sufficiency_observable_weights(0).reshape(2, 2, -1)
    assert torch.all(weights >= 0.0)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights[..., 0]),
        atol=1e-6,
    )

    torch.manual_seed(62)
    key, attention, subject, object_ = _evidence_inputs()
    steering, sufficiency = quantum.token_readouts(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    context = attention & ~(subject | object_)
    context_weights = context[:, None, :].expand_as(sufficiency)
    context_count = context_weights.sum(dim=-1)
    sufficiency_mean = (
        (sufficiency * context_weights).sum(dim=-1) / context_count
    )
    assert torch.allclose(
        sufficiency_mean,
        torch.full_like(sufficiency_mean, 0.35),
        atol=1e-6,
    )
    assert not torch.allclose(steering, sufficiency)

    token_axis = torch.arange(key.shape[2], dtype=key.dtype).view(1, 1, -1)
    loss = (steering * token_axis).sum() + (
        sufficiency * token_axis.flip(-1)
    ).sum()
    loss.backward()
    assert quantum.observable_logits.grad is not None
    assert quantum.observable_logits.grad.abs().sum() > 0.0
    assert quantum.sufficiency_observable_logits is not None
    assert quantum.sufficiency_observable_logits.grad is not None
    assert quantum.sufficiency_observable_logits.grad.abs().sum() > 0.0
    assert quantum.raw_sufficiency_sharpness is not None
    assert quantum.raw_sufficiency_sharpness.grad is not None
    assert quantum.raw_sufficiency_sharpness.grad.abs().sum() > 0.0

    separable = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            **{**config.__dict__, "cross_entanglement": False}
        )
    )
    separable_steering, separable_sufficiency = separable.token_readouts(
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    assert torch.allclose(
        separable_steering.masked_select(context_weights),
        torch.full_like(
            separable_steering.masked_select(context_weights),
            0.5,
        ),
        atol=1e-6,
    )
    assert torch.allclose(
        separable_sufficiency.masked_select(context_weights),
        torch.full_like(
            separable_sufficiency.masked_select(context_weights),
            0.35,
        ),
        atol=1e-6,
    )

    metadata = quantum.metadata()["task_readout"]
    assert metadata["shared_state_preparation"] is True
    assert metadata["additional_state_preparations_per_token"] == 0
    assert metadata["additional_measurement_circuits_per_frame"] == 0
    assert metadata["steering_bank"] == "signed_phase_sensitive"
    assert metadata["sufficiency_bank"] == "positive_connected_projectors"

    with pytest.raises(ValueError, match="connected_relation_token"):
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            evidence_task_readout="dual",
        )


def test_qres_dual_task_readout_checkpoint_and_kernel_reuse(
    tmp_path,
    monkeypatch,
) -> None:
    selector = QuantumRelationEvidenceSelector(
        RelationEvidenceSelectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            evidence_gate_calibration="context_budget",
            evidence_task_readout="dual",
            evidence_readout="connected_relation_token",
            evidence_correlation_mode="phase_selective",
            evidence_weight_mode="signed_centered_l1",
            evidence_measurement_mode="entanglement_phase_offset",
            intervention_mode="direct_bias",
            seed=63,
        )
    )
    kernel = QuantumRelationAttentionScoreKernel(
        RelationScoreKernelConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=4,
            seed=64,
        )
    )
    kernel.attach_evidence_selector(selector)
    query = torch.randn(2, 2, 6, 4)
    key, attention, subject, object_ = _evidence_inputs()

    calls = 0
    original = selector.token_readouts

    def counted_readouts(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(selector, "token_readouts", counted_readouts)
    expected = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        evidence_view="keep",
    )
    assert calls == 1

    checkpoint = tmp_path / "qres-dual-readout.pt"
    save_relation_attention_score_kernel_checkpoint(checkpoint, kernel)
    restored, _ = load_relation_attention_score_kernel_checkpoint(checkpoint)
    assert restored.evidence_selector is not None
    assert restored.evidence_selector.config.evidence_task_readout == "dual"
    actual = restored(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        evidence_view="keep",
    )
    assert torch.equal(actual, expected)
