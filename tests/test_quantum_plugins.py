from __future__ import annotations

from itertools import combinations

import pytest
import torch

from q_attention.adapters import QuantumPluginHookConfig, QuantumPluginSteeringAdapter
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins import (
    PLUGIN_NAMES,
    ComposableQuantumSteering,
    HeadwiseQuantumProjectorConfig,
    HeadwiseQuantumProjectorPlugin,
    QuantumEvidenceGateConfig,
    QuantumEvidenceGatePlugin,
    QuantumExpertBankConfig,
    QuantumExpertBankPlugin,
    QuantumSteeringContext,
    build_quantum_steering,
    load_quantum_steering_checkpoint,
    save_quantum_steering_checkpoint,
)


def relation_masks(batch: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = torch.ones(batch, tokens, dtype=torch.bool)
    subject = torch.zeros(batch, tokens, dtype=torch.bool)
    object_ = torch.zeros(batch, tokens, dtype=torch.bool)
    subject[:, 0] = True
    object_[:, 2] = True
    return attention, subject, object_


def context(keys: torch.Tensor, layer_index: int = 0) -> QuantumSteeringContext:
    attention, subject, object_ = relation_masks(keys.shape[0], keys.shape[1])
    return QuantumSteeringContext(
        keys=keys,
        layer_index=layer_index,
        attention_mask=attention,
        steering_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )


def assert_projector(projector: torch.Tensor, rank: int) -> None:
    assert torch.allclose(projector, projector.transpose(-1, -2), atol=1e-5)
    assert torch.allclose(projector @ projector, projector, atol=1e-5)
    assert torch.allclose(
        torch.diagonal(projector, dim1=-2, dim2=-1).sum(dim=-1),
        torch.full(projector.shape[:-2], float(rank)),
        atol=1e-5,
    )


def test_headwise_quantum_plugin_builds_exact_projectors() -> None:
    plugin = HeadwiseQuantumProjectorPlugin(
        HeadwiseQuantumProjectorConfig(
            num_layers=2,
            num_heads=2,
            head_dim=4,
            depth=2,
            rank=2,
        )
    )

    projectors = plugin.projectors(layer_index=1)

    assert projectors.shape == (2, 4, 4)
    assert_projector(projectors, rank=2)


def test_evidence_gate_is_bounded_and_token_specific() -> None:
    torch.manual_seed(7)
    keys = torch.randn(2, 5, 8)
    plugin = QuantumEvidenceGatePlugin(
        QuantumEvidenceGateConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_qubits=3,
        )
    )

    gates = plugin.gates(context(keys))

    assert gates.shape == keys.shape
    assert torch.all(gates <= 1.0 + 1e-6)
    assert torch.all(gates >= -1.0 - 1e-6)
    assert not torch.allclose(gates[:, 0], gates[:, 1])


def test_evidence_gate_supports_non_power_of_two_head_dimensions() -> None:
    keys = torch.randn(2, 4, 12)
    plugin = QuantumEvidenceGatePlugin(
        QuantumEvidenceGateConfig(
            num_layers=1,
            num_heads=2,
            head_dim=6,
            num_qubits=3,
        )
    )

    gates = plugin.gates(context(keys))

    assert gates.shape == keys.shape


def test_expert_bank_projectors_and_router_are_quantum_normalized() -> None:
    torch.manual_seed(11)
    keys = torch.randn(3, 4, 8)
    plugin = QuantumExpertBankPlugin(
        QuantumExpertBankConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_experts=3,
            router_qubits=3,
            rank=2,
        )
    )
    plugin_context = context(keys)

    projectors = plugin.expert_projectors(layer_index=0)
    weights = plugin.routing_weights(plugin_context)

    assert projectors.shape == (3, 2, 4, 4)
    assert_projector(projectors, rank=2)
    assert weights.shape == (3, 2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 2), atol=1e-6)
    assert torch.all(weights > 0)


@pytest.mark.parametrize(
    "plugin_names",
    [
        names
        for count in range(4)
        for names in combinations(PLUGIN_NAMES, count)
    ],
)
def test_every_plugin_combination_is_finite_and_differentiable(plugin_names) -> None:
    torch.manual_seed(13)
    keys = torch.randn(2, 4, 16, requires_grad=True)
    attention, subject, object_ = relation_masks(2, 4)
    steering = build_quantum_steering(
        plugin_names,
        num_layers=1,
        num_heads=2,
        head_dim=8,
    )

    output = steering(
        keys,
        layer_index=0,
        attention_mask=attention,
        steering_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    assert output.shape == keys.shape
    assert torch.isfinite(output).all()
    if plugin_names:
        assert not torch.allclose(output, keys)
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in steering.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
    else:
        assert torch.equal(output, keys)


def test_builder_seed_controls_plugin_initialization_and_metadata() -> None:
    first = build_quantum_steering(
        "headwise_projector,evidence_gate,expert_bank",
        num_layers=1,
        num_heads=2,
        head_dim=8,
        seed=13,
    )
    second = build_quantum_steering(
        "headwise_projector,evidence_gate,expert_bank",
        num_layers=1,
        num_heads=2,
        head_dim=8,
        seed=29,
    )

    assert not torch.equal(first.plugins[0].angles, second.plugins[0].angles)
    assert not torch.equal(
        first.plugins[1].feature_projection,
        second.plugins[1].feature_projection,
    )
    first_seeds = [item["config"]["seed"] for item in first.metadata()["plugins"]]
    second_seeds = [item["config"]["seed"] for item in second.metadata()["plugins"]]
    assert first_seeds == [13, 14, 15]
    assert second_seeds == [29, 30, 31]


def test_composer_changes_only_selected_tokens() -> None:
    torch.manual_seed(17)
    keys = torch.randn(1, 4, 8)
    mask = torch.tensor([[False, True, False, False]])
    plugin = HeadwiseQuantumProjectorPlugin(
        HeadwiseQuantumProjectorConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            rank=2,
        )
    )
    steering = ComposableQuantumSteering([plugin])

    output = steering(keys, layer_index=0, steering_mask=mask)

    assert not torch.allclose(output[mask], keys[mask])
    assert torch.equal(output[~mask], keys[~mask])


def test_composer_rejects_plugins_with_incompatible_dimensions() -> None:
    first = HeadwiseQuantumProjectorPlugin(
        HeadwiseQuantumProjectorConfig(1, 2, 4, rank=2)
    )
    second = QuantumEvidenceGatePlugin(
        QuantumEvidenceGateConfig(2, 2, 4, num_qubits=3)
    )

    with pytest.raises(ValueError, match="same model dimensions"):
        ComposableQuantumSteering([first, second])


def test_plugin_adapter_attaches_without_modifying_relation_model() -> None:
    torch.manual_seed(19)
    config = RelationTransformerConfig(
        vocab_size=24,
        num_labels=3,
        dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    )
    model = RelationExtractionModel(config)
    steering = build_quantum_steering(
        "headwise_projector,evidence_gate,expert_bank",
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.dim // config.num_heads,
    )
    adapter = QuantumPluginSteeringAdapter(model, model.key_module_paths, steering)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention, subject, object_ = relation_masks(2, 5)

    baseline = model(input_ids, attention, subject, object_)
    hook_config = QuantumPluginHookConfig(
        attention_mask=attention,
        steering_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    with adapter.steering(hook_config):
        steered = model(input_ids, attention, subject, object_)
    restored = model(input_ids, attention, subject, object_)

    assert not adapter.attached
    assert steered.shape == baseline.shape
    assert not torch.allclose(steered, baseline)
    assert torch.allclose(restored, baseline)


def test_task_loss_updates_plugins_while_backbone_stays_frozen() -> None:
    torch.manual_seed(23)
    config = RelationTransformerConfig(
        vocab_size=20,
        num_labels=3,
        dim=16,
        num_layers=1,
        num_heads=2,
        ff_dim=24,
        dropout=0.0,
    )
    model = RelationExtractionModel(config)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    steering = build_quantum_steering(
        "headwise_projector,evidence_gate",
        num_layers=1,
        num_heads=2,
        head_dim=8,
    )
    adapter = QuantumPluginSteeringAdapter(model, model.key_module_paths, steering)
    optimizer = torch.optim.Adam(steering.parameters(), lr=0.05)
    input_ids = torch.randint(0, config.vocab_size, (6, 5))
    attention, subject, object_ = relation_masks(6, 5)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    hook_config = QuantumPluginHookConfig(
        attention_mask=attention,
        steering_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    backbone_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    losses: list[float] = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        with adapter.steering(hook_config):
            logits = model(input_ids, attention, subject, object_)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        losses.append(float(loss.detach().item()))
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]
    assert all(
        torch.equal(value, backbone_before[name])
        for name, value in model.state_dict().items()
    )


def test_plugin_checkpoint_round_trip_is_independent_of_backbone(tmp_path) -> None:
    torch.manual_seed(29)
    keys = torch.randn(2, 4, 16)
    attention, subject, object_ = relation_masks(2, 4)
    steering = build_quantum_steering(
        "headwise_projector,evidence_gate,expert_bank",
        num_layers=1,
        num_heads=2,
        head_dim=8,
    )
    expected = steering(
        keys,
        layer_index=0,
        attention_mask=attention,
        steering_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    checkpoint = tmp_path / "plugins.pt"

    save_quantum_steering_checkpoint(
        checkpoint,
        steering,
        extra_metadata={"base_model": "frozen"},
    )
    restored, metadata = load_quantum_steering_checkpoint(checkpoint)
    actual = restored(
        keys,
        layer_index=0,
        attention_mask=attention,
        steering_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )

    assert restored.active_plugin_names == steering.active_plugin_names
    assert metadata == {"base_model": "frozen"}
    assert torch.allclose(actual, expected)
