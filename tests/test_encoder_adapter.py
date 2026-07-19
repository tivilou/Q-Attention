from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from q_attention.adapters import EncoderKeySteeringAdapter, KeySteeringHookConfig, resolve_module
from q_attention.spans import batched_span_mask


class ToyEncoderLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.last_keys: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        keys = self.key_proj(hidden)
        self.last_keys = keys.detach().clone()
        return keys


class ToyEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyEncoderLayer(dim) for _ in range(num_layers)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_resolve_module_supports_module_list_index() -> None:
    model = ToyEncoder(4)
    assert resolve_module(model, "layers.0.key_proj") is model.layers[0].key_proj


def test_resolve_module_rejects_bad_path() -> None:
    model = ToyEncoder(4)
    with pytest.raises(ValueError):
        resolve_module(model, "layers.1.key_proj")


def test_encoder_adapter_changes_only_masked_keys() -> None:
    torch.manual_seed(11)
    model = ToyEncoder(6)
    hidden = torch.randn(2, 5, 6)
    projector = torch.eye(6)
    mask = batched_span_mask(2, 5, [[(1, 3)], [(4, 5)]])

    baseline = model(hidden)
    baseline_keys = model.layers[0].last_keys

    adapter = EncoderKeySteeringAdapter(model, ["layers.0.key_proj"])
    config = KeySteeringHookConfig(projector=projector, mask=mask, gain=0.5)
    with adapter.steering(config):
        steered = model(hidden)
        steered_keys = model.layers[0].last_keys

    assert not adapter.attached
    assert steered.shape == baseline.shape
    assert not torch.allclose(steered_keys[mask], baseline_keys[mask])
    assert torch.allclose(steered_keys[~mask], baseline_keys[~mask])


def test_adapter_remove_restores_model_output() -> None:
    torch.manual_seed(19)
    model = ToyEncoder(4)
    hidden = torch.randn(1, 3, 4)
    projector = torch.eye(4)
    mask = batched_span_mask(1, 3, [[(0, 1)]])
    adapter = EncoderKeySteeringAdapter(model, ["layers.0.key_proj"])

    baseline = model(hidden)
    adapter.attach(KeySteeringHookConfig(projector=projector, mask=mask, gain=1.0))
    changed = model(hidden)
    adapter.remove()
    restored = model(hidden)

    assert not torch.allclose(changed, baseline)
    assert torch.allclose(restored, baseline)


def test_encoder_adapter_applies_path_specific_projectors() -> None:
    torch.manual_seed(23)
    model = ToyEncoder(4, num_layers=2)
    hidden = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    paths = ["layers.0.key_proj", "layers.1.key_proj"]
    adapter = EncoderKeySteeringAdapter(model, paths)
    baseline = model(hidden)

    config = KeySteeringHookConfig(
        projector={paths[0]: torch.zeros(4, 4), paths[1]: torch.eye(4)},
        mask=mask,
        gain=0.5,
    )
    with adapter.steering(config):
        steered = model(hidden)

    assert torch.allclose(steered, 1.5 * baseline, atol=1e-6)


def test_encoder_adapter_rejects_incomplete_layer_projector_mapping() -> None:
    model = ToyEncoder(4, num_layers=2)
    adapter = EncoderKeySteeringAdapter(model, ["layers.0.key_proj", "layers.1.key_proj"])
    config = KeySteeringHookConfig(
        projector={"layers.0.key_proj": torch.eye(4)},
        mask=torch.ones(1, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="missing"):
        adapter.attach(config)


def test_encoder_adapter_applies_path_specific_gains() -> None:
    torch.manual_seed(29)
    model = ToyEncoder(4, num_layers=2)
    hidden = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    paths = ["layers.0.key_proj", "layers.1.key_proj"]
    adapter = EncoderKeySteeringAdapter(model, paths)
    baseline = model(hidden)

    config = KeySteeringHookConfig(
        projector={path: torch.eye(4) for path in paths},
        mask=mask,
        gain={paths[0]: 0.0, paths[1]: 0.5},
    )
    with adapter.steering(config):
        steered = model(hidden)

    assert torch.allclose(steered, 1.5 * baseline, atol=1e-6)


def test_encoder_adapter_rejects_incomplete_layer_gain_mapping() -> None:
    model = ToyEncoder(4, num_layers=2)
    paths = ["layers.0.key_proj", "layers.1.key_proj"]
    adapter = EncoderKeySteeringAdapter(model, paths)
    config = KeySteeringHookConfig(
        projector={path: torch.eye(4) for path in paths},
        mask=torch.ones(1, 2, dtype=torch.bool),
        gain={paths[0]: 0.1},
    )

    with pytest.raises(ValueError, match="layer gain paths"):
        adapter.attach(config)
