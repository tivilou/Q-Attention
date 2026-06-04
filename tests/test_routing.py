from __future__ import annotations

import pytest
import torch

from q_attention import RouterConfig, apply_key_steering, projector_prototype, route_projectors, stack_projector_bank


def test_apply_key_steering_supports_batch_wise_projectors() -> None:
    keys = torch.ones(2, 3, 2)
    projectors = torch.stack([torch.eye(2), 2.0 * torch.eye(2)], dim=0)
    mask = torch.tensor([[True, False, True], [False, True, True]])

    result = apply_key_steering(keys, projectors, mask=mask, gain=1.0, return_delta=True)

    assert result.changed_count == 4
    assert torch.allclose(result.keys[0, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(result.keys[0, 1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(result.keys[1, 1], torch.tensor([3.0, 3.0]))


def test_projector_bank_routes_to_dynamic_projector_shape() -> None:
    names = ["identity", "double"]
    projectors = [torch.eye(3), 2.0 * torch.eye(3)]
    prototypes = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])]
    bank = stack_projector_bank(names, projectors, prototypes)
    anchors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    routed = route_projectors(anchors, bank, RouterConfig(temperature=0.5))

    assert routed.weights.shape == (2, 2)
    assert routed.projectors.shape == (2, 3, 3)
    assert torch.allclose(routed.weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert routed.entropy.shape == (2,)


def test_projector_prototype_uses_projected_anchor_mean() -> None:
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    projector = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

    prototype = projector_prototype(keys, projector)

    assert prototype.shape == (2,)
    assert torch.allclose(prototype.norm(), torch.tensor(1.0), atol=1e-6)
    assert prototype[0] > prototype[1]


def test_projector_bank_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        stack_projector_bank(["one"], [torch.eye(2), torch.eye(2)], [torch.ones(2)])