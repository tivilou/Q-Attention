from __future__ import annotations

import pytest
import torch

from q_attention import SpectralProjectorConfig, apply_key_steering, batched_span_mask, build_projector, span_mask


def test_span_mask_marks_half_open_spans() -> None:
    mask = span_mask(6, [(1, 3), (4, 5)])
    assert mask.tolist() == [False, True, True, False, True, False]


def test_batched_span_mask_shape() -> None:
    mask = batched_span_mask(2, 5, [[(0, 2)], [(3, 5)]])
    assert mask.shape == (2, 5)
    assert mask.sum().item() == 4


def test_build_projector_shape_and_symmetry() -> None:
    torch.manual_seed(3)
    source = torch.randn(16, 8)
    target = source + 0.1 * torch.randn(16, 8)
    projector = build_projector(source, target, config=SpectralProjectorConfig(rank=3))
    assert projector.shape == (8, 8)
    assert torch.allclose(projector, projector.transpose(0, 1), atol=1e-5)


def test_key_steering_only_changes_masked_positions() -> None:
    torch.manual_seed(5)
    keys = torch.randn(2, 4, 6)
    basis = torch.linalg.qr(torch.randn(6, 3)).Q
    projector = build_projector(basis=basis, singular_values=torch.ones(3), config=SpectralProjectorConfig(rank=2))
    mask = torch.tensor([[True, False, False, True], [False, True, False, False]])

    result = apply_key_steering(keys, projector, mask=mask, gain=0.25, return_delta=True)

    assert result.changed_count == 3
    assert result.keys.shape == keys.shape
    assert not torch.allclose(result.keys[mask], keys[mask])
    assert torch.allclose(result.keys[~mask], keys[~mask])


def test_invalid_span_raises() -> None:
    with pytest.raises(ValueError):
        span_mask(4, [(3, 3)])
