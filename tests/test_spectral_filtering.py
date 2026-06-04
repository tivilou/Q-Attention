from __future__ import annotations

import pytest
import torch

from q_attention import SpectralProjectorConfig, singular_weights, spectral_effective_rank, spectral_filter_diagnostics


def test_spectral_effective_rank_bounds() -> None:
    values = torch.tensor([4.0, 2.0, 1.0, 0.0])
    rank = spectral_effective_rank(values)

    assert 1.0 <= rank <= 4.0


def test_hard_topk_diagnostics_reports_active_rank() -> None:
    values = torch.tensor([5.0, 3.0, 1.0, 0.5])
    config = SpectralProjectorConfig(mode="hard_topk", rank=2)
    diagnostics = spectral_filter_diagnostics(values, config)

    assert diagnostics["active_directions"] == 2
    assert diagnostics["top_filter_weights"][:3] == [1.0, 1.0, 0.0]


def test_smooth_filter_weights_are_bounded() -> None:
    values = torch.tensor([5.0, 3.0, 1.0])
    weights = singular_weights(values, SpectralProjectorConfig(mode="high_pass", threshold=0.5, sharpness=6.0))

    assert torch.all(weights >= 0.0)
    assert torch.all(weights <= 1.0)


def test_band_pass_diagnostics_contains_filter_metadata() -> None:
    values = torch.tensor([5.0, 3.0, 1.0])
    config = SpectralProjectorConfig(mode="band_pass", threshold=0.4, sharpness=8.0)
    diagnostics = spectral_filter_diagnostics(values, config)

    assert diagnostics["mode"] == "band_pass"
    assert diagnostics["threshold"] == 0.4
    assert diagnostics["num_singular_values"] == 3
    assert len(diagnostics["top_singular_values"]) == 3


def test_spectral_diagnostics_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        spectral_filter_diagnostics(torch.tensor([]), SpectralProjectorConfig())