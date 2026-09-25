from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_intervention_identifiability_audit import (  # noqa: E402
    ActionSpec,
    build_action_specs,
    fit_ridge,
    predict_ridge,
    zero_sum_action_residual,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(8)[None, :].repeat(2, 1),
        "attention_mask": torch.ones(2, 8, dtype=torch.bool),
        "subject_mask": torch.tensor(
            [[False, True, False, False, False, False, False, False]] * 2
        ),
        "object_mask": torch.tensor(
            [[False, False, True, False, False, False, False, False]] * 2
        ),
        "labels": torch.tensor([0, 1]),
    }


def test_action_basis_is_complete_and_deterministic() -> None:
    specs = build_action_specs(2, 8, (0, 3, 4, 5, 6, 7))
    assert len(specs) == 2 * 8 * 6 * 2
    assert specs[0] == ActionSpec(0, 0, 0, -1)
    assert specs[-1] == ActionSpec(1, 7, 7, 1)
    assert specs == build_action_specs(2, 8, (0, 3, 4, 5, 6, 7))


def test_action_residual_is_context_only_zero_sum_and_bounded() -> None:
    batch = _batch()
    spec = ActionSpec(layer=0, query=3, key=4, sign=1)
    residual = zero_sum_action_residual(batch, spec, 2, 2.0, torch.float32)
    assert residual.shape == (2, 2, 8, 8)
    assert float(residual.abs().max()) <= 2.0
    assert float(residual[:, :, 3, :].sum(dim=-1).abs().max()) < 1e-6
    assert float(residual[..., 1].abs().max()) == 0.0
    assert float(residual[..., 2].abs().max()) == 0.0
    untouched = torch.cat((residual[:, :, :3], residual[:, :, 4:]), dim=2)
    assert float(untouched.abs().max()) == 0.0


def test_fixed_ridge_recovers_linear_utility() -> None:
    features = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [2.0, 1.0]],
            [[2.0, 0.0], [3.0, 1.0]],
        ]
    )
    target = 2.0 * features[..., 0] - 0.5 * features[..., 1] + 0.25
    model = fit_ridge(features, target, 1e-6)
    predicted = predict_ridge(model, features)
    assert torch.allclose(predicted, target, atol=1e-4)


def test_fixed_ridge_handles_constant_and_duplicate_columns() -> None:
    features = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 2.0], [1.0, 1.0, 3.0]],
        ]
    )
    target = features[..., 2] - 0.5
    model = fit_ridge(features, target, 1e-3)
    predicted = predict_ridge(model, features)
    assert torch.isfinite(predicted).all()
    assert torch.allclose(predicted, target, atol=5e-3)
