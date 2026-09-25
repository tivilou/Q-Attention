from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_full_position_evidence_anchor_prescreen import (  # noqa: E402
    _anchor_distribution,
    _evidence_mask,
    _field_metrics,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 12, 11, 5, 19]], dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.bool),
        "subject_mask": torch.tensor([[True, False, False, False, False, False, False, False]]),
        "object_mask": torch.tensor([[False, True, False, False, False, False, False, False]]),
        "labels": torch.tensor([1]),
    }


def test_evidence_mask_is_audit_only_and_excludes_entity_positions() -> None:
    mask = _evidence_mask(_batch())
    assert mask.tolist() == [[False, False, False, False, False, False, True, False]]


def test_anchor_uses_all_context_positions_and_normalizes() -> None:
    batch = _batch()
    scores = torch.zeros(1, 2, 8, 8)
    scores[..., 0, 6] = 4.0
    anchor = _anchor_distribution(scores, batch)
    assert torch.allclose(anchor.sum(dim=-1), torch.ones(1, 2, 8))
    assert float(anchor[..., 6].mean()) > float(anchor[..., 3].mean())


def test_field_metrics_require_normalized_finite_field() -> None:
    batch = _batch()
    field = torch.full((1, 2, 8, 8), 1.0 / 6.0)
    field[..., :2] = 0.0
    field = field / field.sum(dim=-1, keepdim=True)
    metrics = _field_metrics(field, field, batch, 1)
    assert metrics["finite"]
    assert metrics["field_sum_error"] < 1e-6
    assert metrics["anchor_sum_error"] < 1e-6
