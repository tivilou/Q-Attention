from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
from run_q_epvg_selector_worker import _case_study_scores


def _cross_device_reference() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("meta")


def test_case_study_trace_tensor_moves_to_score_device_and_dtype() -> None:
    device = _cross_device_reference()
    reference = torch.zeros(2, 1, 3, 3, dtype=torch.float32, device=device)
    captured = torch.ones(2, 1, 3, 3, device="cpu", dtype=torch.float64)

    adjustment, steered_scores = _case_study_scores(captured, reference)

    assert adjustment.device == reference.device
    assert adjustment.dtype == reference.dtype
    assert steered_scores.device == reference.device
    assert steered_scores.dtype == reference.dtype
    assert steered_scores.shape == reference.shape
    if device.type != "meta":
        assert torch.allclose(steered_scores, reference + 1.0)


def test_cpu_case_study_alignment_preserves_values() -> None:
    reference = torch.zeros(2, 1, 3, 3, dtype=torch.float32)
    captured = torch.ones(2, 1, 3, 3, dtype=torch.float64)

    adjustment, steered_scores = _case_study_scores(captured, reference)

    assert adjustment.device == reference.device
    assert adjustment.dtype == reference.dtype
    assert torch.allclose(steered_scores, reference + 1.0)


def test_missing_case_study_trace_tensor_uses_reference_zeros() -> None:
    device = _cross_device_reference()
    reference = torch.zeros(2, 1, 3, 3, dtype=torch.float32, device=device)

    adjustment, steered_scores = _case_study_scores(None, reference)

    assert adjustment.device == reference.device
    assert adjustment.dtype == reference.dtype
    assert steered_scores.device == reference.device
    assert steered_scores.dtype == reference.dtype
    assert steered_scores.shape == reference.shape
    if device.type != "meta":
        assert torch.count_nonzero(adjustment) == 0
        assert torch.allclose(steered_scores, reference)
