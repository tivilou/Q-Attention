from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_counterbalanced_routing_headroom_audit_toy as routing  # noqa: E402
import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
from run_q_routing_training_upper_bound_toy import (  # noqa: E402
    CONDITIONS,
    condition_logits,
    hard_selected_model_logits,
    make_condition_split,
)


def test_hard_selected_readout_preserves_classifier_contract() -> None:
    device = torch.device("cpu")
    split = routing.make_counterbalanced_split(7, 12, device)
    model = legacy.build_model(7, device)
    logits = hard_selected_model_logits(model, split)
    assert logits.shape == (12, legacy.NUM_LABELS)
    logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_masked_training_split_exposes_only_selected_evidence_position() -> None:
    split = make_condition_split(
        "masked_routing_query", 10007, 96, torch.device("cpu")
    )
    roles = split["routing_role"]
    left, right = routing.EVIDENCE_POSITIONS
    assert torch.equal(split["attention_mask"][:, left], roles == 0)
    assert torch.equal(split["attention_mask"][:, right], roles == 1)
    assert torch.equal(
        split["input_ids"][roles == 0, left],
        2 + split["selected_evidence_label"][roles == 0],
    )
    assert torch.equal(
        split["input_ids"][roles == 1, right],
        2 + split["selected_evidence_label"][roles == 1],
    )


def test_condition_readouts_are_predeclared() -> None:
    assert len(CONDITIONS) == len(set(CONDITIONS))
    for condition in CONDITIONS:
        logits = condition_logits(condition)
        if "hard_selected" in condition:
            assert logits is hard_selected_model_logits
        else:
            assert logits is not hard_selected_model_logits
