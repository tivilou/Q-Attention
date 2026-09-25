from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SRC = ROOT / "src"
for path in (EXPERIMENTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_q_lass_attention_alignment_audit as audit


def test_config_is_bound_to_frozen_q_lass_protocol() -> None:
    config = audit.load_config(
        ROOT / "configs/q_lass_attention_alignment_audit.json"
    )
    assert tuple(config["source_experiment"]["seeds"]) == (7, 11, 13, 17, 23)
    assert tuple(config["source_experiment"]["splits"]) == ("valid", "test")
    assert tuple(config["source_experiment"]["selectors"]) == audit.AUDIT_SELECTORS
    assert config["device"]["required_type"] == "cuda"


def test_alignment_metrics_detect_evidence_localization_without_labels() -> None:
    before = torch.softmax(torch.tensor([[[0.0, 0.0, 2.0, 0.0]]]), dim=-1)
    after = torch.softmax(torch.tensor([[[2.0, 2.0, 0.0, 0.0]]]), dim=-1)
    result = audit.attention_alignment_metrics(
        before,
        after,
        evidence_slot=torch.tensor([[[0, 1]]]),
        bad_slot=torch.tensor([[2]]),
        active=torch.tensor([[True]]),
        baseline_correct=torch.tensor([[False]]),
        movement_tolerance=1e-8,
    )
    assert result["evidence_mass_delta"] > 0.0
    assert result["distractor_mass_delta"] < 0.0
    assert result["evidence_minus_distractor_margin_delta"] > 0.0
    assert result["evidence_top2_recall_delta"] > 0.0
    assert result["mean_evidence_rank_delta"] < 0.0
    assert result["harmful_movement_rate"] == 0.0


def test_alignment_metrics_count_harmful_movement() -> None:
    before = torch.softmax(torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]), dim=-1)
    after = torch.softmax(torch.tensor([[[0.0, 0.0, 2.0, 0.0]]]), dim=-1)
    result = audit.attention_alignment_metrics(
        before,
        after,
        evidence_slot=torch.tensor([[[0, 1]]]),
        bad_slot=torch.tensor([[2]]),
        active=torch.tensor([[True]]),
        baseline_correct=torch.tensor([[True]]),
        movement_tolerance=1e-8,
    )
    assert result["harmful_movement_queries"] == 1
    assert result["harmful_movement_rate"] == 1.0
    assert result["baseline_correct_harmful_movement_rate"] == 1.0


def test_disabled_selector_is_zero_action_and_alignment_neutral() -> None:
    split = {
        "labels": torch.zeros(2, 2, dtype=torch.long),
    }
    candidate, support, active = audit.select_action("disabled", None, split)
    assert candidate.shape == (2, 2)
    assert support.shape == (2, 2, audit.canary.task.v1.EVIDENCE_KEYS)
    assert not active.any()
    assert not support.any()


def test_gate_requires_cuda_replay_and_both_heldout_splits() -> None:
    config = audit.load_config(
        ROOT / "configs/q_lass_attention_alignment_audit.json"
    )
    baseline = {
        "evidence_mass_delta": 0.1,
        "distractor_mass_delta": -0.1,
        "evidence_minus_distractor_margin_delta": 0.2,
        "evidence_top2_recall_delta": 0.1,
        "mean_evidence_rank_delta": -0.1,
        "harmful_movement_rate": 0.0,
    }
    aggregate = {
        split: {
            "q_consensus_quantum": {"alignment": dict(baseline)}
        }
        for split in audit.HELDOUT_SPLITS
    }
    result = audit.alignment_gate(
        aggregate,
        replay_pass=True,
        source_pass=True,
        device_eligible=True,
        config=config,
    )
    assert result["status"] == "pass"
    result = audit.alignment_gate(
        aggregate,
        replay_pass=True,
        source_pass=True,
        device_eligible=False,
        config=config,
    )
    assert result["status"] == "fail"
