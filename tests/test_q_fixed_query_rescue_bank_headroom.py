from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
from run_q_fixed_query_rescue_bank_headroom_toy import (  # noqa: E402
    BANK_POSITIONS,
    make_fixed_query_rescue_bank_split,
    seed7_gate,
    split_invariants,
)


def test_fixed_query_rescue_bank_preserves_per_example_token_multiset() -> None:
    device = torch.device("cpu")
    current = legacy.make_split(17, 96, device)
    bank = make_fixed_query_rescue_bank_split(17, 96, device)
    report = split_invariants(current, bank)
    assert report["status"] == "pass"
    assert report["rescue_bank_position_counts"]
    assert max(report["rescue_bank_position_counts"]) - min(
        report["rescue_bank_position_counts"]
    ) <= 1
    assert BANK_POSITIONS == (legacy.RESCUE_POS, 4, 5)


def test_fixed_query_rescue_bank_is_deterministic_and_query_cue_unchanged() -> None:
    first = make_fixed_query_rescue_bank_split(23, 96, torch.device("cpu"))
    second = make_fixed_query_rescue_bank_split(23, 96, torch.device("cpu"))
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name])
    assert torch.equal(
        first["input_ids"][:, legacy.EVIDENCE_POS],
        legacy.make_split(23, 96, torch.device("cpu"))["input_ids"][:, legacy.EVIDENCE_POS],
    )


def test_non_rescue_examples_keep_bank_position_unset() -> None:
    bank = make_fixed_query_rescue_bank_split(31, 96, torch.device("cpu"))
    assert torch.equal(
        bank["rescue_bank_position"][~bank["rescue_available"]],
        torch.full_like(
            bank["rescue_bank_position"][~bank["rescue_available"]], -1
        ),
    )


def test_seed7_gate_requires_headroom_after_validity_passes() -> None:
    row = {
        "validity_gate": {
            "split_invariants": True,
            "baseline_accuracy_parity": True,
        },
        "oracle_gate": {
            "minimum_headroom": False,
            "oracle_correct_retention": True,
            "oracle_residual_invariants": True,
        },
    }
    gate = seed7_gate(row)
    assert gate["status"] == "fail"
    assert not gate["five_seed_phase_authorized"]
