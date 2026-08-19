from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
from run_q_counterbalanced_routing_headroom_audit_toy import (  # noqa: E402
    EVIDENCE_POSITIONS,
    ROLE_MARKERS,
    counterbalanced_invariants,
    make_counterbalanced_split,
    role_ablated_batch,
)


def test_counterbalanced_split_preserves_contract_and_exact_marginals() -> None:
    device = torch.device("cpu")
    current = legacy.make_split(17, 96, device)
    routed = make_counterbalanced_split(17, 96, device)
    report = counterbalanced_invariants(current, routed)
    assert report["status"] == "pass"
    assert report["primary_histogram"] == report["distractor_histogram"]
    assert report["left_histogram"] == report["right_histogram"]
    assert report["role_label_counts"][0] == report["role_label_counts"][1]

    roles = routed["routing_role"]
    selected = 2 + routed["selected_evidence_label"]
    distractor = 2 + routed["distractor_label"]
    left = routed["input_ids"][:, EVIDENCE_POSITIONS[0]]
    right = routed["input_ids"][:, EVIDENCE_POSITIONS[1]]
    assert torch.equal(left[roles == 0], selected[roles == 0])
    assert torch.equal(right[roles == 1], selected[roles == 1])
    assert torch.equal(right[roles == 0], distractor[roles == 0])
    assert torch.equal(left[roles == 1], distractor[roles == 1])


def test_role_ablation_only_flips_independent_query_marker() -> None:
    split = make_counterbalanced_split(23, 96, torch.device("cpu"))
    ablated = role_ablated_batch(split)
    expected = split["input_ids"].clone()
    expected[:, legacy.EVIDENCE_POS] = ROLE_MARKERS[0] + (
        1 - split["routing_role"]
    )
    assert torch.equal(ablated["input_ids"], expected)
    for name, value in split.items():
        if name != "input_ids":
            assert torch.equal(ablated[name], value)


def test_counterbalanced_split_is_deterministic() -> None:
    first = make_counterbalanced_split(1007, 192, torch.device("cpu"))
    second = make_counterbalanced_split(1007, 192, torch.device("cpu"))
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name])
