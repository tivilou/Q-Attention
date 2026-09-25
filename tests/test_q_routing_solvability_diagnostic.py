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
from run_q_routing_solvability_diagnostic_toy import (  # noqa: E402
    VARIANTS,
    make_variant,
    variant_contract,
)


def _batches() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device("cpu")
    current = legacy.make_split(10007, 96, device)
    routed = routing.make_counterbalanced_split(10007, 96, device)
    return current, routed


def test_variant_contract_is_exact() -> None:
    current, routed = _batches()
    variants = {name: make_variant(current, routed, name) for name in VARIANTS}
    report = variant_contract(current, routed, variants)
    assert report["status"] == "pass"
    assert report["selected_token_histogram"] == report["distractor_token_histogram"]


def test_neutral_and_duplicate_variants_preserve_selected_cue() -> None:
    current, routed = _batches()
    variants = {name: make_variant(current, routed, name) for name in VARIANTS}
    selected, _ = (
        torch.where(
            routed["routing_role"] == 0,
            routed["input_ids"][:, routing.EVIDENCE_POSITIONS[0]],
            routed["input_ids"][:, routing.EVIDENCE_POSITIONS[1]],
        ),
        None,
    )
    left, right = routing.EVIDENCE_POSITIONS
    assert torch.equal(
        variants["duplicate_selected"]["input_ids"][:, left], selected
    )
    assert torch.equal(
        variants["duplicate_selected"]["input_ids"][:, right], selected
    )
    assert torch.equal(
        variants["query_primary_upper_bound"]["input_ids"][:, legacy.EVIDENCE_POS],
        2 + routed["selected_evidence_label"],
    )


def test_masked_variant_masks_only_the_unselected_position() -> None:
    current, routed = _batches()
    variants = {name: make_variant(current, routed, name) for name in VARIANTS}
    left, right = routing.EVIDENCE_POSITIONS
    roles = routed["routing_role"]
    assert torch.equal(variants["masked_distractor"]["attention_mask"][:, left], roles == 0)
    assert torch.equal(variants["masked_distractor"]["attention_mask"][:, right], roles == 1)
    assert torch.equal(
        variants["full_routing"]["attention_mask"], routed["attention_mask"]
    )
