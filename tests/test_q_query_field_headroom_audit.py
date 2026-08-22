from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_query_field_headroom_audit_toy import (  # noqa: E402
    FIXED_SEEDS,
    SELECTORS,
    headroom_gate,
)


def _config() -> dict:
    return {
        "gate": {
            "minimum_headroom_corrections": 2,
            "required_seed_fraction": 0.8,
            "max_correct_to_wrong_rate": 0.02,
        }
    }


def _row(seed: int, headroom: int, *, safe: bool = True) -> dict:
    return {
        "seed": seed,
        "oracle_headroom_vs_best_restricted": headroom,
        "results": {
            "query_field_oracle": {
                "correct_to_wrong_rate": 0.0 if safe else 0.03,
                "residual_invariants": safe,
            }
        },
    }


def test_checked_in_config_locks_selectors_and_seeds() -> None:
    payload = json.loads(
        (ROOT / "configs/q_query_field_headroom_audit_toy.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(payload["selectors"]) == SELECTORS
    assert tuple(payload["seeds"]) == FIXED_SEEDS


def test_headroom_gate_requires_four_of_five_fixed_seeds() -> None:
    rows = [_row(seed, 2 if index < 4 else 1) for index, seed in enumerate(FIXED_SEEDS)]
    gate = headroom_gate(rows, _config())
    assert gate["status"] == "pass"
    assert gate["eligible_seed_count"] == 4
    assert gate["candidate_transport_basis_authorized"] is True


def test_headroom_gate_fails_for_low_headroom_or_invalid_oracle() -> None:
    low = [_row(seed, 1) for seed in FIXED_SEEDS]
    assert headroom_gate(low, _config())["status"] == "fail"
    invalid = [_row(seed, 2, safe=seed != 13) for seed in FIXED_SEEDS]
    gate = headroom_gate(invalid, _config())
    assert gate["status"] == "fail"
    assert gate["diagnostic_split_redesign_required"] is True
