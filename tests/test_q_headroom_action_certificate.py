from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_headroom_action_certificate import (  # noqa: E402
    FIXED_SEEDS,
    balanced_geometry_certificate,
    current_certificate,
    rescue_bank_certificate,
    routing_certificate,
)


def test_current_certificate_rejects_one_correction_headroom() -> None:
    payload = {
        "results": [
            {"seed": seed, "oracle_headroom_vs_best_restricted": headroom}
            for seed, headroom in zip(FIXED_SEEDS, (1, 1, 0, 1, 1))
        ]
    }
    result = current_certificate(payload)
    assert result["status"] == "fail"
    assert result["mean_headroom"] == 0.8
    assert result["failure_class"] == "insufficient_headroom"


def test_routing_certificate_rejects_invalid_baseline() -> None:
    payload = {
        "gate": {
            "marginals_exact": True,
            "baseline_accuracy_parity": False,
            "oracle_correct_retention": True,
            "oracle_residual_invariants": True,
            "baseline_accuracy_gap": -0.2291667,
        }
    }
    result = routing_certificate(payload)
    assert result["status"] == "fail"
    assert result["failure_class"] == "invalid_baseline"


def test_rescue_certificate_combines_validity_and_headroom() -> None:
    payload = {
        "gate": {
            "split_invariants": True,
            "baseline_accuracy_parity": True,
            "oracle_correct_retention": True,
            "oracle_residual_invariants": True,
        },
        "results": [{"oracle_headroom_vs_best_restricted": 0}],
    }
    result = rescue_bank_certificate(payload)
    assert result["status"] == "fail"
    assert result["conditions"]["baseline_validity"]
    assert not result["conditions"]["minimum_headroom_two"]


def test_balanced_geometry_requires_fixed_seed_classical_slack() -> None:
    rows = []
    for seed in FIXED_SEEDS:
        rows.extend(
            [
                {
                    "selector": "q_causal_transport",
                    "seed": seed,
                    "context_target_mass_gain": 0.1,
                    "target_counterfactual_influence_gain": 0.08,
                },
                {
                    "selector": "classical_causal_transport",
                    "seed": seed,
                    "context_target_mass_gain": 0.0,
                    "target_counterfactual_influence_gain": 0.0,
                },
            ]
        )
    rows[1]["context_target_mass_gain"] = 0.11
    result = balanced_geometry_certificate({"seeds": list(FIXED_SEEDS), "results": rows})
    assert result["status"] == "fail"
    assert not result["conditions"]["classical_slack_mass_all_seeds"]
