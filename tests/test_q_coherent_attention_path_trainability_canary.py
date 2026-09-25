from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_trainability_canary import (  # noqa: E402
    make_splits,
    promotion_gate,
    split_diagnostics,
    train_selector,
)
from run_q_coherent_attention_path_geometry_audit import (  # noqa: E402
    forward,
    predictions,
)


def config() -> dict:
    return json.loads(
        (
            ROOT / "configs" / "q_coherent_attention_path_trainability_canary.json"
        ).read_text(encoding="utf-8")
    )


def test_train_valid_qk_streams_are_balanced_and_disjoint() -> None:
    train, valid = make_splits(config(), torch.device("cpu"))
    diagnostics = split_diagnostics(train, valid)
    assert diagnostics["train_positive_labels"] * 2 == diagnostics["train_examples"]
    assert diagnostics["valid_positive_labels"] * 2 == diagnostics["valid_examples"]
    assert diagnostics["exact_train_valid_score_overlap"] == 0
    assert diagnostics["train_qk"]["maximum_qk_reconstruction_error"] <= 1e-5
    assert diagnostics["valid_qk"]["maximum_qk_reconstruction_error"] <= 1e-5


def test_fixed_budget_training_updates_only_matched_transport() -> None:
    cfg = config()
    cfg["training"]["steps"] = 8
    train, valid = make_splits(cfg, torch.device("cpu"))
    tolerance = float(cfg["mechanism"]["tie_tolerance"])
    baseline_predictions = {
        "train": predictions(forward(None, train)[2], tolerance),
        "valid": predictions(forward(None, valid)[2], tolerance),
    }
    results = [
        train_selector(selector, train, valid, baseline_predictions, cfg)
        for selector in cfg["selectors"]
    ]
    by_selector = {row["selector"]: row for row in results}
    assert by_selector["disabled"]["trainable_parameters"] == 0
    assert {
        by_selector[name]["trainable_parameters"]
        for name in (
            "q_wap_signed",
            "q_wap_unsigned",
            "classical_wap_diffusion",
        )
    } == {1}
    assert by_selector["q_wap_signed"]["transport_increase"] > 0.0
    assert by_selector["q_wap_signed"]["final_train"]["nll"] < (
        by_selector["q_wap_signed"]["initial_train"]["nll"]
    )
    assert all(row["training_curve"]["finite"] for row in results)


def test_promotion_gate_rejects_control_parity() -> None:
    cfg = config()
    metrics = {
        "accuracy": 0.5,
        "nll": 0.693147,
        "corrected_examples": 0,
        "harmed_correct_examples": 0,
        "deterministic_replay": True,
        "residual_invariants": {"status": "pass"},
    }
    result = {
        "trainable_parameters": 1,
        "transport_increase": 1.0,
        "training_curve": {"finite": True},
        "initial_train": metrics,
        "initial_valid": metrics,
        "final_train": metrics,
        "final_valid": metrics,
    }
    results = [
        {**result, "selector": selector, "trainable_parameters": 0 if selector == "disabled" else 1}
        for selector in cfg["selectors"]
    ]
    diagnostics = {
        "train_examples": 192,
        "valid_examples": 96,
        "train_positive_labels": 96,
        "valid_positive_labels": 48,
        "exact_train_valid_score_overlap": 0,
        "train_qk": {
            "finite_query_key": True,
            "maximum_qk_reconstruction_error": 0.0,
            "maximum_hamiltonian_reconstruction_error": 0.0,
        },
        "valid_qk": {
            "finite_query_key": True,
            "maximum_qk_reconstruction_error": 0.0,
            "maximum_hamiltonian_reconstruction_error": 0.0,
        },
    }
    gate = promotion_gate(results, diagnostics, cfg)
    assert gate["status"] == "fail"
    assert gate["heldout_synthetic_attention_utility_authorized"] is False
