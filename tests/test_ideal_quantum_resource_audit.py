from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_ideal_quantum_resource_audit import (  # noqa: E402
    audit,
    load_config,
    qlass_model,
    qvres_model,
)


def config() -> dict:
    return load_config(ROOT / "configs" / "q_ideal_quantum_resource_audit.json")


def test_qlass_counts_match_frozen_architecture() -> None:
    model = qlass_model(config()["qlass"])
    assert model.qubits == 3
    assert model.depth == 2
    assert model.trainable_parameters == 48
    assert model.classical_control_parameters == 48
    assert model.state_preparation_calls_per_query_key == 3
    assert model.state_preparation_calls_per_query == 18
    assert model.two_qubit_gates_per_state == 6
    assert model.one_qubit_rotations_per_state == 9
    result = audit(config())["models"]["qlass"]
    assert result["source"].endswith("q_consensus_quantum_estimator.py")
    assert result["evidence"].endswith("2026-08-20-qlass-evidence-card.json")


def test_qvres_counts_match_formal_selector() -> None:
    model = qvres_model(config()["qvres"])
    assert model.qubits == 2
    assert model.depth == 1
    assert model.trainable_parameters == 72
    assert model.classical_control_parameters == 72
    assert model.state_preparation_calls_per_query_key == 2
    assert model.state_preparation_calls_per_query == "num_layers * num_heads * 2 * keys_per_query"
    assert model.two_qubit_gates_per_state == 2
    assert model.ancilla_qubits == 1


def test_audit_blocks_quantum_resource_claims_but_preserves_utility_axes() -> None:
    result = audit(config())
    assert result["status"] == "complete"
    assert result["cross_model_conclusions"]["parameter_advantage"].startswith(
        "not established"
    )
    assert result["models"]["qlass"]["classification"]["ordinary_method_utility_status"] == (
        "reproducible_synthetic_utility"
    )
    assert result["models"]["qvres"]["classification"]["ordinary_method_utility_status"] == (
        "formal_natural_task_negative"
    )
    assert result["models"]["qvres"]["classification"]["claim_ceiling"] == (
        "no_positive_utility_or_resource_claim"
    )
    assert all(
        item["classification"]["resource_advantage_status"]
        == "resource_advantage_not_established"
        for item in result["models"].values()
    )
