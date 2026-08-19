from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SCRIPTS = ROOT / "scripts"
for path in (EXPERIMENTS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_q_consensus_quantum_estimator_frozen_multiseed as runner
import summarize_q_consensus_quantum_estimator_frozen_multiseed as summarize


def load_config() -> dict:
    return runner.load_config(
        ROOT / "configs/q_consensus_quantum_estimator_frozen_multiseed.json"
    )


def test_protocol_is_exactly_frozen() -> None:
    config = load_config()
    assert tuple(config["seeds"]) == runner.FROZEN_SEEDS == (7, 11, 13, 17, 23)
    assert tuple(config["selectors"]) == runner.FROZEN_SELECTORS
    assert config["training"]["steps"] == 120
    assert config["estimator"]["register_qubits"] == 3
    assert config["aggregate_gate"]["required_seed_count"] == 5


def test_exact_paired_sign_flip_reports_zero_p_for_strict_positive_differences() -> None:
    result = summarize.paired_sign_flip([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result["permutations"] == 32
    assert result["greater_p"] == 1 / 32
    assert result["two_sided_p"] == 2 / 32


def test_confidence_interval_is_student_t_and_uses_sample_std() -> None:
    result = summarize.stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["n"] == 5
    assert result["std"] > 0.0
    assert result["ci95"]["method"].startswith("two-sided Student-t")
    assert result["ci95"]["lower"] < result["mean"] < result["ci95"]["upper"]


def test_dry_run_is_explicitly_non_formal() -> None:
    assert "--dry-run" in runner.parse_args.__code__.co_consts
