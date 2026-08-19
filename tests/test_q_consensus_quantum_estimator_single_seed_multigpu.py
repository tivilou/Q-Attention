from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_consensus_quantum_estimator_single_seed_multigpu as runner


def test_assignments_require_two_distinct_physical_gpus() -> None:
    with pytest.raises(ValueError, match="at least two GPUs"):
        runner.build_assignments([0])
    assignments = runner.build_assignments([3, 5, 7])
    assert assignments == [
        {"stage": "quantum_controls", "gpu_id": 3},
        {"stage": "classical_control", "gpu_id": 5},
    ]


def test_public_cli_cannot_override_seed_training_or_selectors() -> None:
    options = {
        option
        for action in runner.build_parser()._actions
        for option in action.option_strings
    }
    assert "--seed" not in options
    assert "--steps" not in options
    assert "--selectors" not in options
    assert "--gpus" in options
    assert "--dry-run" in options


def test_stage_overlap_requires_actual_concurrency() -> None:
    overlapping = [
        {"started_at_epoch": 10.0, "completed_at_epoch": 20.0},
        {"started_at_epoch": 12.0, "completed_at_epoch": 21.0},
    ]
    sequential = [
        {"started_at_epoch": 10.0, "completed_at_epoch": 11.0},
        {"started_at_epoch": 12.0, "completed_at_epoch": 13.0},
    ]
    assert runner.stage_overlap_seconds(overlapping) == 8.0
    assert runner.stage_overlap_seconds(sequential) == 0.0


def metric(selector: str, delta: float) -> dict:
    return {
        "selector": selector,
        "baseline_accuracy": 0.8,
        "accuracy": 0.8 + delta,
        "accuracy_delta": delta,
        "baseline_wrong_queries": 2,
        "corrected_queries": 1,
        "harmed_correct_queries": 0,
        "wrong_correction_rate": 0.5,
        "harm_rate": 0.0,
        "active_rate": 0.2 if selector != "disabled" else 0.0,
        "active_candidate_accuracy": 1.0 if selector != "disabled" else 0.0,
        "residual_finite": True,
        "residual_zero_sum_error": 0.0,
        "residual_max_abs": 1.0 if selector != "disabled" else 0.0,
    }


def stage_payload(
    stage: str,
    selectors: tuple[str, ...],
    deltas: dict[str, float],
    *,
    kind: str,
    source_hashes: dict[str, str],
) -> dict:
    baseline = {
        split: {"accuracy": 0.8, "replay_error": 0.0, "queries": 10}
        for split in ("train", "calibration", "valid", "test")
    }
    return {
        "schema_version": runner.STAGE_SCHEMA_VERSION,
        "status": "complete",
        "stage": stage,
        "seed": 7,
        "config_sha256": "config-hash",
        "provenance": {
            "git_commit": "commit",
            "git_dirty": False,
            "source_sha256": source_hashes,
        },
        "baseline": baseline,
        "training": {
            kind: {
                "gradient_norm_min": 0.1,
                "gradient_norm_max": 1.0,
            }
        },
        "estimators": {kind: {"type": kind}},
        "trainable_parameters": {kind: 12},
        "selectors": {
            split: {selector: metric(selector, deltas[selector]) for selector in selectors}
            for split in baseline
        },
    }


def test_combiner_reconstructs_the_original_gate_and_authorizes_multiseed() -> None:
    source_hashes = {"source.py": "hash"}
    quantum = stage_payload(
        "quantum_controls",
        runner.STAGE_SELECTORS["quantum_controls"],
        {
            "disabled": 0.0,
            "q_consensus_quantum": 0.10,
            "q_consensus_shuffled_query": 0.02,
            "q_consensus_magnitude": 0.03,
        },
        kind="quantum",
        source_hashes=source_hashes,
    )
    classical = stage_payload(
        "classical_control",
        runner.STAGE_SELECTORS["classical_control"],
        {"classical_consensus_control": 0.05},
        kind="classical",
        source_hashes=source_hashes,
    )
    execution = [
        {
            "stage": "quantum_controls",
            "gpu_id": 0,
            "started_at_epoch": 10.0,
            "completed_at_epoch": 20.0,
            "success": True,
        },
        {
            "stage": "classical_control",
            "gpu_id": 1,
            "started_at_epoch": 11.0,
            "completed_at_epoch": 19.0,
            "success": True,
        },
    ]
    master_config = runner.frozen.load_config(
        ROOT / "configs/q_consensus_quantum_estimator_frozen_multiseed.json"
    )
    payload = runner.combine_stage_payloads(
        {
            "quantum_controls": quantum,
            "classical_control": classical,
        },
        execution,
        master_config,
        config_path=ROOT / "configs/q_consensus_quantum_estimator_frozen_multiseed.json",
        config_sha256="config-hash",
        git_commit="commit",
        expected_source_hashes=source_hashes,
    )
    assert payload["parallelism"]["type"] == "within_seed_stage_parallel"
    assert payload["parallelism"]["physical_gpu_ids"] == [0, 1]
    assert payload["gate"]["status"] == "pass"
    assert payload["gate"]["next_multi_seed_authorized"] is True
    assert payload["trainable_parameters"]["quantum"] == payload["trainable_parameters"]["classical"]
