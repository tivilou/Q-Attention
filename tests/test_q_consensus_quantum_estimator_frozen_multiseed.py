from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


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


def test_each_gpu_has_one_serial_seed_queue() -> None:
    assignments = [
        {"seed": 7, "gpu_id": 0},
        {"seed": 11, "gpu_id": 1},
        {"seed": 13, "gpu_id": 0},
        {"seed": 17, "gpu_id": 1},
        {"seed": 23, "gpu_id": 0},
    ]
    commands = [[str(item["seed"])] for item in assignments]
    schedules = runner.build_gpu_schedules(assignments, commands)
    assert schedules == {
        0: [["7"], ["13"], ["23"]],
        1: [["11"], ["17"]],
    }


def test_formal_multiseed_accepts_only_a_passed_multigpu_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    summary_path = tmp_path / "runs/preflight/run_summary.json"
    summary_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": "q-attention.q-consensus-quantum-estimator-single-seed-multigpu.v1",
        "formal_preflight": True,
        "seed": 7,
        "config_sha256": "config-hash",
        "provenance": {"git_commit": "commit", "git_dirty": False},
        "parallelism": {
            "type": "within_seed_stage_parallel",
            "ddp": False,
            "physical_gpu_ids": [0, 1],
            "stage_time_overlap_seconds": 4.5,
        },
        "stage_execution": [{"success": True}, {"success": True}],
        "gate": {"status": "pass", "next_multi_seed_authorized": True},
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.validate_multigpu_preflight(
        summary_path,
        expected_commit="commit",
        expected_config_sha256="config-hash",
    )
    assert result["physical_gpu_ids"] == [0, 1]
    assert result["gate_status"] == "pass"

    payload["parallelism"]["stage_time_overlap_seconds"] = 0.0
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="concurrent stages"):
        runner.validate_multigpu_preflight(
            summary_path,
            expected_commit="commit",
            expected_config_sha256="config-hash",
        )
