from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

import pytest
import torch

from q_attention.plugins.q_relation_perturbation_echo_curvature import (
    RelationPerturbationEchoConfig,
    RelationPerturbationEchoCurvatureKernel,
    LocalRelationEchoCurvatureControl,
)

from q_attention.experiments.batch_resume import ResumeCompatibilityError


def _runner_module():
    path = Path(__file__).parents[1] / "experiments/run_retacred_qrpec_formal_single_seed.py"
    spec = importlib.util.spec_from_file_location("qrpec_formal_runner_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adaptive_profile(module):
    return {
        "name": "adaptive",
        "adaptive": True,
        "tiers": [dict(item) for item in module.ADAPTIVE_HARDWARE_PROFILES],
    }


def test_adaptive_tiers_start_all_pairs_then_halve_then_micro_batch() -> None:
    runner = _runner_module()
    tiers = list(runner.ADAPTIVE_HARDWARE_PROFILES)
    assert [tier["pair_chunk_divisor"] for tier in tiers[:11]] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    ]
    assert all(tier["micro_batch_size"] == 256 for tier in tiers[:11])
    assert all(tier["gradient_accumulation_steps"] == 1 for tier in tiers[:11])
    assert [(tier["micro_batch_size"], tier["gradient_accumulation_steps"])
            for tier in tiers[11:]] == [(128, 2), (64, 4), (32, 8)]
    assert all(tier["pair_chunk_size"] is None for tier in tiers)
    assert all(not tier["activation_checkpointing"] for tier in tiers[:11])
    assert all(tier["activation_checkpointing"] for tier in tiers[11:])


def test_new_adaptive_state_starts_with_all_pairs(tmp_path: Path) -> None:
    runner = _runner_module()
    profile = _adaptive_profile(runner)
    state = runner._load_or_create_adaptive_memory_state(
        tmp_path, profile, resume=False
    )
    assert state is not None
    assert state["current_profile"] == "adaptive_chunk_1x"
    assert state["pair_chunk_size"] == "all"
    assert state["pair_chunk_divisor"] == 1
    assert state["micro_batch_size"] == 256
    assert state["gradient_accumulation_steps"] == 1


def test_profile_execution_fields_preserves_all_pairs_and_normalizes_numeric_chunk() -> None:
    runner = _runner_module()
    all_pairs = runner._profile_execution_fields(
        {
            "pair_chunk_size": "all",
            "pair_chunk_divisor": 1,
            "micro_batch_size": 256,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": False,
        }
    )
    assert all_pairs["pair_chunk_size"] == "all"

    numeric = runner._profile_execution_fields(
        {
            "pair_chunk_size": "4096",
            "pair_chunk_divisor": 2,
            "micro_batch_size": 128,
            "gradient_accumulation_steps": 2,
            "activation_checkpointing": True,
        }
    )
    assert numeric["pair_chunk_size"] == 4096


def test_retired_adaptive_state_is_rejected(tmp_path: Path) -> None:
    runner = _runner_module()
    path = tmp_path / "adaptive_memory_state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": runner.ADAPTIVE_MEMORY_STATE_SCHEMA,
                "current_tier": 0,
                "current_profile": "adaptive_max",
                "pair_chunk_size": 163840,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResumeCompatibilityError, match="compatible v2 profile"):
        runner._load_or_create_adaptive_memory_state(
            tmp_path, _adaptive_profile(runner), resume=True
        )


def test_oom_retry_is_recorded_only_for_the_failed_selector(tmp_path: Path) -> None:
    runner = _runner_module()
    profile = _adaptive_profile(runner)
    state = runner._load_or_create_adaptive_memory_state(
        tmp_path, profile, resume=False
    )
    assert state is not None
    runner._adaptive_selector_state(tmp_path, state, profile, "q_rpec")
    runner._adaptive_selector_state(tmp_path, state, profile, "classical_local_echo")
    event = runner._record_adaptive_oom_retry(
        tmp_path,
        state,
        selector="q_rpec",
        gpu=0,
        from_tier=0,
        to_tier=1,
        hardware_profile=profile,
        event_type="oom_retry",
    )
    assert event["event"] == "oom_retry"
    assert state["selectors"]["q_rpec"]["current_tier"] == 1
    assert state["selectors"]["q_rpec"]["oom_retries"] == 1
    assert state["selectors"]["classical_local_echo"]["current_tier"] == 0


def test_dashboard_shows_in_progress_without_fake_rate(tmp_path: Path) -> None:
    runner = _runner_module()
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(
        json.dumps(
            {
                "event": "batch_start",
                "phase": "train",
                "epoch": 1,
                "epochs": 12,
                "batch": 1,
                "batches": 229,
                "completed_batches": 0,
                "percent": 0.0,
                "batches_per_second": None,
                "eta_seconds": None,
            }
        ),
        encoding="utf-8",
    )
    output = runner._render_selector_dashboard(
        {
            "q_rpec": {
                "status": "running",
                "gpu": 0,
                "heartbeat_file": str(heartbeat),
                "first_batch_started_monotonic": time.monotonic() - 12,
            }
        },
        {"q_rpec": {"started_monotonic": 0}},
    )
    assert "train" in output
    assert "ETA --:--" in output
    assert "batch/s" not in output


def test_analytical_observable_matches_statevector_reference() -> None:
    config = RelationPerturbationEchoConfig(
        num_layers=1, num_heads=2, head_dim=3, num_qubits=2, seed=17
    )
    for kernel_class in (RelationPerturbationEchoCurvatureKernel, LocalRelationEchoCurvatureControl):
        kernel = kernel_class(
            config,
            pair_chunk_size=None,
            pair_chunk_divisor=1,
            activation_checkpointing=False,
        )
        query = torch.randn(5, 3)
        relation = torch.randn(5, 3)
        key = torch.randn(5, 3)
        for head in range(2):
            analytical = kernel._observable(
                query, relation, key, layer_index=0, head_index=head
            )
            reference = kernel._observable_statevector_reference(
                query, relation, key, layer_index=0, head_index=head
            )
            torch.testing.assert_close(analytical, reference, rtol=1e-5, atol=1e-5)
