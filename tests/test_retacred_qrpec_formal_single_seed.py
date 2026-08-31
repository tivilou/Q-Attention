from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

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


def test_old_adaptive_max_state_is_rejected(tmp_path: Path) -> None:
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
    with pytest.raises(ResumeCompatibilityError, match="retired unsafe adaptive_max"):
        runner._load_or_create_adaptive_memory_state(
            tmp_path, _adaptive_profile(runner), resume=True
        )


def test_throughput_timeout_is_recorded_per_selector(tmp_path: Path) -> None:
    runner = _runner_module()
    profile = _adaptive_profile(runner)
    state = runner._load_or_create_adaptive_memory_state(
        tmp_path, profile, resume=False
    )
    assert state is not None
    runner._adaptive_selector_state(tmp_path, state, profile, "q_rpec")
    event = runner._record_adaptive_retry(
        tmp_path,
        state,
        selector="q_rpec",
        gpu=0,
        from_tier=0,
        to_tier=1,
        hardware_profile=profile,
        event_type="throughput_timeout",
        event_fields={"first_batch_elapsed_seconds": 300.5},
    )
    assert event["event"] == "throughput_timeout"
    assert event["first_batch_elapsed_seconds"] == 300.5
    assert state["selectors"]["q_rpec"]["throughput_timeouts"] == 1
    assert state["throughput_timeouts"] == 1


def test_dashboard_does_not_claim_rate_or_eta_for_first_batch(tmp_path: Path) -> None:
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
        {"q_rpec": {"started_monotonic": time.monotonic() - 12}},
    )
    assert "IN PROGRESS" in output
    assert "ETA" not in output
    assert "batch/s" not in output


def test_first_batch_timeout_requeues_only_the_stalled_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    profile = _adaptive_profile(runner)
    state = runner._load_or_create_adaptive_memory_state(
        tmp_path, profile, resume=False
    )
    assert state is not None
    selectors = ["q_rpec", "classical_local_echo"]
    for selector in selectors:
        checkpoint_dir = tmp_path / "selectors" / selector / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "latest.pt").write_bytes(b"batch-0")

    processes = []

    class FakeProcess:
        next_pid = 31000

        def __init__(self, command, *, env, **_kwargs):
            self.command = command
            self.selector = command[command.index("--selector") + 1]
            self.returncode = None
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.start_count = sum(item.selector == self.selector for item in processes) + 1
            heartbeat_path = Path(env["Q_ATTENTION_HEARTBEAT_FILE"])
            if self.selector == "q_rpec" and self.start_count == 1:
                payload = {
                    "event": "batch_start",
                    "batch": 1,
                    "batches": 229,
                    "completed_batches": 0,
                }
            else:
                payload = {
                    "event": "batch_progress",
                    "batch": 1,
                    "batches": 229,
                    "completed_batches": 1,
                    "batches_per_second": 1.0,
                }
            heartbeat_path.write_text(json.dumps(payload), encoding="utf-8")
            metrics_path = heartbeat_path.parent / "metrics.json"
            metrics_path.write_text(
                json.dumps({"selector": self.selector, "valid": {}, "test": {}}),
                encoding="utf-8",
            )
            processes.append(self)

        def poll(self):
            if self.selector == "classical_local_echo" and self.returncode is None:
                self.returncode = 0
            if self.selector == "q_rpec" and self.start_count >= 2:
                self.returncode = 0
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

    def fake_terminate(process):
        process.returncode = -9

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(runner, "_terminate_worker", fake_terminate)
    args = SimpleNamespace(
        python_bin="python",
        log_every_batches=50,
        checkpoint_every_batches=50,
        first_batch_timeout_seconds=0,
    )
    statuses = runner.run_selector_workers(
        selectors=selectors,
        gpu_ids=[0, 1],
        args=args,
        config_path=tmp_path / "config.json",
        baseline_dir=tmp_path / "baseline",
        data_dir=tmp_path / "data",
        run_dir=tmp_path,
        seed=13,
        hardware_profile=profile,
        resume=False,
        adaptive_memory_state=state,
    )
    assert statuses["q_rpec"]["status"] == "complete"
    assert statuses["classical_local_echo"]["status"] == "complete"
    assert state["selectors"]["q_rpec"]["current_tier"] == 1
    assert state["selectors"]["classical_local_echo"]["current_tier"] == 0
    timeout_events = [
        event for event in state["events"] if event["event"] == "throughput_timeout"
    ]
    assert [event["selector"] for event in timeout_events] == ["q_rpec"]
