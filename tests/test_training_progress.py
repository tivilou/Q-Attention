from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from q_attention.experiments import progress
from q_attention.experiments.health import EpochHealthMonitor, require_finite_values


def _counterfactual_training_module():
    path = Path(__file__).parents[1] / "experiments/train_relation_counterfactual_evidence.py"
    spec = importlib.util.spec_from_file_location("counterfactual_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_limit_batches_caps_iterable_without_consuming_the_source() -> None:
    source = [1, 2, 3, 4]

    limited, total = progress.limit_batches(source, 2)

    assert total == 2
    assert list(limited) == [1, 2]


def test_limit_batches_zero_preserves_full_iterable() -> None:
    source = [1, 2, 3]

    limited, total = progress.limit_batches(source, 0)

    assert total == 3
    assert limited is source


def test_tracked_batches_reports_interval_final_batch_and_eta(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readings = iter((0.0, 2.0, 4.0, 6.0, 6.0))
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(readings))

    result = list(
        progress.tracked_batches(
            ["a", "b", "c"],
            total_batches=3,
            stage="selector_classical_strong",
            phase="train",
            log_every_batches=2,
            epoch=5,
            epochs=10,
        )
    )

    assert result == ["a", "b", "c"]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "phase_start",
        "batch_progress",
        "batch_progress",
        "batch_progress",
        "phase_complete",
    ]
    assert [event["batch"] for event in events[1:4]] == [1, 2, 3]
    assert events[1]["eta_seconds"] == 4.0
    assert events[1]["estimated_completion_time"] is not None
    assert events[3]["percent"] == 100.0
    assert events[4]["completed_batches"] == 3


def test_tracked_batches_can_emit_human_readable_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readings = iter((0.0, 2.0, 2.0))
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(readings))
    monkeypatch.setenv("Q_ATTENTION_PROGRESS_FORMAT", "both")

    assert list(
        progress.tracked_batches(
            ["a"],
            total_batches=1,
            stage="selector_quantum",
            phase="train",
            log_every_batches=1,
            epoch=1,
            epochs=10,
        )
    ) == ["a"]

    lines = capsys.readouterr().out.splitlines()
    json_events = [json.loads(line) for line in lines if line.startswith("{")]
    text_events = [line for line in lines if line.startswith("[")]
    assert [event["event"] for event in json_events] == [
        "phase_start",
        "batch_progress",
        "phase_complete",
    ]
    assert any("[####################]" in line for line in text_events)
    assert any("ETA 00:00" in line and "finish " in line for line in text_events)


def test_tracked_batches_rejects_invalid_log_interval() -> None:
    with pytest.raises(ValueError, match="log_every_batches"):
        list(
            progress.tracked_batches(
                [1],
                total_batches=1,
                stage="baseline",
                phase="train",
                log_every_batches=0,
            )
        )


def test_log_event_updates_configured_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    heartbeat = tmp_path / "status" / "selector.heartbeat"
    monkeypatch.setenv("Q_ATTENTION_HEARTBEAT_FILE", str(heartbeat))

    progress.log_event("phase_start", stage="selector_quantum", phase="train")

    assert heartbeat.is_file()
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["event"] == "phase_start"
    assert json.loads(capsys.readouterr().out)["event"] == "phase_start"


def test_format_gpu_memory_converts_mib_to_gib() -> None:
    rendered = progress.format_gpu_memory(
        [
            {
                "physical_index": 2,
                "nvidia_used_mib": 4096,
                "nvidia_total_mib": 8192,
                "nvidia_free_mib": 4096,
                "allocated_mib": 1024,
                "reserved_mib": 2048,
                "peak_reserved_mib": 3072,
            }
        ]
    )

    assert rendered == (
        "GPU 2 VRAM 4.0/8.0 GiB used, free 4.0 GiB | "
        "proc 1.0/2.0 GiB alloc/reserved, peak 3.0 GiB"
    )


def test_gpu_memory_snapshot_is_cached_for_five_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter((10.0, 12.0, 16.0))
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(readings))
    progress._GPU_MEMORY_CACHE = None
    progress._GPU_MEMORY_CACHE_AT = None
    progress._GPU_MEMORY_CACHE_KEY = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _index: 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _index: 2 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _index: 3 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _index: 4 * 1024 * 1024)
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=0, stdout="2, 4096, 4096, 8192\n")

    monkeypatch.setattr(progress.subprocess, "run", fake_run)
    first = progress._gpu_memory_snapshot(force=True)
    second = progress._gpu_memory_snapshot()
    third = progress._gpu_memory_snapshot()

    assert first == second
    assert third == second
    assert len(calls) == 2


def test_health_monitor_warns_without_stopping_training(capsys: pytest.CaptureFixture[str]) -> None:
    monitor = EpochHealthMonitor("selector_quantum", patience=3)
    for epoch in range(1, 4):
        monitor.observe(epoch=epoch, valid_loss=1.0, mechanism_pass=False)

    summary = monitor.summary()
    assert summary["no_improvement_epochs"] == 2
    assert summary["consecutive_mechanism_failures"] == 3
    assert [item["warning"] for item in summary["warnings"]] == [
        "mechanism_selectivity_failure",
    ]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["event"] == "health_warning"


def test_finite_value_guard_checks_nested_metrics() -> None:
    require_finite_values({"valid": {"loss": 1.0}, "history": [0, 2.5]}, "run")
    with pytest.raises(FloatingPointError, match="run.valid.loss"):
        require_finite_values({"valid": {"loss": float("inf")}}, "run")


def test_counterfactual_training_guards_fail_fast_on_nonfinite_values() -> None:
    training = _counterfactual_training_module()
    training._require_finite_tensor(
        torch.ones(()),
        "objective",
        stage="selector_quantum",
        epoch=1,
        batch_index=0,
    )

    with pytest.raises(FloatingPointError, match="non-finite objective"):
        training._require_finite_tensor(
            torch.tensor(float("nan")),
            "objective",
            stage="selector_quantum",
            epoch=1,
            batch_index=0,
        )

    module = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.ones(()))
    module.register_parameter("weight", parameter)
    parameter.grad = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite gradients"):
        training._require_finite_gradients(
            module,
            stage="selector_quantum",
            epoch=1,
            batch_index=0,
        )

    parameter.grad = None
    parameter.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite parameters"):
        training._require_finite_parameters(
            module,
            stage="selector_quantum",
            epoch=1,
            batch_index=0,
        )
