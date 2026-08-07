from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from q_attention.experiments import progress


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
    assert events[3]["percent"] == 100.0
    assert events[4]["completed_batches"] == 3


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
