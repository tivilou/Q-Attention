from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from experiments.run_relation_qness_proportional_gate import (
    _CONSOLE_BROKEN,
    _console_print,
    _gpu_assignment_manifest,
    _render_selector_dashboard,
    _run_completion_errors,
    _resolve_gpus,
    _run_stage,
    _run_selector_workers,
    _selector_dashboard_snapshot,
    _selector_command,
    _stage_summary,
    summarize_existing_run,
    proportional_gate_decision,
)


def stage(loss, macro_f1, *, task_epoch=None, selectivity=True, resources=None):
    return {
        "validation_metrics": {
            "loss": loss,
            "macro_f1": macro_f1,
            "correct_label_margin": 0.1,
        },
        "best_task_epoch": task_epoch,
        "checkpoint_selection": {
            "selected": "best_task_valid" if task_epoch is not None else "best_valid",
        },
        "selectivity_pass": selectivity,
        "diagnostic_means": {
            "off_diagonal_density_norm": None,
            "mutual_information": None,
            "observable_commutator_norm": None,
            "complement_error": 0.1,
            "keep_advantage": 0.1,
            "drop_advantage": 0.1,
            **(resources or {}),
        },
    }


def test_proportional_gate_requires_task_gain_and_resource_controls():
    stages = {
        "baseline": stage(2.0, 0.20),
        "core_quantum": stage(1.50, 0.20),
        "selector_qness": stage(
            1.20,
            0.21,
            task_epoch=2,
            resources={
                "observable_commutator_norm": 1.2,
                "off_diagonal_density_norm": 0.2,
                "mutual_information": 0.3,
            },
        ),
        "selector_qness_classical": stage(1.30, 0.20),
        "selector_qness_commuting": stage(
            1.4, 0.20, resources={"observable_commutator_norm": 0.0}
        ),
        "selector_qness_separable": stage(
            1.4, 0.20, resources={"mutual_information": 0.0}
        ),
        "selector_qness_phase_scrambled": stage(1.4, 0.20),
        "selector_qness_dephased": stage(
            1.4, 0.20, resources={"off_diagonal_density_norm": 0.0}
        ),
    }
    decision = proportional_gate_decision(stages)
    assert decision["gate_pass"]

    stages["selector_qness"]["best_task_epoch"] = None
    stages["selector_qness"]["checkpoint_selection"]["selected"] = "best_valid"
    assert not proportional_gate_decision(stages)["gate_pass"]


def test_proportional_gate_can_screen_without_controls():
    stages = {
        "baseline": stage(2.0, 0.20),
        "core_quantum": stage(1.50, 0.20),
        "selector_qness": stage(
            1.20,
            0.21,
            task_epoch=2,
            resources={
                "observable_commutator_norm": 1.2,
                "off_diagonal_density_norm": 0.2,
                "mutual_information": 0.3,
            },
        ),
        "selector_qness_classical": stage(1.30, 0.20),
    }
    assert proportional_gate_decision(stages, controls_requested=False)["gate_pass"]


def test_qness_command_uses_compatible_connected_fixed_measurement():
    args = Namespace(
        random_repeats=1,
        diagnostic_batches=2,
        quantum_diagnostic_limit=64,
        selector_epochs=1,
        plugin_batch_size=8,
        log_every_batches=1,
        seed=13,
        device="cpu",
    )
    paths = {"train": Path("train.jsonl"), "valid": Path("valid.jsonl")}
    command = _selector_command(
        args,
        "qness",
        paths,
        Path("runs/example/selector/qness"),
    )
    assert command[command.index("--evidence_correlation_mode") + 1] == "connected"
    assert command[command.index("--evidence_measurement_mode") + 1] == "fixed"


def test_baseline_stage_summary_does_not_require_diagnostics(tmp_path):
    (tmp_path / "metrics.json").write_text(
        '{"best_valid":{"loss":1.0,"macro_f1":0.2,"correct_label_margin":0.1}}',
        encoding="utf-8",
    )
    summary = _stage_summary(tmp_path)
    assert summary["diagnostics_path"] is None
    assert summary["checkpoint_selection"] == {
        "policy": "best_valid",
        "requested": "best_valid",
        "selected": "best_valid",
        "fallback": False,
        "epoch": None,
    }


def test_selector_stage_summary_prefers_task_checkpoint_and_records_fallback(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_epoch": 1,
                "best_valid": {
                    "loss": 1.2,
                    "macro_f1": 0.3,
                    "correct_label_margin": 0.1,
                },
                "best_task_epoch": 3,
                "best_task_valid": {
                    "loss": 1.1,
                    "macro_f1": 0.4,
                    "correct_label_margin": 0.2,
                },
                "best_selectivity": {
                    "selectivity_pass": False,
                    "metrics": {
                        "mutual_information": {"mean": 0.1},
                    },
                },
                "best_task_selectivity": {
                    "selectivity_pass": True,
                    "metrics": {
                        "mutual_information": {"mean": 0.2},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = _stage_summary(tmp_path, prefer_task=True)
    assert summary["validation_metrics"]["loss"] == pytest.approx(1.1)
    assert summary["selectivity_pass"]
    assert summary["diagnostic_means"]["mutual_information"] == pytest.approx(0.2)
    assert summary["checkpoint_selection"] == {
        "policy": "best_task_valid_with_best_valid_fallback",
        "requested": "best_task_valid",
        "selected": "best_task_valid",
        "fallback": False,
        "epoch": 3,
    }

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["best_task_valid"] = None
    payload["best_task_epoch"] = None
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = _stage_summary(tmp_path, prefer_task=True)
    assert summary["validation_metrics"]["loss"] == pytest.approx(1.2)
    assert not summary["selectivity_pass"]
    assert summary["diagnostic_means"]["mutual_information"] == pytest.approx(0.1)
    assert summary["checkpoint_selection"] == {
        "policy": "best_task_valid_with_best_valid_fallback",
        "requested": "best_task_valid",
        "selected": "best_valid",
        "fallback": True,
        "epoch": 1,
    }


def test_existing_run_uses_one_checkpoint_policy_for_all_selectors(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "seed": 13,
                    "revision": "test-revision",
                    "run_controls": "always",
                },
                "subset_manifest": {},
            }
        ),
        encoding="utf-8",
    )

    def write_stage(relative_path, best_loss, task_loss=None):
        stage_dir = tmp_path / relative_path
        stage_dir.mkdir(parents=True)
        payload = {
            "best_epoch": 1,
            "best_valid": {
                "loss": best_loss,
                "macro_f1": 0.2,
                "correct_label_margin": 0.1,
            },
        }
        if task_loss is not None:
            payload.update(
                {
                    "best_task_epoch": 2,
                    "best_task_valid": {
                        "loss": task_loss,
                        "macro_f1": 0.2,
                        "correct_label_margin": 0.1,
                    },
                }
            )
        (stage_dir / "metrics.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    write_stage("baseline", 2.0)
    write_stage("core/quantum", 1.5)
    write_stage("selector/qness", 1.4, task_loss=1.2)
    write_stage("selector/qness_classical", 1.6, task_loss=1.1)
    write_stage("selector/qness_commuting", 1.7, task_loss=1.3)
    write_stage("selector/qness_separable", 1.8)
    write_stage("selector/qness_phase_scrambled", 1.9, task_loss=1.4)
    write_stage("selector/qness_dephased", 2.0)

    summary = summarize_existing_run(tmp_path)

    assert summary["schema_version"] == "retacred-qness-proportional-gate.v2"
    assert summary["checkpoint_policy"] == {
        "primary_metric": "validation_loss",
        "baseline_and_core": "best_valid",
        "selectors": "best_task_valid_with_best_valid_fallback",
        "macro_f1_role": "guardrail_only",
    }
    assert summary["stages"]["selector_qness"]["validation_metrics"]["loss"] == 1.2
    assert (
        summary["stages"]["selector_qness_classical"]["validation_metrics"]["loss"]
        == 1.1
    )
    assert (
        summary["stages"]["selector_qness_commuting"]["checkpoint_selection"][
            "selected"
        ]
        == "best_task_valid"
    )
    assert summary["stages"]["selector_qness_separable"]["checkpoint_selection"][
        "fallback"
    ]
    assert summary["stages"]["selector_qness_dephased"]["checkpoint_selection"][
        "fallback"
    ]
    assert not summary["decision"]["task_checks"][
        "qness_improves_over_classical_control"
    ]


def test_gpu_list_parser_preserves_order_and_rejects_duplicates():
    assert _resolve_gpus("2, 0,1") == [2, 0, 1]
    with pytest.raises(ValueError, match="more than once"):
        _resolve_gpus("0,1,0")
    with pytest.raises(ValueError, match="non-negative integers"):
        _resolve_gpus("0,bad")


def test_gpu_list_parser_resolves_auto_with_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="0\n2\n", stderr=""
        ),
    )
    assert _resolve_gpus("auto") == [0, 2]


def test_gpu_assignment_manifest_pins_shared_stages_to_primary_gpu():
    manifest = _gpu_assignment_manifest(
        "2,5", [2, 5], ["qness", "qness_classical"]
    )
    assert manifest["ddp"] is False
    assert manifest["stages"]["baseline"]["gpu_id"] == 2
    assert manifest["stages"]["core_quantum"]["gpu_id"] == 2
    assert manifest["stages"]["selector_qness"]["gpu_id"] is None


def test_selector_workers_let_the_first_free_gpu_take_the_next_job():
    first_pair = threading.Barrier(2)
    assignments = {}
    assignments_lock = threading.Lock()

    def run_selector(selector, gpu_id):
        with assignments_lock:
            assignments[selector] = gpu_id
        if selector in {"slow", "fast"}:
            first_pair.wait(timeout=2)
        if selector == "slow":
            time.sleep(0.05)

    outcomes = _run_selector_workers(
        ["slow", "fast", "next"], [0, 1], run_selector
    )

    assert all(outcome["status"] == "complete" for outcome in outcomes)
    assert assignments["slow"] != assignments["fast"]
    assert assignments["next"] == assignments["fast"]


def test_run_stage_records_gpu_status_and_required_output(tmp_path):
    artifact = tmp_path / "artifact.txt"
    assignments = _gpu_assignment_manifest("7", [7], ["qness"])
    assignments_path = tmp_path / "gpu_assignments.json"
    assignments_path.write_text(json.dumps(assignments), encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib; "
            "assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'; "
            f"pathlib.Path({str(artifact)!r}).write_text('ok', encoding='utf-8'); "
            "print('stage progress')"
        ),
    ]

    _run_stage(
        "baseline",
        command,
        tmp_path,
        None,
        7,
        assignments,
        assignments_path,
        threading.Lock(),
        threading.Lock(),
        required_paths=(artifact,),
    )

    status = (tmp_path / "status" / "baseline.env").read_text(encoding="utf-8")
    recorded = json.loads(assignments_path.read_text(encoding="utf-8"))
    assert "STATUS=complete" in status
    assert "GPU_ID=7" in status
    assert "DURATION_SECONDS=" in status
    assert recorded["stages"]["baseline"]["status"] == "complete"
    assert recorded["stages"]["baseline"]["gpu_id"] == 7


def test_broken_console_pipe_does_not_fail_training(monkeypatch):
    _CONSOLE_BROKEN.clear()

    def broken_print(*args, **kwargs):
        raise BrokenPipeError("closed")

    monkeypatch.setattr("builtins.print", broken_print)
    _console_print("progress\n", threading.Lock())
    assert _CONSOLE_BROKEN.is_set()
    _CONSOLE_BROKEN.clear()


def test_selector_dashboard_reads_status_and_heartbeat(tmp_path):
    assignments = _gpu_assignment_manifest(
        "0,1,2",
        [0, 1, 2],
        ["qness", "qness_classical", "qness_commuting", "qness_separable"],
    )
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    (status_dir / "selector_qness.env").write_text(
        "STATUS=running\nGPU_ID=1\n", encoding="utf-8"
    )
    (status_dir / "selector_qness.heartbeat").write_text(
        json.dumps(
            {
                "event": "batch_progress",
                "phase": "train",
                "epoch": 2,
                "epochs": 5,
                "batch": 64,
                "batches": 128,
                "eta_seconds": 2120,
                "batches_per_second": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (status_dir / "selector_qness_classical.env").write_text(
        "STATUS=complete\nGPU_ID=0\n", encoding="utf-8"
    )
    (status_dir / "selector_qness_commuting.env").write_text(
        "STATUS=failed\nGPU_ID=2\n", encoding="utf-8"
    )

    snapshot = _selector_dashboard_snapshot(
        tmp_path,
        ["qness", "qness_classical", "qness_commuting", "qness_separable"],
        assignments,
    )
    rendered = _render_selector_dashboard(snapshot)

    assert snapshot["counts"] == {
        "complete": 1,
        "running": 1,
        "pending": 1,
        "failed": 1,
    }
    assert "GPU 1 | selector_qness | train | epoch 2/5 | batch 64/128" in rendered
    assert "ETA 35:20" in rendered
    assert "Completed: selector_qness_classical" in rendered
    assert "Pending: selector_qness_separable" in rendered
    assert "Failed: selector_qness_commuting" in rendered


def test_completion_gate_requires_all_outputs_and_markers(tmp_path):
    selectors = ["qness", "qness_classical"]
    assignments = _gpu_assignment_manifest("0", [0], selectors)
    (tmp_path / "gpu_assignments.json").write_text(
        json.dumps(assignments), encoding="utf-8"
    )
    for stage in ["baseline", "core_quantum", *[f"selector_{s}" for s in selectors]]:
        status_dir = tmp_path / "status"
        status_dir.mkdir(exist_ok=True)
        (status_dir / f"{stage}.env").write_text(
            "STATUS=complete\n", encoding="utf-8"
        )
        assignments["stages"][stage]["status"] = "complete"
    for selector in selectors:
        selector_dir = tmp_path / "selector" / selector
        selector_dir.mkdir(parents=True)
        (selector_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (selector_dir / "diagnostics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "gpu_assignments.json").write_text(
        json.dumps(assignments), encoding="utf-8"
    )

    errors = _run_completion_errors(
        tmp_path, selectors, require_summary=True, require_marker=True
    )
    assert "missing run_summary.json" in errors
    assert "missing run_summary.md" in errors
    assert "missing RUN_COMPLETE" in errors

    (tmp_path / "run_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_summary.md").write_text("ok\n", encoding="utf-8")
    (tmp_path / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
    assert _run_completion_errors(
        tmp_path, selectors, require_summary=True, require_marker=True
    ) == []
