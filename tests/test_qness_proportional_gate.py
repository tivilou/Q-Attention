from argparse import Namespace
from pathlib import Path

from experiments.run_relation_qness_proportional_gate import (
    _selector_command,
    _stage_summary,
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
