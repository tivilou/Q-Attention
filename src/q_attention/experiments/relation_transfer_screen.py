"""Aggregate validation-to-test evidence for the real-data transfer screen."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SPLITS = ("valid", "test")
STAGES = ("core", "evidence", "routing")


def metric_gains(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
) -> dict[str, float]:
    """Express all tracked changes so positive values are improvements."""
    return {
        "loss_reduction": float(reference["loss"] - candidate["loss"]),
        "correct_label_margin_gain": float(
            candidate["correct_label_margin"]
            - reference["correct_label_margin"]
        ),
        "macro_f1_gain": float(candidate["macro_f1"] - reference["macro_f1"]),
    }


def _task_direction_pass(gains: Mapping[str, float]) -> bool:
    return gains["loss_reduction"] > 0.0 and gains["correct_label_margin_gain"] > 0.0


def _direction_agrees(valid: Mapping[str, float], test: Mapping[str, float]) -> bool:
    return all(
        (valid[name] > 0.0) == (test[name] > 0.0)
        for name in ("loss_reduction", "correct_label_margin_gain")
    )


def _stage_summary(
    quantum: Mapping[str, Mapping[str, float]],
    classical: Mapping[str, Mapping[str, float]],
    quantum_reference: Mapping[str, Mapping[str, float]],
    classical_reference: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in SPLITS:
        quantum_increment = metric_gains(quantum[split], quantum_reference[split])
        classical_increment = metric_gains(classical[split], classical_reference[split])
        quantum_over_classical = metric_gains(quantum[split], classical[split])
        splits[split] = {
            "quantum": dict(quantum[split]),
            "classical": dict(classical[split]),
            "quantum_increment": quantum_increment,
            "classical_increment": classical_increment,
            "quantum_over_classical": quantum_over_classical,
            "quantum_increment_pass": _task_direction_pass(quantum_increment),
            "quantum_over_classical_pass": _task_direction_pass(
                quantum_over_classical
            ),
        }

    return {
        "splits": splits,
        "increment_direction_agreement": _direction_agrees(
            splits["valid"]["quantum_increment"],
            splits["test"]["quantum_increment"],
        ),
        "matched_control_direction_agreement": _direction_agrees(
            splits["valid"]["quantum_over_classical"],
            splits["test"]["quantum_over_classical"],
        ),
        "transfer_pass": all(
            splits[split]["quantum_increment_pass"]
            and splits[split]["quantum_over_classical_pass"]
            for split in SPLITS
        ),
    }


def _metrics_close(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    tolerance: float,
) -> bool:
    return all(
        abs(float(left[name]) - float(right[name])) <= tolerance
        for name in ("loss", "correct_label_margin", "macro_f1")
    )


def summarize_transfer_screen(
    *,
    baseline: Mapping[str, Mapping[str, float]],
    stages: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    routing_uniform: Mapping[str, Mapping[str, Mapping[str, float]]],
    parameter_counts: Mapping[str, Mapping[str, int]],
    identity_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Build the strict core/evidence/routing transfer decision."""
    missing_stages = set(STAGES) - set(stages)
    if missing_stages:
        raise ValueError(f"missing transfer stages: {sorted(missing_stages)}")

    core = _stage_summary(
        stages["core"]["quantum"],
        stages["core"]["classical"],
        baseline,
        baseline,
    )
    evidence = _stage_summary(
        stages["evidence"]["quantum"],
        stages["evidence"]["classical"],
        stages["core"]["quantum"],
        stages["core"]["classical"],
    )
    routing = _stage_summary(
        stages["routing"]["quantum"],
        stages["routing"]["classical"],
        routing_uniform["quantum"],
        routing_uniform["classical"],
    )

    parameter_match = {
        stage: (
            int(parameter_counts[stage]["quantum"])
            == int(parameter_counts[stage]["classical"])
            and int(parameter_counts[stage]["quantum"]) > 0
        )
        for stage in STAGES
    }
    routing_uniform_identity = {
        family: {
            split: _metrics_close(
                routing_uniform[family][split],
                stages["evidence"][family][split],
                tolerance=identity_tolerance,
            )
            for split in SPLITS
        }
        for family in ("quantum", "classical")
    }
    stage_summaries = {
        "core": core,
        "evidence": evidence,
        "routing": routing,
    }
    return {
        "baseline": {split: dict(baseline[split]) for split in SPLITS},
        "stages": stage_summaries,
        "parameter_counts": {
            stage: {family: int(count) for family, count in counts.items()}
            for stage, counts in parameter_counts.items()
        },
        "parameter_match": parameter_match,
        "routing_uniform_identity": routing_uniform_identity,
        "screen_pass": (
            all(parameter_match.values())
            and all(
                routing_uniform_identity[family][split]
                for family in routing_uniform_identity
                for split in SPLITS
            )
            and all(summary["transfer_pass"] for summary in stage_summaries.values())
        ),
    }
