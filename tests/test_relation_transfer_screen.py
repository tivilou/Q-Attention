from __future__ import annotations

from q_attention.experiments.relation_transfer_screen import summarize_transfer_screen


def _metrics(loss: float, margin: float, macro_f1: float) -> dict[str, float]:
    return {
        "loss": loss,
        "correct_label_margin": margin,
        "macro_f1": macro_f1,
    }


def test_transfer_screen_requires_positive_quantum_transfer_and_matched_advantage() -> None:
    baseline = {
        "valid": _metrics(1.0, -0.5, 0.20),
        "test": _metrics(1.1, -0.6, 0.18),
    }
    stages = {
        "core": {
            "quantum": {
                "valid": _metrics(0.90, -0.40, 0.22),
                "test": _metrics(1.00, -0.50, 0.20),
            },
            "classical": {
                "valid": _metrics(0.95, -0.45, 0.21),
                "test": _metrics(1.05, -0.55, 0.19),
            },
        },
        "evidence": {
            "quantum": {
                "valid": _metrics(0.85, -0.35, 0.23),
                "test": _metrics(0.95, -0.45, 0.21),
            },
            "classical": {
                "valid": _metrics(0.93, -0.43, 0.21),
                "test": _metrics(1.03, -0.53, 0.19),
            },
        },
        "routing": {
            "quantum": {
                "valid": _metrics(0.82, -0.32, 0.24),
                "test": _metrics(0.92, -0.42, 0.22),
            },
            "classical": {
                "valid": _metrics(0.92, -0.42, 0.21),
                "test": _metrics(1.02, -0.52, 0.19),
            },
        },
    }
    routing_uniform = {
        family: dict(stages["evidence"][family])
        for family in ("quantum", "classical")
    }
    summary = summarize_transfer_screen(
        baseline=baseline,
        stages=stages,
        routing_uniform=routing_uniform,
        parameter_counts={
            "core": {"quantum": 62, "classical": 62},
            "evidence": {"quantum": 70, "classical": 70},
            "routing": {"quantum": 18, "classical": 18},
        },
    )

    assert summary["screen_pass"]
    assert all(stage["transfer_pass"] for stage in summary["stages"].values())


def test_transfer_screen_rejects_test_reversal_and_parameter_mismatch() -> None:
    baseline = {
        "valid": _metrics(1.0, -0.5, 0.20),
        "test": _metrics(1.0, -0.5, 0.20),
    }
    stages = {
        stage: {
            "quantum": {
                "valid": _metrics(0.90, -0.40, 0.22),
                "test": _metrics(1.10, -0.60, 0.18),
            },
            "classical": {
                "valid": _metrics(0.95, -0.45, 0.21),
                "test": _metrics(1.05, -0.55, 0.19),
            },
        }
        for stage in ("core", "evidence", "routing")
    }
    routing_uniform = {
        family: dict(stages["evidence"][family])
        for family in ("quantum", "classical")
    }
    summary = summarize_transfer_screen(
        baseline=baseline,
        stages=stages,
        routing_uniform=routing_uniform,
        parameter_counts={
            "core": {"quantum": 62, "classical": 63},
            "evidence": {"quantum": 70, "classical": 70},
            "routing": {"quantum": 18, "classical": 18},
        },
    )

    assert not summary["screen_pass"]
    assert not summary["parameter_match"]["core"]
    assert not summary["stages"]["core"]["increment_direction_agreement"]
