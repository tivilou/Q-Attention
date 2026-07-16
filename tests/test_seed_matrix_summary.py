from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path("experiments/summarize_relation_seed_matrix.py")
    spec = importlib.util.spec_from_file_location("summarize_relation_seed_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_summary(seed: int, macro_f1: float, delta: float) -> dict:
    return {
        "run_dir": f"runs/seed_{seed}",
        "pipeline": {"reproducibility": {"global_seed": seed}},
        "rows": [
            {
                "variant": "quantum_steering",
                "macro_f1": macro_f1,
                "delta_macro_f1": delta,
            }
        ],
    }


def test_seed_summary_reports_variance_confidence_and_direction() -> None:
    module = load_module()
    aggregate = module.aggregate_run_summaries(
        [run_summary(13, 0.30, 0.01), run_summary(17, 0.28, -0.01), run_summary(23, 0.32, 0.02)]
    )
    quantum = aggregate["variants"]["quantum_steering"]
    delta = quantum["metrics"]["delta_macro_f1"]

    assert aggregate["num_runs"] == 3
    assert quantum["positive_delta_macro_f1"] == 2
    assert delta["n"] == 3
    assert delta["std"] is not None
    assert delta["ci95_low"] < delta["mean"] < delta["ci95_high"]


def test_seed_parser_rejects_duplicates() -> None:
    path = Path("experiments/run_relation_seed_matrix.py")
    spec = importlib.util.spec_from_file_location("run_relation_seed_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.parse_seed_csv("13,17,23") == [13, 17, 23]
    try:
        module.parse_seed_csv("13,13")
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate seeds must be rejected")
