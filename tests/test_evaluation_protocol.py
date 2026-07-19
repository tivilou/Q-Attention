from __future__ import annotations

import importlib.util
from pathlib import Path


def load_experiment(name: str):
    path = Path("experiments") / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_seed_override_reaches_seeded_stages() -> None:
    module = load_experiment("run_relation_smoke_pipeline.py")
    config = {
        "baseline": {"epochs": 5, "seed": 13},
        "classical_steering": {"gain": 0.5},
        "quantum_projector": {"seed": 13, "feature_seed": 17},
    }

    baseline = module.merged_stage_options(config, "baseline", seed_override=29)
    steering = module.merged_stage_options(config, "classical_steering", seed_override=29)
    quantum = module.merged_stage_options(config, "quantum_projector", seed_override=29)

    assert baseline["epochs"] == 5
    assert baseline["seed"] == 29
    assert "seed" not in steering
    assert quantum["seed"] == 29
    assert quantum["feature_seed"] == 29


def test_split_path_supports_direct_and_nested_configs() -> None:
    module = load_experiment("run_relation_smoke_pipeline.py")

    assert module.split_path_from_config({"test_path": "direct.jsonl"}, "test") == "direct.jsonl"
    assert module.split_path_from_config({"splits": {"test": {"path": "nested.jsonl"}}}, "test") == "nested.jsonl"


def test_spectral_selection_uses_macro_f1() -> None:
    module = load_experiment("sweep_relation_spectral_filters.py")
    rows = [
        {"family": "classical", "metrics": {"macro_f1": 0.31}},
        {"family": "quantum", "metrics": {"macro_f1": 0.34}},
    ]

    assert module.select_best_row(rows)["family"] == "quantum"
    assert module.select_best_row([]) is None


def test_supervised_quantum_stages_are_explicit_opt_in() -> None:
    module = load_experiment("run_relation_smoke_pipeline.py")

    assert "supervised_quantum_projector" in module.STAGE_CHOICES
    assert "supervised_quantum_steering" in module.STAGE_CHOICES
    assert "supervised_quantum_gain_selection" in module.STAGE_CHOICES
    assert "supervised_quantum_projector" not in module.DEFAULT_STAGES
    assert "supervised_quantum_steering" not in module.DEFAULT_STAGES
    assert "supervised_quantum_gain_selection" not in module.DEFAULT_STAGES


def test_gain_selection_prefers_validation_metric_then_smaller_intervention() -> None:
    module = load_experiment("select_relation_steering_gain.py")
    rows = [
        {"gain": 0.25, "metrics": {"macro_f1": 0.40}},
        {"gain": 0.10, "metrics": {"macro_f1": 0.40}},
        {"gain": 0.05, "metrics": {"macro_f1": 0.39}},
    ]

    assert module.parse_gain_values("0.0,0.1,0.1,0.25") == [0.0, 0.1, 0.25]
    assert module.select_best_gain(rows, "macro_f1")["gain"] == 0.10
