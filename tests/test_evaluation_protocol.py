from __future__ import annotations

import importlib.util
from pathlib import Path

from q_attention.tasks.relation import RelationRecord


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
        "supervised_quantum_gain_selection": {"bootstrap_seed": 13, "acceptance_seed": 13},
    }

    baseline = module.merged_stage_options(config, "baseline", seed_override=29)
    steering = module.merged_stage_options(config, "classical_steering", seed_override=29)
    quantum = module.merged_stage_options(config, "quantum_projector", seed_override=29)
    selection = module.merged_stage_options(config, "supervised_quantum_gain_selection", seed_override=29)

    assert baseline["epochs"] == 5
    assert baseline["seed"] == 29
    assert "seed" not in steering
    assert quantum["seed"] == 29
    assert quantum["feature_seed"] == 29
    assert selection["bootstrap_seed"] == 29
    assert selection["acceptance_seed"] == 29


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


def test_layer_gain_helpers_build_strict_zero_initialized_map() -> None:
    module = load_experiment("select_relation_steering_gain.py")
    paths = ["layers.0.key_proj", "layers.1.key_proj"]

    gains = module.zero_layer_gains(paths)
    updated = module.set_layer_gain(gains, paths[1], 0.05)

    assert gains == {paths[0]: 0.0, paths[1]: 0.0}
    assert updated == {paths[0]: 0.0, paths[1]: 0.05}


def test_validation_split_is_deterministic_and_non_overlapping() -> None:
    module = load_experiment("select_relation_steering_gain.py")
    records = [
        RelationRecord(tokens=(f"token-{index}",), subject=(0, 1), object=(0, 1), label=f"label-{index % 2}")
        for index in range(10)
    ]

    first_selection, first_acceptance = module.split_validation_records(
        records, acceptance_fraction=0.4, seed=17
    )
    second_selection, second_acceptance = module.split_validation_records(
        records, acceptance_fraction=0.4, seed=17
    )

    assert first_selection == second_selection
    assert first_acceptance == second_acceptance
    assert len(first_selection) == 6
    assert len(first_acceptance) == 4
    assert {id(record) for record in first_selection}.isdisjoint({id(record) for record in first_acceptance})


def test_paired_bootstrap_accepts_stable_macro_f1_gain_and_rejects_no_change() -> None:
    module = load_experiment("select_relation_steering_gain.py")
    labels = [0, 1] * 20
    baseline = [1, 0] * 20
    steered = list(labels)

    improved = module.paired_bootstrap_macro_f1_delta(
        baseline,
        steered,
        labels,
        num_labels=2,
        num_samples=100,
        confidence=0.95,
        seed=7,
    )
    unchanged = module.paired_bootstrap_macro_f1_delta(
        steered,
        steered,
        labels,
        num_labels=2,
        num_samples=100,
        confidence=0.95,
        seed=7,
    )

    assert improved["lower"] > 0.0
    assert unchanged["lower"] == 0.0
    assert unchanged["upper"] == 0.0
