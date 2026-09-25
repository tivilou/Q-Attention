from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_q_coherence_destruction_differential.py"
SPEC = importlib.util.spec_from_file_location("qcdd_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def frozen_config() -> dict:
    return json.loads(
        (ROOT / "configs/q_coherence_destruction_differential.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_config_uses_seed_7_and_explicit_selector_allowlist() -> None:
    config = runner.load_config(
        ROOT / "configs/q_coherence_destruction_differential.json"
    )
    assert config["seed"] == 7
    assert tuple(config["selectors"]) == runner.SELECTORS
    assert config["shot_estimate"]["maximum_shots_per_candidate_pair"] == 4096
    assert config["training"]["pair_loss_weight"] == 0.5


def test_config_rejects_selector_expansion(tmp_path: Path) -> None:
    config = frozen_config()
    config["selectors"].append("attention_action")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist"):
        runner.load_config(path)


def test_pair_auc_assigns_average_ranks_to_ties() -> None:
    scores = torch.zeros(2, 3, 15)
    target = torch.zeros_like(scores, dtype=torch.bool)
    target[..., 4] = True
    assert runner.pair_auc(scores, target) == 0.5


def test_exact_disjoint_stream_check_detects_overlap() -> None:
    streams = {
        "train": {"query": torch.tensor([[[1.0, 2.0]]])},
        "valid": {"query": torch.tensor([[[1.0, 2.0]]])},
        "test": {"query": torch.tensor([[[3.0, 4.0]]])},
    }
    overlaps = runner.exact_stream_overlaps(streams)
    assert overlaps["train:valid"] == 1
    assert overlaps["train:test"] == 0


def test_runner_source_has_no_attention_action_call() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "apply_pair_actions(" not in source
    assert '"next_attention_action_authorized": False' in source


def test_shot_estimate_accepts_real_task_score_dimensions() -> None:
    config = frozen_config()
    device = torch.device("cpu")
    frames = runner.task.v1.relation_frames(device)
    model = runner.build_model("quantum", 7, frames, config, device)
    split = runner.task.make_split(10007, 2, device)
    estimate = runner.shot_estimate(model, split, config)
    assert estimate["candidate_pair_count"] == 4
    assert estimate["settings"] == list(runner.PAULI_SETTINGS)
