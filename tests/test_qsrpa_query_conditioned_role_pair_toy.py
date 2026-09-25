from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_qsrpa_query_conditioned_role_pair_toy.py"
SPEC = importlib.util.spec_from_file_location("qsrpa_query_role_toy", RUNNER)
assert SPEC is not None and SPEC.loader is not None
toy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(toy)


def _config() -> dict:
    return toy.load_config(ROOT / "configs" / "qsrpa_query_conditioned_role_pair_toy.json")


def test_role_router_is_label_free_and_query_conditioned() -> None:
    config = _config()
    split = toy.make_split(7, 24, torch.device("cpu"), config)
    changed = dict(split)
    changed["query"] = -split["query"]
    first = toy.role_weights("query_conditioned_soft_role", split, config)
    second = toy.role_weights("query_conditioned_soft_role", changed, config)
    assert first.shape == (24, toy.QUERIES, toy.ROLES, toy.KEYS)
    assert not torch.allclose(first, second)
    relabeled = dict(split)
    relabeled["role_slots"] = torch.roll(split["role_slots"], shifts=1, dims=-1)
    assert torch.equal(
        toy.role_weights("query_conditioned_soft_role", relabeled, config), first
    )
    assert not torch.equal(split["role_slots"], relabeled["role_slots"])
    assert torch.equal(split["key"][:, 0], split["key"][:, 1])
    assert torch.allclose(split["query"][:, 0], -split["query"][:, 1])
    assert torch.equal(split["role_slots"][:, 0], split["role_slots"][:, 1].flip(-1))


def test_role_action_is_bounded_zero_sum_and_query_reversal_is_structural() -> None:
    config = _config()
    split = toy.make_split(11, 24, torch.device("cpu"), config)
    weights = toy.role_weights("query_conditioned_soft_role", split, config)
    _scores, residual = toy.apply_role_action(split, weights, config)
    assert residual.abs().max() <= config["dataset"]["action_delta"] + 1e-6
    assert residual.sum(dim=-1).abs().max() <= 1e-6
    assert toy.reversal_consistency("query_conditioned_soft_role", split, config) > 0.9


def test_config_has_predeclared_five_seed_gate() -> None:
    config = _config()
    assert config["seeds"] == [7, 11, 13, 17, 23]
    assert config["gate"]["required_seed_fraction"] == 0.8
