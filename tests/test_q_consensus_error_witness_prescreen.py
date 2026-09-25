from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
RUNNER = EXPERIMENTS / "run_q_consensus_error_witness_prescreen_toy.py"
SPEC = importlib.util.spec_from_file_location("consensus_prescreen", RUNNER)
assert SPEC is not None and SPEC.loader is not None
prescreen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prescreen)


def test_fixed_error_witness_recovers_constructed_score_regime() -> None:
    split = prescreen.make_split(7, 96, torch.device("cpu"))
    witness = prescreen.error_witness(split["scores"])
    assert torch.equal(witness, split["hard"])
    assert witness.any() and (~witness).any()


def test_consensus_action_is_query_local_bounded_and_zero_sum() -> None:
    split = prescreen.make_split(11, 48, torch.device("cpu"))
    frames = prescreen.v1.relation_frames(torch.device("cpu"))
    candidate, support, active, confidence = prescreen.select_actions(
        "candidate_relative_consensus", split, frames
    )
    _scores, residual = prescreen.apply_pair_actions(
        split["scores"], support, active
    )
    assert candidate.shape == active.shape == confidence.shape == (48, prescreen.v1.QUERIES)
    assert support.shape == (48, prescreen.v1.QUERIES, prescreen.v1.EVIDENCE_KEYS)
    assert residual.abs().max() <= prescreen.v1.MAX_DELTA + 1e-6
    assert residual.sum(dim=-1).abs().max() <= 1e-6
    assert torch.equal(residual[~active], torch.zeros_like(residual[~active]))


def test_complete_pair_oracle_and_label_free_controls() -> None:
    split = prescreen.make_split(19, 32, torch.device("cpu"))
    frames = prescreen.v1.relation_frames(torch.device("cpu"))
    logits, _ = prescreen.v1.baseline_logits(
        split["scores"], split["key"], split["query"], frames
    )
    oracle = prescreen.pair_action_oracle(split, frames, logits)
    assert oracle["action_count_per_query"] == len(prescreen.PAIR_ACTIONS) * 2
    for selector in prescreen.SELECTORS:
        metrics, tensors = prescreen.evaluate_selector(selector, split, frames, logits)
        assert metrics["selector"] == selector
        assert tensors["prediction"].shape == split["labels"].shape
        assert tensors["residual"].shape == split["scores"].shape
