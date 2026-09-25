from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "run_q_identifiable_query_local_prescreen_toy.py"
SPEC = importlib.util.spec_from_file_location("identifiable_prescreen", RUNNER)
assert SPEC is not None and SPEC.loader is not None
prescreen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prescreen)


def test_dynamic_address_split_has_no_fixed_label_token() -> None:
    split = prescreen.make_split(7, 48, torch.device("cpu"))
    assert split["labels"].shape == (48, prescreen.QUERIES)
    assert split["query"].shape == (48, prescreen.QUERIES, prescreen.DIM)
    assert split["key"].shape == (48, prescreen.QUERIES, prescreen.KEYS, prescreen.DIM)
    assert split["evidence_slot"].unique().numel() == prescreen.KEYS
    assert torch.all(split["evidence_slot"] != split["bad_slot"].unsqueeze(-1))


def test_candidate_relative_observable_is_query_local_and_label_free() -> None:
    split = prescreen.make_split(11, 32, torch.device("cpu"))
    frames = prescreen.relation_frames(torch.device("cpu"))
    field = prescreen.compatibility(split["query"], split["key"], frames)
    shuffled = prescreen.compatibility(torch.roll(split["query"], 1, 0), split["key"], frames)
    assert field.shape == (32, prescreen.QUERIES, prescreen.CLASSES, prescreen.KEYS)
    assert not torch.allclose(field, shuffled)
    # Labels are absent from the observable's function signature and changing
    # them cannot change a recomputed field.
    relabeled = dict(split)
    relabeled["labels"] = (split["labels"] + 1) % prescreen.CLASSES
    assert torch.equal(field, prescreen.compatibility(relabeled["query"], relabeled["key"], frames))


def test_bounded_action_and_complete_oracle_bank() -> None:
    split = prescreen.make_split(19, 24, torch.device("cpu"))
    frames = prescreen.relation_frames(torch.device("cpu"))
    logits, _ = prescreen.baseline_logits(split["scores"], split["key"], split["query"], frames)
    candidate, key_index, sign = prescreen.selector_action(
        "candidate_relative", split, frames, logits
    )
    _scores, residual = prescreen.apply_query_actions(split["scores"], key_index, sign)
    assert candidate.shape == key_index.shape == sign.shape == (24, prescreen.QUERIES)
    assert residual.abs().max() <= prescreen.MAX_DELTA + 1e-6
    assert residual.sum(dim=-1).abs().max() <= 1e-6
    bank = prescreen.all_action_utilities(split, frames, logits)
    assert bank["utility"].shape == (24, prescreen.QUERIES, prescreen.KEYS, 2)
    assert bank["prediction"].shape == (24, prescreen.QUERIES, prescreen.KEYS, 2)
