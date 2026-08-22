from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_trained_baseline_gate import split_diagnostics  # noqa: E402
from run_q_partial_evidence_triad_gate import make_split  # noqa: E402


def test_partial_evidence_is_deterministic_balanced_and_label_correct() -> None:
    first = make_split(6701, 64, 64, 0.25, torch.device("cpu"), set())
    second = make_split(6701, 64, 64, 0.25, torch.device("cpu"), set())
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["labels"], first["cycle_labels"])
    assert torch.equal(first["labels"], second["labels"])
    assert float(first["labels"].float().mean()) == 0.5
    assert bool(first["relocated_roles"].any())
    assert bool((~first["relocated_roles"]).any())


def test_partial_evidence_streams_are_exactly_disjoint() -> None:
    seen: set[tuple[int, ...]] = set()
    train = make_split(6701, 64, 64, 0.25, torch.device("cpu"), seen)
    valid = make_split(6702, 32, 64, 0.25, torch.device("cpu"), seen)
    test = make_split(6703, 32, 64, 0.25, torch.device("cpu"), seen)
    diagnostics = split_diagnostics({"train": train, "valid": valid, "test": test})
    assert diagnostics["exact_split_overlap"] == 0
