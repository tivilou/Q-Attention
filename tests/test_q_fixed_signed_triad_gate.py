from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_fixed_signed_triad_gate import make_split  # noqa: E402
from run_q_coherent_attention_path_trained_baseline_gate import split_diagnostics  # noqa: E402


def expected_label(row: torch.Tensor) -> int:
    return int(int(((row == 3) | (row == 5) | (row == 7)).sum()) % 2 == 1)


def test_fixed_signed_triad_is_balanced_deterministic_and_label_correct() -> None:
    first = make_split(3701, 64, 64, torch.device("cpu"), set())
    second = make_split(3701, 64, 64, torch.device("cpu"), set())
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert float(first["labels"].float().mean()) == 0.5
    expected = torch.tensor([expected_label(row) for row in first["input_ids"]])
    assert torch.equal(first["labels"], expected)
    assert torch.all(first["subject_mask"][:, 1])
    assert torch.all(first["object_mask"][:, 2])
    assert torch.all(first["bridge_mask"][:, 3])


def test_fixed_signed_triad_streams_are_exactly_disjoint() -> None:
    seen: set[tuple[int, ...]] = set()
    train = make_split(3701, 64, 64, torch.device("cpu"), seen)
    valid = make_split(3702, 32, 64, torch.device("cpu"), seen)
    diagnostics = split_diagnostics({"train": train, "valid": valid})
    assert diagnostics["exact_split_overlap"] == 0
