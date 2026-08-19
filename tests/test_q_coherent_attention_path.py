from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from q_attention.plugins.q_coherent_attention_path import (  # noqa: E402
    CoherentAttentionPathConfig,
    build_coherent_attention_path_kernel,
)
from run_q_coherent_attention_path_motif_toy import (  # noqa: E402
    evaluate,
    make_split,
)


def config() -> dict:
    return json.loads(
        (ROOT / "configs" / "q_coherent_attention_path_motif_toy.json").read_text(
            encoding="utf-8"
        )
    )


def test_q_wap_controls_are_parameter_matched() -> None:
    counts = []
    for kind in ("quantum_signed", "quantum_unsigned", "classical_diffusion"):
        kernel = build_coherent_attention_path_kernel(
            kind, CoherentAttentionPathConfig(num_layers=1, num_heads=1)
        )
        counts.append(sum(parameter.numel() for parameter in kernel.parameters()))
    assert counts == [1, 1, 1]


def test_q_wap_rejects_non_square_scores() -> None:
    kernel = build_coherent_attention_path_kernel(
        "quantum_signed", CoherentAttentionPathConfig(num_layers=1, num_heads=1)
    )
    with pytest.raises(ValueError, match="square scores"):
        kernel(
            torch.zeros(2, 1, 4, 4),
            torch.zeros(2, 1, 5, 4),
            scores=torch.zeros(2, 1, 4, 5),
            layer_index=0,
            attention_mask=torch.ones(2, 5, dtype=torch.bool),
            subject_mask=torch.zeros(2, 5, dtype=torch.bool),
            object_mask=torch.zeros(2, 5, dtype=torch.bool),
        )


def test_signed_q_wap_residual_is_context_only_zero_sum_and_deterministic() -> None:
    split = make_split(7, 16, torch.device("cpu"))
    kernel = build_coherent_attention_path_kernel(
        "quantum_signed", CoherentAttentionPathConfig(num_layers=1, num_heads=1)
    )
    first = kernel(
        split["query"],
        split["key"],
        split["value"],
        scores=split["scores"],
        layer_index=0,
        attention_mask=split["attention_mask"],
        subject_mask=split["subject_mask"],
        object_mask=split["object_mask"],
    )
    second = kernel(
        split["query"],
        split["key"],
        split["value"],
        scores=split["scores"],
        layer_index=0,
        attention_mask=split["attention_mask"],
        subject_mask=split["subject_mask"],
        object_mask=split["object_mask"],
    )
    context = split["attention_mask"] & ~(
        split["subject_mask"] | split["object_mask"]
    )
    assert torch.equal(first, second)
    assert torch.allclose(
        (first * context[:, None, None, :]).sum(dim=-1),
        torch.zeros_like(first[..., 0]),
        atol=1e-6,
    )
    assert torch.equal(first[..., 1], torch.zeros_like(first[..., 1]))
    assert torch.equal(first[..., 2], torch.zeros_like(first[..., 2]))
    assert first.abs().max() <= 2.0 + 1e-6


def test_signed_phase_separates_motif_from_unsigned_and_diffusion() -> None:
    cfg = config()
    split = make_split(7, 32, torch.device("cpu"))
    baseline = evaluate(
        "disabled",
        split,
        torch.zeros(32, dtype=torch.long),
        cfg,
    )
    prediction = torch.zeros(32, dtype=torch.long)
    signed = evaluate("q_wap_signed", split, prediction, cfg)
    unsigned = evaluate("q_wap_unsigned", split, prediction, cfg)
    classical = evaluate("classical_wap_diffusion", split, prediction, cfg)
    assert baseline["accuracy"] == 0.5
    assert signed["accuracy"] >= 0.95
    assert signed["target_minus_distractor_attention"] > 0.0
    assert unsigned["accuracy"] <= 0.55
    assert classical["accuracy"] <= 0.55

