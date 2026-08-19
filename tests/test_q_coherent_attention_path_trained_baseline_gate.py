from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from q_attention.models import RelationExtractionModel, RelationTransformerConfig  # noqa: E402
from run_q_coherent_attention_path_trained_baseline_gate import (  # noqa: E402
    ChiralRelationContrastKernel,
    DirectedRelationContrastKernel,
    build_selector,
    geometry_diagnostics,
    make_split,
    score_intervention,
    split_diagnostics,
    stage_b_gate,
)


def config() -> dict:
    return json.loads(
        (ROOT / "configs" / "q_coherent_attention_path_trained_baseline_gate.json").read_text(
            encoding="utf-8"
        )
    )


def test_structural_streams_are_balanced_deterministic_and_disjoint() -> None:
    seen: set[tuple[int, ...]] = set()
    train = make_split(1701, 64, 32, torch.device("cpu"), seen)
    valid = make_split(1702, 32, 32, torch.device("cpu"), seen)
    replay = make_split(1701, 64, 32, torch.device("cpu"), set())
    assert torch.equal(train["input_ids"], replay["input_ids"])
    assert float(train["labels"].float().mean()) == 0.5
    assert float(valid["labels"].float().mean()) == 0.5
    diagnostics = split_diagnostics({"train": train, "valid": valid})
    assert diagnostics["exact_split_overlap"] == 0


def test_score_capture_geometry_detects_signed_cycles_and_asymmetry() -> None:
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=39,
            num_labels=2,
            dim=16,
            num_layers=1,
            num_heads=2,
            ff_dim=32,
            dropout=0.0,
            max_length=7,
        )
    )
    split = make_split(1701, 16, 32, torch.device("cpu"), set())
    batch = {key: value for key, value in split.items() if isinstance(value, torch.Tensor)}
    captured: list[dict[str, torch.Tensor]] = []

    def hook(_module, inputs, _output):
        captured.append({"scores": inputs[0].detach()})

    handle = model.encoder.layers[0].attn.score_intervention.register_forward_hook(hook)
    try:
        _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["subject_mask"],
            batch["object_mask"],
        )
    finally:
        handle.remove()
    diagnostics = geometry_diagnostics(captured, batch, config()["mechanism"]["walk_time"])
    assert diagnostics["nonzero_cycle_fraction"] > 0.9
    assert diagnostics["raw_score_asymmetry_max"] > 0.0
    assert diagnostics["signed_unsigned_probability_tv_max"] >= 0.0


def test_subject_object_query_scope_masks_non_anchor_rows() -> None:
    class ConstantKernel(torch.nn.Module):
        def forward(self, _query, _key, _value, *, scores, **_kwargs):
            return torch.ones_like(scores)

    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=39,
            num_labels=2,
            dim=16,
            num_layers=1,
            num_heads=2,
            ff_dim=32,
            dropout=0.0,
            max_length=7,
        )
    )
    split = make_split(1701, 4, 32, torch.device("cpu"), set())
    batch = {key: value for key, value in split.items() if isinstance(value, torch.Tensor)}
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    try:
        with score_intervention(
            model,
            ConstantKernel(),
            batch,
            query_scope="subject_object",
        ):
            handle = model.encoder.layers[0].attn.score_intervention.register_forward_hook(hook)
            _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["subject_mask"],
                batch["object_mask"],
            )
    finally:
        handle.remove()
    query_mask = (batch["subject_mask"] | batch["object_mask"])[:, None, :, None]
    original: list[torch.Tensor] = []
    handle = model.encoder.layers[0].attn.score_intervention.register_forward_hook(
        lambda _module, _inputs, output: original.append(output.detach())
    )
    try:
        _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["subject_mask"],
            batch["object_mask"],
        )
    finally:
        handle.remove()
    delta = captured[0] - original[0]
    assert torch.allclose(
        delta.masked_select(query_mask.expand_as(delta)),
        torch.ones_like(delta.masked_select(query_mask.expand_as(delta))),
        atol=1e-6,
    )
    assert torch.allclose(
        delta.masked_select(~query_mask.expand_as(delta)),
        torch.zeros_like(delta.masked_select(~query_mask.expand_as(delta))),
        atol=1e-6,
    )


def test_relation_contrast_is_anchor_directed_zero_sum_and_parameter_matched() -> None:
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=39,
            num_labels=2,
            dim=16,
            num_layers=1,
            num_heads=2,
            ff_dim=32,
            dropout=0.0,
            max_length=7,
        )
    )
    cfg = config()
    cfg["mechanism"]["readout"] = "relation_contrast"
    kernels = [
        build_selector(name, model, cfg)
        for name in (
            "q_wap_signed",
            "q_wap_unsigned",
            "classical_wap_diffusion",
            "direct_row",
        )
    ]
    assert len({sum(parameter.numel() for parameter in kernel.parameters()) for kernel in kernels}) == 1
    split = make_split(1701, 4, 32, torch.device("cpu"), set())
    batch = {key: value for key, value in split.items() if isinstance(value, torch.Tensor)}
    scores = torch.randn(4, 2, 7, 7)
    query = torch.randn(4, 2, 7, 8)
    key = torch.randn(4, 2, 7, 8)
    residual = kernels[0](
        query,
        key,
        scores=scores,
        layer_index=0,
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
    )
    query_mask = (batch["subject_mask"] | batch["object_mask"])[:, None, :, None]
    context = batch["attention_mask"] & ~(
        batch["subject_mask"] | batch["object_mask"]
    )
    assert torch.allclose(
        residual.masked_select(~query_mask.expand_as(residual)),
        torch.zeros_like(residual.masked_select(~query_mask.expand_as(residual))),
        atol=1e-6,
    )
    context_sum = (residual * context[:, None, None, :]).sum(dim=-1)
    assert float(context_sum.abs().max()) <= 1e-6


def test_directed_complex_graph_is_hermitian_and_transition_asymmetric() -> None:
    from q_attention.plugins.q_coherent_attention_path import CoherentAttentionPathConfig

    kernel = DirectedRelationContrastKernel(
        CoherentAttentionPathConfig(
            num_layers=1,
            num_heads=1,
            max_transport=2.0,
            initial_transport=0.05,
            walk_time=config()["mechanism"]["walk_time"],
        )
    )
    scores = torch.tensor(
        [[[[0.0, 1.0, -0.4], [-0.2, 0.0, 0.8], [0.6, -0.3, 0.0]]]]
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    graph = kernel._hermitian_graph(scores, mask)
    assert torch.allclose(graph, graph.transpose(-1, -2).conj(), atol=1e-7)
    path = kernel._path_probabilities(graph)
    assert torch.allclose(path.sum(dim=-1), torch.ones_like(path.sum(dim=-1)), atol=1e-6)
    assert float((path - path.transpose(-1, -2)).abs().max()) > 1e-4


def test_stage_b_compares_against_lowest_nll_control() -> None:
    cfg = config()
    results = []
    nlls = {
        "disabled": (0.3, 0.3),
        "q_wap_signed": (0.28, 0.28),
        "q_wap_unsigned": (0.29, 0.29),
        "classical_wap_diffusion": (0.27, 0.27),
        "direct_row": (0.31, 0.31),
        "shuffled_anchor": (0.32, 0.32),
    }
    for selector, (valid_nll, test_nll) in nlls.items():
        results.append(
            {
                "selector": selector,
                "trainable_parameters": 0 if selector == "disabled" else 2,
                "training": {"finite": True},
                "final": {
                    "valid": {"nll": valid_nll, "accuracy": 0.9},
                    "test": {"nll": test_nll, "accuracy": 0.9},
                },
            }
        )
    gate = stage_b_gate(results, cfg)
    assert abs(gate["quantum_over_best_control_nll_advantage"]["valid"] + 0.01) <= 1e-8
    assert gate["valid_control_advantage"] is False


def test_chiral_graph_uses_only_skew_and_is_hermitian() -> None:
    from q_attention.plugins.q_coherent_attention_path import CoherentAttentionPathConfig

    kernel = ChiralRelationContrastKernel(
        CoherentAttentionPathConfig(
            num_layers=1,
            num_heads=1,
            max_transport=2.0,
            initial_transport=0.05,
            walk_time=config()["mechanism"]["walk_time"],
        )
    )
    scores = torch.tensor(
        [[[[0.0, 1.0, -0.4], [-0.2, 0.0, 0.8], [0.6, -0.3, 0.0]]]]
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    graph = kernel._hermitian_graph(scores, mask)
    assert torch.allclose(graph.real, torch.zeros_like(graph.real), atol=1e-7)
    assert torch.allclose(graph, graph.transpose(-1, -2).conj(), atol=1e-7)
    path = kernel._path_probabilities(graph)
    assert torch.allclose(path.sum(dim=-1), torch.ones_like(path.sum(dim=-1)), atol=1e-6)
    assert float((path - path.transpose(-1, -2)).abs().max()) > 1e-4
