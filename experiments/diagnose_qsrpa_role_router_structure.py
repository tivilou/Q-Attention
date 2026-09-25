#!/usr/bin/env python3
"""Run a cheap, label-free structural audit of the Q-SRPA role router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins import (  # noqa: E402
    RelationScoreKernelConfig,
    build_relation_attention_score_kernel,
)


def run_diagnostics(*, seed: int = 13, batch: int = 2, heads: int = 3, tokens: int = 7, dim: int = 4) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    key = torch.randn(batch, heads, tokens, dim, generator=generator)
    query_a = torch.randn(batch, heads, tokens, dim, generator=generator)
    query_b = torch.randn(batch, heads, tokens, dim, generator=generator)
    attention_mask = torch.ones(batch, tokens, dtype=torch.bool)
    attention_mask[1, -2:] = False
    subject_mask = torch.zeros(batch, tokens, dtype=torch.bool)
    object_mask = torch.zeros(batch, tokens, dtype=torch.bool)
    subject_mask[:, 0] = True
    object_mask[:, 1] = True

    config = RelationScoreKernelConfig(
        num_layers=2,
        num_heads=heads,
        head_dim=dim,
        num_qubits=4,
        depth=2,
        score_readout="fidelity",
        input_encoding="joint",
        query_scope="all",
        relation_anchor_mode="soft_role_pair",
        role_router_temperature=1.0,
        role_entropy_floor=0.35,
        seed=seed,
    )
    kernel = build_relation_attention_score_kernel("quantum", config)
    kernel.eval()

    diagnostics = kernel.relation_role_diagnostics(key, attention_mask)
    weights = diagnostics["weights"]
    role_context = torch.einsum("bhkr,bhkd->bhrd", weights, key)
    anchor_a = kernel._relation_anchor(key, attention_mask, subject_mask, object_mask)
    flipped_subject = ~subject_mask & attention_mask[:, None]
    flipped_object = ~object_mask & attention_mask[:, None]
    anchor_b = kernel._relation_anchor(key, attention_mask, flipped_subject, flipped_object)
    anchor_query_a = kernel._relation_anchor(key, attention_mask, subject_mask, object_mask)
    anchor_query_b = kernel._relation_anchor(key, attention_mask, subject_mask, object_mask)

    centered = kernel.unmodulated_centered_kernel(
        query_a,
        key,
        layer_index=0,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
    )
    residual = kernel.score_residual(centered, 0)
    role_weight_sum_error = (weights.sum(dim=2) - 1.0).abs().max().item()
    valid_counts = attention_mask.sum(dim=-1).to(weights.dtype)
    effective_fraction = diagnostics["effective_tokens"].mean(dim=(1, 2)) / valid_counts
    query_feature_a, _ = kernel._relation_features(query_a, key, anchor_query_a)
    query_feature_b, _ = kernel._relation_features(query_b, key, anchor_query_b)

    return {
        "schema_version": "qsrpa.role_router_structural_diagnostic.v1",
        "seed": seed,
        "trained_replay": False,
        "checkpoint_available": False,
        "label_free_action_path": True,
        "role_weight_sum_max_error": role_weight_sum_error,
        "mean_normalized_entropy": diagnostics["normalized_entropy"].mean().item(),
        "mean_role_overlap": diagnostics["overlap"].mean().item(),
        "mean_effective_tokens": diagnostics["effective_tokens"].mean().item(),
        "mean_effective_token_fraction": effective_fraction.mean().item(),
        "role_router_antisymmetry_max_error": (
            kernel.role_router[:, 0] + kernel.role_router[:, 1]
        ).abs().max().item(),
        "role_bias_difference_max": (
            kernel.role_router_bias[:, 0] - kernel.role_router_bias[:, 1]
        ).abs().max().item(),
        "span_invariant_anchor": bool(torch.allclose(anchor_a, anchor_b)),
        "query_independent_role_context": bool(torch.allclose(anchor_query_a, anchor_query_b)),
        "query_features_change_when_query_changes": bool(
            not torch.allclose(query_feature_a, query_feature_b)
        ),
        "mean_abs_centered_score": centered.abs().mean().item(),
        "mean_abs_initial_residual": residual.abs().mean().item(),
        "max_abs_initial_residual": residual.abs().max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_diagnostics()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
