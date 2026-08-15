#!/usr/bin/env python3
"""Bounded causal evidence screen for Q-VRES v1 transport."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_value_evidence_toy import (  # noqa: E402
    CONTEXT_POSITIONS,
    HEAD_DIM,
    HEADS,
    QUERY_POSITIONS,
    TOKENS,
    batches,
    make_split,
)
from q_attention.plugins.q_causal_value_evidence import (  # noqa: E402
    CausalValueTransportConfig,
    build_causal_value_transport_kernel,
)
from q_attention.experiments.parameter_efficiency import (  # noqa: E402
    QuantumResourceLedger,
    build_parameter_efficiency_manifest,
)


SELECTORS = (
    "disabled",
    "q_causal_transport",
    "classical_causal_transport",
    "q_causal_key_only",
)
PROTOCOLS = ("fixed", "balanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--valid-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--output-root", default="runs/q_causal_value_evidence_toy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="fixed")
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list without duplicates")
    return seeds


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def make_balanced_split(
    seed: int,
    size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Break fixed role/position couplings while keeping the task query-local."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    alternating = torch.arange(size) % 2 == 0
    relation_sign = torch.where(alternating, 1.0, -1.0)
    relation_sign = relation_sign[torch.randperm(size, generator=generator)]
    query_type = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    query_type = query_type.repeat((size + 1) // 2, 1)[:size]
    query_type = query_type[torch.randperm(size, generator=generator)]
    context_swap = torch.arange(size) % 2 == 1
    context_swap = context_swap[torch.randperm(size, generator=generator)]

    query = torch.randn(size, HEADS, TOKENS, HEAD_DIM, generator=generator) * 0.01
    key = torch.randn(size, HEADS, TOKENS, HEAD_DIM, generator=generator) * 0.01
    query[:, 0, QUERY_POSITIONS, 0] = query_type
    key[:, 0, 0, 1] = relation_sign
    key[:, 0, 1, 1] = relation_sign
    labels = (query_type * relation_sign[:, None] < 0).long()

    value = torch.zeros(size, HEADS, TOKENS, HEAD_DIM)
    swap_float = context_swap.to(torch.float32)
    value[:, 0, CONTEXT_POSITIONS[0], 0] = 1.0 - swap_float
    value[:, 0, CONTEXT_POSITIONS[0], 1] = swap_float
    value[:, 0, CONTEXT_POSITIONS[1], 0] = swap_float
    value[:, 0, CONTEXT_POSITIONS[1], 1] = 1.0 - swap_float
    first_context = torch.full((size,), CONTEXT_POSITIONS[0], dtype=torch.long)
    second_context = torch.full((size,), CONTEXT_POSITIONS[1], dtype=torch.long)
    value_zero_position = torch.where(context_swap, second_context, first_context)
    value_one_position = torch.where(context_swap, first_context, second_context)
    target_key = torch.where(
        labels == 0,
        value_zero_position[:, None].expand(-1, len(QUERY_POSITIONS)),
        value_one_position[:, None].expand(-1, len(QUERY_POSITIONS)),
    )
    attention_mask = torch.ones(size, TOKENS, dtype=torch.bool)
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, 0] = True
    object_mask[:, 1] = True
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) / HEAD_DIM**0.5
    return {
        "query": query.to(device),
        "key": key.to(device),
        "value": value.to(device),
        "labels": labels.to(device),
        "query_type": query_type.to(device),
        "relation_sign": relation_sign.to(device),
        "target_key": target_key.to(device),
        "attention_mask": attention_mask.to(device),
        "subject_mask": subject_mask.to(device),
        "object_mask": object_mask.to(device),
        "scores": scores.to(device),
    }


def make_protocol_split(
    seed: int,
    size: int,
    device: torch.device,
    protocol: str,
) -> dict[str, torch.Tensor]:
    if protocol == "fixed":
        return make_split(seed, size, device)
    if protocol == "balanced":
        return make_balanced_split(seed, size, device)
    raise ValueError(f"unknown protocol: {protocol}")


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_kernel(selector: str, seed: int) -> nn.Module | None:
    if selector == "disabled":
        return None
    kind = "classical" if selector == "classical_causal_transport" else "quantum"
    mode = "key_only" if selector == "q_causal_key_only" else "leave_one_out"
    return build_causal_value_transport_kernel(
        kind,
        CausalValueTransportConfig(
            num_layers=1,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            register_qubits=2,
            depth=2,
            max_transport=0.75,
            initial_transport=0.25,
            value_feature_mode=mode,
            seed=seed + 5000,
        ),
    )


def build_parameter_efficiency_manifests(
    revision: str,
    protocol: str,
) -> list[dict[str, Any]]:
    manifests = []
    for selector in SELECTORS:
        kernel = build_kernel(selector, seed=5000)
        modules: dict[str, object] = {"classifier": nn.Linear(2, 2)}
        if kernel is not None:
            modules["intervention"] = kernel
        manifest = build_parameter_efficiency_manifest(
            candidate_id=f"q_vres_{protocol}_{selector}",
            mechanism=selector,
            modules=modules,
            resources=QuantumResourceLedger(
                assumptions=(
                    "CPU statevector simulation for mechanism screening only",
                    "ideal-quantum state preparation, overlap estimation, and readout costs are not measured here",
                ),
            ),
            controls=tuple(value for value in SELECTORS if value != selector),
            code_revision=revision,
            dataset_identity=f"synthetic_causal_value_evidence_{protocol}_v1",
            metadata={
                "protocol": protocol,
                "num_layers": 1,
                "num_heads": HEADS,
                "head_dim": HEAD_DIM,
                "register_qubits": 2,
                "depth": 2,
                "value_feature_mode": (
                    "key_only" if selector == "q_causal_key_only" else "leave_one_out"
                ),
            },
        )
        manifests.append(manifest.to_dict())
    return manifests


def forward(
    kernel: nn.Module | None,
    classifier: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kernel is None:
        residual = torch.zeros_like(batch["scores"])
    else:
        residual = kernel(
            batch["query"],
            batch["key"],
            batch["value"],
            scores=batch["scores"],
            layer_index=0,
            attention_mask=batch["attention_mask"],
            subject_mask=batch["subject_mask"],
            object_mask=batch["object_mask"],
        )
    attention = torch.softmax(batch["scores"] + residual, dim=-1)
    pooled = torch.einsum("bhqk,bhkd->bhqd", attention, batch["value"])
    query_repr = pooled[:, :, QUERY_POSITIONS, :].reshape(-1, HEAD_DIM)
    logits = classifier(query_repr[:, :2])
    return residual, attention, logits


def _target_values(
    attention: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    query_attention = attention[:, :, QUERY_POSITIONS, :]
    base_full_attention = torch.softmax(batch["scores"], dim=-1)
    base_attention = base_full_attention[:, :, QUERY_POSITIONS, :]
    context = batch["attention_mask"] & ~(
        batch["subject_mask"] | batch["object_mask"]
    )
    context_mask = context[:, None, None, :]
    target_indices = batch["target_key"][:, None, :, None].expand(
        -1, query_attention.shape[1], -1, -1
    )
    context_attention = query_attention * context_mask
    base_context_attention = base_attention * context_mask
    context_total = context_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    base_context_total = base_context_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_context_mass_by_query = (
        context_attention / context_total
    ).gather(-1, target_indices).squeeze(-1).mean(dim=1)
    base_target_context_mass_by_query = (
        base_context_attention / base_context_total
    ).gather(-1, target_indices).squeeze(-1).mean(dim=1)
    context_target_top1_by_query = (
        context_attention.masked_fill(~context_mask, -torch.inf)
        .argmax(dim=-1)
        == batch["target_key"][:, None, :]
    ).float().mean(dim=1)
    base_context_target_top1_by_query = (
        base_context_attention.masked_fill(~context_mask, -torch.inf)
        .argmax(dim=-1)
        == batch["target_key"][:, None, :]
    ).float().mean(dim=1)

    output = torch.einsum("bhqk,bhkd->bhqd", query_attention, batch["value"])
    leave_one_out_delta = (
        query_attention.unsqueeze(-1)
        / (1.0 - query_attention).clamp_min(0.05).unsqueeze(-1)
        * (batch["value"][:, :, None, :, :] - output[:, :, :, None, :])
    )
    influence = leave_one_out_delta.norm(dim=-1)
    target_influence_by_query = influence.gather(-1, target_indices).squeeze(-1).mean(dim=1)
    context_influence_top1_by_query = (
        influence.masked_fill(~context_mask, -torch.inf)
        .argmax(dim=-1)
        == batch["target_key"][:, None, :]
    ).float().mean(dim=1)
    base_output = torch.einsum("bhqk,bhkd->bhqd", base_full_attention, batch["value"])
    base_leave_one_out_delta = (
        base_attention.unsqueeze(-1)
        / (1.0 - base_attention).clamp_min(0.05).unsqueeze(-1)
        * (
            batch["value"][:, :, None, :, :]
            - base_output[:, :, QUERY_POSITIONS, :][:, :, :, None, :]
        )
    )
    base_influence = base_leave_one_out_delta.norm(dim=-1)
    base_target_influence_by_query = (
        base_influence.gather(-1, target_indices).squeeze(-1).mean(dim=1)
    )
    context_mass_error_by_query = (
        (query_attention * context_mask).sum(dim=-1)
        - (base_attention * context_mask).sum(dim=-1)
    ).abs().amax(dim=1)
    entity_mask = (batch["subject_mask"] | batch["object_mask"])[:, None, None, :]
    entity_mass_error_by_query = (
        (query_attention * entity_mask).sum(dim=-1)
        - (base_attention * entity_mask).sum(dim=-1)
    ).abs().amax(dim=1)
    return {
        "target_context_mass": target_context_mass_by_query.mean(dim=1),
        "baseline_target_context_mass": base_target_context_mass_by_query.mean(dim=1),
        "context_target_top1": context_target_top1_by_query.mean(dim=1),
        "baseline_context_target_top1": base_context_target_top1_by_query.mean(dim=1),
        "target_influence": target_influence_by_query.mean(dim=1),
        "context_influence_top1": context_influence_top1_by_query.mean(dim=1),
        "baseline_target_influence": base_target_influence_by_query.mean(dim=1),
        "context_mass_error": context_mass_error_by_query.amax(dim=1),
        "entity_mass_error": entity_mass_error_by_query.amax(dim=1),
        "target_context_mass_by_query": target_context_mass_by_query,
        "baseline_target_context_mass_by_query": base_target_context_mass_by_query,
        "context_target_top1_by_query": context_target_top1_by_query,
        "baseline_context_target_top1_by_query": base_context_target_top1_by_query,
        "target_influence_by_query": target_influence_by_query,
        "context_influence_top1_by_query": context_influence_top1_by_query,
        "baseline_target_influence_by_query": base_target_influence_by_query,
        "context_mass_error_by_query": context_mass_error_by_query,
        "entity_mass_error_by_query": entity_mass_error_by_query,
    }


DIAGNOSTIC_METRICS = (
    "query_accuracy",
    "context_target_top1",
    "baseline_context_target_top1",
    "context_target_top1_gain",
    "context_target_mass",
    "baseline_context_target_mass",
    "context_target_mass_gain",
    "target_counterfactual_influence",
    "baseline_target_counterfactual_influence",
    "target_counterfactual_influence_gain",
    "context_influence_top1",
    "context_mass_error",
    "entity_mass_error",
)


def _summarize_group_metrics(
    values: dict[str, torch.Tensor],
    groups: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for group_name, group_mask in groups.items():
        selected = group_mask.reshape(-1)
        count = int(selected.sum().item())
        stats: dict[str, float | int | None] = {"count": count}
        for metric_name, tensor in values.items():
            flattened = tensor.reshape(-1)
            stats[metric_name] = (
                float(flattened[selected].mean().item()) if count else None
            )
        summary[group_name] = stats
    return summary


def evaluate(
    kernel: nn.Module | None,
    classifier: nn.Module,
    split: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, float]:
    classifier.eval()
    if kernel is not None:
        kernel.eval()
    total = 0
    correct = 0
    metrics: list[dict[str, torch.Tensor]] = []
    residual_rms: list[torch.Tensor] = []
    query_correct: list[torch.Tensor] = []
    relation_signs: list[torch.Tensor] = []
    target_keys: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches(split, batch_size):
            residual, attention, logits = forward(kernel, classifier, batch)
            labels = batch["labels"].reshape(-1)
            correct += int((logits.argmax(dim=-1) == labels).sum())
            total += labels.numel()
            metrics.append(_target_values(attention, batch))
            residual_rms.append(residual.square().mean().sqrt().expand(labels.shape[0]))
            batch_size_actual = batch["labels"].shape[0]
            logits_by_query = logits.reshape(batch_size_actual, len(QUERY_POSITIONS), -1)
            query_correct.append(
                logits_by_query.argmax(dim=-1).eq(batch["labels"]).to(torch.float32)
            )
            relation_signs.append(batch["relation_sign"])
            target_keys.append(batch["target_key"])
    metric_values = {
        key: torch.cat([item[key] for item in metrics])
        for key in metrics[0]
    }
    target_mass = metric_values["target_context_mass"].mean().item()
    base_mass = metric_values["baseline_target_context_mass"].mean().item()
    target_top1 = metric_values["context_target_top1"].mean().item()
    base_top1 = metric_values["baseline_context_target_top1"].mean().item()
    target_influence = metric_values["target_influence"].mean().item()
    influence_top1 = metric_values["context_influence_top1"].mean().item()
    base_target_influence = metric_values["baseline_target_influence"].mean().item()
    context_mass_error = metric_values["context_mass_error"].mean().item()
    entity_mass_error = metric_values["entity_mass_error"].mean().item()
    query_values = {
        "query_accuracy": torch.cat(query_correct),
        "context_target_top1": metric_values["context_target_top1_by_query"],
        "baseline_context_target_top1": metric_values[
            "baseline_context_target_top1_by_query"
        ],
        "context_target_mass": metric_values["target_context_mass_by_query"],
        "baseline_context_target_mass": metric_values[
            "baseline_target_context_mass_by_query"
        ],
        "target_counterfactual_influence": metric_values["target_influence_by_query"],
        "baseline_target_counterfactual_influence": metric_values[
            "baseline_target_influence_by_query"
        ],
        "context_influence_top1": metric_values["context_influence_top1_by_query"],
        "context_mass_error": metric_values["context_mass_error_by_query"],
        "entity_mass_error": metric_values["entity_mass_error_by_query"],
    }
    query_values["context_target_top1_gain"] = (
        query_values["context_target_top1"]
        - query_values["baseline_context_target_top1"]
    )
    query_values["context_target_mass_gain"] = (
        query_values["context_target_mass"]
        - query_values["baseline_context_target_mass"]
    )
    query_values["target_counterfactual_influence_gain"] = (
        query_values["target_counterfactual_influence"]
        - query_values["baseline_target_counterfactual_influence"]
    )
    total_examples = query_values["query_accuracy"].shape[0]
    diagnostic_device = query_values["query_accuracy"].device
    query_groups = {
        str(position): torch.arange(len(QUERY_POSITIONS), device=diagnostic_device)[None, :]
        .eq(index)
        .expand(total_examples, -1)
        for index, position in enumerate(QUERY_POSITIONS)
    }
    relation_sign = torch.cat(relation_signs).to(torch.int64)
    target_key = torch.cat(target_keys).to(torch.int64)
    relation_groups = {
        str(sign): relation_sign[:, None].eq(sign).expand(-1, len(QUERY_POSITIONS))
        for sign in (-1, 1)
    }
    target_groups = {
        str(position): target_key.eq(position)
        for position in CONTEXT_POSITIONS
    }
    return {
        "query_accuracy": correct / max(total, 1),
        "context_target_top1": target_top1,
        "baseline_context_target_top1": base_top1,
        "context_target_top1_gain": target_top1 - base_top1,
        "context_target_mass": target_mass,
        "baseline_context_target_mass": base_mass,
        "context_target_mass_gain": target_mass - base_mass,
        "target_counterfactual_influence": target_influence,
        "baseline_target_counterfactual_influence": base_target_influence,
        "target_counterfactual_influence_gain": target_influence - base_target_influence,
        "context_influence_top1": influence_top1,
        "context_mass_error": context_mass_error,
        "entity_mass_error": entity_mass_error,
        "residual_rms": torch.cat(residual_rms).mean().item(),
        "diagnostics": {
            "query_position": _summarize_group_metrics(query_values, query_groups),
            "relation_sign": _summarize_group_metrics(query_values, relation_groups),
            "target_position": _summarize_group_metrics(query_values, target_groups),
        },
    }


def run_selector(selector: str, seed: int, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(seed)
    protocol = getattr(args, "protocol", "fixed")
    train = make_protocol_split(seed, args.train_size, device, protocol)
    valid = make_protocol_split(seed + 10000, args.valid_size, device, protocol)
    kernel = build_kernel(selector, seed)
    classifier = nn.Linear(2, 2).to(device)
    modules = [classifier] + ([] if kernel is None else [kernel])
    optimizer = torch.optim.AdamW(
        [parameter for module in modules for parameter in module.parameters()], lr=args.lr
    )
    train_batches = list(batches(train, args.batch_size))
    started = time.perf_counter()
    for step in range(args.steps):
        for module in modules:
            module.train()
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        _residual, _attention, logits = forward(kernel, classifier, batch)
        loss = F.cross_entropy(logits, batch["labels"].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss selector={selector} step={step}")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for module in modules for parameter in module.parameters()
        ):
            raise FloatingPointError(f"non-finite gradient selector={selector} step={step}")
        optimizer.step()
    metrics = evaluate(kernel, classifier, valid, args.batch_size)
    metrics.update(
        {
            "selector": selector,
            "seed": seed,
            "steps": args.steps,
            "intervention_parameters": 0 if kernel is None else sum(p.numel() for p in kernel.parameters()),
            "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
            "total_trainable_parameters": sum(
                p.numel() for module in modules for p in module.parameters()
            ),
            "runtime_seconds": time.perf_counter() - started,
            "finite_parameters": all(
                torch.isfinite(p).all().item()
                for module in modules for p in module.parameters()
            ),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    if min(args.steps, args.train_size, args.valid_size, args.batch_size) <= 0:
        raise ValueError("steps, sizes, and batch-size must be positive")
    device = choose_device(args.device)
    seeds = parse_seeds(args.seeds)
    output = Path(args.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for selector in SELECTORS:
        for seed in seeds:
            results.append(run_selector(selector, seed, device, args))
            print(json.dumps(results[-1], sort_keys=True), flush=True)
    summary = {
        "schema_version": "q-attention.causal-value-transport-screen.v2",
        "revision": git_revision(),
        "device": str(device),
        "protocol": args.protocol,
        "parameter_efficiency_manifests": build_parameter_efficiency_manifests(
            git_revision(), args.protocol
        ),
        "selectors": list(SELECTORS),
        "seeds": seeds,
        "query_positions": list(QUERY_POSITIONS),
        "context_positions": list(CONTEXT_POSITIONS),
        "steps": args.steps,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
        "batch_size": args.batch_size,
        "results": results,
        "interpretation": {
            "context_target_metrics_are_primary": True,
            "counterfactual_influence_is_reported": True,
            "entity_attention_is_transport_invariant": True,
            "classical_control_is_parameter_matched": True,
            "key_only_is_value_ablation": True,
            "query_position_diagnostics": True,
            "relation_sign_diagnostics": True,
            "target_position_diagnostics": True,
            "task_is_synthetic_screening_only": True,
            "no_hardware_quantum_claim": True,
        },
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "parameter_efficiency_manifests.json").write_text(
        json.dumps(summary["parameter_efficiency_manifests"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
