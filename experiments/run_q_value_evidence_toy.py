#!/usr/bin/env python3
"""Bounded query-local screen for the value-aware Q-VRES intervention."""

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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins.q_value_evidence import (  # noqa: E402
    ValueEvidenceKernelConfig,
    build_value_evidence_score_kernel,
)


TOKENS = 6
HEADS = 1
HEAD_DIM = 4
QUERY_POSITIONS = (2, 3)
CONTEXT_POSITIONS = (4, 5)
SELECTORS = (
    "disabled",
    "q_vres",
    "classical_value_control",
    "q_vres_key_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--valid-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--output-root", default="runs/q_value_evidence_toy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
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


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def make_split(seed: int, size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Create query/relation pairs where the correct value token is query-local."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    relation_sign = torch.randint(0, 2, (size,), generator=generator).float().mul(2.0).sub(1.0)
    query = torch.randn(size, HEADS, TOKENS, HEAD_DIM, generator=generator) * 0.01
    key = torch.randn(size, HEADS, TOKENS, HEAD_DIM, generator=generator) * 0.01
    query[:, 0, 2, 0] = 1.0
    query[:, 0, 3, 0] = -1.0
    key[:, 0, 0, 1] = relation_sign
    key[:, 0, 1, 1] = relation_sign
    query_type = torch.tensor([1.0, -1.0])[None, :].expand(size, -1)
    labels = (query_type * relation_sign[:, None] < 0).long()
    target_key = torch.where(
        labels == 0,
        torch.full_like(labels, CONTEXT_POSITIONS[0]),
        torch.full_like(labels, CONTEXT_POSITIONS[1]),
    )
    value = torch.zeros(size, HEADS, TOKENS, HEAD_DIM)
    value[:, 0, CONTEXT_POSITIONS[0], 0] = 1.0
    value[:, 0, CONTEXT_POSITIONS[1], 1] = 1.0
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


def batches(split: dict[str, torch.Tensor], batch_size: int):
    size = split["labels"].shape[0]
    for start in range(0, size, batch_size):
        yield {name: value[start : start + batch_size] for name, value in split.items()}


def build_kernel(selector: str, seed: int) -> nn.Module | None:
    if selector == "disabled":
        return None
    kind = "classical" if selector == "classical_value_control" else "quantum"
    mode = "key_only" if selector == "q_vres_key_only" else "leave_one_out"
    return build_value_evidence_score_kernel(
        kind,
        ValueEvidenceKernelConfig(
            num_layers=1,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            register_qubits=2,
            depth=2,
            value_feature_mode=mode,
            initial_gain=0.05,
            max_gain=0.5,
            seed=seed + 3000,
        ),
    )


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
    target_hits = 0
    target_mass: list[torch.Tensor] = []
    target_influence: list[torch.Tensor] = []
    base_target_mass: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches(split, batch_size):
            residual, attention, logits = forward(kernel, classifier, batch)
            labels = batch["labels"].reshape(-1)
            correct += int((logits.argmax(dim=-1) == labels).sum())
            total += labels.numel()
            query_attention = attention[:, :, QUERY_POSITIONS, :]
            base_attention = torch.softmax(batch["scores"], dim=-1)[:, :, QUERY_POSITIONS, :]
            targets = batch["target_key"]
            target_indices = targets[:, None, :, None].expand(
                -1, query_attention.shape[1], -1, -1
            )
            target_mass.append(
                query_attention.gather(-1, target_indices).squeeze(-1).mean(dim=1)
            )
            base_target_mass.append(
                base_attention.gather(-1, target_indices).squeeze(-1).mean(dim=1)
            )
            target_hits += int(
                (query_attention.argmax(dim=-1) == targets[:, None, :]).sum()
            )
            output = torch.einsum(
                "bhqk,bhkd->bhqd", query_attention, batch["value"]
            )
            influence = query_attention.unsqueeze(-1) * (
                batch["value"][:, :, None, :, :] - output[:, :, :, None, :]
            )
            target_influence.append(
                influence.norm(dim=-1)
                .gather(-1, target_indices)
                .squeeze(-1)
                .mean(dim=1)
            )
    target_mass_mean = torch.cat(target_mass).mean().item()
    base_mass_mean = torch.cat(base_target_mass).mean().item()
    return {
        "query_accuracy": correct / max(total, 1),
        "target_top1": target_hits / max(total, 1),
        "target_mass": target_mass_mean,
        "baseline_target_mass": base_mass_mean,
        "target_mass_gain": target_mass_mean - base_mass_mean,
        "target_influence": torch.cat(target_influence).mean().item(),
        "residual_rms": float(residual.square().mean().sqrt().item()),
    }


def run_selector(selector: str, seed: int, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(seed)
    train = make_split(seed, args.train_size, device)
    valid = make_split(seed + 10000, args.valid_size, device)
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
        "schema_version": "q-attention.value-evidence-screen.v1",
        "revision": git_revision(),
        "device": str(device),
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
            "value_aware_metric_is_primary": True,
            "classical_value_control_is_parameter_matched": True,
            "key_only_is_value_ablation": True,
            "task_is_synthetic_screening_only": True,
            "no_hardware_quantum_claim": True,
        },
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
