#!/usr/bin/env python3
"""Run the bounded Q-RPEC mechanism preflight on synthetic tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from q_attention.plugins import (
    LocalRelationEchoCurvatureControl,
    RelationPerturbationEchoConfig,
    RelationPerturbationEchoCurvatureKernel,
)


def _inputs(seed: int, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = int(config["batch_size"])
    query_tokens = int(config["query_tokens"])
    key_tokens = int(config["key_tokens"])
    heads = int(config["num_heads"])
    dim = int(config["head_dim"])
    query = torch.randn(batch, heads, query_tokens, dim, generator=generator, requires_grad=True)
    key = torch.randn(batch, heads, key_tokens, dim, generator=generator, requires_grad=True)
    attention = torch.ones(batch, key_tokens, dtype=torch.bool)
    attention[1, -1] = False
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 1] = True
    object_[:, 3] = True
    object_[1, 3] = False
    object_[1, 2] = True
    return {
        "query": query,
        "key": key,
        "attention_mask": attention,
        "subject_mask": subject,
        "object_mask": object_,
    }


def _kernel(config: dict[str, Any], cls: type[RelationPerturbationEchoCurvatureKernel]) -> RelationPerturbationEchoCurvatureKernel:
    return cls(
        RelationPerturbationEchoConfig(
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            head_dim=int(config["head_dim"]),
            num_qubits=int(config["num_qubits"]),
            perturbation=float(config["perturbation"]),
            seed=271,
        )
    )


def _residual(kernel: RelationPerturbationEchoCurvatureKernel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return kernel(
        batch["query"],
        batch["key"],
        layer_index=0,
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fatal: list[str] = []
    for seed in config["seeds"]:
        batch = _inputs(int(seed), config)
        quantum = _kernel(config, RelationPerturbationEchoCurvatureKernel)
        control = _kernel(config, LocalRelationEchoCurvatureControl)
        control.load_state_dict(quantum.state_dict())
        q_residual = _residual(quantum, batch)
        c_residual = _residual(control, batch)
        loss = q_residual.square().mean()
        loss.backward()
        gradients = [p.grad for p in quantum.parameters() if p.requires_grad]
        finite = bool(torch.isfinite(q_residual).all() and torch.isfinite(c_residual).all())
        grad_finite = bool(gradients and all(g is not None and torch.isfinite(g).all() for g in gradients))
        grad_norm = float(torch.sqrt(sum(g.square().sum() for g in gradients if g is not None)).item())
        attention = batch["attention_mask"]
        entity = batch["subject_mask"] | batch["object_mask"]
        context = attention & ~entity
        context_sum = (q_residual * context[:, None, None, :].to(q_residual.dtype)).sum(dim=-1).abs().max()
        entity_max = (q_residual * entity[:, None, None, :].to(q_residual.dtype)).abs().max()
        masked_max = (q_residual * (~attention)[:, None, None, :].to(q_residual.dtype)).abs().max()
        gap = (q_residual - c_residual).abs().max()
        permuted_key = batch["key"].flip(dims=(2,))
        permuted = dict(batch)
        permuted["key"] = permuted_key
        shuffle_gap = (_residual(quantum, permuted) - q_residual).abs().mean()
        replay = _residual(quantum, batch).detach()
        replay_error = (replay - q_residual.detach()).abs().max()
        row = {
            "seed": int(seed),
            "finite": finite,
            "gradient_finite": grad_finite,
            "gradient_norm": grad_norm,
            "action_norm": float(q_residual.norm().item()),
            "context_zero_sum_max": float(context_sum.item()),
            "entity_action_max": float(entity_max.item()),
            "masked_action_max": float(masked_max.item()),
            "quantum_control_gap_max": float(gap.item()),
            "relation_shuffle_gap_mean": float(shuffle_gap.item()),
            "deterministic_replay_error": float(replay_error.item()),
            "parameter_count_quantum": quantum.parameter_count,
            "parameter_count_control": control.parameter_count,
        }
        rows.append(row)
        if not finite or not grad_finite:
            fatal.append(f"seed={seed}: nonfinite score or gradient")
        if row["action_norm"] <= 1e-8:
            fatal.append(f"seed={seed}: zero action")
        if row["context_zero_sum_max"] > 1e-5 or row["entity_action_max"] > 1e-7 or row["masked_action_max"] > 1e-7:
            fatal.append(f"seed={seed}: mask/gauge invariant failure")
        if row["quantum_control_gap_max"] <= 1e-7:
            fatal.append(f"seed={seed}: exact local-control replay")
        if row["relation_shuffle_gap_mean"] <= 1e-7:
            fatal.append(f"seed={seed}: relation-shuffle insensitivity")
        if row["deterministic_replay_error"] > 1e-7:
            fatal.append(f"seed={seed}: nondeterministic replay")
        if row["parameter_count_quantum"] != row["parameter_count_control"]:
            fatal.append(f"seed={seed}: parameter mismatch")
    return {
        "experiment": config.get("experiment", "q_rpec_toy_preflight"),
        "status": "fail" if fatal else "pass",
        "seeds": [int(seed) for seed in config["seeds"]],
        "rows": rows,
        "fatal": fatal,
        "claim_ceiling": "mechanism_preflight_only",
        "formal_training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/q_rpec_toy_preflight.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run(config)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
