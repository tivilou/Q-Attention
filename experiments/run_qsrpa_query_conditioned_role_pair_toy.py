#!/usr/bin/env python3
"""Mechanism-only screen for query-conditioned soft role-pair routing.

The router sees only query/key vectors and padding-equivalent validity masks.
Latent role slots are retained for offline mechanism metrics and never enter
the router or score action construction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
QUERIES = 2
KEYS = 6
DIM = 4
ROLES = 2
METHODS = (
    "disabled",
    "global_soft_role",
    "query_conditioned_soft_role",
    "classical_bilinear_role",
    "query_shuffled",
)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.qsrpa-query-conditioned-role-pair-toy.v1":
        raise ValueError("unsupported Q-SRPA query-conditioned toy schema")
    required = {"seeds", "dataset", "router", "gate", "output_root"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if tuple(payload["seeds"]) != (7, 11, 13, 17, 23):
        raise ValueError("the mechanism screen is predeclared for seeds 7,11,13,17,23")
    dataset = payload["dataset"]
    expected = {
        "queries": QUERIES,
        "keys": KEYS,
        "dim": DIM,
    }
    if any(dataset.get(name) != value for name, value in expected.items()):
        raise ValueError("dataset dimensions differ from the predeclared contract")
    return payload


def make_split(seed: int, size: int, device: torch.device, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_query = torch.randn(size, DIM, generator=generator)
    base_query = base_query / base_query.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    query = torch.stack((base_query, -base_query), dim=1)
    shared_key = torch.randn(
        size,
        KEYS,
        DIM,
        generator=generator,
    ) * float(config["dataset"]["distractor_scale"])
    shared_role_slots = torch.stack(
        [torch.randperm(KEYS, generator=generator)[:ROLES] for _ in range(size)]
    )
    role_slots = torch.stack(
        (shared_role_slots, shared_role_slots.flip(dims=(-1,))), dim=1
    )
    rows = torch.arange(size)
    noise = float(config["dataset"]["noise"])
    positive = base_query + noise * torch.randn(size, DIM, generator=generator)
    negative = -base_query + noise * torch.randn(size, DIM, generator=generator)
    shared_key[rows, shared_role_slots[:, 0]] = positive
    shared_key[rows, shared_role_slots[:, 1]] = negative
    key = shared_key[:, None, :, :].expand(-1, QUERIES, -1, -1).clone()
    distractor_slot = torch.stack(
        [torch.tensor([slot for slot in range(KEYS) if slot not in pair.tolist()][0]) for pair in shared_role_slots]
    )
    distractor_slot = distractor_slot[:, None].expand(-1, QUERIES)
    rows2 = torch.arange(size)[:, None]
    queries = torch.arange(QUERIES)[None, :]
    scores = 0.03 * torch.randn(size, QUERIES, KEYS, generator=generator)
    scores[rows2, queries, distractor_slot] += float(config["dataset"]["baseline_bias"])
    return {
        "query": query.to(device),
        "key": key.to(device),
        "scores": scores.to(device),
        "role_slots": role_slots.to(device),
        "shared_role_slots": shared_role_slots.to(device),
        "attention_mask": torch.ones(size, QUERIES, KEYS, dtype=torch.bool, device=device),
    }


def role_logits(method: str, split: dict[str, torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    query = split["query"]
    key = split["key"]
    if method == "query_shuffled":
        query = torch.roll(query, shifts=1, dims=0)
        method = "query_conditioned_soft_role"
    if method == "disabled":
        return torch.zeros(
            query.shape[0], query.shape[1], ROLES, KEYS,
            device=query.device,
            dtype=query.dtype,
        )
    if method == "global_soft_role":
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["router"]["global_projection_seed_offset"])
        )
        projection = torch.randn(DIM, generator=generator)
        projection = projection / projection.norm().clamp_min(1e-8)
        base = torch.einsum("bqkd,d->bqk", key, projection.to(key.device))
        return torch.stack((base, -base), dim=2)
    compatibility = torch.einsum("bqd,bqkd->bqk", query, key)
    if method == "query_conditioned_soft_role":
        return torch.stack((compatibility, -compatibility), dim=2)
    if method == "classical_bilinear_role":
        # Strong matched control: the same query-key bilinear statistic with
        # an identity frame. This intentionally tests classical replicability.
        compatibility = torch.einsum("bqd,bqkd->bqk", query, key)
        return torch.stack((compatibility, -compatibility), dim=2)
    raise ValueError(f"unknown method: {method}")


def role_weights(method: str, split: dict[str, torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    logits = role_logits(method, split, config)
    if method == "disabled":
        return torch.full_like(logits, 1.0 / KEYS)
    temperature = float(config["router"]["temperature"])
    return torch.softmax(logits / temperature, dim=-1)


def apply_role_action(split: dict[str, torch.Tensor], weights: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    source = weights[:, :, 0].argmax(dim=-1)
    target = weights[:, :, 1].argmax(dim=-1)
    residual = torch.zeros_like(split["scores"])
    rows = torch.arange(split["scores"].shape[0], device=split["scores"].device)[:, None]
    queries = torch.arange(QUERIES, device=split["scores"].device)[None, :]
    delta = float(config["dataset"]["action_delta"])
    residual[rows, queries, source] += delta
    residual[rows, queries, target] -= delta
    return split["scores"] + residual, residual


def attention_mass(scores: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(scores, dim=-1)
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    queries = torch.arange(QUERIES, device=scores.device)[None, :]
    return weights[rows, queries, slots]


def evaluate(method: str, split: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, Any]:
    weights = role_weights(method, split, config)
    role_slots = split["role_slots"]
    rows = torch.arange(weights.shape[0], device=weights.device)[:, None]
    queries = torch.arange(QUERIES, device=weights.device)[None, :]
    target_mass = weights.gather(-1, role_slots.unsqueeze(-1)).squeeze(-1)
    top = weights.argmax(dim=-1)
    role_hit = (top == role_slots).float().mean(dim=(0, 1))
    pair_distinct = (top[:, :, 0] != top[:, :, 1]).float().mean()
    steered_scores, residual = apply_role_action(split, weights, config)
    base_source_mass = attention_mass(split["scores"], role_slots[:, :, 0])
    base_target_mass = attention_mass(split["scores"], role_slots[:, :, 1])
    steered_source_mass = attention_mass(steered_scores, role_slots[:, :, 0])
    steered_target_mass = attention_mass(steered_scores, role_slots[:, :, 1])
    return {
        "role_hit_source": float(role_hit[0].item()),
        "role_hit_target": float(role_hit[1].item()),
        "role_hit_mean": float(role_hit.mean().item()),
        "target_role_mass_mean": float(target_mass.mean().item()),
        "pair_distinct_rate": float(pair_distinct.item()),
        "source_attention_gain": float((steered_source_mass - base_source_mass).mean().item()),
        "target_attention_delta": float((steered_target_mass - base_target_mass).mean().item()),
        "action_alignment_gain": float(((steered_source_mass - steered_target_mass) - (base_source_mass - base_target_mass)).mean().item()),
        "zero_sum_error": float(residual.sum(dim=-1).abs().max().item()),
        "max_abs_residual": float(residual.abs().max().item()),
        "active_rate": float((top[:, :, 0] != top[:, :, 1]).float().mean().item()),
    }


def reversal_consistency(method: str, split: dict[str, torch.Tensor], config: dict[str, Any]) -> float:
    reversed_split = dict(split)
    reversed_split["query"] = -split["query"]
    first = role_weights(method, split, config)
    second = role_weights(method, reversed_split, config)
    # Query reversal should swap source and target distributions for a query-local router.
    swap_error = 0.5 * (
        (first[:, :, 0] - second[:, :, 1]).abs().mean()
        + (first[:, :, 1] - second[:, :, 0]).abs().mean()
    )
    return float((1.0 - swap_error).item())


def evaluate_seed(seed: int, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    train = make_split(seed, int(config["dataset"]["train_size"]), device, config)
    valid = make_split(seed + 10000, int(config["dataset"]["valid_size"]), device, config)
    methods = {method: evaluate(method, valid, config) for method in METHODS}
    reversal = {method: reversal_consistency(method, valid, config) for method in METHODS}
    gate = config["gate"]
    candidate = methods["query_conditioned_soft_role"]
    classical = methods["classical_bilinear_role"]
    global_method = methods["global_soft_role"]
    conditions = {
        "query_role_hit": candidate["role_hit_mean"] >= float(gate["minimum_query_role_hit"]),
        "query_reversal_consistency": reversal["query_conditioned_soft_role"] >= float(gate["minimum_query_reversal_consistency"]),
        "action_alignment": candidate["action_alignment_gain"] >= float(gate["minimum_action_alignment_gain"]),
        "beats_global_role_hit": candidate["role_hit_mean"] - global_method["role_hit_mean"] >= float(gate["minimum_query_vs_global_role_hit_gain"]),
        "classical_gap_within_tolerance": candidate["role_hit_mean"] + float(gate["maximum_candidate_vs_classical_role_hit_gap"]) >= classical["role_hit_mean"],
        "residual_invariants": candidate["zero_sum_error"] <= 1e-6 and candidate["max_abs_residual"] <= float(config["dataset"]["action_delta"]) + 1e-6,
    }
    return {
        "seed": seed,
        "train_examples": int(train["query"].shape[0]),
        "valid_examples": int(valid["query"].shape[0]),
        "methods": methods,
        "reversal_consistency": reversal,
        "gate": {**conditions, "status": "pass" if all(conditions.values()) else "fail"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qsrpa_query_conditioned_role_pair_toy.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = choose_device(args.device or str(config.get("device", "cpu")))
    seed_rows = [evaluate_seed(int(seed), config, device) for seed in config["seeds"]]
    required = math.ceil(
        len(seed_rows) * float(config["gate"]["required_seed_fraction"])
    )
    passed = sum(row["gate"]["status"] == "pass" for row in seed_rows)
    gate = {
        "passed_seeds": passed,
        "total_seeds": len(seed_rows),
        "required_passed_seeds": required,
        "status": "pass" if passed >= required else "fail",
        "next_plugin_design_authorized": False,
        "quantum_estimator_run": False,
        "real_data_run": False,
    }
    if gate["status"] == "pass":
        gate["next_plugin_design_authorized"] = True
    payload = {
        "schema_version": "q-attention.qsrpa-query-conditioned-role-pair-toy.v1",
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "dataset_identity": config["dataset"]["identity"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "label_free_action_path": True,
        "latent_role_slots_offline_only": True,
        "trainable_router_parameters": {
            "global_soft_role": 0,
            "query_conditioned_soft_role": 0,
            "classical_bilinear_role": 0,
            "query_shuffled": 0
        },
        "seed_rows": seed_rows,
        "gate": gate,
        "limitations": [
            "Mechanism-only synthetic evidence; no task utility or natural transfer claim.",
            "The classical bilinear control is an attribution control, not a quantum advantage claim.",
            "All routers are fixed zero-trainable-parameter structural controls; the screen tests query conditioning, not learned capacity.",
            "Role slots are used only for offline evaluation and never for router construction.",
        ],
    }
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
