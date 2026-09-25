#!/usr/bin/env python3
"""Seed-7 readout-only gate for the QCDD coherent-minus-dephased witness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_q_consensus_error_witness_prescreen_toy as task  # noqa: E402
from q_attention.plugins.q_coherence_destruction_differential import (  # noqa: E402
    CoherenceDifferentialConfig,
    build_coherence_differential,
)
from q_attention.plugins.q_connected_consensus_witness import unordered_pair_index  # noqa: E402


SELECTORS = (
    "qcdd_coherent_differential",
    "qcdd_dephased_null",
    "qcdd_product_null",
    "qcdd_sincos_control",
    "qcdd_key_shuffle",
    "qccw_raw_xx",
)
PAULI_SETTINGS = ("YYYY", "YIYI", "IYIY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/q_coherence_destruction_differential.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.q-coherence-destruction-differential.v1":
        raise ValueError("unsupported QCDD config schema")
    required = {"selectors", "seed", "device", "dataset", "estimator", "training", "shot_estimate", "gate", "output_root"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if tuple(payload["selectors"]) != SELECTORS:
        raise ValueError("selectors must match the explicit QCDD allowlist")
    if int(payload["estimator"]["num_key_pairs"]) != 15:
        raise ValueError("QCDD requires all fifteen unordered key pairs")
    if int(payload["training"]["steps"]) <= 0 or float(payload["training"]["lr"]) <= 0:
        raise ValueError("training steps and learning rate must be positive")
    if int(payload["shot_estimate"]["maximum_shots_per_candidate_pair"]) != 4096:
        raise ValueError("the frozen QCDD shot ceiling must remain 4096")
    return payload


def batches(split: dict[str, torch.Tensor], batch_size: int):
    for start in range(0, split["labels"].shape[0], batch_size):
        yield {name: value[start : start + batch_size] for name, value in split.items()}


def pair_auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    scores = scores.detach().reshape(-1).float()
    positive = positive.detach().reshape(-1).bool()
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    labels = positive[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(dim=0).to(torch.float32)
    starts = ends - counts.to(torch.float32) + 1.0
    ranks = torch.repeat_interleave(0.5 * (starts + ends), counts)
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def exact_stream_overlaps(
    streams: dict[str, dict[str, torch.Tensor]],
) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    names = tuple(streams)
    for left_index, left_name in enumerate(names):
        left = streams[left_name]["query"].reshape(
            streams[left_name]["query"].shape[0], -1
        )
        for right_name in names[left_index + 1 :]:
            right = streams[right_name]["query"].reshape(
                streams[right_name]["query"].shape[0], -1
            )
            overlaps[f"{left_name}:{right_name}"] = int(
                ((left[:, None] == right[None, :]).all(dim=-1)).sum()
            )
    return overlaps


def build_model(kind: str, seed: int, frames: torch.Tensor, config: dict[str, Any], device: torch.device):
    estimator = config["estimator"]
    model = build_coherence_differential(
        kind,
        CoherenceDifferentialConfig(
            num_candidates=task.v1.CLASSES,
            head_dim=task.v1.DIM,
            num_key_pairs=int(estimator["num_key_pairs"]),
            angle_scale=float(estimator["angle_scale"]),
            seed=seed + int(estimator["seed_offset"]),
        ),
        frames,
    )
    return model.to(device)


def qcdd_training_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], pair_loss_weight: float) -> torch.Tensor:
    pair_scores = model.pair_scores(batch["query"], batch["key"])
    candidate_scores = pair_scores.max(dim=-1).values
    candidate_loss = F.cross_entropy(candidate_scores.reshape(-1, candidate_scores.shape[-1]), batch["labels"].reshape(-1))
    target_pair = unordered_pair_index(batch["evidence_slot"], model.pair_indices)
    target_score = pair_scores.gather(2, batch["labels"][:, :, None, None].expand(-1, -1, 1, pair_scores.shape[-1])).squeeze(2)
    target_score = target_score.gather(-1, target_pair[..., None]).squeeze(-1)
    masked = pair_scores.clone()
    masked.scatter_(-1, target_pair[..., None, None].expand(-1, -1, pair_scores.shape[2], 1), -torch.inf)
    non_target = masked.max(dim=-1).values.gather(-1, batch["labels"][..., None]).squeeze(-1)
    return candidate_loss + pair_loss_weight * F.softplus(non_target - target_score).mean()


def train_model(model: torch.nn.Module, train: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    train_batches = list(batches(train, int(config["dataset"]["batch_size"])))
    losses: list[float] = []
    gradients: list[float] = []
    pair_loss_weight = float(config["training"]["pair_loss_weight"])
    for step in range(int(config["training"]["steps"])):
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = qcdd_training_loss(model, batch, pair_loss_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite QCDD loss at step {step}")
        loss.backward()
        current = [p.grad for p in model.parameters() if p.grad is not None]
        if not current or any(not torch.isfinite(g).all() for g in current):
            raise FloatingPointError(f"missing or non-finite QCDD gradient at step {step}")
        gradients.append(float(torch.sqrt(sum(g.detach().square().sum() for g in current))))
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "steps": len(losses),
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "gradient_norm_min": min(gradients),
        "gradient_norm_max": max(gradients),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
    }


def _selector_scores(
    selector: str,
    model: torch.nn.Module,
    product_model: torch.nn.Module,
    control_model: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    key_second = torch.roll(key, shifts=1, dims=2) if selector == "qcdd_key_shuffle" else None
    components = model.pair_score_components(query, key, key_second=key_second)
    if selector == "qcdd_dephased_null":
        score = components["dephased_yyyy"]
    elif selector == "qcdd_product_null":
        score = product_model.pair_scores(query, key)
    elif selector == "qcdd_sincos_control":
        score = control_model.pair_scores(query, key)
    elif selector == "qccw_raw_xx":
        score = components["raw_qccw_xx"]
    else:
        score = components["differential"]
    return score, components


def evaluate_selector(
    selector: str,
    model: torch.nn.Module,
    product_model: torch.nn.Module,
    control_model: torch.nn.Module,
    split: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, Any]:
    scores_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    candidate_correct: list[torch.Tensor] = []
    component_max: dict[str, float] = {}
    replay_error = 0.0
    with torch.no_grad():
        for batch in batches(split, batch_size):
            full_score, components = _selector_scores(
                selector, model, product_model, control_model, batch["query"], batch["key"]
            )
            replay, _ = _selector_scores(
                selector, model, product_model, control_model, batch["query"], batch["key"]
            )
            replay_error = max(replay_error, float((replay - full_score).abs().max()))
            candidate_correct.append(full_score.max(dim=-1).values.argmax(dim=-1).eq(batch["labels"]))
            score = full_score.gather(
                2,
                batch["labels"][:, :, None, None].expand(-1, -1, 1, full_score.shape[-1]),
            ).squeeze(2)
            scores_all.append(score)
            target_pair = unordered_pair_index(batch["evidence_slot"], model.pair_indices)
            targets_all.append(torch.arange(score.shape[-1], device=score.device)[None, None, :] == target_pair[..., None])
            for name, value in components.items():
                component_max[name] = max(component_max.get(name, 0.0), float(value.abs().max()))
    scores = torch.cat(scores_all)
    targets = torch.cat(targets_all)
    choices = scores.argmax(dim=-1)
    target_pair = targets.to(torch.int64).argmax(dim=-1)
    target_score = scores.gather(-1, target_pair[..., None]).squeeze(-1)
    hardest_negative = scores.masked_fill(targets, -torch.inf).max(dim=-1).values
    margins = target_score - hardest_negative
    return {
        "selector": selector,
        "pair_auc": pair_auc(scores, targets),
        "score_abs_max": float(scores.abs().max()),
        "score_abs_mean": float(scores.abs().mean()),
        "candidate_accuracy": float(torch.cat(candidate_correct).float().mean()),
        "pair_top1_accuracy": float(choices.eq(target_pair).float().mean()),
        "pair_margin_mean": float(margins.mean()),
        "pair_margin_min": float(margins.min()),
        "pair_margin_nonpositive_rate": float((margins <= 0).float().mean()),
        "finite": bool(torch.isfinite(scores).all()),
        "deterministic_replay_error": replay_error,
        "components_abs_max": component_max,
    }


def shot_estimate(model: torch.nn.Module, split: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        components = model.pair_score_components(split["query"], split["key"])
    target_pair = unordered_pair_index(split["evidence_slot"], model.pair_indices)
    labels = split["labels"]
    scores = components["differential"].gather(
        2, labels[:, :, None, None].expand(-1, -1, 1, components["differential"].shape[-1])
    ).squeeze(2)
    target_mask = F.one_hot(target_pair, scores.shape[-1]).bool()
    hardest = scores.masked_fill(target_mask, -torch.inf).argmax(dim=-1)
    pair_rows = torch.stack((target_pair, hardest), dim=-1)
    moments = {
        name: components[component].gather(
            2, labels[:, :, None, None].expand(-1, -1, 1, components[component].shape[-1])
        ).squeeze(2)
        for name, component in (
            ("YYYY", "yyyy_moment"),
            ("YIYI", "register_a_yy_moment"),
            ("IYIY", "register_b_yy_moment"),
        )
    }
    z = float(config["shot_estimate"]["confidence_z"])
    selected = {
        name: torch.stack(
            (
                value.gather(-1, pair_rows[..., 0, None]).squeeze(-1),
                value.gather(-1, pair_rows[..., 1, None]).squeeze(-1),
            ),
            dim=-1,
        )
        for name, value in moments.items()
    }
    variances = {name: (1.0 - value.square()).clamp_min(0.0) for name, value in selected.items()}
    coefficient = (
        variances["YYYY"]
        + selected["IYIY"].square() * variances["YIYI"]
        + selected["YIYI"].square() * variances["IYIY"]
    ).sum(dim=-1)
    positive = scores.gather(-1, target_pair[..., None]).squeeze(-1)
    negative = scores.gather(-1, hardest[..., None]).squeeze(-1)
    margin = positive - negative
    shots_per_setting = torch.ceil(z * z * coefficient / margin.square().clamp_min(1e-16))
    shots_per_pair = int(config["shot_estimate"]["measurement_settings"]) * shots_per_setting
    shots_per_pair = torch.where(margin > 0.0, shots_per_pair, torch.full_like(shots_per_pair, torch.inf))
    ordered = torch.sort(shots_per_pair.reshape(-1)).values

    def nearest_rank(quantile: float) -> float | None:
        index = max(0, math.ceil(quantile * ordered.numel()) - 1)
        value = float(ordered[index])
        return value if math.isfinite(value) else None

    ceiling = int(config["shot_estimate"]["maximum_shots_per_candidate_pair"])
    p95 = nearest_rank(0.95)
    return {
        "settings": list(PAULI_SETTINGS),
        "confidence_z": z,
        "statistic": "nearest-rank p95 total shots per candidate pair across three Pauli settings",
        "candidate_pair_count": int(shots_per_pair.numel()),
        "nonpositive_margin_count": int((margin <= 0.0).sum()),
        "median_shots": nearest_rank(0.50),
        "p95_shots": p95,
        "worst_shots": nearest_rank(1.0),
        "maximum_shots_per_candidate_pair": ceiling,
        "gate_value": p95,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    device = task.v1.choose_device(args.device or str(config["device"]))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    dataset = config["dataset"]
    streams = {
        "train": task.make_split(seed, int(dataset["train_size"]), device),
        "calibration": task.make_split(seed + 1000, int(dataset["calibration_size"]), device),
        "valid": task.make_split(seed + 10000, int(dataset["valid_size"]), device),
        "test": task.make_split(seed + 20000, int(dataset["test_size"]), device),
    }
    overlaps = exact_stream_overlaps(streams)
    frames = task.v1.relation_frames(device)
    quantum = build_model("quantum", seed, frames, config, device)
    control = build_model("sincos", seed, frames, config, device)
    product = build_model("product", seed, frames, config, device)
    train_quantum = train_model(quantum, streams["train"], config)
    train_control = train_model(control, streams["train"], config)
    product.load_state_dict(quantum.state_dict(), strict=False)
    selectors: dict[str, dict[str, Any]] = {}
    for name in streams:
        selectors[name] = {}
        for selector in SELECTORS:
            selectors[name][selector] = evaluate_selector(
                selector,
                quantum,
                product,
                control,
                streams[name],
                int(dataset["batch_size"]),
            )
    valid, test = selectors["valid"], selectors["test"]
    shots = shot_estimate(quantum, {k: torch.cat((streams["valid"][k], streams["test"][k])) for k in streams["valid"]}, config)
    gate_cfg = config["gate"]
    conditions = {
        "exact_disjoint_streams": all(count == 0 for count in overlaps.values()),
        "differential_nontrivial": max(valid["qcdd_coherent_differential"]["score_abs_max"], test["qcdd_coherent_differential"]["score_abs_max"]) >= float(gate_cfg["minimum_differential_abs"]),
        "product_null": max(valid["qcdd_product_null"]["score_abs_max"], test["qcdd_product_null"]["score_abs_max"]) <= float(gate_cfg["maximum_null_abs"]),
        "dephased_null": max(valid["qcdd_dephased_null"]["score_abs_max"], test["qcdd_dephased_null"]["score_abs_max"]) <= float(gate_cfg["maximum_null_abs"]),
        "pair_auc": valid["qcdd_coherent_differential"]["pair_auc"] >= float(gate_cfg["minimum_pair_auc"]) and test["qcdd_coherent_differential"]["pair_auc"] >= float(gate_cfg["minimum_pair_auc"]),
        "dephased_auc": valid["qcdd_dephased_null"]["pair_auc"] <= float(gate_cfg["maximum_dephased_auc"]) and test["qcdd_dephased_null"]["pair_auc"] <= float(gate_cfg["maximum_dephased_auc"]),
        "quantum_control_margin": valid["qcdd_coherent_differential"]["pair_auc"] - valid["qcdd_sincos_control"]["pair_auc"] >= float(gate_cfg["minimum_quantum_control_auc_margin"]) and test["qcdd_coherent_differential"]["pair_auc"] - test["qcdd_sincos_control"]["pair_auc"] >= float(gate_cfg["minimum_quantum_control_auc_margin"]),
        "key_shuffle_drop": valid["qcdd_coherent_differential"]["pair_auc"] - valid["qcdd_key_shuffle"]["pair_auc"] >= float(gate_cfg["minimum_key_shuffle_auc_drop"]) and test["qcdd_coherent_differential"]["pair_auc"] - test["qcdd_key_shuffle"]["pair_auc"] >= float(gate_cfg["minimum_key_shuffle_auc_drop"]),
        "gradient_finite": train_quantum["gradient_norm_min"] > 0.0 and train_quantum["gradient_norm_max"] < float(gate_cfg["maximum_gradient_norm"]) and train_control["gradient_norm_min"] > 0.0 and train_control["gradient_norm_max"] < float(gate_cfg["maximum_gradient_norm"]),
        "four_parameter_budget": train_quantum["trainable_parameter_count"] == 4 and train_control["trainable_parameter_count"] == 4,
        "finite_deterministic_readout": all(
            item["finite"] and item["deterministic_replay_error"] == 0.0
            for split in selectors.values()
            for item in split.values()
        ),
        "shot_ceiling": shots["gate_value"] is not None and shots["gate_value"] <= shots["maximum_shots_per_candidate_pair"],
    }
    plugin_path = ROOT / "src/q_attention/plugins/q_coherence_destruction_differential.py"
    return {
        "schema_version": "q-attention.q-coherence-destruction-differential.v1",
        "status": "complete",
        "experiment_name": "q_coherence_destruction_differential",
        "dataset_identity": dataset["identity"],
        "seed": seed,
        "split_policy": "exact disjoint streams seed, seed+1000, seed+10000, seed+20000",
        "exact_stream_overlaps": overlaps,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "plugin_sha256": hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
        "plugin_plan_sha256": hashlib.sha256((ROOT / "configs/q_coherence_destruction_differential.plugin-plan.json").read_bytes()).hexdigest(),
        "circuit_sha256": hashlib.sha256((ROOT / config["estimator"]["circuit_path"]).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "training": {"quantum": train_quantum, "sincos_control": train_control},
        "estimators": {"quantum": quantum.metadata(), "product": product.metadata(), "sincos": control.metadata()},
        "selectors": selectors,
        "shot_estimate": shots,
        "gate": {**conditions, "status": "pass" if all(conditions.values()) else "fail", "next_five_seed_authorized": bool(all(conditions.values())), "next_attention_action_authorized": False, "next_real_data_authorized": False, "hardware_claim": False},
    }


def main() -> None:
    args = parse_args()
    payload = run(args)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": payload["gate"], "valid": payload["selectors"]["valid"], "test": payload["selectors"]["test"], "shot_estimate": payload["shot_estimate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
