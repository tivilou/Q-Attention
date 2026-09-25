#!/usr/bin/env python3
"""Seed-7 Stage-0 gate for the QCCW connected-correlation witness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
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
from q_attention.plugins.q_connected_consensus_witness import (  # noqa: E402
    ConnectedConsensusWitnessConfig,
    build_connected_consensus_witness,
    unordered_pair_index,
)
from q_attention.plugins.q_consensus_quantum_estimator import (  # noqa: E402
    ConsensusQuantumEstimatorConfig,
    build_consensus_estimator,
)


SELECTORS = (
    "disabled",
    "qccw_entangled",
    "qccw_product_null",
    "qccw_bilinear",
    "qccw_entangler_cut",
    "qccw_key_shuffle",
    "independent_key_quantum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/q_connected_consensus_witness_stage0.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.q-connected-consensus-witness-stage0.v1":
        raise ValueError("unsupported QCCW Stage-0 config schema")
    required = {"seed", "device", "dataset", "estimator", "training", "gate", "output_root"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if tuple(payload["selectors"]) != SELECTORS:
        raise ValueError("selectors must match the explicit QCCW Stage-0 allowlist")
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
    labels = positive[order]
    ranks = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def build_qccw(kind: str, seed: int, frames: torch.Tensor, config: dict[str, Any], device: torch.device):
    estimator = config["estimator"]
    model = build_connected_consensus_witness(
        kind,
        ConnectedConsensusWitnessConfig(
            num_candidates=task.v1.CLASSES,
            head_dim=task.v1.DIM,
            num_key_pairs=int(estimator["num_key_pairs"]),
            angle_scale=float(estimator["angle_scale"]),
            seed=seed + int(estimator["seed_offset"]),
        ),
        frames,
    )
    return model.to(device)


def build_independent(seed: int, frames: torch.Tensor, config: dict[str, Any], device: torch.device):
    model = build_consensus_estimator(
        "quantum",
        ConsensusQuantumEstimatorConfig(
            num_candidates=task.v1.CLASSES,
            head_dim=task.v1.DIM,
            register_qubits=3,
            depth=2,
            angle_scale=1.0,
            seed=seed + 7331,
        ),
        frames,
    )
    return model.to(device)


def qccw_training_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    pair_indices: torch.Tensor,
    pair_loss_weight: float,
) -> torch.Tensor:
    pair_scores = model.pair_scores(batch["query"], batch["key"])
    candidate_scores = pair_scores.max(dim=-1).values
    candidate_loss = F.cross_entropy(
        candidate_scores.reshape(-1, candidate_scores.shape[-1]), batch["labels"].reshape(-1)
    )
    target_pair = unordered_pair_index(batch["evidence_slot"], pair_indices)
    target_candidate = batch["labels"]
    target_score = pair_scores.gather(2, target_candidate[:, :, None, None].expand(-1, -1, 1, pair_scores.shape[-1]))
    target_score = target_score.squeeze(2).gather(-1, target_pair[..., None]).squeeze(-1)
    masked = pair_scores.clone()
    masked.scatter_(-1, target_pair[..., None, None].expand(-1, -1, pair_scores.shape[2], 1), -torch.inf)
    non_target = masked.max(dim=-1).values.gather(-1, target_candidate[..., None]).squeeze(-1)
    pair_loss = F.softplus(non_target - target_score).mean()
    return candidate_loss + pair_loss_weight * pair_loss


def train_qccw(
    model: torch.nn.Module,
    train: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    pair_indices = model.pair_indices
    train_batches = list(batches(train, int(config["dataset"]["batch_size"])))
    losses: list[float] = []
    gradient_norms: list[float] = []
    for step in range(int(config["training"]["steps"])):
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = qccw_training_loss(
            model, batch, pair_indices, float(config["training"]["pair_loss_weight"])
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite QCCW loss at step {step}")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"missing or non-finite QCCW gradient at step {step}")
        gradient_norm = torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients))
        gradient_norms.append(float(gradient_norm))
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "steps": len(losses),
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def train_independent(model: torch.nn.Module, train: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0001)
    train_batches = list(batches(train, int(config["dataset"]["batch_size"])))
    losses: list[float] = []
    gradient_norms: list[float] = []
    for step in range(int(config["training"]["steps"])):
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        scores = model.candidate_scores(batch["query"], batch["key"])
        loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), batch["labels"].reshape(-1))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        gradient_norms.append(float(torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients))))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "steps": len(losses),
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def support_from_pair_choice(pair_indices: torch.Tensor, pair_choice: torch.Tensor) -> torch.Tensor:
    return pair_indices.to(pair_choice.device)[pair_choice]


def evaluate_qccw(
    selector: str,
    model: torch.nn.Module | None,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    labels_all: list[torch.Tensor] = []
    baseline_all: list[torch.Tensor] = []
    prediction_all: list[torch.Tensor] = []
    candidate_all: list[torch.Tensor] = []
    active_all: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    auc_scores: list[torch.Tensor] = []
    auc_targets: list[torch.Tensor] = []
    pair_indices = (
        model.pair_indices
        if model is not None and hasattr(model, "pair_indices")
        else torch.tensor(list(combinations(range(task.v1.KEYS), 2)), device=split["labels"].device)
    )
    with torch.no_grad():
        for batch in batches(split, batch_size):
            baseline_logits, _ = task.v1.baseline_logits(batch["scores"], batch["key"], batch["query"], frames)
            if selector == "disabled":
                candidate = torch.zeros_like(batch["labels"])
                support = torch.zeros(batch["labels"].shape[0], task.v1.QUERIES, task.v1.EVIDENCE_KEYS, dtype=torch.long, device=batch["labels"].device)
                active = torch.zeros_like(batch["labels"], dtype=torch.bool)
                pair_scores = None
            elif selector == "independent_key_quantum":
                field = model.field(batch["query"], batch["key"])
                top_support = field.topk(task.v1.EVIDENCE_KEYS, dim=-1)
                candidate_scores = top_support.values.mean(dim=-1)
                candidate = candidate_scores.argmax(dim=-1)
                support = top_support.indices.gather(2, candidate[:, :, None, None].expand(-1, -1, 1, task.v1.EVIDENCE_KEYS)).squeeze(2)
                active = task.error_witness(batch["scores"])
                pair_scores = None
            else:
                key_second = None
                if selector == "qccw_key_shuffle":
                    key_second = torch.roll(batch["key"], shifts=1, dims=2)
                pair_scores = model.pair_scores(batch["query"], batch["key"], key_second=key_second)
                candidate_scores, pair_choice = pair_scores.max(dim=-1)
                candidate = candidate_scores.argmax(dim=-1)
                chosen_pair = pair_choice.gather(2, candidate[:, :, None]).squeeze(2)
                support = support_from_pair_choice(pair_indices, chosen_pair)
                active = task.error_witness(batch["scores"])
                true_candidate_scores = pair_scores.gather(2, batch["labels"][:, :, None, None].expand(-1, -1, 1, pair_scores.shape[-1])).squeeze(2)
                target_pair = unordered_pair_index(batch["evidence_slot"], pair_indices)
                auc_scores.append(true_candidate_scores)
                auc_targets.append(torch.arange(pair_scores.shape[-1], device=pair_scores.device)[None, None, :] == target_pair[..., None])
            steered_scores, residual = task.apply_pair_actions(batch["scores"], support, active)
            logits, _ = task.v1.baseline_logits(steered_scores, batch["key"], batch["query"], frames)
            labels_all.append(batch["labels"])
            baseline_all.append(baseline_logits.argmax(dim=-1))
            prediction_all.append(logits.argmax(dim=-1))
            candidate_all.append(candidate)
            active_all.append(active)
            residuals.append(residual)
    labels = torch.cat(labels_all)
    baseline_prediction = torch.cat(baseline_all)
    prediction = torch.cat(prediction_all)
    candidate = torch.cat(candidate_all)
    active = torch.cat(active_all)
    residual = torch.cat(residuals)
    wrong = baseline_prediction.ne(labels)
    correct = ~wrong
    corrected = wrong & prediction.eq(labels)
    metrics: dict[str, Any] = {
        "selector": selector,
        "baseline_accuracy": float(baseline_prediction.eq(labels).float().mean()),
        "accuracy": float(prediction.eq(labels).float().mean()),
        "accuracy_delta": float(prediction.eq(labels).float().mean() - baseline_prediction.eq(labels).float().mean()),
        "baseline_wrong_queries": int(wrong.sum()),
        "corrected_queries": int(corrected.sum()),
        "harmed_correct_queries": int((correct & prediction.ne(labels)).sum()),
        "wrong_correction_rate": float(corrected.sum() / wrong.sum().clamp_min(1)),
        "harm_rate": float((correct & prediction.ne(labels)).sum() / correct.sum().clamp_min(1)),
        "active_rate": float(active.float().mean()),
        "active_candidate_accuracy": float(candidate[active].eq(labels[active]).float().mean()) if active.any() else 0.0,
        "residual_finite": bool(torch.isfinite(residual).all()),
        "residual_zero_sum_error": float(residual.sum(dim=-1).abs().max()),
        "residual_max_abs": float(residual.abs().max()),
    }
    if auc_scores:
        metrics["pair_consistency_auc"] = pair_auc(torch.cat(auc_scores), torch.cat(auc_targets))
        metrics["connected_score_abs_max"] = float(torch.cat(auc_scores).abs().max())
    else:
        metrics["pair_consistency_auc"] = None
        metrics["connected_score_abs_max"] = None
    return metrics, residual


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    device_name = args.device or str(config["device"])
    device = task.v1.choose_device(device_name)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    dataset = config["dataset"]
    streams = {
        "train": task.make_split(seed, int(dataset["train_size"]), device),
        "calibration": task.make_split(seed + 1000, int(dataset["calibration_size"]), device),
        "valid": task.make_split(seed + 10000, int(dataset["valid_size"]), device),
        "test": task.make_split(seed + 20000, int(dataset["test_size"]), device),
    }
    frames = task.v1.relation_frames(device)
    qccw = build_qccw("quantum", seed, frames, config, device)
    bilinear = build_qccw("bilinear", seed, frames, config, device)
    product = build_qccw("product", seed, frames, config, device)
    independent = build_independent(seed, frames, config, device)
    qccw_training = train_qccw(qccw, streams["train"], config)
    bilinear_training = train_qccw(bilinear, streams["train"], config)
    product.load_state_dict(qccw.state_dict(), strict=False)
    independent_training = train_independent(independent, streams["train"], config)
    baseline: dict[str, Any] = {}
    selectors: dict[str, dict[str, Any]] = {}
    for split_name, split in streams.items():
        logits, _ = task.v1.baseline_logits(split["scores"], split["key"], split["query"], frames)
        replay, _ = task.v1.baseline_logits(split["scores"], split["key"], split["query"], frames)
        baseline_prediction = logits.argmax(dim=-1)
        baseline[split_name] = {
            "accuracy": float(baseline_prediction.eq(split["labels"]).float().mean()),
            "replay_error": float((replay - logits).abs().max()),
            "queries": int(split["labels"].numel()),
        }
        selectors[split_name] = {
            "disabled": evaluate_qccw("disabled", None, split, frames, int(dataset["batch_size"]))[0],
            "qccw_entangled": evaluate_qccw("qccw_entangled", qccw, split, frames, int(dataset["batch_size"]))[0],
            "qccw_product_null": evaluate_qccw("qccw_product_null", product, split, frames, int(dataset["batch_size"]))[0],
            "qccw_bilinear": evaluate_qccw("qccw_bilinear", bilinear, split, frames, int(dataset["batch_size"]))[0],
            "qccw_entangler_cut": evaluate_qccw("qccw_entangler_cut", product, split, frames, int(dataset["batch_size"]))[0],
            "qccw_key_shuffle": evaluate_qccw("qccw_key_shuffle", qccw, split, frames, int(dataset["batch_size"]))[0],
            "independent_key_quantum": evaluate_qccw("independent_key_quantum", independent, split, frames, int(dataset["batch_size"]))[0],
        }
    gate_config = config["gate"]
    valid = selectors["valid"]
    test = selectors["test"]
    q = "qccw_entangled"
    conditions = {
        "baseline_replay": all(item["replay_error"] == 0.0 for item in baseline.values()),
        "baseline_non_saturated": float(gate_config["baseline_accuracy_min"]) <= baseline["valid"]["accuracy"] <= float(gate_config["baseline_accuracy_max"]) and float(gate_config["baseline_accuracy_min"]) <= baseline["test"]["accuracy"] <= float(gate_config["baseline_accuracy_max"]),
        "product_connected_null": max(valid["qccw_product_null"]["connected_score_abs_max"] or 0.0, test["qccw_product_null"]["connected_score_abs_max"] or 0.0) <= float(gate_config["maximum_product_connected_abs"]),
        "connected_signal_nontrivial": max(valid[q]["connected_score_abs_max"] or 0.0, test[q]["connected_score_abs_max"] or 0.0) >= float(gate_config["minimum_connected_abs"]),
        "pair_consistency_auc": valid[q]["pair_consistency_auc"] >= float(gate_config["minimum_pair_auc"]) and test[q]["pair_consistency_auc"] >= float(gate_config["minimum_pair_auc"]),
        "qccw_heldout_gain": valid[q]["accuracy_delta"] >= float(gate_config["minimum_accuracy_delta"]) and test[q]["accuracy_delta"] >= float(gate_config["minimum_accuracy_delta"]),
        "qccw_no_harm": valid[q]["harm_rate"] <= float(gate_config["maximum_harm_rate"]) and test[q]["harm_rate"] <= float(gate_config["maximum_harm_rate"]),
        "qccw_beats_bilinear": valid[q]["accuracy_delta"] - valid["qccw_bilinear"]["accuracy_delta"] >= float(gate_config["minimum_quantum_bilinear_margin"]) and test[q]["accuracy_delta"] - test["qccw_bilinear"]["accuracy_delta"] >= float(gate_config["minimum_quantum_bilinear_margin"]),
        "entangler_cut_drop": valid[q]["accuracy_delta"] - valid["qccw_entangler_cut"]["accuracy_delta"] >= float(gate_config["minimum_entangler_cut_drop"]) and test[q]["accuracy_delta"] - test["qccw_entangler_cut"]["accuracy_delta"] >= float(gate_config["minimum_entangler_cut_drop"]),
        "key_shuffle_drop": valid[q]["accuracy_delta"] - valid["qccw_key_shuffle"]["accuracy_delta"] >= float(gate_config["minimum_key_shuffle_drop"]) and test[q]["accuracy_delta"] - test["qccw_key_shuffle"]["accuracy_delta"] >= float(gate_config["minimum_key_shuffle_drop"]),
        "training_gradients_finite": qccw_training["gradient_norm_min"] > 0.0 and qccw_training["gradient_norm_max"] < float(gate_config["maximum_gradient_norm"]) and bilinear_training["gradient_norm_min"] > 0.0 and bilinear_training["gradient_norm_max"] < float(gate_config["maximum_gradient_norm"]),
        "residual_invariants": all(item["qccw_entangled"]["residual_finite"] and item["qccw_entangled"]["residual_zero_sum_error"] <= float(gate_config["residual_zero_sum_tolerance"]) and item["qccw_entangled"]["residual_max_abs"] <= task.v1.MAX_DELTA + 1e-6 for item in selectors.values()),
    }
    plugin_path = ROOT / "src/q_attention/plugins/q_connected_consensus_witness.py"
    return {
        "schema_version": "q-attention.q-connected-consensus-witness-stage0.v1",
        "status": "complete",
        "experiment_name": "q_connected_consensus_witness_stage0",
        "dataset_identity": dataset["identity"],
        "seed": seed,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "plugin_sha256": hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "baseline": baseline,
        "training": {"qccw_entangled": qccw_training, "qccw_bilinear": bilinear_training, "independent_key_quantum": independent_training},
        "estimators": {"qccw_entangled": qccw.metadata(), "qccw_product_null": product.metadata(), "qccw_bilinear": bilinear.metadata(), "independent_key_quantum": independent.metadata()},
        "selectors": selectors,
        "gate": {**conditions, "status": "pass" if all(conditions.values()) else "fail", "next_five_seed_authorized": bool(all(conditions.values())), "next_real_data_authorized": False, "hardware_claim": False},
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
    print(json.dumps({"output": str(output), "gate": payload["gate"], "baseline": {name: payload["baseline"][name] for name in ("valid", "test")}, "qccw": {name: payload["selectors"][name]["qccw_entangled"] for name in ("valid", "test")}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
