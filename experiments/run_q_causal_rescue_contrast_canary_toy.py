#!/usr/bin/env python3
"""Seed-7 causal rescue-contrast canary for gain-only Q-AOC."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
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

import run_q_antisymmetric_observable_contrast_toy as q_aoc  # noqa: E402
import run_q_candidate_attention_transport_toy as qcat  # noqa: E402
import run_q_instance_conditioned_gain_canary_toy as gain_canary  # noqa: E402
import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
import run_q_rde_stage0_action_support_audit_toy as stage0  # noqa: E402
import run_q_relative_evidence_field_toy as q_rde  # noqa: E402


SELECTORS = (
    "disabled",
    "q_crc_gain_quantum",
    "q_crc_gain_classical",
    "q_crc_gain_shuffled_candidate",
    "q_gain_task_only_quantum",
    "q_crc_aoc_quantum",
    "q_cat_gold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_causal_rescue_contrast_canary_toy.json"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_name",
        "selectors",
        "seed",
        "device",
        "readout",
        "dataset",
        "baseline",
        "evidence",
        "training",
        "gate",
        "q_cat_config",
        "output_root",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if payload["schema_version"] != "q-attention.q-crc-canary-config.v1":
        raise ValueError("unsupported causal rescue contrast config")
    if tuple(payload["selectors"]) != SELECTORS:
        raise ValueError(f"selectors must equal {SELECTORS}")
    if payload["readout"] != "query":
        raise ValueError("causal rescue contrast requires query-indexed readout")
    return payload


def rescue_ablated_batch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    ablated = dict(batch)
    input_ids = batch["input_ids"].clone()
    input_ids[:, legacy.RESCUE_POS] = 4
    ablated["input_ids"] = input_ids
    return ablated


def causal_loss(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    candidates: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    with torch.no_grad():
        baseline_logits = legacy.model_logits(model, batch)
        baseline_prediction = baseline_logits.argmax(dim=-1)
    candidate_labels = labels if candidates is None else candidates
    original = q_rde.run_rde(
        model, kernel, batch, candidate_labels, baseline_prediction
    )
    ablated = q_rde.run_rde(
        model,
        kernel,
        rescue_ablated_batch(batch),
        candidate_labels,
        baseline_prediction,
    )
    row = torch.arange(labels.shape[0], device=labels.device)
    original_margin = original[row, labels] - original[row, baseline_prediction]
    ablated_margin = ablated[row, labels] - ablated[row, baseline_prediction]
    rescued = batch["primary_corrupt"] & batch["rescue_available"]
    target = float(config["training"]["margin_target"])
    ordinary_hinge = F.relu(target - original_margin)
    causal_hinge = F.relu(target - (original_margin - ablated_margin))
    per_example = torch.where(rescued, causal_hinge, ordinary_hinge)
    loss = F.cross_entropy(original, labels) + float(
        config["training"]["margin_weight"]
    ) * per_example.mean()
    return loss, {
        "rescued_examples": int(rescued.sum().item()),
        "mean_causal_hinge": float(causal_hinge[rescued].mean().detach().item())
        if rescued.any()
        else 0.0,
        "mean_rescue_margin_drop": float(
            (original_margin - ablated_margin)[rescued].mean().detach().item()
        )
        if rescued.any()
        else 0.0,
    }


def causal_trainability_diagnostics(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, float]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel.zero_grad(set_to_none=True)
    loss, metrics = causal_loss(model, kernel, batch, config)
    loss.backward()

    def grad_norm(name: str) -> float:
        parameter = getattr(kernel, name)
        if parameter is None or parameter.grad is None:
            return 0.0
        return float(parameter.grad.detach().norm().item())

    diagnostics = {
        "loss": float(loss.detach().item()),
        "candidate_embedding_gradient_norm": grad_norm("candidate_embeddings"),
        "evidence_scale_gradient_norm": grad_norm("evidence_scales"),
        "evidence_bias_gradient_norm": grad_norm("evidence_biases"),
        "global_gain_gradient_norm": grad_norm("raw_gains"),
        "instance_gain_gradient_norm": grad_norm("instance_gain_weights"),
        "instance_gain_bias_gradient_norm": grad_norm("instance_gain_biases"),
        **metrics,
    }
    kernel.zero_grad(set_to_none=True)
    return diagnostics


def train_causal(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    train: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, float]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel.train()
    optimizer = torch.optim.AdamW(
        kernel.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batches = list(
        legacy.batches(train, int(config["dataset"]["batch_size"]))
    )
    losses = []
    causal_hinges = []
    rescue_drops = []
    for step in range(int(config["training"]["steps"])):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = causal_loss(model, kernel, batch, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite causal rescue loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in kernel.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(
            not torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise FloatingPointError(
                f"missing or non-finite causal rescue gradient at step {step}"
            )
        torch.nn.utils.clip_grad_norm_(
            kernel.parameters(), float(config["training"]["gradient_clip"])
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        causal_hinges.append(float(metrics["mean_causal_hinge"]))
        rescue_drops.append(float(metrics["mean_rescue_margin_drop"]))
    return {
        "final_training_loss": losses[-1],
        "min_training_loss": min(losses),
        "final_causal_hinge": causal_hinges[-1],
        "min_causal_hinge": min(causal_hinges),
        "final_training_rescue_margin_drop": rescue_drops[-1],
        "max_training_rescue_margin_drop": max(rescue_drops),
    }


def counterfactual_metrics(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    valid: dict[str, torch.Tensor],
    batch_size: int,
    selector: str,
) -> dict[str, Any]:
    margin_drops = []
    original_predictions = []
    ablated_predictions = []
    rescue_flags = []
    for batch in legacy.batches(valid, batch_size):
        labels = batch["labels"]
        with torch.no_grad():
            baseline = legacy.model_logits(model, batch)
            baseline_prediction = baseline.argmax(dim=-1)
            candidates = labels
            if selector.endswith("_shuffled_candidate"):
                candidates = (labels + 1) % legacy.NUM_LABELS
            original = q_rde.run_rde(
                model, kernel, batch, candidates, baseline_prediction
            )
            ablated = q_rde.run_rde(
                model,
                kernel,
                rescue_ablated_batch(batch),
                candidates,
                baseline_prediction,
            )
        row = torch.arange(labels.shape[0], device=labels.device)
        margin_drops.append(
            (original[row, labels] - original[row, baseline_prediction])
            - (ablated[row, labels] - ablated[row, baseline_prediction])
        )
        original_predictions.append(original.argmax(dim=-1))
        ablated_predictions.append(ablated.argmax(dim=-1))
        rescue_flags.append(batch["primary_corrupt"] & batch["rescue_available"])
    margin_drop = torch.cat(margin_drops)
    original_prediction = torch.cat(original_predictions)
    ablated_prediction = torch.cat(ablated_predictions)
    rescued = torch.cat(rescue_flags)
    return {
        "rescue_examples": int(rescued.sum().item()),
        "rescue_ablation_gold_margin_drop_mean": float(
            margin_drop[rescued].mean().item()
        )
        if rescued.any()
        else None,
        "rescue_ablation_prediction_change_fraction": float(
            original_prediction[rescued]
            .ne(ablated_prediction[rescued])
            .float()
            .mean()
            .item()
        )
        if rescued.any()
        else None,
        "all_example_margin_drop_mean": float(margin_drop.mean().item()),
    }


def evaluate(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    valid: dict[str, torch.Tensor],
    selector: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    row = q_rde.evaluate_rde(
        model,
        kernel,
        valid,
        int(config["dataset"]["batch_size"]),
        selector,
        float(config["evidence"]["max_delta"]),
    )
    row["counterfactual"] = counterfactual_metrics(
        model,
        kernel,
        valid,
        int(config["dataset"]["batch_size"]),
        selector,
    )
    return row


def promotion_gate(
    rows: dict[str, dict[str, Any]],
    diagnostics: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    quantum = rows["q_crc_gain_quantum"]
    classical = rows["q_crc_gain_classical"]
    shuffled = rows["q_crc_gain_shuffled_candidate"]
    task_only = rows["q_gain_task_only_quantum"]
    fixed = rows["q_crc_aoc_quantum"]
    qcat_row = rows["q_cat_gold"]
    counterfactual = quantum["counterfactual"]
    shuffled_counterfactual = shuffled["counterfactual"]
    conditions = {
        "minimum_corrections": quantum["corrected_examples"]
        >= int(config["gate"]["min_corrected_examples"]),
        "improves_task_only_gain": quantum["corrected_examples"]
        > task_only["corrected_examples"],
        "improves_causal_fixed_q_aoc": quantum["corrected_examples"]
        > fixed["corrected_examples"],
        "corrective_expansion_beyond_q_cat": quantum["corrected_examples"]
        > qcat_row["corrected_examples"],
        "quantum_exceeds_classical_corrections": quantum["corrected_examples"]
        > classical["corrected_examples"],
        "positive_accuracy_delta": quantum["accuracy_delta"] > 0.0,
        "correct_retention": quantum["correct_to_wrong_rate"]
        <= float(config["gate"]["max_correct_to_wrong_rate"]),
        "aligned_beats_shuffled": quantum["accuracy"] > shuffled["accuracy"],
        "positive_rescue_ablation_necessity": counterfactual[
            "rescue_ablation_gold_margin_drop_mean"
        ]
        is not None
        and counterfactual["rescue_ablation_gold_margin_drop_mean"] > 0.0,
        "aligned_necessity_exceeds_shuffled": counterfactual[
            "rescue_ablation_gold_margin_drop_mean"
        ]
        > shuffled_counterfactual["rescue_ablation_gold_margin_drop_mean"],
        "causal_instance_gain_gradient": diagnostics[
            "instance_gain_gradient_norm"
        ]
        > float(config["gate"]["minimum_gradient_norm"]),
        "residual_invariants": quantum["residual_invariants"]
        and classical["residual_invariants"]
        and fixed["residual_invariants"],
        "parameter_matched": quantum["action_parameters"]
        == classical["action_parameters"],
    }
    return {
        **conditions,
        "status": "pass" if all(conditions.values()) else "fail",
        "next_label_path_audit_authorized": bool(all(conditions.values())),
        "does_not_establish_task_utility": True,
        "does_not_authorize_real_data": True,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    seed = int(config["seed"] if args.seed is None else args.seed)
    device = legacy.choose_device(args.device or str(config["device"]))
    legacy.model_logits = stage0.query_indexed_model_logits
    dataset = config["dataset"]
    train = legacy.make_split(seed, int(dataset["train_size"]), device)
    calibration = legacy.make_split(
        seed + 1000, int(dataset["calibration_size"]), device
    )
    valid = legacy.make_split(seed + 10000, int(dataset["valid_size"]), device)
    model = legacy.build_model(seed, device)
    baseline_info = legacy.train_baseline(
        model,
        train,
        calibration,
        argparse.Namespace(
            baseline_epochs=int(config["baseline"]["epochs"]),
            baseline_lr=float(config["baseline"]["lr"]),
            batch_size=int(dataset["batch_size"]),
        ),
    )
    baseline_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    first_batch = next(legacy.batches(train, int(dataset["batch_size"])))
    diagnostic_kernel = gain_canary.build_gain_kernel(
        "quantum", seed, config, device
    )
    diagnostics = causal_trainability_diagnostics(
        model, diagnostic_kernel, first_batch, config
    )
    rows: dict[str, dict[str, Any]] = {
        "disabled": {
            "selector": "disabled",
            "corrected_examples": 0,
            "harmed_correct_examples": 0,
        }
    }
    for kernel_type, selector in (
        ("quantum", "q_crc_gain_quantum"),
        ("classical", "q_crc_gain_classical"),
    ):
        model.load_state_dict(baseline_state)
        kernel = gain_canary.build_gain_kernel(kernel_type, seed, config, device)
        training = train_causal(model, kernel, train, config)
        row = evaluate(model, kernel, valid, selector, config)
        row.update(training)
        rows[selector] = row

    model.load_state_dict(baseline_state)
    kernel = gain_canary.build_gain_kernel("quantum", seed, config, device)
    training = train_causal(model, kernel, train, config)
    shuffled = evaluate(
        model, kernel, valid, "q_crc_gain_shuffled_candidate", config
    )
    shuffled.update(training)
    rows["q_crc_gain_shuffled_candidate"] = shuffled

    model.load_state_dict(baseline_state)
    kernel = gain_canary.build_gain_kernel("quantum", seed, config, device)
    training = q_rde.train_rde(model, kernel, train, config)
    task_only = evaluate(model, kernel, valid, "q_gain_task_only_quantum", config)
    task_only.update(training)
    rows["q_gain_task_only_quantum"] = task_only

    model.load_state_dict(baseline_state)
    kernel = q_aoc.build_aoc("quantum", seed, config, device)
    training = train_causal(model, kernel, train, config)
    fixed = evaluate(model, kernel, valid, "q_crc_aoc_quantum", config)
    fixed.update(training)
    rows["q_crc_aoc_quantum"] = fixed

    model.load_state_dict(baseline_state)
    qcat_config = qcat.load_config(ROOT / str(config["q_cat_config"]))
    qcat_kernel = qcat.build_kernel("quantum", seed, qcat_config, device)
    qcat.train_kernel(model, qcat_kernel, train, qcat_config)
    rows["q_cat_gold"] = q_rde.evaluate_qcat(
        model, qcat_kernel, valid, int(dataset["batch_size"])
    )

    gate = promotion_gate(rows, diagnostics, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.q-crc-canary.v1",
        "status": "complete",
        "run_type": "gold_candidate_causal_rescue_contrast_canary",
        "revision": q_rde.git_revision(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "dataset_identity": dataset["identity"],
        "readout": config["readout"],
        "seed": seed,
        "selectors": list(SELECTORS),
        "baseline": baseline_info,
        "trainability_diagnostics": diagnostics,
        "results": rows,
        "promotion_gate": gate,
        "training_contract": {
            "ordinary_objective": "cross entropy plus existing gold-margin hinge",
            "rescued_hard_objective": "cross entropy plus paired original-minus-rescue-ablated gold-margin hinge",
            "new_trainable_parameters": 0,
            "new_scalar_settings": 0,
            "rescue_ablation_token": 4,
        },
        "limitations": [
            "Gold candidate labels, corruption flags, and rescue availability condition this diagnostic training objective.",
            "One seed cannot establish statistically stable task utility.",
            "Matched classical comparison is descriptive and does not establish quantum advantage.",
            "No scalar sweep, five-seed, real-data, collaborator, manuscript, or hardware claim is authorized.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "promotion_gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
