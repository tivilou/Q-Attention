#!/usr/bin/env python3
"""Matched-training upper bounds for the counterbalanced routing split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_counterbalanced_routing_headroom_audit_toy as routing  # noqa: E402
import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
import run_q_rde_stage0_action_support_audit_toy as stage0  # noqa: E402
import run_q_routing_solvability_diagnostic_toy as solvability  # noqa: E402


SEED = 7
CONDITIONS = (
    "current_query",
    "full_routing_query",
    "masked_routing_query",
    "full_routing_hard_selected",
    "masked_routing_hard_selected",
)
LogitFunction = Callable[
    [torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_routing_training_upper_bound_toy.json"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_name",
        "seed",
        "device",
        "readout",
        "conditions",
        "dataset",
        "baseline",
        "gate",
        "output_root",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if payload["schema_version"] != "q-attention.q-routing-training-upper-bound.v1":
        raise ValueError("unsupported routing training upper-bound config")
    if tuple(payload["conditions"]) != CONDITIONS:
        raise ValueError(f"conditions must equal {CONDITIONS}")
    if int(payload["seed"]) != SEED:
        raise ValueError("training upper-bound audit is fixed to seed 7")
    if payload["readout"] != "query":
        raise ValueError("reference readout must remain query-indexed")
    return payload


def hard_selected_model_logits(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Read the role-selected position without adding trainable parameters."""
    hidden = model.encoder(batch["input_ids"], batch["attention_mask"])
    subject_repr = model._masked_mean(hidden, batch["subject_mask"])
    object_repr = model._masked_mean(hidden, batch["object_mask"])
    selected_position = torch.where(
        batch["routing_role"] == 0,
        torch.full_like(batch["routing_role"], routing.EVIDENCE_POSITIONS[0]),
        torch.full_like(batch["routing_role"], routing.EVIDENCE_POSITIONS[1]),
    )
    row = torch.arange(hidden.shape[0], device=hidden.device)
    selected_repr = hidden[row, selected_position]
    return model.classifier(
        torch.cat([subject_repr, object_repr, selected_repr], dim=-1)
    )


def make_condition_split(
    condition: str,
    seed: int,
    size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    current = legacy.make_split(seed, size, device)
    if condition == "current_query":
        return current
    routed = routing.make_counterbalanced_split(seed, size, device)
    if condition in ("full_routing_query", "full_routing_hard_selected"):
        return routed
    if condition in ("masked_routing_query", "masked_routing_hard_selected"):
        return solvability.make_variant(current, routed, "masked_distractor")
    raise ValueError(f"unknown condition: {condition}")


def condition_logits(condition: str) -> LogitFunction:
    if condition in (
        "current_query",
        "full_routing_query",
        "masked_routing_query",
    ):
        return stage0.query_indexed_model_logits
    if condition in (
        "full_routing_hard_selected",
        "masked_routing_hard_selected",
    ):
        return hard_selected_model_logits
    raise ValueError(f"unknown condition: {condition}")


def train_condition(
    condition: str,
    seed: int,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    dataset = config["dataset"]
    train = make_condition_split(
        condition, seed, int(dataset["train_size"]), device
    )
    calibration = make_condition_split(
        condition, seed + 1000, int(dataset["calibration_size"]), device
    )
    valid = make_condition_split(
        condition, seed + 10000, int(dataset["valid_size"]), device
    )
    logits_function = condition_logits(condition)
    model = legacy.build_model(seed, device)
    original_logits = legacy.model_logits
    legacy.model_logits = logits_function
    try:
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
        model.eval()
        with torch.no_grad():
            valid_logits = logits_function(model, valid)
    finally:
        legacy.model_logits = original_logits
    labels = valid["labels"]
    prediction = valid_logits.argmax(dim=-1)
    accuracy = float(prediction.eq(labels).float().mean().item())
    row = torch.arange(labels.shape[0], device=labels.device)
    predicted_margin = valid_logits[row, labels] - valid_logits[row, prediction]
    return {
        "condition": condition,
        "selected_epoch": int(baseline_info["selected_epoch"]),
        "calibration_accuracy": float(baseline_info["calibration_accuracy"]),
        "valid_accuracy": accuracy,
        "valid_wrong_examples": int(prediction.ne(labels).sum().item()),
        "mean_gold_margin": float(predicted_margin.mean().item()),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classifier_parameters": sum(
            parameter.numel() for parameter in model.classifier.parameters()
        ),
        "readout": "hard_role_selected" if "hard_selected" in condition else "query",
        "distractor_attention": "masked" if condition.startswith("masked") else "visible",
    }


def upper_bound_gate(
    results: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    tolerance = float(config["gate"]["reference_tolerance"])
    reference = results["current_query"]["valid_accuracy"]
    threshold = reference - tolerance
    conditions = {
        "matched_model_parameter_counts": len(
            {row["model_parameters"] for row in results.values()}
        ) == 1,
        "matched_classifier_parameter_counts": len(
            {row["classifier_parameters"] for row in results.values()}
        ) == 1,
        "full_routing_query_parity": results["full_routing_query"]["valid_accuracy"]
        >= threshold,
        "masked_routing_query_parity": results["masked_routing_query"]["valid_accuracy"]
        >= threshold,
        "full_hard_selected_parity": results["full_routing_hard_selected"]["valid_accuracy"]
        >= threshold,
        "masked_hard_selected_parity": results["masked_routing_hard_selected"]["valid_accuracy"]
        >= threshold,
    }
    if conditions["masked_routing_query_parity"]:
        interpretation = "distractor_attention_interference"
    elif conditions["full_hard_selected_parity"]:
        interpretation = "query_role_binding_bottleneck"
    elif conditions["masked_hard_selected_parity"]:
        interpretation = "distractor_contaminates_selected_representation"
    else:
        interpretation = "routed_task_not_solved_under_frozen_contract"
    return {
        **conditions,
        "reference_accuracy": reference,
        "parity_threshold": threshold,
        "interpretation": interpretation,
        "status": "pass"
        if conditions["matched_model_parameter_counts"]
        and conditions["matched_classifier_parameter_counts"]
        else "fail",
        "new_attention_mechanism_authorized": False,
        "real_data_authorized": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    if args.seed != SEED:
        raise ValueError("this audit is predeclared for seed 7 only")
    device = legacy.choose_device(args.device or str(config["device"]))
    results: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        results[condition] = train_condition(condition, args.seed, device, config)
        print(json.dumps(results[condition], sort_keys=True), flush=True)
    gate = upper_bound_gate(results, config)

    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.q-routing-training-upper-bound.v1",
        "status": "complete",
        "revision": routing.q_rde.git_revision(),
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
        "dataset_identity": config["dataset"]["identity"],
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "results": results,
        "diagnostic_gate": gate,
        "contract": {
            "model_initialization": "identical seed per condition",
            "optimizer_and_budget": "identical fixed baseline contract",
            "train_calibration_valid_offsets": [0, 1000, 10000],
            "hard_selected_trainable_parameters_added": 0,
            "parameter_sweep": "prohibited",
        },
        "limitations": [
            "Hard-selected readout uses the known synthetic routing role and is an upper bound only.",
            "This one-seed diagnostic does not establish task utility or generalization.",
            "No quantum mechanism or real-data run is authorized by this diagnostic alone.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "diagnostic_gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
