#!/usr/bin/env python3
"""Decompose counterbalanced routing failure into input and readout bottlenecks."""

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


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_counterbalanced_routing_headroom_audit_toy as routing  # noqa: E402
import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
import run_q_rde_stage0_action_support_audit_toy as stage0  # noqa: E402


SEED = 7
VARIANTS = (
    "current_reference",
    "full_routing",
    "neutral_distractor",
    "masked_distractor",
    "duplicate_selected",
    "query_primary_upper_bound",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_routing_solvability_diagnostic_toy.json"
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
        "dataset",
        "baseline",
        "gate",
        "output_root",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if payload["schema_version"] != "q-attention.q-routing-solvability-diagnostic.v1":
        raise ValueError("unsupported routing solvability diagnostic config")
    if payload["readout"] != "query":
        raise ValueError("routing solvability diagnostic requires query readout")
    if tuple(payload.get("variants", ())) != VARIANTS:
        raise ValueError(f"variants must equal {VARIANTS}")
    return payload


def _clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _selected_and_distractor(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    roles = batch["routing_role"]
    selected = torch.where(
        roles == 0,
        batch["input_ids"][:, routing.EVIDENCE_POSITIONS[0]],
        batch["input_ids"][:, routing.EVIDENCE_POSITIONS[1]],
    )
    distractor = torch.where(
        roles == 0,
        batch["input_ids"][:, routing.EVIDENCE_POSITIONS[1]],
        batch["input_ids"][:, routing.EVIDENCE_POSITIONS[0]],
    )
    return selected, distractor


def make_variant(
    current: dict[str, torch.Tensor],
    routed: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, torch.Tensor]:
    if variant == "current_reference":
        return _clone_batch(current)
    if variant == "full_routing":
        return _clone_batch(routed)

    batch = _clone_batch(routed)
    selected, distractor = _selected_and_distractor(batch)
    left, right = routing.EVIDENCE_POSITIONS
    if variant == "neutral_distractor":
        batch["input_ids"][:, left] = torch.where(
            batch["routing_role"] == 0, selected, torch.full_like(selected, 4)
        )
        batch["input_ids"][:, right] = torch.where(
            batch["routing_role"] == 1, selected, torch.full_like(selected, 4)
        )
        return batch
    if variant == "masked_distractor":
        batch["attention_mask"][:, left] = batch["routing_role"] == 0
        batch["attention_mask"][:, right] = batch["routing_role"] == 1
        return batch
    if variant == "duplicate_selected":
        batch["input_ids"][:, left] = selected
        batch["input_ids"][:, right] = selected
        return batch
    if variant == "query_primary_upper_bound":
        primary = 2 + batch["selected_evidence_label"]
        batch["input_ids"][:, legacy.EVIDENCE_POS] = primary
        return batch
    raise ValueError(f"unknown variant: {variant}")


def variant_contract(
    current: dict[str, torch.Tensor],
    routed: dict[str, torch.Tensor],
    variant_batches: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    selected, distractor = _selected_and_distractor(routed)
    checks = {
        "full_routing_uses_selected_and_distractor": bool(
            torch.equal(
                variant_batches["full_routing"]["input_ids"][:, routing.EVIDENCE_POSITIONS[0]],
                torch.where(
                    routed["routing_role"] == 0, selected, distractor
                ),
            )
            and torch.equal(
                variant_batches["full_routing"]["input_ids"][:, routing.EVIDENCE_POSITIONS[1]],
                torch.where(
                    routed["routing_role"] == 1, selected, distractor
                ),
            )
        ),
        "neutral_distractor_hides_only_unselected_cue": bool(
            torch.equal(
                variant_batches["neutral_distractor"]["input_ids"][:, routing.EVIDENCE_POSITIONS[0]],
                torch.where(
                    routed["routing_role"] == 0, selected, torch.full_like(selected, 4)
                ),
            )
            and torch.equal(
                variant_batches["neutral_distractor"]["input_ids"][:, routing.EVIDENCE_POSITIONS[1]],
                torch.where(
                    routed["routing_role"] == 1, selected, torch.full_like(selected, 4)
                ),
            )
        ),
        "masked_distractor_masks_only_unselected_position": bool(
            torch.equal(
                variant_batches["masked_distractor"]["attention_mask"][:, routing.EVIDENCE_POSITIONS[0]],
                routed["routing_role"] == 0,
            )
            and torch.equal(
                variant_batches["masked_distractor"]["attention_mask"][:, routing.EVIDENCE_POSITIONS[1]],
                routed["routing_role"] == 1,
            )
        ),
        "query_primary_uses_original_primary_cue": bool(
            torch.equal(
                variant_batches["query_primary_upper_bound"]["input_ids"][:, legacy.EVIDENCE_POS],
                2 + routed["selected_evidence_label"],
            )
        ),
        "current_reference_matches_legacy": bool(
            torch.equal(
                variant_batches["current_reference"]["input_ids"],
                current["input_ids"],
            )
        ),
    }
    return {
        **checks,
        "status": "pass" if all(checks.values()) else "fail",
        "selected_token_histogram": torch.bincount(
            selected.cpu(), minlength=legacy.VOCAB_SIZE
        ).tolist(),
        "distractor_token_histogram": torch.bincount(
            distractor.cpu(), minlength=legacy.VOCAB_SIZE
        ).tolist(),
    }


def evaluate_variant(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    reference_accuracy: float,
) -> dict[str, Any]:
    with torch.no_grad():
        logits = stage0.query_indexed_model_logits(model, batch)
    labels = batch["labels"]
    prediction = logits.argmax(dim=-1)
    accuracy = float(prediction.eq(labels).float().mean().item())
    row = torch.arange(labels.shape[0], device=labels.device)
    margin = logits[row, labels] - logits[row, prediction]
    return {
        "examples": int(labels.shape[0]),
        "accuracy": accuracy,
        "accuracy_delta_to_current": accuracy - reference_accuracy,
        "wrong_examples": int(prediction.ne(labels).sum().item()),
        "prediction_entropy": float(
            torch.softmax(logits, dim=-1).mul(
                torch.log_softmax(logits, dim=-1)
            ).sum(dim=-1).neg().mean().item()
        ),
        "mean_gold_margin": float(margin.mean().item()),
        "attention_mask_true_fraction": float(batch["attention_mask"].float().mean().item()),
    }


def diagnostic_gate(
    results: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    reference = results["current_reference"]["accuracy"]
    parity = float(gate["reference_tolerance"])
    hard_route_best = max(
        results["neutral_distractor"]["accuracy"],
        results["masked_distractor"]["accuracy"],
    )
    conditions = {
        "variant_contract": contract["status"] == "pass",
        "hard_route_recovers_reference": hard_route_best >= reference - parity,
        "duplicate_route_recovers_reference": results["duplicate_selected"]["accuracy"]
        >= reference - parity,
        "query_upper_bound_reaches_reference": results["query_primary_upper_bound"]["accuracy"]
        >= reference - parity,
    }
    if conditions["hard_route_recovers_reference"]:
        interpretation = "distractor_interference"
    elif conditions["duplicate_route_recovers_reference"]:
        interpretation = "position_competition"
    elif conditions["query_upper_bound_reaches_reference"]:
        interpretation = "query_representation_bottleneck"
    else:
        interpretation = "distributed_or_training_bottleneck"
    return {
        **conditions,
        "reference_accuracy": reference,
        "hard_route_best_accuracy": hard_route_best,
        "interpretation": interpretation,
        "status": "pass" if conditions["variant_contract"] else "fail",
        "mechanism_authorized": False,
        "real_data_authorized": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    seed = int(args.seed)
    if seed != int(config["seed"]):
        raise ValueError("this diagnostic is predeclared for seed 7 only")
    device = legacy.choose_device(args.device or str(config["device"]))
    legacy.model_logits = stage0.query_indexed_model_logits
    dataset = config["dataset"]
    valid_size = int(dataset["valid_size"])
    current = legacy.make_split(seed + 10000, valid_size, device)
    routed = routing.make_counterbalanced_split(seed + 10000, valid_size, device)
    current_train = legacy.make_split(seed, int(dataset["train_size"]), device)
    current_calibration = legacy.make_split(
        seed + 1000, int(dataset["calibration_size"]), device
    )
    routed_train = routing.make_counterbalanced_split(
        seed, int(dataset["train_size"]), device
    )
    routed_calibration = routing.make_counterbalanced_split(
        seed + 1000, int(dataset["calibration_size"]), device
    )
    current_model = routing.train_baseline(
        seed, current_train, current_calibration, device, config
    )[0]
    routed_model = routing.train_baseline(
        seed, routed_train, routed_calibration, device, config
    )[0]
    variants = {
        name: make_variant(current, routed, name) for name in VARIANTS
    }
    contract = variant_contract(current, routed, variants)
    reference_accuracy = evaluate_variant(
        current_model, variants["current_reference"], 0.0
    )["accuracy"]
    results = {
        "current_reference": evaluate_variant(
            current_model, variants["current_reference"], reference_accuracy
        ),
        **{
            name: evaluate_variant(routed_model, variants[name], reference_accuracy)
            for name in VARIANTS
            if name != "current_reference"
        },
    }
    gate = diagnostic_gate(results, contract, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.q-routing-solvability-diagnostic.v1",
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
        "seed": seed,
        "readout": config["readout"],
        "results": results,
        "variant_contract": contract,
        "diagnostic_gate": gate,
        "interpretation": {
            "full_routing": "frozen routed split as trained",
            "neutral_distractor": "replace only the unselected cue with neutral token 4",
            "masked_distractor": "mask only the unselected evidence position from attention",
            "duplicate_selected": "copy the selected cue into both evidence positions",
            "query_primary_upper_bound": "restore the original primary cue at the fixed query position",
        },
        "limitations": [
            "This is a one-seed frozen-model solvability decomposition, not task utility evidence.",
            "All variants are evaluated on the same trained routed model and valid split.",
            "No new quantum mechanism, scalar sweep, or real-data run is authorized.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "diagnostic_gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
