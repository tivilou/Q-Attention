#!/usr/bin/env python3
"""Single-seed quantum-estimator canary after the consensus pre-screen gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from q_attention.plugins.q_consensus_quantum_estimator import (  # noqa: E402
    ConsensusQuantumEstimatorConfig,
    build_consensus_estimator,
)


SELECTORS = (
    "disabled",
    "q_consensus_quantum",
    "classical_consensus_control",
    "q_consensus_shuffled_query",
    "q_consensus_magnitude",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_consensus_quantum_estimator_canary.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.q-consensus-quantum-estimator-canary.v1":
        raise ValueError("unsupported quantum-estimator config schema")
    required = {"seed", "device", "dataset", "estimator", "training", "gate", "output_root"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if tuple(payload["selectors"]) != SELECTORS:
        raise ValueError("selectors must match the explicit canary allowlist")
    return payload


def batches(split: dict[str, torch.Tensor], batch_size: int):
    for start in range(0, split["labels"].shape[0], batch_size):
        yield {name: value[start : start + batch_size] for name, value in split.items()}


def build_estimator(kind: str, seed: int, frames: torch.Tensor, config: dict[str, Any], device: torch.device):
    estimator = config["estimator"]
    kernel = build_consensus_estimator(
        kind,
        ConsensusQuantumEstimatorConfig(
            num_candidates=task.v1.CLASSES,
            head_dim=task.v1.DIM,
            register_qubits=int(estimator["register_qubits"]),
            depth=int(estimator["depth"]),
            angle_scale=float(estimator["angle_scale"]),
            seed=seed + int(estimator["seed_offset"]),
        ),
        frames,
    )
    return kernel.to(device)


def train_estimator(
    estimator: torch.nn.Module,
    train: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, Any]:
    for parameter in estimator.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        estimator.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    train_batches = list(batches(train, int(config["dataset"]["batch_size"])))
    losses = []
    gradient_norms = []
    for step in range(int(config["training"]["steps"])):
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        scores = estimator.candidate_scores(batch["query"], batch["key"])
        loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), batch["labels"].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite estimator loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in estimator.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"missing or non-finite estimator gradient at step {step}")
        gradient_norm = torch.sqrt(
            sum(gradient.detach().square().sum() for gradient in gradients)
        )
        gradient_norms.append(float(gradient_norm))
        torch.nn.utils.clip_grad_norm_(
            estimator.parameters(), float(config["training"]["gradient_clip"])
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "steps": len(losses),
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
    }


def evaluate(
    selector: str,
    estimator: torch.nn.Module | None,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    labels_all = []
    baseline_all = []
    prediction_all = []
    candidate_all = []
    active_all = []
    residuals = []
    with torch.no_grad():
        for batch in batches(split, batch_size):
            baseline_logits, _ = task.v1.baseline_logits(
                batch["scores"], batch["key"], batch["query"], frames
            )
            if selector == "disabled":
                candidate = torch.zeros_like(batch["labels"])
                support = torch.zeros(
                    batch["labels"].shape[0], task.v1.QUERIES, task.v1.EVIDENCE_KEYS,
                    dtype=torch.long, device=labels_all[0].device if labels_all else batch["labels"].device,
                )
                active = torch.zeros_like(batch["labels"], dtype=torch.bool)
            else:
                query = batch["query"]
                if selector == "q_consensus_shuffled_query":
                    query = query[torch.roll(torch.arange(query.shape[0], device=query.device), 1)]
                field = estimator.field(query, batch["key"])
                if selector == "q_consensus_magnitude":
                    field = field.abs()
                top_support = field.topk(task.v1.EVIDENCE_KEYS, dim=-1)
                candidate_scores = top_support.values.mean(dim=-1)
                candidate = candidate_scores.argmax(dim=-1)
                support = top_support.indices.gather(
                    2,
                    candidate[:, :, None, None].expand(-1, -1, 1, task.v1.EVIDENCE_KEYS),
                ).squeeze(2)
                active = task.error_witness(batch["scores"])
            steered_scores, residual = task.apply_pair_actions(
                batch["scores"], support, active
            )
            logits, _ = task.v1.baseline_logits(
                steered_scores, batch["key"], batch["query"], frames
            )
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
    harmed = correct & prediction.ne(labels)
    return {
        "selector": selector,
        "baseline_accuracy": float(baseline_prediction.eq(labels).float().mean()),
        "accuracy": float(prediction.eq(labels).float().mean()),
        "accuracy_delta": float(
            prediction.eq(labels).float().mean()
            - baseline_prediction.eq(labels).float().mean()
        ),
        "baseline_wrong_queries": int(wrong.sum()),
        "corrected_queries": int(corrected.sum()),
        "harmed_correct_queries": int(harmed.sum()),
        "wrong_correction_rate": float(corrected.sum() / wrong.sum().clamp_min(1)),
        "harm_rate": float(harmed.sum() / correct.sum().clamp_min(1)),
        "active_rate": float(active.float().mean()),
        "active_candidate_accuracy": float(candidate[active].eq(labels[active]).float().mean())
        if active.any()
        else 0.0,
        "residual_finite": bool(torch.isfinite(residual).all()),
        "residual_zero_sum_error": float(residual.sum(dim=-1).abs().max()),
        "residual_max_abs": float(residual.abs().max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    device = task.v1.choose_device(args.device or str(config["device"]))
    seed = int(config["seed"])
    dataset = config["dataset"]
    frames = task.v1.relation_frames(device)
    streams = {
        "train": task.make_split(seed, int(dataset["train_size"]), device),
        "calibration": task.make_split(seed + 1000, int(dataset["calibration_size"]), device),
        "valid": task.make_split(seed + 10000, int(dataset["valid_size"]), device),
        "test": task.make_split(seed + 20000, int(dataset["test_size"]), device),
    }
    estimators = {
        "quantum": build_estimator("quantum", seed, frames, config, device),
        "classical": build_estimator("classical", seed, frames, config, device),
    }
    training = {
        name: train_estimator(estimator, streams["train"], config)
        for name, estimator in estimators.items()
    }
    baseline = {}
    selectors: dict[str, dict[str, Any]] = {}
    for split_name, split in streams.items():
        logits, _ = task.v1.baseline_logits(
            split["scores"], split["key"], split["query"], frames
        )
        replay, _ = task.v1.baseline_logits(
            split["scores"], split["key"], split["query"], frames
        )
        baseline_prediction = logits.argmax(dim=-1)
        baseline[split_name] = {
            "accuracy": float(baseline_prediction.eq(split["labels"]).float().mean()),
            "replay_error": float((replay - logits).abs().max()),
            "queries": int(split["labels"].numel()),
        }
        selectors[split_name] = {
            "disabled": evaluate("disabled", None, split, frames, int(dataset["batch_size"])),
            "q_consensus_quantum": evaluate(
                "q_consensus_quantum", estimators["quantum"], split, frames, int(dataset["batch_size"])
            ),
            "classical_consensus_control": evaluate(
                "classical_consensus_control", estimators["classical"], split, frames, int(dataset["batch_size"])
            ),
            "q_consensus_shuffled_query": evaluate(
                "q_consensus_shuffled_query", estimators["quantum"], split, frames, int(dataset["batch_size"])
            ),
            "q_consensus_magnitude": evaluate(
                "q_consensus_magnitude", estimators["quantum"], split, frames, int(dataset["batch_size"])
            ),
        }
    gate_config = config["gate"]
    quantum = "q_consensus_quantum"
    classical = "classical_consensus_control"
    shuffled = "q_consensus_shuffled_query"
    valid = selectors["valid"]
    test = selectors["test"]
    conditions = {
        "baseline_replay": all(item["replay_error"] == 0.0 for item in baseline.values()),
        "baseline_non_saturated": float(gate_config["baseline_accuracy_min"])
        <= baseline["valid"]["accuracy"]
        <= float(gate_config["baseline_accuracy_max"])
        and float(gate_config["baseline_accuracy_min"])
        <= baseline["test"]["accuracy"]
        <= float(gate_config["baseline_accuracy_max"]),
        "quantum_heldout_gain": valid[quantum]["accuracy_delta"]
        >= float(gate_config["minimum_accuracy_delta"])
        and test[quantum]["accuracy_delta"]
        >= float(gate_config["minimum_accuracy_delta"]),
        "quantum_no_harm": valid[quantum]["harm_rate"]
        <= float(gate_config["maximum_harm_rate"])
        and test[quantum]["harm_rate"]
        <= float(gate_config["maximum_harm_rate"]),
        "quantum_beats_classical": valid[quantum]["accuracy_delta"]
        - valid[classical]["accuracy_delta"]
        >= float(gate_config["minimum_quantum_control_margin"])
        and test[quantum]["accuracy_delta"]
        - test[classical]["accuracy_delta"]
        >= float(gate_config["minimum_quantum_control_margin"]),
        "quantum_beats_shuffled": valid[quantum]["accuracy_delta"]
        - valid[shuffled]["accuracy_delta"]
        >= float(gate_config["minimum_control_margin"])
        and test[quantum]["accuracy_delta"]
        - test[shuffled]["accuracy_delta"]
        >= float(gate_config["minimum_control_margin"]),
        "training_gradients_finite": all(
            info["gradient_norm_min"] > 0.0
            and info["gradient_norm_max"] < float(gate_config["maximum_gradient_norm"])
            for info in training.values()
        ),
        "residual_invariants": all(
            item[quantum]["residual_finite"]
            and item[quantum]["residual_zero_sum_error"] <= 1e-5
            and item[quantum]["residual_max_abs"] <= task.v1.MAX_DELTA + 1e-6
            for item in selectors.values()
        ),
    }
    return {
        "schema_version": "q-attention.q-consensus-quantum-estimator-canary.v1",
        "status": "complete",
        "experiment_name": "q_consensus_quantum_estimator_canary",
        "dataset_identity": config["dataset"]["identity"],
        "seed": seed,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "baseline": baseline,
        "training": training,
        "estimators": {name: estimator.metadata() for name, estimator in estimators.items()},
        "selectors": selectors,
        "gate": {
            **conditions,
            "status": "pass" if all(conditions.values()) else "fail",
            "next_multi_seed_authorized": False,
            "next_real_data_authorized": False,
            "hardware_claim": False,
        },
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
    (output / "run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "gate": payload["gate"],
                "baseline": payload["baseline"],
                "quantum": {
                    split: payload["selectors"][split]["q_consensus_quantum"]
                    for split in ("valid", "test")
                },
                "classical": {
                    split: payload["selectors"][split]["classical_consensus_control"]
                    for split in ("valid", "test")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
