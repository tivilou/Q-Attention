#!/usr/bin/env python3
"""Seed-7 oracle-anchored continuation canary for gain-only Q-AOC."""

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
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_q_antisymmetric_observable_contrast_toy as q_aoc  # noqa: E402
import run_q_candidate_attention_transport_toy as qcat  # noqa: E402
import run_q_instance_conditioned_field_capacity_toy as capacity  # noqa: E402
import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
import run_q_rde_oracle_field_fit_toy as oracle_fit  # noqa: E402
import run_q_rde_stage0_action_support_audit_toy as stage0  # noqa: E402
import run_q_relative_evidence_field_toy as q_rde  # noqa: E402


SELECTORS = (
    "disabled",
    "q_oracle_warm_gain_quantum",
    "q_oracle_warm_gain_classical",
    "q_oracle_warm_gain_shuffled_candidate",
    "q_gain_task_only_quantum",
    "q_gain_task_only_classical",
    "q_aoc_quantum",
    "q_cat_gold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_oracle_anchored_gain_continuation_canary_toy.json"
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
        "oracle",
        "fit",
        "training",
        "gate",
        "q_cat_config",
        "output_root",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if payload["schema_version"] != "q-attention.q-oracle-continuation-canary-config.v1":
        raise ValueError("unsupported oracle continuation config")
    if tuple(payload["selectors"]) != SELECTORS:
        raise ValueError(f"selectors must equal {SELECTORS}")
    if payload["readout"] != "query":
        raise ValueError("oracle continuation requires query-indexed readout")
    return payload


def build_gain(
    kernel_type: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    kernel = capacity.build_kernel(
        f"gain_only_{kernel_type}", seed, config, device
    )
    return kernel


def task_train_and_evaluate(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    train: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    selector: str,
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    training = q_rde.train_rde(model, kernel, train, config)
    row = q_rde.evaluate_rde(
        model,
        kernel,
        valid,
        int(config["dataset"]["batch_size"]),
        selector,
        float(config["fit"]["max_delta"]),
    )
    row.update(training)
    return training, row


def warm_start_and_continue(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    train: dict[str, torch.Tensor],
    calibration_records: list[dict[str, torch.Tensor]],
    valid_records: list[dict[str, torch.Tensor]],
    valid_oracle_corrected: int,
    valid: dict[str, torch.Tensor],
    selector: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_info = capacity.fit_kernel(kernel, calibration_records, config)
    pre_continuation = capacity.evaluate_kernel(
        model, kernel, valid_records, valid_oracle_corrected
    )
    training, task_row = task_train_and_evaluate(
        model, kernel, train, valid, selector, config
    )
    post_continuation = capacity.evaluate_kernel(
        model, kernel, valid_records, valid_oracle_corrected
    )
    task_row["oracle_fit"] = fit_info
    task_row["pre_continuation_field"] = pre_continuation
    task_row["post_continuation_field"] = post_continuation
    return training, task_row


def promotion_gate(
    rows: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    quantum = rows["q_oracle_warm_gain_quantum"]
    classical = rows["q_oracle_warm_gain_classical"]
    shuffled = rows["q_oracle_warm_gain_shuffled_candidate"]
    random_gain = rows["q_gain_task_only_quantum"]
    fixed = rows["q_aoc_quantum"]
    qcat_row = rows["q_cat_gold"]
    post_field = quantum["post_continuation_field"]
    conditions = {
        "minimum_corrections": quantum["corrected_examples"]
        >= int(config["gate"]["min_corrected_examples"]),
        "improves_random_task_only_gain": quantum["corrected_examples"]
        > random_gain["corrected_examples"],
        "improves_fixed_q_aoc": quantum["corrected_examples"]
        > fixed["corrected_examples"],
        "corrective_expansion_beyond_q_cat": quantum["corrected_examples"]
        > qcat_row["corrected_examples"],
        "quantum_exceeds_classical_corrections": quantum["corrected_examples"]
        > classical["corrected_examples"],
        "positive_accuracy_delta": quantum["accuracy_delta"] > 0.0,
        "correct_retention": quantum["correct_to_wrong_rate"]
        <= float(config["gate"]["max_correct_to_wrong_rate"]),
        "aligned_beats_shuffled": quantum["accuracy"] > shuffled["accuracy"],
        "positive_rescue_support": post_field["rescue_support_margin_mean"]
        is not None
        and post_field["rescue_support_margin_mean"] > 0.0,
        "oracle_fit_positive": quantum["pre_continuation_field"][
            "field_cosine_mean"
        ]
        > 0.0,
        "post_field_finite": post_field["residual_invariants"],
        "quantum_not_classically_matched": quantum["accuracy"]
        > classical["accuracy"],
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
    calibration = legacy.make_split(seed + 1000, int(dataset["calibration_size"]), device)
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    calibration_records, calibration_info = oracle_fit.collect_oracle_records(
        model, calibration, config
    )
    valid_records, valid_info = oracle_fit.collect_oracle_records(
        model, valid, config
    )
    rows: dict[str, dict[str, Any]] = {
        "disabled": {
            "selector": "disabled",
            "corrected_examples": 0,
            "harmed_correct_examples": 0,
        }
    }

    for kernel_type, selector in (
        ("quantum", "q_oracle_warm_gain_quantum"),
        ("classical", "q_oracle_warm_gain_classical"),
    ):
        model.load_state_dict(baseline_state)
        kernel = build_gain(kernel_type, seed, config, device)
        _, row = warm_start_and_continue(
            model,
            kernel,
            train,
            calibration_records,
            valid_records,
            int(valid_info["oracle_corrected_examples"]),
            valid,
            selector,
            config,
        )
        rows[selector] = row

    model.load_state_dict(baseline_state)
    kernel = build_gain("quantum", seed, config, device)
    _, row = task_train_and_evaluate(
        model, kernel, train, valid, "q_gain_task_only_quantum", config
    )
    rows["q_gain_task_only_quantum"] = row

    model.load_state_dict(baseline_state)
    kernel = build_gain("classical", seed, config, device)
    _, row = task_train_and_evaluate(
        model, kernel, train, valid, "q_gain_task_only_classical", config
    )
    rows["q_gain_task_only_classical"] = row

    model.load_state_dict(baseline_state)
    kernel = build_gain("quantum", seed, config, device)
    _, row = task_train_and_evaluate(
        model,
        kernel,
        train,
        valid,
        "q_oracle_warm_gain_shuffled_candidate",
        config,
    )
    rows["q_oracle_warm_gain_shuffled_candidate"] = row

    model.load_state_dict(baseline_state)
    kernel = q_aoc.build_aoc("quantum", seed, config, device)
    _, row = task_train_and_evaluate(model, kernel, train, valid, "q_aoc_quantum", config)
    rows["q_aoc_quantum"] = row

    model.load_state_dict(baseline_state)
    qcat_config = qcat.load_config(ROOT / str(config["q_cat_config"]))
    qcat_kernel = qcat.build_kernel("quantum", seed, qcat_config, device)
    qcat.train_kernel(model, qcat_kernel, train, qcat_config)
    rows["q_cat_gold"] = q_rde.evaluate_qcat(
        model, qcat_kernel, valid, int(dataset["batch_size"])
    )

    gate = promotion_gate(rows, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.q-oracle-continuation-canary.v1",
        "status": "complete",
        "run_type": "gold_candidate_oracle_anchored_gain_continuation_canary",
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
        "oracle_source": calibration_info,
        "oracle_target": valid_info,
        "results": rows,
        "promotion_gate": gate,
        "continuation_contract": {
            "warm_start": "80-step calibration oracle residual fit",
            "continuation": "unchanged 80-step task-loss training",
            "new_trainable_parameters": 0,
            "new_scalar_settings": 0,
            "oracle_target": "gold-conditioned Stage-0 residual on calibration only",
        },
        "limitations": [
            "Gold candidate labels, oracle residuals, corruption flags, and rescue availability condition this diagnostic.",
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
