#!/usr/bin/env python3
"""Five-seed query-field oracle headroom audit for Q-AOC recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
import run_q_instance_conditioned_gain_canary_toy as gain_canary  # noqa: E402
import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
import run_q_rde_oracle_field_fit_toy as oracle_fit  # noqa: E402
import run_q_rde_stage0_action_support_audit_toy as stage0  # noqa: E402
import run_q_relative_evidence_field_toy as q_rde  # noqa: E402


FIXED_SEEDS = (7, 11, 13, 17, 23)
SELECTORS = (
    "disabled",
    "query_field_oracle",
    "q_aoc_quantum",
    "q_gain_quantum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_query_field_headroom_audit_toy.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_name",
        "selectors",
        "seeds",
        "device",
        "readout",
        "dataset",
        "baseline",
        "oracle",
        "fit",
        "evidence",
        "training",
        "gate",
        "output_root",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if payload["schema_version"] != "q-attention.q-headroom-audit-config.v1":
        raise ValueError("unsupported query-field headroom config")
    if tuple(payload["selectors"]) != SELECTORS:
        raise ValueError(f"selectors must equal {SELECTORS}")
    if tuple(int(seed) for seed in payload["seeds"]) != FIXED_SEEDS:
        raise ValueError(f"seeds must equal the fixed set {FIXED_SEEDS}")
    if payload["readout"] != "query":
        raise ValueError("headroom audit requires query-indexed readout")
    return payload


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_oracle_records(
    model: torch.nn.Module,
    records: list[dict[str, torch.Tensor]],
    oracle_info: dict[str, Any],
    max_delta: float,
) -> dict[str, Any]:
    labels_all = []
    baseline_all = []
    prediction_all = []
    invariants = []
    residual_maxima = []
    for record in records:
        with torch.no_grad():
            logits = stage0.run_with_residual(
                model, record, record["oracle_residual"]
            )
        labels_all.append(record["labels"])
        baseline_all.append(record["baseline_prediction"])
        prediction_all.append(logits.argmax(dim=-1))
        context = record["attention_mask"] & ~(
            record["subject_mask"] | record["object_mask"]
        )
        invariants.append(
            q_rde.residual_invariants(
                record["oracle_residual"], context, max_delta
            )
        )
        residual_maxima.append(record["oracle_residual"].abs().max())

    labels = torch.cat(labels_all)
    baseline = torch.cat(baseline_all)
    prediction = torch.cat(prediction_all)
    wrong = baseline.ne(labels)
    correct = ~wrong
    corrected = wrong & prediction.eq(labels)
    harmed = correct & prediction.ne(labels)
    invariant_pass = all(
        item["finite"]
        and item["zero_sum_error_max"] <= 1e-5
        and item["non_context_max_abs"] <= 1e-7
        and item["amplitude_bound_pass"]
        for item in invariants
    )
    return {
        "selector": "query_field_oracle",
        "baseline_accuracy": float(baseline.eq(labels).float().mean().item()),
        "accuracy": float(prediction.eq(labels).float().mean().item()),
        "accuracy_delta": float(
            prediction.eq(labels).float().mean().sub(
                baseline.eq(labels).float().mean()
            ).item()
        ),
        "baseline_wrong_examples": int(wrong.sum().item()),
        "corrected_examples": int(corrected.sum().item()),
        "harmed_correct_examples": int(harmed.sum().item()),
        "wrong_correction_rate": _rate(
            int(corrected.sum().item()), int(wrong.sum().item())
        ),
        "correct_to_wrong_rate": _rate(
            int(harmed.sum().item()), int(correct.sum().item())
        ),
        "mean_best_margin_gain": float(oracle_info["oracle_mean_margin_gain"]),
        "residual_invariants": invariant_pass,
        "residual_max_abs": float(torch.stack(residual_maxima).max().item()),
    }


def train_restricted(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    train: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    selector: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    training = q_rde.train_rde(model, kernel, train, config)
    row = q_rde.evaluate_rde(
        model,
        kernel,
        valid,
        int(config["dataset"]["batch_size"]),
        selector,
        float(config["evidence"]["max_delta"]),
    )
    row.update(training)
    return row


def headroom_gate(
    seed_rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    minimum_gap = int(config["gate"]["minimum_headroom_corrections"])
    required_fraction = float(config["gate"]["required_seed_fraction"])
    required_seed_count = math.ceil(required_fraction * len(FIXED_SEEDS))
    eligible = [
        row for row in seed_rows
        if row["oracle_headroom_vs_best_restricted"] >= minimum_gap
    ]
    conditions = {
        "fixed_seed_set": tuple(row["seed"] for row in seed_rows) == FIXED_SEEDS,
        "sufficient_headroom_seed_count": len(eligible) >= required_seed_count,
        "oracle_correct_retention": all(
            row["results"]["query_field_oracle"]["correct_to_wrong_rate"]
            <= float(config["gate"]["max_correct_to_wrong_rate"])
            for row in seed_rows
        ),
        "oracle_residual_invariants": all(
            row["results"]["query_field_oracle"]["residual_invariants"]
            for row in seed_rows
        ),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "minimum_headroom_corrections": minimum_gap,
        "required_seed_count": required_seed_count,
        "eligible_seed_count": len(eligible),
        "eligible_seeds": [row["seed"] for row in eligible],
        "status": "pass" if passed else "fail",
        "candidate_transport_basis_authorized": passed,
        "diagnostic_split_redesign_required": not passed,
        "does_not_establish_task_utility": True,
        "does_not_authorize_real_data": True,
    }


def run_seed(
    seed: int, device: torch.device, config: dict[str, Any]
) -> dict[str, Any]:
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records, oracle_info = oracle_fit.collect_oracle_records(model, valid, config)
    oracle_row = evaluate_oracle_records(
        model,
        records,
        oracle_info,
        float(config["evidence"]["max_delta"]),
    )

    model.load_state_dict(baseline_state)
    aoc_row = train_restricted(
        model,
        q_aoc.build_aoc("quantum", seed, config, device),
        train,
        valid,
        "q_aoc_quantum",
        config,
    )

    model.load_state_dict(baseline_state)
    gain_row = train_restricted(
        model,
        gain_canary.build_gain_kernel("quantum", seed, config, device),
        train,
        valid,
        "q_gain_quantum",
        config,
    )

    best_restricted = max(
        int(aoc_row["corrected_examples"]), int(gain_row["corrected_examples"])
    )
    return {
        "seed": seed,
        "baseline": baseline_info,
        "results": {
            "disabled": {
                "selector": "disabled",
                "accuracy": oracle_row["baseline_accuracy"],
                "corrected_examples": 0,
                "harmed_correct_examples": 0,
            },
            "query_field_oracle": oracle_row,
            "q_aoc_quantum": aoc_row,
            "q_gain_quantum": gain_row,
        },
        "best_restricted_corrected_examples": best_restricted,
        "oracle_headroom_vs_best_restricted": int(
            oracle_row["corrected_examples"]
        )
        - best_restricted,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = legacy.choose_device(args.device or str(config["device"]))
    legacy.model_logits = stage0.query_indexed_model_logits
    seed_rows = []
    for seed in FIXED_SEEDS:
        row = run_seed(seed, device, config)
        seed_rows.append(row)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "oracle_corrected": row["results"]["query_field_oracle"][
                        "corrected_examples"
                    ],
                    "q_aoc_corrected": row["results"]["q_aoc_quantum"][
                        "corrected_examples"
                    ],
                    "q_gain_corrected": row["results"]["q_gain_quantum"][
                        "corrected_examples"
                    ],
                    "headroom": row["oracle_headroom_vs_best_restricted"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    gate = headroom_gate(seed_rows, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.q-query-field-headroom-audit.v1",
        "status": "complete",
        "run_type": "fixed_five_seed_query_field_oracle_headroom_audit",
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
        "dataset_identity": config["dataset"]["identity"],
        "readout": config["readout"],
        "seeds": list(FIXED_SEEDS),
        "results": seed_rows,
        "aggregate": {
            "mean_oracle_corrected_examples": sum(
                row["results"]["query_field_oracle"]["corrected_examples"]
                for row in seed_rows
            )
            / len(seed_rows),
            "mean_best_restricted_corrected_examples": sum(
                row["best_restricted_corrected_examples"] for row in seed_rows
            )
            / len(seed_rows),
            "mean_headroom_corrections": sum(
                row["oracle_headroom_vs_best_restricted"] for row in seed_rows
            )
            / len(seed_rows),
        },
        "promotion_gate": gate,
        "audit_contract": {
            "seed_replacement": "prohibited",
            "parameter_sweep": "prohibited",
            "oracle_action": "per-example bounded context-only zero-sum query-field residual",
            "restricted_controls": ["task-only Q-AOC", "task-only gain-only Q-AOC"],
        },
        "limitations": [
            "Gold labels condition the query-field oracle and restricted controls.",
            "The audit measures available action headroom, not label-free task utility.",
            "Five fixed synthetic seeds do not establish real-data generalization or significance.",
            "No quantum advantage, hardware speedup, collaborator, or manuscript claim is authorized.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "promotion_gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
