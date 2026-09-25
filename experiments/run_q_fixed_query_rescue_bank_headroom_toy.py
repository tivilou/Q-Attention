#!/usr/bin/env python3
"""Baseline-valid fixed-query rescue-bank headroom audit."""

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
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT / "src", EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_q_antisymmetric_observable_contrast_toy as q_aoc  # noqa: E402
import run_q_instance_conditioned_gain_canary_toy as gain_canary  # noqa: E402
import run_q_margin_credit_self_conditioned_toy as legacy  # noqa: E402
import run_q_query_field_headroom_audit_toy as headroom  # noqa: E402
import run_q_rde_oracle_field_fit_toy as oracle_fit  # noqa: E402
import run_q_rde_stage0_action_support_audit_toy as stage0  # noqa: E402
import run_q_relative_evidence_field_toy as q_rde  # noqa: E402


FIXED_SEEDS = (7, 11, 13, 17, 23)
BANK_POSITIONS = (legacy.RESCUE_POS, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_fixed_query_rescue_bank_headroom_toy.json"
    )
    parser.add_argument("--phase", choices=("seed7", "five_seed"), default="seed7")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_name",
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
    if payload["schema_version"] != "q-attention.fixed-query-rescue-bank-headroom.v1":
        raise ValueError("unsupported fixed-query rescue-bank config")
    if tuple(int(seed) for seed in payload["seeds"]) != FIXED_SEEDS:
        raise ValueError(f"seeds must equal {FIXED_SEEDS}")
    if payload["readout"] != "query":
        raise ValueError("fixed-query rescue-bank audit requires query readout")
    if tuple(payload["dataset"].get("bank_positions", ())) != BANK_POSITIONS:
        raise ValueError(f"bank positions must equal {BANK_POSITIONS}")
    return payload


def make_fixed_query_rescue_bank_split(
    seed: int,
    size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move the existing rescue cue among three positions without changing tokens."""
    split = legacy.make_split(seed, size, device)
    rescue_cpu = split["rescue_available"].cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed + 7171)
    rescue_indices = torch.where(rescue_cpu)[0]
    bank_slot = torch.full((size,), -1, dtype=torch.long)
    if rescue_indices.numel():
        order = torch.randperm(rescue_indices.numel(), generator=generator)
        slots = torch.arange(3, dtype=torch.long).repeat(
            (rescue_indices.numel() + 2) // 3
        )[: rescue_indices.numel()]
        slots = slots[torch.randperm(slots.numel(), generator=generator)]
        for index, slot in zip(rescue_indices[order].tolist(), slots.tolist()):
            bank_slot[index] = slot

    input_ids = split["input_ids"].clone()
    for index in rescue_indices.tolist():
        slot = int(bank_slot[index].item())
        target = BANK_POSITIONS[slot]
        if target != legacy.RESCUE_POS:
            rescue_token = input_ids[index, legacy.RESCUE_POS].clone()
            input_ids[index, legacy.RESCUE_POS] = input_ids[index, target]
            input_ids[index, target] = rescue_token
    split["input_ids"] = input_ids
    split["rescue_bank_position"] = torch.tensor(
        [BANK_POSITIONS[slot] if slot >= 0 else -1 for slot in bank_slot.tolist()],
        dtype=torch.long,
        device=device,
    )
    return split


def split_invariants(
    current: dict[str, torch.Tensor],
    bank: dict[str, torch.Tensor],
) -> dict[str, Any]:
    preserved = (
        "labels",
        "primary_corrupt",
        "rescue_available",
        "attention_mask",
        "subject_mask",
        "object_mask",
    )
    field_checks = {
        name: bool(torch.equal(current[name], bank[name])) for name in preserved
    }
    current_sorted = torch.sort(current["input_ids"], dim=1).values
    bank_sorted = torch.sort(bank["input_ids"], dim=1).values
    rescue = bank["rescue_available"]
    positions = bank["rescue_bank_position"]
    labels = bank["labels"]
    expected_rescue = 8 + labels
    row = torch.arange(labels.shape[0], device=labels.device)
    selected_rescue = torch.full_like(labels, -1)
    selected_rescue[rescue] = bank["input_ids"][row[rescue], positions[rescue]]
    rescue_counts = torch.stack(
        [(positions[rescue] == position).sum() for position in BANK_POSITIONS]
    )
    checks = {
        "legacy_fields_preserved": all(field_checks.values()),
        "per_example_token_multiset_preserved": bool(
            torch.equal(current_sorted, bank_sorted)
        ),
        "query_cue_preserved": bool(
            torch.equal(
                current["input_ids"][:, legacy.EVIDENCE_POS],
                bank["input_ids"][:, legacy.EVIDENCE_POS],
            )
        ),
        "rescue_cue_remains_label_relevant": bool(
            torch.equal(selected_rescue[rescue], expected_rescue[rescue])
        ),
        "non_rescue_bank_position_unset": bool((positions[~rescue] == -1).all()),
        "bank_position_counts_balanced": bool(
            int(rescue_counts.max().item()) - int(rescue_counts.min().item()) <= 1
        ),
        "no_nonbank_positions_changed": bool(
            torch.equal(
                current["input_ids"][:, [0, 1, 2, 6, 7]],
                bank["input_ids"][:, [0, 1, 2, 6, 7]],
            )
        ),
    }
    return {
        **checks,
        "status": "pass" if all(checks.values()) else "fail",
        "field_checks": field_checks,
        "rescue_bank_position_counts": rescue_counts.tolist(),
        "rescue_examples": int(rescue.sum().item()),
    }


def train_baseline(
    seed: int,
    train: dict[str, torch.Tensor],
    calibration: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = legacy.build_model(seed, device)
    info = legacy.train_baseline(
        model,
        train,
        calibration,
        argparse.Namespace(
            baseline_epochs=int(config["baseline"]["epochs"]),
            baseline_lr=float(config["baseline"]["lr"]),
            batch_size=int(config["dataset"]["batch_size"]),
        ),
    )
    return model, info


def baseline_accuracy(
    model: torch.nn.Module,
    split: dict[str, torch.Tensor],
) -> float:
    with torch.no_grad():
        logits = stage0.query_indexed_model_logits(model, split)
    return float(logits.argmax(dim=-1).eq(split["labels"]).float().mean().item())


def evaluate_oracle(
    model: torch.nn.Module,
    split: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    records, oracle_info = oracle_fit.collect_oracle_records(model, split, config)
    row = headroom.evaluate_oracle_records(
        model,
        records,
        oracle_info,
        float(config["evidence"]["max_delta"]),
    )
    return row, records


def run_seed(
    seed: int,
    device: torch.device,
    config: dict[str, Any],
    *,
    run_controls: bool,
) -> dict[str, Any]:
    dataset = config["dataset"]
    specs = {
        "train": (seed, int(dataset["train_size"])),
        "calibration": (seed + 1000, int(dataset["calibration_size"])),
        "valid": (seed + 10000, int(dataset["valid_size"])),
    }
    current = {
        name: legacy.make_split(split_seed, size, device)
        for name, (split_seed, size) in specs.items()
    }
    bank = {
        name: make_fixed_query_rescue_bank_split(split_seed, size, device)
        for name, (split_seed, size) in specs.items()
    }
    reports = {
        name: split_invariants(current[name], bank[name]) for name in specs
    }
    current_model, current_info = train_baseline(
        seed, current["train"], current["calibration"], device, config
    )
    bank_model, bank_info = train_baseline(
        seed, bank["train"], bank["calibration"], device, config
    )
    current_valid_accuracy = baseline_accuracy(current_model, current["valid"])
    bank_valid_accuracy = baseline_accuracy(bank_model, bank["valid"])
    parity_gap = bank_valid_accuracy - current_valid_accuracy
    gate = config["gate"]
    parity_pass = abs(parity_gap) <= float(gate["baseline_accuracy_tolerance"])
    row: dict[str, Any] = {
        "seed": seed,
        "split_invariants": reports,
        "current_baseline": {
            **current_info,
            "valid_accuracy": current_valid_accuracy,
        },
        "bank_baseline": {
            **bank_info,
            "valid_accuracy": bank_valid_accuracy,
        },
        "baseline_accuracy_gap": parity_gap,
        "validity_gate": {
            "split_invariants": all(
                report["status"] == "pass" for report in reports.values()
            ),
            "baseline_accuracy_parity": parity_pass,
            "status": "pass"
            if all(report["status"] == "pass" for report in reports.values())
            and parity_pass
            else "fail",
            "headroom_audit_authorized": False,
        },
    }
    if not run_controls or not parity_pass or row["validity_gate"]["split_invariants"] is False:
        return row

    for parameter in bank_model.parameters():
        parameter.requires_grad_(False)
    oracle_row, records = evaluate_oracle(bank_model, bank["valid"], config)
    baseline_state = {
        key: value.detach().clone() for key, value in bank_model.state_dict().items()
    }
    bank_model.load_state_dict(baseline_state)
    aoc_row = headroom.train_restricted(
        bank_model,
        q_aoc.build_aoc("quantum", seed, config, device),
        bank["train"],
        bank["valid"],
        "q_aoc_quantum",
        config,
    )
    bank_model.load_state_dict(baseline_state)
    gain_row = headroom.train_restricted(
        bank_model,
        gain_canary.build_gain_kernel("quantum", seed, config, device),
        bank["train"],
        bank["valid"],
        "q_gain_quantum",
        config,
    )
    best_restricted = max(
        int(aoc_row["corrected_examples"]), int(gain_row["corrected_examples"])
    )
    headroom_gap = int(oracle_row["corrected_examples"]) - best_restricted
    oracle_gate = {
        "oracle_correct_retention": oracle_row["correct_to_wrong_rate"]
        <= float(gate["max_correct_to_wrong_rate"]),
        "oracle_residual_invariants": bool(oracle_row["residual_invariants"]),
        "minimum_headroom": headroom_gap >= int(gate["minimum_headroom_corrections"]),
    }
    row.update(
        {
            "oracle": oracle_row,
            "q_aoc": aoc_row,
            "q_gain": gain_row,
            "best_restricted_corrected_examples": best_restricted,
            "oracle_headroom_vs_best_restricted": headroom_gap,
            "oracle_gate": {
                **oracle_gate,
                "status": "pass" if all(oracle_gate.values()) else "fail",
            },
        }
    )
    return row


def five_seed_gate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["gate"]
    eligible = [
        row
        for row in rows
        if row.get("oracle_headroom_vs_best_restricted", -10**9)
        >= int(gate["minimum_headroom_corrections"])
    ]
    required = math.ceil(float(gate["required_seed_fraction"]) * len(FIXED_SEEDS))
    conditions = {
        "fixed_seed_set": tuple(row["seed"] for row in rows) == FIXED_SEEDS,
        "all_validity_gates": all(
            row["validity_gate"]["status"] == "pass" for row in rows
        ),
        "sufficient_headroom_seed_count": len(eligible) >= required,
        "oracle_correct_retention": all(
            row.get("oracle_gate", {}).get("oracle_correct_retention", False)
            for row in rows
        ),
        "oracle_residual_invariants": all(
            row.get("oracle_gate", {}).get("oracle_residual_invariants", False)
            for row in rows
        ),
    }
    return {
        **conditions,
        "required_seed_count": required,
        "eligible_seed_count": len(eligible),
        "eligible_seeds": [row["seed"] for row in eligible],
        "status": "pass" if all(conditions.values()) else "fail",
        "new_mechanism_authorized": False,
        "real_data_authorized": False,
    }


def seed7_gate(row: dict[str, Any]) -> dict[str, Any]:
    oracle = row.get("oracle_gate", {})
    conditions = {
        "split_invariants": bool(row["validity_gate"]["split_invariants"]),
        "baseline_accuracy_parity": bool(
            row["validity_gate"]["baseline_accuracy_parity"]
        ),
        "minimum_headroom": bool(oracle.get("minimum_headroom", False)),
        "oracle_correct_retention": bool(
            oracle.get("oracle_correct_retention", False)
        ),
        "oracle_residual_invariants": bool(
            oracle.get("oracle_residual_invariants", False)
        ),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "status": "pass" if passed else "fail",
        "five_seed_phase_authorized": passed,
        "new_mechanism_authorized": False,
        "real_data_authorized": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = legacy.choose_device(args.device or str(config["device"]))
    legacy.model_logits = stage0.query_indexed_model_logits
    seeds = (FIXED_SEEDS[0],) if args.phase == "seed7" else FIXED_SEEDS
    rows = []
    for seed in seeds:
        row = run_seed(seed, device, config, run_controls=True)
        rows.append(row)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "current_accuracy": row["current_baseline"]["valid_accuracy"],
                    "bank_accuracy": row["bank_baseline"]["valid_accuracy"],
                    "parity_gap": row["baseline_accuracy_gap"],
                    "bank_validity": row["validity_gate"]["status"],
                    "headroom": row.get("oracle_headroom_vs_best_restricted"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.phase == "seed7" and row["validity_gate"]["status"] != "pass":
            break
    if args.phase == "seed7":
        gate = seed7_gate(rows[0])
    else:
        gate = five_seed_gate(rows, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / args.phase / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.fixed-query-rescue-bank-headroom.v1",
        "status": "complete",
        "phase": args.phase,
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
        "seeds": [row["seed"] for row in rows],
        "results": rows,
        "gate": gate,
        "design_contract": {
            "bank_positions": list(BANK_POSITIONS),
            "query_cue_position_unchanged": legacy.EVIDENCE_POS,
            "per_example_token_multiset": "exactly preserved",
            "role_marker": "none",
            "sequence_length_and_model_dimensions": "unchanged",
            "seed_replacement": "prohibited",
            "parameter_sweep": "prohibited",
        },
        "limitations": [
            "This is a synthetic split-validity and action-headroom diagnostic.",
            "Oracle and restricted controls are gold-conditioned diagnostics, not label-free inference.",
            "A passing split would require a new mechanism gate and matched classical control; it would not establish utility or quantum advantage.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
