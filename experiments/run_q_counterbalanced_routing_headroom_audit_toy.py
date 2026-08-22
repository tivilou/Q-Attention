#!/usr/bin/env python3
"""Counterbalanced routing split validity and five-seed headroom audit."""

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
import run_q_query_field_headroom_audit_toy as headroom  # noqa: E402
import run_q_rde_oracle_field_fit_toy as oracle_fit  # noqa: E402
import run_q_rde_stage0_action_support_audit_toy as stage0  # noqa: E402
import run_q_relative_evidence_field_toy as q_rde  # noqa: E402


FIXED_SEEDS = (7, 11, 13, 17, 23)
ROLE_MARKERS = (28, 29)
EVIDENCE_POSITIONS = (4, 5)
CURRENT_MAKE_SPLIT = legacy.make_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/q_counterbalanced_routing_headroom_audit_toy.json",
    )
    parser.add_argument(
        "--phase", choices=("seed7", "five_seed"), default="seed7"
    )
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
    if payload["schema_version"] != "q-attention.q-counterbalanced-routing.v2":
        raise ValueError("unsupported counterbalanced routing config")
    if tuple(int(seed) for seed in payload["seeds"]) != FIXED_SEEDS:
        raise ValueError(f"seeds must equal the fixed set {FIXED_SEEDS}")
    if payload["readout"] != "query":
        raise ValueError("counterbalanced routing requires query-indexed readout")
    if tuple(payload["dataset"].get("evidence_positions", ())) != EVIDENCE_POSITIONS:
        raise ValueError(f"evidence positions must equal {EVIDENCE_POSITIONS}")
    if tuple(payload["dataset"].get("role_markers", ())) != ROLE_MARKERS:
        raise ValueError(f"role markers must equal {ROLE_MARKERS}")
    return payload


def _symmetric_distractors(
    primary_label: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Match each cue to a different cue while preserving both marginals."""
    primary = primary_label.cpu()
    counts = torch.bincount(primary, minlength=legacy.NUM_LABELS)
    size = int(primary.shape[0])
    if legacy.NUM_LABELS != 3 or int(counts.max().item()) > size // 2:
        raise ValueError("symmetric three-label counterbalancing is infeasible")

    buckets: list[list[int]] = []
    for label in range(legacy.NUM_LABELS):
        indices = torch.where(primary == label)[0]
        order = torch.randperm(indices.numel(), generator=generator)
        buckets.append(indices[order].tolist())

    c0, c1, c2 = (int(value) for value in counts.tolist())
    edge_counts = {
        (0, 1): (c0 + c1 - c2) // 2,
        (0, 2): (c0 + c2 - c1) // 2,
        (1, 2): (c1 + c2 - c0) // 2,
    }
    cursors = [0, 0, 0]
    distractor = torch.full_like(primary, -1)
    pairs: list[tuple[int, int]] = []
    for (left_label, right_label), count in edge_counts.items():
        for _ in range(count):
            left = buckets[left_label][cursors[left_label]]
            right = buckets[right_label][cursors[right_label]]
            cursors[left_label] += 1
            cursors[right_label] += 1
            distractor[left] = right_label
            distractor[right] = left_label
            pairs.append((left, right))
    if any(cursors[label] != int(counts[label]) for label in range(3)):
        raise AssertionError("counterbalanced pair allocation was incomplete")
    if (distractor < 0).any() or distractor.eq(primary).any():
        raise AssertionError("distractors must be assigned and differ from cues")
    return distractor, pairs


def _label_balanced_roles(
    pairs: list[tuple[int, int]],
    labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign one role per symmetric pair with exact label-role balance."""
    labels_cpu = labels.cpu()
    label_counts = torch.bincount(labels_cpu, minlength=legacy.NUM_LABELS)
    if bool((label_counts % 2).any()):
        raise ValueError("each label count must be even for exact role balance")
    target = tuple(int(value // 2) for value in label_counts.tolist())
    target_pairs = len(pairs) // 2
    if len(pairs) % 2:
        raise ValueError("routing pair count must be even")

    order = torch.randperm(len(pairs), generator=generator).tolist()
    ordered_pairs = [pairs[index] for index in order]
    states: dict[tuple[int, int, int], int] = {(0, 0, 0): 0}
    for pair_index, (left, right) in enumerate(ordered_pairs):
        contribution = torch.bincount(
            labels_cpu[torch.tensor([left, right])], minlength=legacy.NUM_LABELS
        )
        add0, add1, add2 = (int(value) for value in contribution.tolist())
        updated = dict(states)
        for (selected, count0, count1), mask in states.items():
            next_selected = selected + 1
            next0 = count0 + add0
            next1 = count1 + add1
            next2 = 2 * next_selected - next0 - next1
            key = (next_selected, next0, next1)
            if (
                next_selected <= target_pairs
                and next0 <= target[0]
                and next1 <= target[1]
                and next2 <= target[2]
                and key not in updated
            ):
                updated[key] = mask | (1 << pair_index)
        states = updated
    final_key = (target_pairs, target[0], target[1])
    if final_key not in states:
        raise ValueError("could not construct an exactly label-balanced role assignment")

    selected_mask = states[final_key]
    roles = torch.zeros(labels_cpu.shape[0], dtype=torch.long)
    pair_ids = torch.empty(labels_cpu.shape[0], dtype=torch.long)
    for pair_index, (left, right) in enumerate(ordered_pairs):
        role = 1 if selected_mask & (1 << pair_index) else 0
        roles[left] = role
        roles[right] = role
        pair_ids[left] = pair_index
        pair_ids[right] = pair_index
    return roles, pair_ids


def make_counterbalanced_split(
    seed: int,
    size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a matched evidence/distractor split selected by a query role."""
    split = CURRENT_MAKE_SPLIT(seed, size, device)
    primary_label = (split["input_ids"][:, legacy.EVIDENCE_POS] - 2).cpu()
    labels = split["labels"].cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed + 4242)
    distractor, pairs = _symmetric_distractors(primary_label, generator)
    roles, pair_ids = _label_balanced_roles(pairs, labels, generator)

    input_ids = split["input_ids"].clone()
    roles_device = roles.to(device)
    primary_device = primary_label.to(device)
    distractor_device = distractor.to(device)
    input_ids[:, legacy.EVIDENCE_POS] = ROLE_MARKERS[0] + roles_device
    input_ids[:, EVIDENCE_POSITIONS[0]] = torch.where(
        roles_device == 0, 2 + primary_device, 2 + distractor_device
    )
    input_ids[:, EVIDENCE_POSITIONS[1]] = torch.where(
        roles_device == 1, 2 + primary_device, 2 + distractor_device
    )
    split["input_ids"] = input_ids
    split["routing_role"] = roles_device
    split["selected_evidence_label"] = primary_device
    split["distractor_label"] = distractor_device
    split["routing_pair_id"] = pair_ids.to(device)
    return split


def counterbalanced_invariants(
    current: dict[str, torch.Tensor],
    routed: dict[str, torch.Tensor],
) -> dict[str, Any]:
    preserved_fields = (
        "labels",
        "primary_corrupt",
        "rescue_available",
        "attention_mask",
        "subject_mask",
        "object_mask",
    )
    field_checks = {
        name: bool(torch.equal(current[name], routed[name]))
        for name in preserved_fields
    }
    roles = routed["routing_role"]
    labels = routed["labels"]
    selected = routed["selected_evidence_label"]
    distractor = routed["distractor_label"]
    left = routed["input_ids"][:, EVIDENCE_POSITIONS[0]] - 2
    right = routed["input_ids"][:, EVIDENCE_POSITIONS[1]] - 2
    primary = current["input_ids"][:, legacy.EVIDENCE_POS] - 2
    hist = lambda values: torch.bincount(  # noqa: E731
        values.cpu(), minlength=legacy.NUM_LABELS
    )
    role_label_counts = torch.stack(
        [torch.bincount(labels[roles == role].cpu(), minlength=legacy.NUM_LABELS)
         for role in (0, 1)]
    )
    conditions = {
        "legacy_fields_preserved": all(field_checks.values()),
        "rescue_tokens_preserved": bool(
            torch.equal(
                current["input_ids"][:, legacy.RESCUE_POS],
                routed["input_ids"][:, legacy.RESCUE_POS],
            )
        ),
        "selected_evidence_preserved": bool(torch.equal(selected, primary)),
        "evidence_distractor_disjoint_per_example": bool(selected.ne(distractor).all()),
        "evidence_distractor_marginals_matched": bool(
            torch.equal(hist(selected), hist(distractor))
        ),
        "left_right_marginals_matched": bool(torch.equal(hist(left), hist(right))),
        "position_marginals_match_primary": bool(
            torch.equal(hist(left), hist(primary))
            and torch.equal(hist(right), hist(primary))
        ),
        "role_counts_balanced": bool(
            torch.equal(
                torch.bincount(roles.cpu(), minlength=2),
                torch.tensor([roles.numel() // 2, roles.numel() // 2]),
            )
        ),
        "role_label_counts_balanced": bool(
            torch.equal(role_label_counts[0], role_label_counts[1])
        ),
        "role_markers_non_label_coded": bool(
            torch.equal(
                routed["input_ids"][:, legacy.EVIDENCE_POS],
                ROLE_MARKERS[0] + roles,
            )
            and min(ROLE_MARKERS) >= legacy.NUM_LABELS + 2
        ),
    }
    return {
        **conditions,
        "status": "pass" if all(conditions.values()) else "fail",
        "field_checks": field_checks,
        "primary_histogram": hist(primary).tolist(),
        "distractor_histogram": hist(distractor).tolist(),
        "left_histogram": hist(left).tolist(),
        "right_histogram": hist(right).tolist(),
        "role_label_counts": role_label_counts.tolist(),
    }


def role_ablated_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ablated = dict(batch)
    input_ids = batch["input_ids"].clone()
    input_ids[:, legacy.EVIDENCE_POS] = ROLE_MARKERS[0] + (
        1 - batch["routing_role"]
    )
    ablated["input_ids"] = input_ids
    return ablated


def baseline_and_role_ablation(
    model: torch.nn.Module,
    valid: dict[str, torch.Tensor],
) -> dict[str, Any]:
    with torch.no_grad():
        baseline = stage0.query_indexed_model_logits(model, valid)
        ablated = stage0.query_indexed_model_logits(model, role_ablated_batch(valid))
    labels = valid["labels"]
    baseline_prediction = baseline.argmax(dim=-1)
    ablated_prediction = ablated.argmax(dim=-1)
    row = torch.arange(labels.shape[0], device=labels.device)
    baseline_margin = baseline[row, labels] - baseline[row, baseline_prediction]
    ablated_margin = ablated[row, labels] - ablated[row, baseline_prediction]
    return {
        "baseline_accuracy": float(baseline_prediction.eq(labels).float().mean()),
        "ablated_accuracy": float(ablated_prediction.eq(labels).float().mean()),
        "accuracy_delta": float(
            ablated_prediction.eq(labels).float().mean()
            - baseline_prediction.eq(labels).float().mean()
        ),
        "mean_gold_margin_change": float((ablated_margin - baseline_margin).mean()),
        "prediction_changed_examples": int(
            ablated_prediction.ne(baseline_prediction).sum()
        ),
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


def validity_gate(
    current_accuracy: float,
    routed_metrics: dict[str, Any],
    marginal_reports: dict[str, dict[str, Any]],
    oracle_row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    conditions = {
        "marginals_exact": all(
            report["status"] == "pass" for report in marginal_reports.values()
        ),
        "baseline_accuracy_parity": abs(
            routed_metrics["baseline_accuracy"] - current_accuracy
        ) <= float(gate["baseline_accuracy_tolerance"]),
        "query_role_ablation_nonnull": (
            routed_metrics["prediction_changed_examples"]
            >= int(gate["minimum_ablation_changed_examples"])
            and abs(routed_metrics["mean_gold_margin_change"]) > 1e-8
        ),
        "oracle_correct_retention": oracle_row["correct_to_wrong_rate"]
        <= float(gate["max_correct_to_wrong_rate"]),
        "oracle_residual_invariants": bool(oracle_row["residual_invariants"]),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "baseline_accuracy_gap": routed_metrics["baseline_accuracy"]
        - current_accuracy,
        "status": "pass" if passed else "fail",
        "five_seed_headroom_authorized": passed,
        "parameter_repair_authorized": False,
        "does_not_authorize_real_data": True,
    }


def run_seed7_validity(
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    seed = FIXED_SEEDS[0]
    dataset = config["dataset"]
    offsets = {"train": 0, "calibration": 1000, "valid": 10000}
    sizes = {
        "train": int(dataset["train_size"]),
        "calibration": int(dataset["calibration_size"]),
        "valid": int(dataset["valid_size"]),
    }
    current = {
        name: CURRENT_MAKE_SPLIT(seed + offsets[name], sizes[name], device)
        for name in offsets
    }
    routed = {
        name: make_counterbalanced_split(seed + offsets[name], sizes[name], device)
        for name in offsets
    }
    reports = {
        name: counterbalanced_invariants(current[name], routed[name])
        for name in offsets
    }

    current_model, current_info = train_baseline(
        seed, current["train"], current["calibration"], device, config
    )
    with torch.no_grad():
        current_logits = stage0.query_indexed_model_logits(
            current_model, current["valid"]
        )
    current_accuracy = float(
        current_logits.argmax(dim=-1).eq(current["valid"]["labels"]).float().mean()
    )

    routed_model, routed_info = train_baseline(
        seed, routed["train"], routed["calibration"], device, config
    )
    routed_metrics = baseline_and_role_ablation(routed_model, routed["valid"])
    for parameter in routed_model.parameters():
        parameter.requires_grad_(False)
    records, oracle_info = oracle_fit.collect_oracle_records(
        routed_model, routed["valid"], config
    )
    oracle_row = headroom.evaluate_oracle_records(
        routed_model,
        records,
        oracle_info,
        float(config["evidence"]["max_delta"]),
    )
    gate = validity_gate(
        current_accuracy, routed_metrics, reports, oracle_row, config
    )
    return {
        "seed": seed,
        "current": {**current_info, "valid_accuracy": current_accuracy},
        "counterbalanced": {**routed_info, **routed_metrics},
        "marginal_reports": reports,
        "query_field_oracle": oracle_row,
        "validity_gate": gate,
    }


def run_headroom_seed(
    seed: int,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    dataset = config["dataset"]
    split_specs = {
        "train": (seed, int(dataset["train_size"])),
        "calibration": (seed + 1000, int(dataset["calibration_size"])),
        "valid": (seed + 10000, int(dataset["valid_size"])),
    }
    routed = {
        name: make_counterbalanced_split(split_seed, size, device)
        for name, (split_seed, size) in split_specs.items()
    }
    reports = {
        name: counterbalanced_invariants(
            CURRENT_MAKE_SPLIT(split_seed, size, device), routed[name]
        )
        for name, (split_seed, size) in split_specs.items()
    }
    if not all(report["status"] == "pass" for report in reports.values()):
        raise AssertionError("counterbalanced split invariants failed")

    model, baseline_info = train_baseline(
        seed, routed["train"], routed["calibration"], device, config
    )
    baseline_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records, oracle_info = oracle_fit.collect_oracle_records(
        model, routed["valid"], config
    )
    oracle_row = headroom.evaluate_oracle_records(
        model,
        records,
        oracle_info,
        float(config["evidence"]["max_delta"]),
    )

    model.load_state_dict(baseline_state)
    aoc_row = headroom.train_restricted(
        model,
        q_aoc.build_aoc("quantum", seed, config, device),
        routed["train"],
        routed["valid"],
        "q_aoc_quantum",
        config,
    )
    model.load_state_dict(baseline_state)
    gain_row = headroom.train_restricted(
        model,
        gain_canary.build_gain_kernel("quantum", seed, config, device),
        routed["train"],
        routed["valid"],
        "q_gain_quantum",
        config,
    )
    best_restricted = max(
        int(aoc_row["corrected_examples"]), int(gain_row["corrected_examples"])
    )
    return {
        "seed": seed,
        "baseline": baseline_info,
        "routing_marginals_exact": True,
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
        ) - best_restricted,
    }


def five_seed_gate(
    seed_rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    minimum = int(config["gate"]["minimum_headroom_corrections"])
    required = math.ceil(
        float(config["gate"]["required_seed_fraction"]) * len(FIXED_SEEDS)
    )
    eligible = [
        row for row in seed_rows
        if row["oracle_headroom_vs_best_restricted"] >= minimum
    ]
    conditions = {
        "fixed_seed_set": tuple(row["seed"] for row in seed_rows) == FIXED_SEEDS,
        "routing_marginals_exact": all(
            row["routing_marginals_exact"] for row in seed_rows
        ),
        "sufficient_headroom_seed_count": len(eligible) >= required,
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
        "minimum_headroom_corrections": minimum,
        "required_seed_count": required,
        "eligible_seed_count": len(eligible),
        "eligible_seeds": [row["seed"] for row in eligible],
        "status": "pass" if passed else "fail",
        "candidate_transport_basis_authorized": passed,
        "does_not_authorize_real_data": True,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = legacy.choose_device(args.device or str(config["device"]))
    legacy.model_logits = stage0.query_indexed_model_logits

    if args.phase == "seed7":
        results: Any = run_seed7_validity(device, config)
        gate = results["validity_gate"]
    else:
        rows = []
        for seed in FIXED_SEEDS:
            row = run_headroom_seed(seed, device, config)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "oracle_corrected": row["results"]["query_field_oracle"]["corrected_examples"],
                        "q_aoc_corrected": row["results"]["q_aoc_quantum"]["corrected_examples"],
                        "q_gain_corrected": row["results"]["q_gain_quantum"]["corrected_examples"],
                        "headroom": row["oracle_headroom_vs_best_restricted"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        gate = five_seed_gate(rows, config)
        results = {
            "seeds": rows,
            "aggregate": {
                "mean_oracle_corrected_examples": sum(
                    row["results"]["query_field_oracle"]["corrected_examples"]
                    for row in rows
                ) / len(rows),
                "mean_best_restricted_corrected_examples": sum(
                    row["best_restricted_corrected_examples"] for row in rows
                ) / len(rows),
                "mean_headroom_corrections": sum(
                    row["oracle_headroom_vs_best_restricted"] for row in rows
                ) / len(rows),
            },
            "promotion_gate": gate,
        }

    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / args.phase / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.q-counterbalanced-routing-headroom-audit.v2",
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
        "results": results,
        "gate": gate,
        "design_contract": {
            "role_markers": list(ROLE_MARKERS),
            "evidence_positions": list(EVIDENCE_POSITIONS),
            "selected_evidence": "legacy noisy primary cue",
            "distractor": "different cue with the exact selected-cue marginal",
            "position_counterbalance": "symmetric cue pairs share a role",
            "label_counterbalance": "each label has equal role-0 and role-1 counts",
            "seed_replacement": "prohibited",
            "parameter_sweep": "prohibited",
        },
        "limitations": [
            "This is a synthetic data-validity and action-headroom diagnostic.",
            "Gold labels condition the query-field oracle and restricted controls.",
            "Passing does not establish label-free task utility or quantum advantage.",
            "No real-data run is authorized by this audit.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
