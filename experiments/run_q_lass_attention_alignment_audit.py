#!/usr/bin/env python3
"""Replay frozen Q-LASS seeds and audit evidence-localized attention movement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch


ROOT = Path(
    os.environ.get("Q_ATTENTION_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
EXPERIMENTS = ROOT / "experiments"
SRC = ROOT / "src"
for path in (EXPERIMENTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_q_consensus_quantum_estimator_canary as canary  # noqa: E402
import run_q_consensus_quantum_estimator_frozen_multiseed as frozen  # noqa: E402


SCHEMA_VERSION = "q-attention.q-lass-attention-alignment-audit.v1"
RESULT_SCHEMA_VERSION = "q-attention.q-lass-attention-alignment-result.v1"
AUDIT_SELECTORS = (
    "disabled",
    "q_consensus_quantum",
    "classical_consensus_control",
)
HELDOUT_SPLITS = ("valid", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_lass_attention_alignment_audit.json"
    )
    parser.add_argument(
        "--reference-run",
        required=True,
        help="original frozen multi-seed run directory containing seed_N summaries",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def resolve_path(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_absolute():
        result = ROOT / result
    return result.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Q-LASS alignment-audit config schema")
    required = {
        "experiment_name",
        "source_experiment",
        "device",
        "replay_compatibility",
        "alignment_gate",
        "output_root",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    source = payload["source_experiment"]
    if tuple(source.get("seeds", ())) != frozen.FROZEN_SEEDS:
        raise ValueError("audit seeds must remain the frozen Q-LASS seeds")
    if tuple(source.get("splits", ())) != HELDOUT_SPLITS:
        raise ValueError("audit splits must be valid and test")
    if tuple(source.get("selectors", ())) != AUDIT_SELECTORS:
        raise ValueError("audit selectors must be disabled, Q-LASS, and product control")
    if payload["device"].get("required_type") != "cuda":
        raise ValueError("evidence-eligible replay must require CUDA")
    return payload


def load_reference_summaries(
    reference_root: Path, config: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    source = config["source_experiment"]
    summaries: dict[int, dict[str, Any]] = {}
    for seed in source["seeds"]:
        path = reference_root / f"seed_{seed}" / "run_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != source["seed_summary_schema"]:
            raise ValueError(f"reference seed {seed} summary schema differs")
        if payload.get("seed") != seed:
            raise ValueError(f"reference seed {seed} identity differs")
        provenance = payload.get("provenance", {})
        if provenance.get("git_commit") != source["git_commit"]:
            raise ValueError(f"reference seed {seed} commit differs")
        if provenance.get("git_dirty") is not False:
            raise ValueError(f"reference seed {seed} was not clean")
        if provenance.get("master_config_sha256") != source["master_config_sha256"]:
            raise ValueError(f"reference seed {seed} master config differs")
        summaries[seed] = payload
    return summaries


def source_replay_check(
    references: dict[int, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    expected_by_seed = {
        seed: payload["provenance"].get("source_sha256", {})
        for seed, payload in references.items()
    }
    expected = expected_by_seed[frozen.FROZEN_SEEDS[0]]
    consistent = all(item == expected for item in expected_by_seed.values())
    current = {
        relative: sha256(ROOT / relative)
        for relative in expected
        if (ROOT / relative).is_file()
    }
    missing = sorted(set(expected) - set(current))
    mismatched = sorted(
        relative
        for relative, digest in expected.items()
        if current.get(relative) != digest
    )
    frozen_config = ROOT / "configs/q_consensus_quantum_estimator_frozen_multiseed.json"
    current_master_hash = sha256(frozen_config)
    master_match = current_master_hash == config["source_experiment"][
        "master_config_sha256"
    ]
    return {
        "reference_source_hashes_consistent": consistent,
        "expected_source_sha256": expected,
        "current_source_sha256": current,
        "missing_source_files": missing,
        "mismatched_source_files": mismatched,
        "current_master_config_sha256": current_master_hash,
        "master_config_match": master_match,
        "status": "pass"
        if consistent and not missing and not mismatched and master_match
        else "fail",
    }


def _gather(attention: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    return attention.gather(-1, slots)


def attention_alignment_metrics(
    baseline_attention: torch.Tensor,
    steered_attention: torch.Tensor,
    *,
    evidence_slot: torch.Tensor,
    bad_slot: torch.Tensor,
    active: torch.Tensor,
    baseline_correct: torch.Tensor,
    movement_tolerance: float,
) -> dict[str, Any]:
    """Compute offline localization metrics; selectors never receive gold slots."""
    if evidence_slot.shape != (*baseline_attention.shape[:2], canary.task.v1.EVIDENCE_KEYS):
        raise ValueError("evidence_slot shape differs from the frozen task")
    if bad_slot.shape != baseline_attention.shape[:2]:
        raise ValueError("bad_slot shape differs from the frozen task")

    baseline_evidence = _gather(baseline_attention, evidence_slot).sum(dim=-1)
    steered_evidence = _gather(steered_attention, evidence_slot).sum(dim=-1)
    baseline_distractor = _gather(
        baseline_attention, bad_slot.unsqueeze(-1)
    ).squeeze(-1)
    steered_distractor = _gather(
        steered_attention, bad_slot.unsqueeze(-1)
    ).squeeze(-1)
    baseline_margin = baseline_evidence - baseline_distractor
    steered_margin = steered_evidence - steered_distractor
    movement = steered_margin - baseline_margin

    def localization(attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = attention.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            1, attention.shape[-1] + 1, device=attention.device
        ).view(1, 1, -1)
        ranks.scatter_(-1, order, rank_values.expand_as(order))
        evidence_ranks = _gather(ranks, evidence_slot)
        top2_recall = evidence_ranks.le(canary.task.v1.EVIDENCE_KEYS).float().mean(dim=-1)
        both_top2 = evidence_ranks.le(canary.task.v1.EVIDENCE_KEYS).all(dim=-1).float()
        return top2_recall, both_top2, evidence_ranks.float().mean(dim=-1)

    baseline_recall, baseline_both, baseline_rank = localization(baseline_attention)
    steered_recall, steered_both, steered_rank = localization(steered_attention)
    harmful = movement < -float(movement_tolerance)
    helpful = movement > float(movement_tolerance)
    total = movement.numel()
    active_count = int(active.sum())
    correct_count = int(baseline_correct.sum())

    def mean(value: torch.Tensor) -> float:
        return float(value.float().mean())

    return {
        "queries": total,
        "active_queries": active_count,
        "baseline_evidence_mass": mean(baseline_evidence),
        "steered_evidence_mass": mean(steered_evidence),
        "evidence_mass_delta": mean(steered_evidence - baseline_evidence),
        "baseline_distractor_mass": mean(baseline_distractor),
        "steered_distractor_mass": mean(steered_distractor),
        "distractor_mass_delta": mean(steered_distractor - baseline_distractor),
        "baseline_evidence_minus_distractor_margin": mean(baseline_margin),
        "steered_evidence_minus_distractor_margin": mean(steered_margin),
        "evidence_minus_distractor_margin_delta": mean(movement),
        "baseline_evidence_top2_recall": mean(baseline_recall),
        "steered_evidence_top2_recall": mean(steered_recall),
        "evidence_top2_recall_delta": mean(steered_recall - baseline_recall),
        "baseline_both_evidence_top2_rate": mean(baseline_both),
        "steered_both_evidence_top2_rate": mean(steered_both),
        "both_evidence_top2_rate_delta": mean(steered_both - baseline_both),
        "baseline_mean_evidence_rank": mean(baseline_rank),
        "steered_mean_evidence_rank": mean(steered_rank),
        "mean_evidence_rank_delta": mean(steered_rank - baseline_rank),
        "harmful_movement_queries": int(harmful.sum()),
        "harmful_movement_rate": float(harmful.sum() / max(total, 1)),
        "active_harmful_movement_rate": float(
            (harmful & active).sum() / max(active_count, 1)
        ),
        "baseline_correct_harmful_movement_rate": float(
            (harmful & baseline_correct).sum() / max(correct_count, 1)
        ),
        "helpful_movement_queries": int(helpful.sum()),
        "helpful_movement_rate": float(helpful.sum() / max(total, 1)),
    }


def select_action(
    selector: str,
    estimator: torch.nn.Module | None,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if selector == "disabled":
        candidate = torch.zeros_like(batch["labels"])
        support = torch.zeros(
            *batch["labels"].shape,
            canary.task.v1.EVIDENCE_KEYS,
            dtype=torch.long,
            device=batch["labels"].device,
        )
        active = torch.zeros_like(batch["labels"], dtype=torch.bool)
        return candidate, support, active
    if estimator is None:
        raise ValueError(f"selector {selector} requires an estimator")
    field = estimator.field(batch["query"], batch["key"])
    top_support = field.topk(canary.task.v1.EVIDENCE_KEYS, dim=-1)
    candidate_scores = top_support.values.mean(dim=-1)
    candidate = candidate_scores.argmax(dim=-1)
    support = top_support.indices.gather(
        2,
        candidate[:, :, None, None].expand(
            -1, -1, 1, canary.task.v1.EVIDENCE_KEYS
        ),
    ).squeeze(2)
    active = canary.task.error_witness(batch["scores"])
    return candidate, support, active


def evaluate_alignment(
    selector: str,
    estimator: torch.nn.Module | None,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    *,
    batch_size: int,
    movement_tolerance: float,
) -> dict[str, Any]:
    labels_all: list[torch.Tensor] = []
    baseline_predictions: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    actives: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    baseline_attentions: list[torch.Tensor] = []
    steered_attentions: list[torch.Tensor] = []
    evidence_slots: list[torch.Tensor] = []
    bad_slots: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in canary.batches(split, batch_size):
            baseline_logits, baseline_attention = canary.task.v1.baseline_logits(
                batch["scores"], batch["key"], batch["query"], frames
            )
            candidate, support, active = select_action(selector, estimator, batch)
            steered_scores, residual = canary.task.apply_pair_actions(
                batch["scores"], support, active
            )
            logits, steered_attention = canary.task.v1.baseline_logits(
                steered_scores, batch["key"], batch["query"], frames
            )
            labels_all.append(batch["labels"])
            baseline_predictions.append(baseline_logits.argmax(dim=-1))
            predictions.append(logits.argmax(dim=-1))
            candidates.append(candidate)
            actives.append(active)
            residuals.append(residual)
            baseline_attentions.append(baseline_attention)
            steered_attentions.append(steered_attention)
            evidence_slots.append(batch["evidence_slot"])
            bad_slots.append(batch["bad_slot"])

    labels = torch.cat(labels_all)
    baseline_prediction = torch.cat(baseline_predictions)
    prediction = torch.cat(predictions)
    candidate = torch.cat(candidates)
    active = torch.cat(actives)
    residual = torch.cat(residuals)
    wrong = baseline_prediction.ne(labels)
    correct = ~wrong
    corrected = wrong & prediction.eq(labels)
    harmed = correct & prediction.ne(labels)
    task_metrics = {
        "baseline_accuracy": float(baseline_prediction.eq(labels).float().mean()),
        "accuracy": float(prediction.eq(labels).float().mean()),
        "accuracy_delta": float(
            prediction.eq(labels).float().mean()
            - baseline_prediction.eq(labels).float().mean()
        ),
        "baseline_wrong_queries": int(wrong.sum()),
        "corrected_queries": int(corrected.sum()),
        "harmed_correct_queries": int(harmed.sum()),
        "active_rate": float(active.float().mean()),
        "active_candidate_accuracy": float(
            candidate[active].eq(labels[active]).float().mean()
        )
        if active.any()
        else 0.0,
        "residual_finite": bool(torch.isfinite(residual).all()),
        "residual_zero_sum_error": float(residual.sum(dim=-1).abs().max()),
        "residual_max_abs": float(residual.abs().max()),
    }
    alignment = attention_alignment_metrics(
        torch.cat(baseline_attentions),
        torch.cat(steered_attentions),
        evidence_slot=torch.cat(evidence_slots),
        bad_slot=torch.cat(bad_slots),
        active=active,
        baseline_correct=correct,
        movement_tolerance=movement_tolerance,
    )
    return {"task": task_metrics, "alignment": alignment}


def _count(rate: float, total: int) -> int:
    return int(round(float(rate) * total))


def replay_compatibility(
    replay: dict[str, Any],
    reference: dict[str, Any],
    training: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    limits = config["replay_compatibility"]
    failures: list[str] = []
    comparisons: dict[str, Any] = {}
    for estimator_name in ("quantum", "classical"):
        difference = abs(
            float(training[estimator_name]["final_loss"])
            - float(reference["training"][estimator_name]["final_loss"])
        )
        comparisons[f"{estimator_name}_final_loss_abs_difference"] = difference
        if difference > float(limits["maximum_final_loss_absolute_difference"]):
            failures.append(f"{estimator_name}:final_loss")

    for split_name in HELDOUT_SPLITS:
        total = replay[split_name]["disabled"]["alignment"]["queries"]
        for selector in AUDIT_SELECTORS:
            current = replay[split_name][selector]["task"]
            expected = reference["selectors"][split_name][selector]
            prefix = f"{split_name}:{selector}"
            baseline_difference = abs(
                _count(current["baseline_accuracy"], total)
                - _count(expected["baseline_accuracy"], total)
            )
            accuracy_difference = abs(
                _count(current["accuracy"], total)
                - _count(expected["accuracy"], total)
            )
            corrected_difference = abs(
                current["corrected_queries"] - int(expected["corrected_queries"])
            )
            harmed_difference = abs(
                current["harmed_correct_queries"]
                - int(expected["harmed_correct_queries"])
            )
            current_active = _count(current["active_rate"], total)
            expected_active = _count(expected["active_rate"], total)
            active_difference = abs(current_active - expected_active)
            current_candidate = _count(
                current["active_candidate_accuracy"], current_active
            )
            expected_candidate = _count(
                expected["active_candidate_accuracy"], expected_active
            )
            candidate_difference = abs(current_candidate - expected_candidate)
            comparisons[prefix] = {
                "baseline_accuracy_count_difference": baseline_difference,
                "accuracy_count_difference": accuracy_difference,
                "corrected_query_count_difference": corrected_difference,
                "harmed_query_count_difference": harmed_difference,
                "active_query_count_difference": active_difference,
                "active_candidate_count_difference": candidate_difference,
            }
            if limits["require_exact_baseline_count"] and baseline_difference:
                failures.append(f"{prefix}:baseline")
            if accuracy_difference > int(limits["maximum_accuracy_count_difference"]):
                failures.append(f"{prefix}:accuracy")
            if corrected_difference > int(
                limits["maximum_corrected_query_count_difference"]
            ):
                failures.append(f"{prefix}:corrected")
            if limits["require_exact_harmed_query_count"] and harmed_difference:
                failures.append(f"{prefix}:harmed")
            if limits["require_exact_active_query_count"] and active_difference:
                failures.append(f"{prefix}:active")
            if candidate_difference > int(
                limits["maximum_active_candidate_count_difference"]
            ):
                failures.append(f"{prefix}:candidate")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "comparisons": comparisons,
    }


def aggregate_results(seed_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for split_name in HELDOUT_SPLITS:
        aggregate[split_name] = {}
        for selector in AUDIT_SELECTORS:
            alignment_items = [
                seed_results[seed]["splits"][split_name][selector]["alignment"]
                for seed in frozen.FROZEN_SEEDS
            ]
            task_items = [
                seed_results[seed]["splits"][split_name][selector]["task"]
                for seed in frozen.FROZEN_SEEDS
            ]
            alignment_keys = [
                key
                for key, value in alignment_items[0].items()
                if isinstance(value, (int, float))
                and key not in {"queries", "active_queries"}
            ]
            task_keys = [
                "baseline_accuracy",
                "accuracy",
                "accuracy_delta",
                "active_rate",
                "active_candidate_accuracy",
            ]
            aggregate[split_name][selector] = {
                "alignment": {
                    key: sum(float(item[key]) for item in alignment_items)
                    / len(alignment_items)
                    for key in alignment_keys
                },
                "task": {
                    key: sum(float(item[key]) for item in task_items) / len(task_items)
                    for key in task_keys
                },
            }
    return aggregate


def alignment_gate(
    aggregate: dict[str, Any],
    *,
    replay_pass: bool,
    source_pass: bool,
    device_eligible: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    limits = config["alignment_gate"]
    conditions: dict[str, bool] = {
        "replay_compatible": replay_pass,
        "source_replay_exact": source_pass,
        "device_eligible": device_eligible,
    }
    for split_name in HELDOUT_SPLITS:
        metrics = aggregate[split_name]["q_consensus_quantum"]["alignment"]
        conditions[f"{split_name}_evidence_mass_increased"] = metrics[
            "evidence_mass_delta"
        ] > float(limits["minimum_evidence_mass_delta"])
        conditions[f"{split_name}_distractor_mass_decreased"] = metrics[
            "distractor_mass_delta"
        ] < float(limits["maximum_distractor_mass_delta"])
        conditions[f"{split_name}_margin_increased"] = metrics[
            "evidence_minus_distractor_margin_delta"
        ] > float(limits["minimum_evidence_minus_distractor_margin_delta"])
        conditions[f"{split_name}_top2_localization_increased"] = metrics[
            "evidence_top2_recall_delta"
        ] > float(limits["minimum_evidence_top2_recall_delta"])
        conditions[f"{split_name}_mean_evidence_rank_improved"] = metrics[
            "mean_evidence_rank_delta"
        ] < float(limits["maximum_mean_evidence_rank_delta"])
        conditions[f"{split_name}_harmful_movement_bounded"] = metrics[
            "harmful_movement_rate"
        ] <= float(limits["maximum_harmful_movement_rate"])
    passed = all(conditions.values())
    return {
        "status": "pass" if passed else "fail",
        "conditions": conditions,
        "attention_alignment_axis": "pass" if passed else "fail",
        "task_utility_axis": "unchanged_L2_reproducible_utility",
        "classical_attribution_axis": "fail_unchanged",
        "natural_transfer_authorized": False,
        "claim_ceiling": "direct_synthetic_attention_alignment"
        if passed
        else "reproducible_synthetic_task_utility_only",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    reference_root = resolve_path(args.reference_run)
    references = load_reference_summaries(reference_root, config)
    source_check = source_replay_check(references, config)
    frozen_config = frozen.load_config(
        ROOT / "configs/q_consensus_quantum_estimator_frozen_multiseed.json"
    )
    device_name = args.device or config["device"]["required_type"]
    device = canary.task.v1.choose_device(device_name)
    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    device_eligible = (
        device.type == config["device"]["required_type"]
        and config["device"]["required_name_substring"] in str(cuda_name)
    )
    movement_tolerance = float(config["alignment_gate"]["movement_tolerance"])
    dataset = frozen_config["dataset"]
    seed_results: dict[int, dict[str, Any]] = {}
    all_replay_pass = True

    for seed in frozen.FROZEN_SEEDS:
        frames = canary.task.v1.relation_frames(device)
        streams = {
            "train": canary.task.make_split(
                seed, int(dataset["train_size"]), device
            ),
            "valid": canary.task.make_split(
                seed + 10000, int(dataset["valid_size"]), device
            ),
            "test": canary.task.make_split(
                seed + 20000, int(dataset["test_size"]), device
            ),
        }
        estimators = {
            name: canary.build_estimator(name, seed, frames, frozen_config, device)
            for name in ("quantum", "classical")
        }
        training = {
            name: canary.train_estimator(estimator, streams["train"], frozen_config)
            for name, estimator in estimators.items()
        }
        split_results: dict[str, Any] = {}
        for split_name in HELDOUT_SPLITS:
            split_results[split_name] = {
                "disabled": evaluate_alignment(
                    "disabled",
                    None,
                    streams[split_name],
                    frames,
                    batch_size=int(dataset["batch_size"]),
                    movement_tolerance=movement_tolerance,
                ),
                "q_consensus_quantum": evaluate_alignment(
                    "q_consensus_quantum",
                    estimators["quantum"],
                    streams[split_name],
                    frames,
                    batch_size=int(dataset["batch_size"]),
                    movement_tolerance=movement_tolerance,
                ),
                "classical_consensus_control": evaluate_alignment(
                    "classical_consensus_control",
                    estimators["classical"],
                    streams[split_name],
                    frames,
                    batch_size=int(dataset["batch_size"]),
                    movement_tolerance=movement_tolerance,
                ),
            }
        replay = replay_compatibility(
            split_results, references[seed], training, config
        )
        all_replay_pass = all_replay_pass and replay["status"] == "pass"
        seed_results[seed] = {
            "training": training,
            "splits": split_results,
            "replay_compatibility": replay,
        }

    aggregate = aggregate_results(seed_results)
    gate = alignment_gate(
        aggregate,
        replay_pass=all_replay_pass,
        source_pass=source_check["status"] == "pass",
        device_eligible=device_eligible,
        config=config,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_path": config_path.as_posix(),
        "config_sha256": sha256(config_path),
        "reference_run": reference_root.as_posix(),
        "source_experiment": config["source_experiment"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": cuda_name,
            "evidence_eligible": device_eligible,
        },
        "metric_definitions": {
            "evidence_mass": "sum of softmax attention on the two generated gold evidence slots",
            "distractor_mass": "softmax attention on the generated structured bad-relation slot",
            "margin": "evidence mass minus distractor mass",
            "top2_recall": "fraction of the two evidence slots ranked in the top two keys",
            "mean_evidence_rank": "mean 1-based rank of the two evidence slots; lower is better",
            "harmful_movement": "post-minus-pre margin below negative movement_tolerance",
            "leakage_boundary": "gold evidence and bad slots are consumed only after selector actions are fixed",
        },
        "source_replay_check": source_check,
        "seeds": seed_results,
        "aggregate": aggregate,
        "gate": gate,
    }


def main() -> int:
    args = parse_args()
    payload = run(args)
    config = load_config(resolve_path(args.config))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(
        args.output_dir or Path(config["output_root"]) / stamp
    )
    if output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {output_dir}")
    write_json(output_dir / "run_summary.json", payload)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "gate": payload["gate"],
                "environment": payload["environment"],
                "quantum_alignment": {
                    split: payload["aggregate"][split]["q_consensus_quantum"][
                        "alignment"
                    ]
                    for split in HELDOUT_SPLITS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
