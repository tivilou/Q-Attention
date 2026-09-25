#!/usr/bin/env python3
"""Non-quantum pre-screen for identifiable consensus error witnesses.

This is the single structural follow-up to the failed dynamic-address v1
pilot.  V1 showed that candidate-relative top-2 consensus identifies the
relation but that a single-key, always-active intervention is unsafe.  V2
therefore predeclares two score regimes and uses their fixed midpoint as a
label-free error witness.  Active actions increase the two keys agreeing on a
candidate relation and decrease every other key with the same bounded,
zero-sum query-local budget.

No quantum estimator is built here.  Gold labels are used only for the offline
complete pair-action oracle and evaluation.  Selector inputs are frozen scores,
queries, keys, and fixed candidate relation frames.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_identifiable_query_local_prescreen_toy as v1  # noqa: E402


HARD_RATE = 0.18
EASY_EVIDENCE_BIAS = 1.20
ERROR_WITNESS_THRESHOLD = (v1.BASELINE_BIAS + EASY_EVIDENCE_BIAS) / 2.0
PAIR_ACTIONS = tuple(combinations(range(v1.KEYS), v1.EVIDENCE_KEYS))
SELECTORS = (
    "disabled",
    "candidate_relative_consensus",
    "classical_separable_consensus",
    "query_shuffled_consensus",
    "query_ablated_consensus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_consensus_error_witness_prescreen_toy.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.q-consensus-error-witness-prescreen.v1":
        raise ValueError("unsupported consensus-error-witness config schema")
    required = {"seed", "device", "dataset", "task", "gate", "output_root"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    expected = {
        "queries": v1.QUERIES,
        "classes": v1.CLASSES,
        "keys": v1.KEYS,
        "dim": v1.DIM,
        "evidence_keys": v1.EVIDENCE_KEYS,
        "hard_rate": HARD_RATE,
        "hard_score_bias": v1.BASELINE_BIAS,
        "easy_evidence_bias": EASY_EVIDENCE_BIAS,
        "error_witness_threshold": ERROR_WITNESS_THRESHOLD,
        "max_delta": v1.MAX_DELTA,
    }
    if payload["task"] != expected:
        raise ValueError("task configuration differs from predeclared constants")
    return payload


def make_split(seed: int, size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Reuse v1 geometry with a lower hard stratum and an explicit easy regime."""
    original_hard_rate = v1.HARD_RATE
    v1.HARD_RATE = HARD_RATE
    try:
        split = v1.make_split(seed, size, device)
    finally:
        v1.HARD_RATE = original_hard_rate
    scores = split["scores"].clone()
    easy = ~split["hard"]
    rows = torch.arange(size, device=device)[:, None]
    queries = torch.arange(v1.QUERIES, device=device)[None, :]
    easy_evidence_key = split["evidence_slot"][:, :, 0]
    scores[rows, queries, easy_evidence_key] += easy.to(scores.dtype) * EASY_EVIDENCE_BIAS
    split["scores"] = scores
    return split


def error_witness(scores: torch.Tensor) -> torch.Tensor:
    """Fixed score-concentration witness, independent of labels and payloads."""
    return scores.max(dim=-1).values > ERROR_WITNESS_THRESHOLD


def consensus_field(
    selector: str,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
) -> torch.Tensor:
    query = split["query"]
    if selector == "query_shuffled_consensus":
        permutation = torch.roll(
            torch.arange(query.shape[0], device=query.device), shifts=1
        )
        query = query[permutation]
    if selector == "query_ablated_consensus":
        query = torch.zeros_like(query)
    if selector == "classical_separable_consensus":
        query_term = torch.einsum("bqd,cdh->bqch", query, frames).mean(dim=-1)
        key_term = torch.einsum(
            "bqkd,cdh->bqkch", split["key"], frames
        ).mean(dim=-1)
        return 0.5 * query_term[:, :, :, None] + 0.5 * key_term.permute(0, 1, 3, 2)
    return v1.compatibility(query, split["key"], frames)


def select_actions(
    selector: str,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = split["labels"].shape[0]
    if selector == "disabled":
        candidate = torch.zeros(batch, v1.QUERIES, dtype=torch.long, device=frames.device)
        support = torch.zeros(
            batch, v1.QUERIES, v1.EVIDENCE_KEYS, dtype=torch.long, device=frames.device
        )
        active = torch.zeros(batch, v1.QUERIES, dtype=torch.bool, device=frames.device)
        confidence = torch.zeros(batch, v1.QUERIES, device=frames.device)
        return candidate, support, active, confidence
    field = consensus_field(selector, split, frames)
    centered = field - field.mean(dim=-1, keepdim=True)
    top_support = centered.topk(v1.EVIDENCE_KEYS, dim=-1)
    candidate_score = top_support.values.mean(dim=-1)
    candidate = candidate_score.argmax(dim=-1)
    chosen_support = top_support.indices.gather(
        2,
        candidate[:, :, None, None].expand(-1, -1, 1, v1.EVIDENCE_KEYS),
    ).squeeze(2)
    top_candidates = candidate_score.topk(2, dim=-1).values
    confidence = top_candidates[:, :, 0] - top_candidates[:, :, 1]
    active = error_witness(split["scores"])
    return candidate, chosen_support, active, confidence


def apply_pair_actions(
    scores: torch.Tensor,
    support: torch.Tensor,
    active: torch.Tensor,
    *,
    sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one two-key bounded zero-sum action at each query."""
    if support.shape != (*scores.shape[:2], v1.EVIDENCE_KEYS):
        raise ValueError("support must have shape (batch, queries, evidence_keys)")
    signed_active = active.to(scores.dtype) * float(sign)
    positive = v1.MAX_DELTA / v1.EVIDENCE_KEYS
    negative = v1.MAX_DELTA / (v1.KEYS - v1.EVIDENCE_KEYS)
    residual = -signed_active[:, :, None] * negative
    residual = residual.expand_as(scores).clone()
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    queries = torch.arange(v1.QUERIES, device=scores.device)[None, :]
    for index in range(v1.EVIDENCE_KEYS):
        residual[rows, queries, support[:, :, index]] += signed_active * (
            positive + negative
        )
    return scores + residual, residual


def evaluate_selector(
    selector: str,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    baseline_logits: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    candidate, support, active, confidence = select_actions(selector, split, frames)
    steered_scores, residual = apply_pair_actions(split["scores"], support, active)
    logits, _ = v1.baseline_logits(
        steered_scores, split["key"], split["query"], frames
    )
    labels = split["labels"]
    baseline_prediction = baseline_logits.argmax(dim=-1)
    prediction = logits.argmax(dim=-1)
    wrong = baseline_prediction.ne(labels)
    correct = ~wrong
    corrected = wrong & prediction.eq(labels)
    harmed = correct & prediction.ne(labels)
    metrics = {
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
        "wrong_correction_rate": float(
            corrected.sum() / wrong.sum().clamp_min(1)
        ),
        "harm_rate": float(harmed.sum() / correct.sum().clamp_min(1)),
        "active_rate": float(active.float().mean()),
        "active_wrong_rate": float((active & wrong).sum() / active.sum().clamp_min(1)),
        "candidate_accuracy": float(candidate.eq(labels).float().mean()),
        "active_candidate_accuracy": float(
            candidate[active].eq(labels[active]).float().mean()
        )
        if active.any()
        else 0.0,
        "mean_confidence": float(confidence.mean()),
        "active_mean_confidence": float(confidence[active].mean())
        if active.any()
        else 0.0,
    }
    tensors = {
        "candidate": candidate,
        "support": support,
        "active": active,
        "residual": residual,
        "prediction": prediction,
    }
    return metrics, tensors


def pair_action_oracle(
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    baseline_logits: torch.Tensor,
) -> dict[str, Any]:
    labels = split["labels"]
    baseline_prediction = baseline_logits.argmax(dim=-1)
    base_gold = baseline_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    base_other = baseline_logits.clone()
    base_other.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
    base_margin = base_gold - base_other.max(dim=-1).values
    best_gain = torch.zeros_like(base_margin)
    best_prediction = baseline_prediction.clone()
    batch = labels.shape[0]
    for query_index in range(v1.QUERIES):
        for pair in PAIR_ACTIONS:
            support = torch.tensor(pair, device=labels.device).view(1, 1, -1)
            support = support.expand(batch, v1.QUERIES, -1).clone()
            active = torch.zeros(batch, v1.QUERIES, dtype=torch.bool, device=labels.device)
            active[:, query_index] = True
            for sign in (-1.0, 1.0):
                scores, _ = apply_pair_actions(split["scores"], support, active, sign=sign)
                logits, _ = v1.baseline_logits(
                    scores, split["key"], split["query"], frames
                )
                gold = logits[:, query_index].gather(
                    -1, labels[:, query_index, None]
                ).squeeze(-1)
                other = logits[:, query_index].clone()
                other.scatter_(-1, labels[:, query_index, None], -torch.inf)
                gain = gold - other.max(dim=-1).values - base_margin[:, query_index]
                improve = gain > best_gain[:, query_index]
                best_gain[improve, query_index] = gain[improve]
                prediction = logits[:, query_index].argmax(dim=-1)
                best_prediction[improve, query_index] = prediction[improve]
    wrong = baseline_prediction.ne(labels)
    corrected = wrong & best_prediction.eq(labels)
    return {
        "action_count_per_query": len(PAIR_ACTIONS) * 2,
        "baseline_wrong_queries": int(wrong.sum()),
        "oracle_corrected_queries": int(corrected.sum()),
        "oracle_harmed_correct_queries": 0,
        "oracle_mean_best_margin_gain": float(best_gain.mean()),
        "oracle_hard_corrected_queries": int((corrected & split["hard"]).sum()),
        "oracle_easy_corrected_queries": int((corrected & ~split["hard"]).sum()),
    }


def residual_invariants(residual: torch.Tensor) -> dict[str, Any]:
    return {
        "finite": bool(torch.isfinite(residual).all()),
        "zero_sum_error": float(residual.sum(dim=-1).abs().max()),
        "max_abs_residual": float(residual.abs().max()),
        "bounded": bool(residual.abs().max() <= v1.MAX_DELTA + 1e-6),
        "query_local": True,
        "support_size": v1.EVIDENCE_KEYS,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    device = v1.choose_device(args.device or str(config["device"]))
    seed = int(config["seed"])
    dataset = config["dataset"]
    streams = {
        "train": make_split(seed, int(dataset["train_size"]), device),
        "calibration": make_split(seed + 1000, int(dataset["calibration_size"]), device),
        "valid": make_split(seed + 10000, int(dataset["valid_size"]), device),
        "test": make_split(seed + 20000, int(dataset["test_size"]), device),
    }
    frames = v1.relation_frames(device)
    baseline = {}
    oracle = {}
    selectors = {}
    invariants = {}
    exact_overlaps = {}
    split_names = tuple(streams)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            left_query = streams[left]["query"].reshape(streams[left]["query"].shape[0], -1)
            right_query = streams[right]["query"].reshape(streams[right]["query"].shape[0], -1)
            exact_overlaps[f"{left}:{right}"] = int(
                ((left_query[:, None] == right_query[None, :]).all(dim=-1)).sum()
            )
    for name, split in streams.items():
        logits, _ = v1.baseline_logits(
            split["scores"], split["key"], split["query"], frames
        )
        replay, _ = v1.baseline_logits(
            split["scores"], split["key"], split["query"], frames
        )
        labels = split["labels"]
        witness = error_witness(split["scores"])
        baseline_prediction = logits.argmax(dim=-1)
        baseline[name] = {
            "accuracy": float(baseline_prediction.eq(labels).float().mean()),
            "replay_error": float((replay - logits).abs().max()),
            "queries": int(labels.numel()),
            "hard_rate": float(split["hard"].float().mean()),
            "witness_rate": float(witness.float().mean()),
            "witness_hard_agreement": float(witness.eq(split["hard"]).float().mean()),
            "witness_wrong_precision": float(
                (witness & baseline_prediction.ne(labels)).sum()
                / witness.sum().clamp_min(1)
            ),
        }
        oracle[name] = pair_action_oracle(split, frames, logits)
        selectors[name] = {}
        for selector in SELECTORS:
            metrics, tensors = evaluate_selector(selector, split, frames, logits)
            selectors[name][selector] = metrics
            if selector == "candidate_relative_consensus":
                invariants[name] = residual_invariants(tensors["residual"])
    gate_config = config["gate"]
    valid = selectors["valid"]
    test = selectors["test"]
    primary = "candidate_relative_consensus"
    shuffled = "query_shuffled_consensus"
    classical = "classical_separable_consensus"
    gate_conditions = {
        "exact_disjoint_streams": all(count == 0 for count in exact_overlaps.values()),
        "baseline_replay": all(item["replay_error"] == 0.0 for item in baseline.values()),
        "baseline_non_saturated": float(gate_config["baseline_accuracy_min"])
        <= baseline["valid"]["accuracy"]
        <= float(gate_config["baseline_accuracy_max"])
        and float(gate_config["baseline_accuracy_min"])
        <= baseline["test"]["accuracy"]
        <= float(gate_config["baseline_accuracy_max"]),
        "oracle_headroom": oracle["valid"]["oracle_corrected_queries"] > 0
        and oracle["test"]["oracle_corrected_queries"] > 0,
        "witness_identifies_need": baseline["valid"]["witness_wrong_precision"]
        >= float(gate_config["minimum_witness_wrong_precision"])
        and baseline["test"]["witness_wrong_precision"]
        >= float(gate_config["minimum_witness_wrong_precision"]),
        "label_free_heldout_gain": valid[primary]["accuracy_delta"]
        >= float(gate_config["minimum_accuracy_delta"])
        and test[primary]["accuracy_delta"]
        >= float(gate_config["minimum_accuracy_delta"]),
        "label_free_no_harm": valid[primary]["harm_rate"]
        <= float(gate_config["maximum_harm_rate"])
        and test[primary]["harm_rate"]
        <= float(gate_config["maximum_harm_rate"]),
        "beats_query_shuffled": valid[primary]["accuracy_delta"]
        - valid[shuffled]["accuracy_delta"]
        >= float(gate_config["minimum_control_advantage"])
        and test[primary]["accuracy_delta"]
        - test[shuffled]["accuracy_delta"]
        >= float(gate_config["minimum_control_advantage"]),
        "beats_classical_separable": valid[primary]["accuracy_delta"]
        - valid[classical]["accuracy_delta"]
        >= float(gate_config["minimum_control_advantage"])
        and test[primary]["accuracy_delta"]
        - test[classical]["accuracy_delta"]
        >= float(gate_config["minimum_control_advantage"]),
        "residual_invariants": all(
            item["finite"] and item["bounded"] and item["zero_sum_error"] <= 1e-5
            for item in invariants.values()
        ),
    }
    return {
        "schema_version": "q-attention.q-consensus-error-witness-prescreen.v1",
        "status": "complete",
        "experiment_name": "q_consensus_error_witness_prescreen_toy",
        "dataset_identity": config["dataset"]["identity"],
        "seed": seed,
        "split_policy": "exact disjoint streams seed, seed+1000, seed+10000, seed+20000",
        "exact_stream_overlaps": exact_overlaps,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "baseline": baseline,
        "oracle": oracle,
        "selectors": selectors,
        "residual_invariants": invariants,
        "gate": {
            **gate_conditions,
            "status": "pass" if all(gate_conditions.values()) else "fail",
            "next_quantum_estimator_design_authorized": bool(
                all(gate_conditions.values())
            ),
            "quantum_estimator_run": False,
            "multi_seed_run": False,
            "real_data_run": False,
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
    summary_path = output / "run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "gate": payload["gate"],
                "baseline": {
                    name: payload["baseline"][name] for name in ("valid", "test")
                },
                "oracle": {
                    name: payload["oracle"][name] for name in ("valid", "test")
                },
                "candidate_relative_consensus": {
                    name: payload["selectors"][name][
                        "candidate_relative_consensus"
                    ]
                    for name in ("valid", "test")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
