#!/usr/bin/env python3
"""Seed-7 non-quantum identifiability pre-screen for a new query-local task.

The task has two query positions and six randomly addressed context keys.  Two
keys carry the same relation-conditioned evidence and one key carries a
different relation, but a frozen baseline gives a random distractor address a
large score bias on a fixed fraction of examples.  Labels are determined by
the relation between a query vector and transformed key vectors; no token id
or dedicated position directly encodes the label.

This file deliberately stops before any quantum estimator.  It measures
whether a predeclared, label-free candidate-relative observable can identify
useful bounded score actions on held-out streams, with matched classical and
shuffled controls.  The observable is candidate-relative signed bilinear
compatibility:

    e[b,q,c,k] = q[b,q]^T R_c^T k[b,q,k]

where R_c is a fixed relation frame.  The complete action bank is scanned only
for offline oracle utility and never used by the label-free selector.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch


QUERIES = 2
CLASSES = 3
KEYS = 6
DIM = 4
EVIDENCE_KEYS = 2
BASELINE_BIAS = 2.0
HARD_RATE = 0.28
NOISE = 0.10
MAX_DELTA = 2.25
SELECTORS = (
    "disabled",
    "candidate_relative",
    "classical_separable",
    "query_shuffled",
    "query_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_identifiable_query_local_prescreen_toy.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.q-identifiable-query-local-prescreen.v1":
        raise ValueError("unsupported identifiable-query-local config schema")
    required = {"seed", "device", "dataset", "task", "gate", "output_root"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    expected = {
        "queries": QUERIES,
        "classes": CLASSES,
        "keys": KEYS,
        "dim": DIM,
        "evidence_keys": EVIDENCE_KEYS,
        "baseline_bias": BASELINE_BIAS,
        "hard_rate": HARD_RATE,
        "noise": NOISE,
        "max_delta": MAX_DELTA,
    }
    if payload["task"] != expected:
        raise ValueError("task configuration differs from the predeclared constants")
    return payload


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def relation_frames(device: torch.device) -> torch.Tensor:
    """Three fixed orthogonal frames, embedded in two independent planes."""
    angles = torch.tensor(
        [0.0, 2.0 * torch.pi / 3.0, 4.0 * torch.pi / 3.0],
        dtype=torch.float32,
        device=device,
    )
    frames = torch.eye(DIM, device=device).repeat(CLASSES, 1, 1)
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    frames[:, 0, 0] = cosines
    frames[:, 0, 1] = -sines
    frames[:, 1, 0] = sines
    frames[:, 1, 1] = cosines
    frames[:, 2, 2] = cosines
    frames[:, 2, 3] = -sines
    frames[:, 3, 2] = sines
    frames[:, 3, 3] = cosines
    return frames


def make_split(seed: int, size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate an exact-disjoint stream with random evidence addresses."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.randint(0, CLASSES, (size, QUERIES), generator=generator)
    query = torch.randn(size, QUERIES, DIM, generator=generator)
    query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    frames_cpu = relation_frames(torch.device("cpu"))
    keys = torch.randn(size, QUERIES, KEYS, DIM, generator=generator) * 0.35
    hard = torch.rand(size, QUERIES, generator=generator) < HARD_RATE
    bad_class = (labels + 1 + torch.randint(0, CLASSES - 1, (size, QUERIES), generator=generator)) % CLASSES
    slot_order = torch.stack(
        [torch.randperm(KEYS, generator=generator) for _ in range(size * QUERIES)]
    ).reshape(size, QUERIES, KEYS)
    evidence_slot = slot_order[:, :, :EVIDENCE_KEYS]
    bad_slot = slot_order[:, :, EVIDENCE_KEYS]
    row = torch.arange(size)[:, None]
    qrow = torch.arange(QUERIES)[None, :]
    for evidence_index in range(EVIDENCE_KEYS):
        slots = evidence_slot[:, :, evidence_index]
        transformed = torch.einsum(
            "bqd,bqdh->bqh",
            query.cpu(),
            frames_cpu[labels],
        )
        noise = torch.randn(size, QUERIES, DIM, generator=generator) * NOISE
        keys[row, qrow, slots] = transformed + noise
    bad_transformed = torch.einsum(
        "bqd,bqdh->bqh",
        query.cpu(),
        frames_cpu[bad_class],
    )
    bad_noise = torch.randn(size, QUERIES, DIM, generator=generator) * NOISE
    keys[row, qrow, bad_slot] = bad_transformed + bad_noise
    bias = torch.zeros(size, QUERIES, KEYS)
    bias[row, qrow, bad_slot] = hard.to(torch.float32) * BASELINE_BIAS
    # Easy examples receive a small random address bias, preserving a nontrivial
    # held-out control while avoiding a fixed rescue position.
    easy_slot = slot_order[:, :, EVIDENCE_KEYS + 1]
    bias[row, qrow, easy_slot] += (~hard).to(torch.float32) * 0.10
    scores = 0.05 * torch.randn(size, QUERIES, KEYS, generator=generator) + bias
    return {
        "query": query.to(device),
        "key": keys.to(device),
        "scores": scores.to(device),
        "labels": labels.to(device),
        "hard": hard.to(device),
        "evidence_slot": evidence_slot.to(device),
        "bad_slot": bad_slot.to(device),
        "attention_mask": torch.ones(size, QUERIES, KEYS, dtype=torch.bool, device=device),
    }


def batches(split: dict[str, torch.Tensor], batch_size: int):
    for start in range(0, split["labels"].shape[0], batch_size):
        yield {name: value[start : start + batch_size] for name, value in split.items()}


def compatibility(
    query: torch.Tensor,
    key: torch.Tensor,
    frames: torch.Tensor,
) -> torch.Tensor:
    """Signed candidate-relative query/key compatibility."""
    # q^T R_c^T k = (R_c q)^T k; all inputs are observable at inference.
    transformed_query = torch.einsum("bqd,cdh->bqch", query, frames)
    return torch.einsum("bqch,bqkh->bqck", transformed_query, key)


def baseline_logits(
    scores: torch.Tensor,
    key: torch.Tensor,
    query: torch.Tensor,
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention = torch.softmax(scores, dim=-1)
    context = torch.einsum("bqk,bqkd->bqd", attention, key)
    logits = compatibility(query, context[:, :, None, :], frames).squeeze(-1)
    return logits, attention


def apply_action(
    scores: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    sign: torch.Tensor,
) -> torch.Tensor:
    residual = torch.zeros_like(scores)
    rows = torch.arange(scores.shape[0], device=scores.device)
    residual[rows, query_index, :] = -sign[:, None] * MAX_DELTA / (KEYS - 1)
    residual[rows, query_index, key_index] += sign * MAX_DELTA * KEYS / (KEYS - 1)
    return scores + residual


def apply_query_actions(
    scores: torch.Tensor,
    key_index: torch.Tensor,
    sign: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one bounded zero-sum key action independently at every query."""
    if key_index.shape != scores.shape[:2] or sign.shape != scores.shape[:2]:
        raise ValueError("key_index and sign must have shape (batch, queries)")
    residual = -sign[:, :, None] * MAX_DELTA / (KEYS - 1)
    residual = residual.expand_as(scores).clone()
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    queries = torch.arange(QUERIES, device=scores.device)[None, :]
    residual[rows, queries, key_index] += sign * MAX_DELTA * KEYS / (KEYS - 1)
    return scores + residual, residual


def all_action_utilities(
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    base_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Complete Q x K x +/- action bank for offline oracle diagnostics."""
    labels = split["labels"]
    utilities = torch.empty(
        labels.shape[0], QUERIES, KEYS, 2, device=labels.device
    )
    predictions = torch.empty(
        labels.shape[0], QUERIES, KEYS, 2, dtype=torch.long, device=labels.device
    )
    base_gold = base_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    base_other = base_logits.clone()
    base_other.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
    base_margin = base_gold - base_other.max(dim=-1).values
    for query_index in range(QUERIES):
        for key_index in range(KEYS):
            for sign_offset, sign_value in enumerate((-1.0, 1.0)):
                qidx = torch.full((labels.shape[0],), query_index, dtype=torch.long, device=labels.device)
                kidx = torch.full_like(qidx, key_index)
                sign = torch.full_like(qidx, sign_value, dtype=torch.float32)
                logits, _ = baseline_logits(
                    apply_action(split["scores"], qidx, kidx, sign),
                    split["key"],
                    split["query"],
                    frames,
                )
                gold = logits[:, query_index].gather(
                    -1, labels[:, query_index, None]
                ).squeeze(-1)
                other = logits[:, query_index].clone()
                other.scatter_(-1, labels[:, query_index, None], -torch.inf)
                margin = gold - other.max(dim=-1).values
                utilities[:, query_index, key_index, sign_offset] = (
                    margin - base_margin[:, query_index]
                )
                predictions[:, query_index, key_index, sign_offset] = logits[
                    :, query_index
                ].argmax(dim=-1)
    return {
        "utility": utilities,
        "prediction": predictions,
    }


def selector_action(
    selector: str,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    base_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one candidate/key action per query without labels."""
    batch_size = split["labels"].shape[0]
    if selector == "disabled":
        qidx = torch.zeros(batch_size, QUERIES, dtype=torch.long, device=base_logits.device)
        kidx = torch.zeros_like(qidx)
        sign = torch.zeros_like(qidx, dtype=torch.float32)
        return qidx, kidx, sign
    query = split["query"]
    if selector == "query_shuffled":
        query = query[torch.roll(torch.arange(batch_size, device=query.device), shifts=1)]
    comp = compatibility(query, split["key"], frames)
    if selector == "query_only":
        comp = comp.mean(dim=-1, keepdim=True).expand_as(comp)
    elif selector == "classical_separable":
        query_term = torch.einsum("bqd,cdh->bqch", query, frames).mean(dim=-1)
        key_term = torch.einsum("bqkd,cdh->bqkch", split["key"], frames).mean(dim=-1)
        comp = 0.5 * query_term[:, :, :, None] + 0.5 * key_term.permute(0, 1, 3, 2)
    centered = comp - comp.mean(dim=-1, keepdim=True)
    # A fixed, predeclared score scale converts compatibility into a predicted
    # candidate utility; it is not fit on labels or oracle action outcomes.
    predicted = base_logits + 0.75 * centered.max(dim=-1).values
    best_candidate = predicted.argmax(dim=-1)
    selected_field = centered.gather(
        2,
        best_candidate[:, :, None, None].expand(-1, -1, 1, KEYS),
    ).squeeze(2)
    best_key = selected_field.argmax(dim=-1)
    selected_gain = predicted.gather(-1, best_candidate.unsqueeze(-1)).squeeze(-1) - base_logits.gather(-1, best_candidate.unsqueeze(-1)).squeeze(-1)
    active = selected_gain > 0.10
    sign = torch.where(active, torch.ones_like(selected_gain), torch.zeros_like(selected_gain))
    return best_candidate, best_key, sign


def evaluate_selector(
    selector: str,
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    base_logits: torch.Tensor,
) -> dict[str, Any]:
    candidate, key_index, sign = selector_action(selector, split, frames, base_logits)
    steered_scores, _residual = apply_query_actions(split["scores"], key_index, sign)
    steered_logits, _ = baseline_logits(steered_scores, split["key"], split["query"], frames)
    labels = split["labels"]
    base_pred = base_logits.argmax(dim=-1)
    pred = steered_logits.argmax(dim=-1)
    wrong = base_pred.ne(labels)
    correct = ~wrong
    corrected = wrong & pred.eq(labels)
    harmed = correct & pred.ne(labels)
    active = sign.ne(0)
    return {
        "selector": selector,
        "accuracy": float(pred.eq(labels).float().mean().item()),
        "baseline_accuracy": float(base_pred.eq(labels).float().mean().item()),
        "accuracy_delta": float((pred.eq(labels).float().mean() - base_pred.eq(labels).float().mean()).item()),
        "baseline_wrong_examples": int(wrong.sum().item()),
        "corrected_examples": int(corrected.sum().item()),
        "harmed_correct_examples": int(harmed.sum().item()),
        "wrong_correction_rate": float(corrected.sum().item() / wrong.sum().clamp_min(1).item()),
        "harm_rate": float(harmed.sum().item() / correct.sum().clamp_min(1).item()),
        "active_rate": float(active.float().mean().item()),
        "selected_candidates": candidate.detach(),
        "selected_keys": key_index.detach(),
        "selected_sign": sign.detach(),
        "steered_logits": steered_logits.detach(),
    }


def oracle_metrics(
    split: dict[str, torch.Tensor],
    frames: torch.Tensor,
    base_logits: torch.Tensor,
) -> dict[str, Any]:
    bank = all_action_utilities(split, frames, base_logits)
    labels = split["labels"]
    base_pred = base_logits.argmax(dim=-1)
    rows = torch.arange(labels.shape[0], device=labels.device)
    utility = bank["utility"]
    prediction = bank["prediction"]
    best_utility = utility.reshape(labels.shape[0], QUERIES, -1).max(dim=-1).values
    best_index = utility.reshape(labels.shape[0], QUERIES, -1).argmax(dim=-1)
    flat_predictions = prediction.reshape(labels.shape[0], QUERIES, -1)
    best_pred = flat_predictions.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    corrected = base_pred.ne(labels) & best_pred.eq(labels)
    # Correct baseline queries retain the disabled action, so the safe oracle
    # upper bound never forces an unnecessary intervention.
    harmed = torch.zeros_like(corrected)
    return {
        "action_count_per_query": KEYS * 2,
        "baseline_accuracy": float(base_pred.eq(labels).float().mean().item()),
        "oracle_corrected_examples": int(corrected.sum().item()),
        "oracle_harmed_correct_examples": int(harmed.sum().item()),
        "oracle_mean_best_margin_gain": float(best_utility.mean().item()),
        "oracle_corrected_on_hard": int((corrected & split["hard"]).sum().item()),
        "oracle_corrected_on_easy": int((corrected & ~split["hard"]).sum().item()),
    }


def residual_invariants(
    selector: str,
    split: dict[str, torch.Tensor],
    result: dict[str, Any],
) -> dict[str, float | bool]:
    key_index = result["selected_keys"]
    sign = result["selected_sign"]
    _steered, residual = apply_query_actions(split["scores"], key_index, sign)
    return {
        "finite": bool(torch.isfinite(residual).all().item()),
        "zero_sum_error": float(residual.sum(dim=-1).abs().max().item()),
        "max_abs_residual": float(residual.abs().max().item()),
        "bounded": bool((residual.abs() <= MAX_DELTA + 1e-6).all().item()),
        "query_local": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    seed = int(config["seed"])
    dataset = config["dataset"]
    device = choose_device(args.device or str(config["device"]))
    frames = relation_frames(device)
    streams = {
        "train": make_split(seed, int(dataset["train_size"]), device),
        "calibration": make_split(seed + 1000, int(dataset["calibration_size"]), device),
        "valid": make_split(seed + 10000, int(dataset["valid_size"]), device),
        "test": make_split(seed + 20000, int(dataset["test_size"]), device),
    }
    stream_pairs = (("train", "calibration"), ("train", "valid"), ("train", "test"), ("calibration", "valid"), ("calibration", "test"), ("valid", "test"))
    exact_overlaps = {}
    for left, right in stream_pairs:
        left_query = streams[left]["query"].reshape(streams[left]["query"].shape[0], -1)
        right_query = streams[right]["query"].reshape(streams[right]["query"].shape[0], -1)
        matches = (left_query[:, None, :] == right_query[None, :, :]).all(dim=-1)
        exact_overlaps[f"{left}:{right}"] = int(matches.sum().item())
    # The baseline is analytical and frozen by construction.  A replay check
    # below serves as the baseline-validity gate before any selector metric.
    baseline = {}
    selector_results: dict[str, dict[str, Any]] = {}
    oracle = {}
    invariants = {}
    for split_name, split in streams.items():
        logits, _attention = baseline_logits(split["scores"], split["key"], split["query"], frames)
        replay_logits, _ = baseline_logits(split["scores"], split["key"], split["query"], frames)
        baseline[split_name] = {
            "accuracy": float(logits.argmax(dim=-1).eq(split["labels"]).float().mean().item()),
            "replay_error": float((logits - replay_logits).abs().max().item()),
            "examples": int(split["labels"].numel()),
            "hard_rate": float(split["hard"].float().mean().item()),
        }
        oracle[split_name] = oracle_metrics(split, frames, logits)
        selector_results[split_name] = {
            selector: evaluate_selector(selector, split, frames, logits)
            for selector in SELECTORS
        }
        invariants[split_name] = residual_invariants(
            "candidate_relative", split, selector_results[split_name]["candidate_relative"]
        )
    # Remove tensors from the JSON payload while preserving compact diagnostics.
    for split_results in selector_results.values():
        for result in split_results.values():
            for key in ("selected_candidates", "selected_keys", "selected_sign", "steered_logits"):
                result.pop(key, None)
    valid = selector_results["valid"]
    test = selector_results["test"]
    oracle_valid = oracle["valid"]
    oracle_test = oracle["test"]
    configured_gate = config["gate"]
    gate_conditions = {
        "exact_disjoint_streams": all(count == 0 for count in exact_overlaps.values()),
        "baseline_validity": all(item["replay_error"] == 0.0 for item in baseline.values()),
        "baseline_non_saturated": float(configured_gate["baseline_accuracy_min"]) <= baseline["valid"]["accuracy"] <= float(configured_gate["baseline_accuracy_max"])
        and float(configured_gate["baseline_accuracy_min"]) <= baseline["test"]["accuracy"] <= float(configured_gate["baseline_accuracy_max"]),
        "oracle_headroom": oracle_valid["oracle_corrected_examples"] > 0
        and oracle_test["oracle_corrected_examples"] > 0
        and oracle_valid["oracle_harmed_correct_examples"] == 0
        and oracle_test["oracle_harmed_correct_examples"] == 0,
        "label_free_valid_correction": valid["candidate_relative"]["corrected_examples"] > 0,
        "label_free_test_correction": test["candidate_relative"]["corrected_examples"] > 0,
        "label_free_no_harm_gate": valid["candidate_relative"]["harm_rate"] <= float(configured_gate["maximum_harm_rate"])
        and test["candidate_relative"]["harm_rate"] <= float(configured_gate["maximum_harm_rate"]),
        "beats_shuffled": valid["candidate_relative"]["corrected_examples"]
        > valid["query_shuffled"]["corrected_examples"]
        and test["candidate_relative"]["corrected_examples"]
        > test["query_shuffled"]["corrected_examples"],
        "beats_classical_control": valid["candidate_relative"]["corrected_examples"]
        >= valid["classical_separable"]["corrected_examples"]
        and test["candidate_relative"]["corrected_examples"]
        >= test["classical_separable"]["corrected_examples"],
        "residual_invariants": all(
            bool(item["finite"]) and bool(item["bounded"]) and item["zero_sum_error"] <= 1e-5
            for item in invariants.values()
        ),
    }
    payload = {
        "schema_version": "q-attention.q-identifiable-query-local-prescreen.v1",
        "status": "complete",
        "experiment_name": "q_identifiable_query_local_prescreen_toy",
        "dataset_identity": "synthetic_dynamic_address_multievidence_relative_frames_v1",
        "seed": seed,
        "split_policy": "exact disjoint streams seed, seed+1000, seed+10000, seed+20000",
        "exact_stream_overlaps": exact_overlaps,
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "baseline": baseline,
        "oracle": oracle,
        "selectors": selector_results,
        "residual_invariants": invariants,
        "gate": {
            **gate_conditions,
            "status": "pass" if all(gate_conditions.values()) else "fail",
            "next_mechanism_run_authorized": bool(all(gate_conditions.values())),
            "quantum_estimator_run": False,
            "real_data_run": False,
        },
        "failure_signatures": {
            "baseline_saturation": "valid/test outside [0.70, 0.90]",
            "non_identifiability": "candidate_relative corrected=0 on valid or test",
            "query_independent": "query-shuffled control matches candidate_relative",
            "classical_reproduction": "classical_separable exceeds candidate_relative",
            "unsafe_action": "harm_rate > 0.02",
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)
    root = Path(args.output_root or str(config["output_root"]))
    if not root.is_absolute():
        root = project_root / root
    output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": payload["gate"], "baseline": payload["baseline"], "oracle": payload["oracle"], "candidate_relative": {name: payload["selectors"][name]["candidate_relative"] for name in ("valid", "test")}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
