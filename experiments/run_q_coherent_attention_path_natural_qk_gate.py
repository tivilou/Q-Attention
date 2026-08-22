#!/usr/bin/env python3
"""Fast-kill gate using scores generated directly by a fixed Q/K feature encoder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins.q_coherent_attention_path import (  # noqa: E402
    CoherentAttentionPathConfig,
    build_coherent_attention_path_kernel,
)


NODES = 7
HEADS = 1
HEAD_DIM = 5
ANCHOR = 0
SUBJECT = 1
OBJECT = 2
TARGETS = (3, 4)
SELECTOR_TYPES = {
    "q_wap_signed": "quantum_signed",
    "q_wap_unsigned": "quantum_unsigned",
    "classical_wap_diffusion": "classical_diffusion",
}

# A fixed role encoder creates a signed two-path cycle from QK scores. Candidate
# nodes 3 and 4 have equal-magnitude paths with opposite orientation, while
# the external value basis makes the selected node the classification target.
Q_BASE = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
K_BASE = torch.tensor(
    [
        [0.0, -1.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0, -1.0],
        [-1.0, 0.0, 0.0, -1.0, 1.0],
        [0.0, -1.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_coherent_attention_path_natural_qk_gate.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.coherent-attention-path-natural-qk.v1":
        raise ValueError("unsupported natural QK config")
    if int(payload.get("seed", -1)) != 7:
        raise ValueError("natural QK gate requires fixed seed 7")
    dataset = payload["dataset"]
    if int(dataset["nodes"]) != NODES or int(dataset["head_dim"]) != HEAD_DIM:
        raise ValueError("natural QK gate requires frozen 7-node, 5D features")
    if abs(float(payload["mechanism"]["walk_time"]) - math.pi / 4) > 1e-12:
        raise ValueError("walk time is frozen at pi/4")
    if tuple(payload.get("selectors", ())) != (
        "disabled",
        "q_wap_signed",
        "q_wap_unsigned",
        "classical_wap_diffusion",
    ):
        raise ValueError("selectors must match the frozen allowlist")
    if dataset.get("trainable_encoder") is not False:
        raise ValueError("natural QK encoder must be frozen")
    return payload


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def make_split(seed: int, size: int, device: torch.device, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    if size <= 0 or size % 2:
        raise ValueError("natural QK size must be positive and even")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(size, dtype=torch.long) % 2
    labels = labels[torch.randperm(size, generator=generator)]
    low = float(config["dataset"]["scale_low"])
    high = float(config["dataset"]["scale_high"])
    scales = low + (high - low) * torch.rand(size, 4, generator=generator)
    queries = []
    keys = []
    values = []
    scores = []
    target_keys = []
    distractor_keys = []
    for row_label, row_scales in zip(labels, scales):
        q_scale = torch.ones(NODES)
        k_scale = torch.ones(NODES)
        q_scale[0] = row_scales[0]
        k_scale[0] = row_scales[0]
        q_scale[1] = row_scales[1]
        # Fixed key-side calibration makes raw QK scores genuinely asymmetric;
        # candidate 3/4 scales remain paired so the control tie is unchanged.
        k_scale[1] = row_scales[1] * 1.07
        q_scale[2] = row_scales[2]
        k_scale[2] = row_scales[2] * 0.93
        q_scale[3] = q_scale[4] = row_scales[3]
        k_scale[3] = k_scale[4] = row_scales[3]
        q = Q_BASE * q_scale[:, None] * HEAD_DIM**0.25
        k = K_BASE * k_scale[:, None] * HEAD_DIM**0.25
        if int(row_label) == 1:
            swap = torch.tensor([0, 1, 2, 4, 3, 5, 6])
            q = q[swap]
            k = k[swap]
        score = q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM)
        value = torch.zeros(NODES, 2)
        value[TARGETS[0], 0] = 1.0
        value[TARGETS[1], 1] = 1.0
        queries.append(q.unsqueeze(0))
        keys.append(k.unsqueeze(0))
        values.append(value.unsqueeze(0))
        scores.append(score.unsqueeze(0))
        target_keys.append(TARGETS[int(row_label)])
        distractor_keys.append(TARGETS[1 - int(row_label)])

    attention_mask = torch.ones(size, NODES, dtype=torch.bool)
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, SUBJECT] = True
    object_mask[:, OBJECT] = True
    return {
        "query": torch.stack(queries).to(device),
        "key": torch.stack(keys).to(device),
        "value": torch.stack(values).to(device),
        "scores": torch.stack(scores).to(device),
        "attention_mask": attention_mask.to(device),
        "subject_mask": subject_mask.to(device),
        "object_mask": object_mask.to(device),
        "labels": labels.to(device),
        "target_key": torch.tensor(target_keys, dtype=torch.long, device=device),
        "distractor_key": torch.tensor(
            distractor_keys, dtype=torch.long, device=device
        ),
        "scales": scales.to(device),
    }


def build_kernel(selector: str, config: dict[str, Any]):
    if selector == "disabled":
        return None
    mechanism = config["mechanism"]
    return build_coherent_attention_path_kernel(
        SELECTOR_TYPES[selector],
        CoherentAttentionPathConfig(
            num_layers=1,
            num_heads=HEADS,
            max_transport=float(mechanism["max_transport"]),
            initial_transport=float(mechanism["initial_transport"]),
            walk_time=float(mechanism["walk_time"]),
        ),
    )


def forward(kernel, split: dict[str, torch.Tensor]):
    if kernel is None:
        residual = torch.zeros_like(split["scores"])
    else:
        residual = kernel(
            split["query"],
            split["key"],
            split["value"],
            scores=split["scores"],
            layer_index=0,
            attention_mask=split["attention_mask"],
            subject_mask=split["subject_mask"],
            object_mask=split["object_mask"],
        )
    attention = torch.softmax(split["scores"] + residual, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention, split["value"])
    row = torch.arange(split["labels"].shape[0], device=split["labels"].device)
    logits = output[row, 0, ANCHOR, :2]
    return residual, attention, logits


def predictions(logits: torch.Tensor, tolerance: float) -> torch.Tensor:
    margin = logits[:, 1] - logits[:, 0]
    return torch.where(
        margin > tolerance,
        torch.ones_like(margin, dtype=torch.long),
        torch.zeros_like(margin, dtype=torch.long),
    )


def residual_invariants(residual: torch.Tensor, split: dict[str, torch.Tensor], maximum: float) -> dict[str, Any]:
    context = split["attention_mask"] & ~(
        split["subject_mask"] | split["object_mask"]
    )
    mask = context[:, None, None, :].to(residual.dtype)
    outside = residual * (1.0 - mask)
    context_sum = (residual * mask).sum(dim=-1)
    checks = {
        "finite": bool(torch.isfinite(residual).all()),
        "context_only": bool(outside.abs().max() <= 1e-7),
        "zero_sum_context": bool(context_sum.abs().max() <= 1e-6),
        "bounded": bool(residual.abs().max() <= maximum + 1e-6),
    }
    return {
        **checks,
        "status": "pass" if all(checks.values()) else "fail",
        "max_abs": float(residual.abs().max()),
        "max_context_sum_error": float(context_sum.abs().max()),
    }


def evaluate(selector: str, split: dict[str, torch.Tensor], baseline: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    kernel = build_kernel(selector, config)
    if kernel is not None:
        kernel = kernel.to(split["scores"].device)
    with torch.no_grad():
        residual, attention, logits = forward(kernel, split)
        replay = forward(kernel, split)
    prediction = predictions(logits, float(config["mechanism"]["tie_tolerance"]))
    labels = split["labels"]
    row = torch.arange(labels.shape[0], device=labels.device)
    anchor_attention = attention[row, 0, ANCHOR, :]
    target_attention = anchor_attention[row, split["target_key"]]
    distractor_attention = anchor_attention[row, split["distractor_key"]]
    corrected = (~baseline.eq(labels)) & prediction.eq(labels)
    harmed = baseline.eq(labels) & (~prediction.eq(labels))
    return {
        "selector": selector,
        "accuracy": float(prediction.eq(labels).float().mean()),
        "corrected_examples": int(corrected.sum()),
        "harmed_correct_examples": int(harmed.sum()),
        "target_attention": float(target_attention.mean()),
        "distractor_attention": float(distractor_attention.mean()),
        "target_minus_distractor_attention": float(
            (target_attention - distractor_attention).mean()
        ),
        "intervention_parameters": 0
        if kernel is None
        else sum(parameter.numel() for parameter in kernel.parameters()),
        "deterministic_replay": bool(
            torch.equal(residual, replay[0])
            and torch.equal(attention, replay[1])
            and torch.equal(logits, replay[2])
        ),
        "residual_invariants": residual_invariants(
            residual, split, float(config["mechanism"]["max_transport"])
        ),
        "metadata": {"id": "disabled", "version": "1.0.0"}
        if kernel is None
        else kernel.metadata(),
    }


def geometry(split: dict[str, torch.Tensor]) -> dict[str, Any]:
    flat = split["scores"].detach().cpu().reshape(split["scores"].shape[0], -1)
    ranks = torch.linalg.matrix_rank(split["scores"][:, 0])
    raw_gap = split["scores"][:, 0, ANCHOR, TARGETS[0]] - split["scores"][:, 0, ANCHOR, TARGETS[1]]
    return {
        "examples": int(split["labels"].shape[0]),
        "positive_labels": int(split["labels"].sum()),
        "unique_scores": int(torch.unique(flat, dim=0).shape[0]),
        "minimum_score_rank": int(ranks.min()),
        "maximum_raw_anchor_candidate_gap": float(raw_gap.abs().max()),
        "maximum_qk_reconstruction_error": float(
            (torch.einsum("bhqd,bhkd->bhqk", split["query"], split["key"]) / math.sqrt(HEAD_DIM) - split["scores"]).abs().max()
        ),
        "maximum_raw_asymmetry": float(
            (split["scores"] - split["scores"].transpose(-1, -2)).abs().max()
        ),
    }


def promotion_gate(results: list[dict[str, Any]], diagnostic: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    by = {row["selector"]: row for row in results}
    disabled = by["disabled"]
    quantum = by["q_wap_signed"]
    unsigned = by["q_wap_unsigned"]
    classical = by["classical_wap_diffusion"]
    gate = config["gate"]
    conditions = {
        "label_balance": diagnostic["positive_labels"] * 2 == diagnostic["examples"],
        "score_diversity": diagnostic["unique_scores"] >= int(gate["minimum_unique_scores"]),
        "score_rank": diagnostic["minimum_score_rank"] >= int(gate["minimum_score_rank"]),
        "qk_reconstruction": diagnostic["maximum_qk_reconstruction_error"] <= float(gate["maximum_reconstruction_error"]),
        "nontrivial_raw_scores": diagnostic["maximum_raw_asymmetry"] > 0.05,
        "baseline_is_balanced_chance": abs(disabled["accuracy"] - 0.5) <= 1e-8,
        "quantum_accuracy": quantum["accuracy"] >= float(gate["minimum_quantum_accuracy"]),
        "unsigned_control": unsigned["accuracy"] <= float(gate["maximum_control_accuracy"]),
        "classical_control": classical["accuracy"] <= float(gate["maximum_control_accuracy"]),
        "quantum_attention_gap": quantum["target_minus_distractor_attention"] >= float(gate["minimum_quantum_attention_gap"]),
        "unsigned_target_tie": abs(unsigned["target_minus_distractor_attention"]) <= float(gate["maximum_control_target_gap"]),
        "classical_target_tie": abs(classical["target_minus_distractor_attention"]) <= float(gate["maximum_control_target_gap"]),
        "minimum_corrected_errors": quantum["corrected_examples"] >= int(gate["minimum_corrected_errors"]),
        "no_harmed_correct": quantum["harmed_correct_examples"] <= int(gate["maximum_harmed_correct"]),
        "parameter_matching": quantum["intervention_parameters"] == unsigned["intervention_parameters"] == classical["intervention_parameters"],
        "all_residual_invariants": all(row["residual_invariants"]["status"] == "pass" for row in results),
        "deterministic_replay": all(row["deterministic_replay"] for row in results),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "status": "pass" if passed else "fail",
        "natural_qk_attention_benchmark_authorized": passed,
        "existing_task_benchmark_authorized": False,
        "five_seed_phase_authorized": False,
        "real_data_authorized": False,
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = choose_device(args.device or str(config["device"]))
    split = make_split(int(config["seed"]), int(config["dataset"]["size"]), device, config)
    tolerance = float(config["mechanism"]["tie_tolerance"])
    with torch.no_grad():
        baseline = predictions(forward(None, split)[2], tolerance)
    results = [evaluate(selector, split, baseline, config) for selector in config["selectors"]]
    diagnostic = geometry(split)
    gate = promotion_gate(results, diagnostic, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / "seed7" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": config["schema_version"],
        "status": "complete",
        "revision": git_revision(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "dataset_identity": config["dataset"]["identity"],
        "seed": int(config["seed"]),
        "selectors": list(config["selectors"]),
        "geometry": diagnostic,
        "results": results,
        "gate": gate,
        "design_contract": {
            "scores_derived_from_fixed_qk_encoder": True,
            "encoder_trainable": False,
            "labels_passed_to_plugin": False,
            "targets_passed_to_plugin": False,
            "values_fixed_external_readout": True,
            "candidate_orientation_is_latent": True,
            "parameter_sweep": False,
            "walk_time": float(config["mechanism"]["walk_time"]),
        },
        "limitations": [
            "The Q/K feature encoder is fixed and synthetic, not learned from natural language.",
            "This gate tests direct attention-score realizability and label-free plugin isolation, not structured-NLP utility.",
            "A pass does not authorize old task splits, five seeds, real data, or hardware-speedup claims.",
        ],
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
