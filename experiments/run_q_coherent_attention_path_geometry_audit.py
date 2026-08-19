#!/usr/bin/env python3
"""Robustness and equivariance audit for nontrivial Q-WAP score geometry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
HEAD_DIM = 2
CANONICAL_ANCHOR = 0
CANONICAL_SUBJECT = 1
CANONICAL_OBJECT = 2
CANONICAL_TARGETS = (3, 4)
SELECTOR_TYPES = {
    "q_wap_signed": "quantum_signed",
    "q_wap_unsigned": "quantum_unsigned",
    "classical_wap_diffusion": "classical_diffusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_coherent_attention_path_geometry_audit.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "q-attention.coherent-attention-path-geometry-audit.v1"
    ):
        raise ValueError("unsupported Q-WAP geometry-audit config")
    if int(payload.get("seed", -1)) != 7:
        raise ValueError("Q-WAP geometry audit requires fixed seed 7")
    if int(payload["dataset"]["nodes"]) != NODES:
        raise ValueError(f"Q-WAP geometry audit requires exactly {NODES} nodes")
    if tuple(payload.get("selectors", ())) != (
        "disabled",
        "q_wap_signed",
        "q_wap_unsigned",
        "classical_wap_diffusion",
    ):
        raise ValueError("Q-WAP selectors must match the frozen allowlist")
    if abs(float(payload["mechanism"]["walk_time"]) - torch.pi / 4) > 1e-12:
        raise ValueError("Q-WAP walk time is frozen at pi/4")
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


def _edge(graph: torch.Tensor, source: int, target: int, weight: float) -> None:
    graph[source, target] = weight
    graph[target, source] = weight


def _jittered_weight(generator: torch.Generator, jitter: float) -> float:
    return 1.0 + jitter * (2.0 * float(torch.rand((), generator=generator)) - 1.0)


def _canonical_graph(
    generator: torch.Generator,
    *,
    label: int,
    weight_jitter: float,
    nuisance_weight: float,
) -> torch.Tensor:
    graph = torch.zeros(NODES, NODES, dtype=torch.float32)
    anchor_left = _jittered_weight(generator, weight_jitter)
    anchor_right = _jittered_weight(generator, weight_jitter)
    target_left = _jittered_weight(generator, weight_jitter)
    target_right = _jittered_weight(generator, weight_jitter)

    _edge(graph, 0, 1, -anchor_left)
    _edge(graph, 0, 2, -anchor_right)
    _edge(graph, 1, 3, -target_left)
    _edge(graph, 2, 3, -target_right)
    _edge(graph, 1, 4, -target_left)
    _edge(graph, 2, 4, target_right)

    # Symmetric low-weight cycles make the suite non-isomorphic while preserving
    # the target tie for every nonnegative control.
    cycle_scale = nuisance_weight * (
        0.8 + 0.4 * float(torch.rand((), generator=generator))
    )
    spectator_scale = nuisance_weight * (
        0.8 + 0.4 * float(torch.rand((), generator=generator))
    )
    _edge(graph, 1, 2, cycle_scale)
    _edge(graph, 0, 5, spectator_scale)
    _edge(graph, 5, 3, spectator_scale)
    _edge(graph, 5, 4, spectator_scale)
    _edge(graph, 0, 6, -cycle_scale)
    _edge(graph, 6, 3, cycle_scale)
    _edge(graph, 6, 4, cycle_scale)
    _edge(graph, 5, 6, nuisance_weight / 2.0)

    if label == 1:
        target_swap = torch.tensor([0, 1, 2, 4, 3, 5, 6])
        graph = graph[target_swap][:, target_swap]
    return graph


def make_split(
    seed: int,
    size: int,
    device: torch.device,
    *,
    weight_jitter: float,
    nuisance_weight: float,
) -> dict[str, torch.Tensor]:
    if size <= 0 or size % 2:
        raise ValueError("geometry-audit size must be positive and even")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(size, dtype=torch.long) % 2
    labels = labels[torch.randperm(size, generator=generator)]

    canonical_graphs = []
    gauged_graphs = []
    transformed_graphs = []
    permutations = []
    gauges = []
    anchors = []
    targets = []
    distractors = []
    queries = []
    keys = []
    values = []
    subject_masks = []
    object_masks = []

    for label_tensor in labels:
        label = int(label_tensor)
        canonical = _canonical_graph(
            generator,
            label=label,
            weight_jitter=weight_jitter,
            nuisance_weight=nuisance_weight,
        )
        gauge = torch.where(
            torch.rand(NODES, generator=generator) >= 0.5,
            torch.ones(NODES),
            -torch.ones(NODES),
        )
        gauge[CANONICAL_ANCHOR] = 1.0
        gauged = gauge[:, None] * canonical * gauge[None, :]
        permutation = torch.randperm(NODES, generator=generator)
        transformed = gauged[permutation][:, permutation]

        canonical_value = torch.zeros(NODES, HEAD_DIM)
        canonical_value[CANONICAL_TARGETS[0], 0] = 1.0
        canonical_value[CANONICAL_TARGETS[1], 1] = 1.0
        canonical_subject = torch.zeros(NODES, dtype=torch.bool)
        canonical_object = torch.zeros(NODES, dtype=torch.bool)
        canonical_subject[CANONICAL_SUBJECT] = True
        canonical_object[CANONICAL_OBJECT] = True
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(NODES)
        target_canonical = CANONICAL_TARGETS[label]
        distractor_canonical = CANONICAL_TARGETS[1 - label]

        canonical_graphs.append(canonical)
        gauged_graphs.append(gauged)
        transformed_graphs.append(transformed)
        permutations.append(permutation)
        gauges.append(gauge)
        anchors.append(inverse[CANONICAL_ANCHOR])
        targets.append(inverse[target_canonical])
        distractors.append(inverse[distractor_canonical])
        queries.append(torch.zeros(HEADS, NODES, HEAD_DIM))
        keys.append(torch.zeros(HEADS, NODES, HEAD_DIM))
        values.append(canonical_value[permutation].unsqueeze(0))
        subject_masks.append(canonical_subject[permutation])
        object_masks.append(canonical_object[permutation])

    scores = torch.stack(transformed_graphs).unsqueeze(1)
    return {
        "query": torch.stack(queries).to(device),
        "key": torch.stack(keys).to(device),
        "value": torch.stack(values).to(device),
        "scores": scores.to(device),
        "attention_mask": torch.ones(size, NODES, dtype=torch.bool).to(device),
        "subject_mask": torch.stack(subject_masks).to(device),
        "object_mask": torch.stack(object_masks).to(device),
        "labels": labels.to(device),
        "anchor": torch.stack(anchors).to(device),
        "target_key": torch.stack(targets).to(device),
        "distractor_key": torch.stack(distractors).to(device),
        "canonical_scores": torch.stack(canonical_graphs).unsqueeze(1).to(device),
        "gauged_canonical_scores": torch.stack(gauged_graphs).unsqueeze(1).to(device),
        "permutations": torch.stack(permutations).to(device),
        "gauges": torch.stack(gauges).to(device),
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
    row = torch.arange(output.shape[0], device=output.device)
    logits = output[row, 0, split["anchor"], :2]
    return residual, attention, logits


def predictions(logits: torch.Tensor, tolerance: float) -> torch.Tensor:
    margin = logits[:, 1] - logits[:, 0]
    return torch.where(
        margin > tolerance,
        torch.ones_like(margin, dtype=torch.long),
        torch.zeros_like(margin, dtype=torch.long),
    )


def residual_invariants(
    residual: torch.Tensor,
    split: dict[str, torch.Tensor],
    max_transport: float,
) -> dict[str, Any]:
    context = split["attention_mask"] & ~(
        split["subject_mask"] | split["object_mask"]
    )
    context_mask = context[:, None, None, :].to(dtype=residual.dtype)
    outside = residual * (1.0 - context_mask)
    context_sum = (residual * context_mask).sum(dim=-1)
    checks = {
        "finite": bool(torch.isfinite(residual).all()),
        "context_only": bool(outside.abs().max() <= 1e-7),
        "zero_sum_context": bool(context_sum.abs().max() <= 1e-6),
        "bounded": bool(residual.abs().max() <= max_transport + 1e-6),
    }
    return {
        **checks,
        "status": "pass" if all(checks.values()) else "fail",
        "max_abs": float(residual.abs().max()),
        "max_context_sum_error": float(context_sum.abs().max()),
    }


def evaluate(
    selector: str,
    split: dict[str, torch.Tensor],
    baseline_prediction: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    kernel = build_kernel(selector, config)
    if kernel is not None:
        kernel = kernel.to(split["scores"].device)
    with torch.no_grad():
        residual, attention, logits = forward(kernel, split)
        replay = forward(kernel, split)
    prediction = predictions(logits, float(config["mechanism"]["tie_tolerance"]))
    labels = split["labels"]
    row = torch.arange(labels.shape[0], device=labels.device)
    anchor_attention = attention[row, 0, split["anchor"], :]
    target_attention = anchor_attention[row, split["target_key"]]
    distractor_attention = anchor_attention[row, split["distractor_key"]]
    corrected = (~baseline_prediction.eq(labels)) & prediction.eq(labels)
    harmed = baseline_prediction.eq(labels) & (~prediction.eq(labels))
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


def _target_gap(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    row = torch.arange(labels.shape[0], device=labels.device)
    target = torch.where(labels == 0, CANONICAL_TARGETS[0], CANONICAL_TARGETS[1])
    distractor = torch.where(
        labels == 0, CANONICAL_TARGETS[1], CANONICAL_TARGETS[0]
    )
    return probabilities[row, target] - probabilities[row, distractor]


def _permuted_split(
    split: dict[str, torch.Tensor], permutation: torch.Tensor
) -> dict[str, torch.Tensor]:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(NODES, device=permutation.device)
    result = dict(split)
    result["query"] = split["query"][:, :, permutation, :]
    result["key"] = split["key"][:, :, permutation, :]
    result["value"] = split["value"][:, :, permutation, :]
    result["scores"] = split["scores"][:, :, permutation][:, :, :, permutation]
    for key in ("attention_mask", "subject_mask", "object_mask"):
        result[key] = split[key][:, permutation]
    for key in ("anchor", "target_key", "distractor_key"):
        result[key] = inverse[split[key]]
    return result


def geometry_diagnostics(
    split: dict[str, torch.Tensor], config: dict[str, Any]
) -> dict[str, Any]:
    kernel = build_kernel("q_wap_signed", config).to(split["scores"].device)
    canonical_mask = torch.ones(
        split["labels"].shape[0], NODES, dtype=torch.bool, device=split["scores"].device
    )
    with torch.no_grad():
        canonical = kernel._hermitian_graph(
            split["canonical_scores"], canonical_mask
        )
        gauged = kernel._hermitian_graph(
            split["gauged_canonical_scores"], canonical_mask
        )
        signed = kernel._quantum_probabilities(canonical)[:, 0, CANONICAL_ANCHOR]
        gauged_signed = kernel._quantum_probabilities(gauged)[
            :, 0, CANONICAL_ANCHOR
        ]
        unsigned = kernel._quantum_probabilities(canonical.abs())[
            :, 0, CANONICAL_ANCHOR
        ]
        classical = kernel._classical_probabilities(canonical)[
            :, 0, CANONICAL_ANCHOR
        ]
        original_residual = forward(kernel, split)[0]
        permutation = torch.tensor(
            [2, 5, 0, 6, 1, 4, 3], device=split["scores"].device
        )
        permuted = _permuted_split(split, permutation)
        permuted_residual = forward(kernel, permuted)[0]
        expected_residual = original_residual[:, :, permutation][
            :, :, :, permutation
        ]

    signed_gap = _target_gap(signed, split["labels"])
    unsigned_gap = _target_gap(unsigned, split["labels"])
    classical_gap = _target_gap(classical, split["labels"])
    flattened = split["scores"].detach().cpu().reshape(split["scores"].shape[0], -1)
    return {
        "examples": int(split["labels"].shape[0]),
        "positive_labels": int(split["labels"].sum()),
        "unique_weighted_graphs": int(torch.unique(flattened, dim=0).shape[0]),
        "unique_permutations": int(
            torch.unique(split["permutations"].detach().cpu(), dim=0).shape[0]
        ),
        "minimum_signed_target_gap": float(signed_gap.min()),
        "mean_signed_target_gap": float(signed_gap.mean()),
        "maximum_absolute_unsigned_target_gap": float(unsigned_gap.abs().max()),
        "maximum_absolute_classical_target_gap": float(classical_gap.abs().max()),
        "maximum_gauge_probability_error": float(
            (signed - gauged_signed).abs().max()
        ),
        "maximum_permutation_residual_error": float(
            (permuted_residual - expected_residual).abs().max()
        ),
    }


def promotion_gate(
    results: list[dict[str, Any]],
    geometry: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_selector = {row["selector"]: row for row in results}
    disabled = by_selector["disabled"]
    quantum = by_selector["q_wap_signed"]
    unsigned = by_selector["q_wap_unsigned"]
    classical = by_selector["classical_wap_diffusion"]
    gate = config["gate"]
    size = int(config["dataset"]["size"])
    conditions = {
        "label_balance": geometry["positive_labels"] * 2 == size,
        "graph_diversity": geometry["unique_weighted_graphs"]
        >= int(gate["minimum_unique_graphs"]),
        "signed_geometry_margin": geometry["minimum_signed_target_gap"]
        >= float(gate["minimum_signed_path_gap"]),
        "unsigned_geometry_symmetry": geometry[
            "maximum_absolute_unsigned_target_gap"
        ]
        <= float(gate["maximum_control_path_gap"]),
        "classical_geometry_symmetry": geometry[
            "maximum_absolute_classical_target_gap"
        ]
        <= float(gate["maximum_control_path_gap"]),
        "gauge_invariance": geometry["maximum_gauge_probability_error"]
        <= float(gate["maximum_gauge_error"]),
        "permutation_equivariance": geometry["maximum_permutation_residual_error"]
        <= float(gate["maximum_permutation_error"]),
        "baseline_is_balanced_chance": abs(disabled["accuracy"] - 0.5) <= 1e-8,
        "quantum_accuracy": quantum["accuracy"]
        >= float(gate["minimum_quantum_accuracy"]),
        "unsigned_phase_ablation": unsigned["accuracy"]
        <= float(gate["maximum_control_accuracy"]),
        "classical_diffusion_control": classical["accuracy"]
        <= float(gate["maximum_control_accuracy"]),
        "minimum_corrected_errors": quantum["corrected_examples"]
        >= int(gate["minimum_corrected_errors"]),
        "no_harmed_correct": quantum["harmed_correct_examples"]
        <= int(gate["maximum_harmed_correct"]),
        "quantum_attention_gap": quantum["target_minus_distractor_attention"]
        >= float(gate["minimum_quantum_attention_gap"]),
        "parameter_matching": quantum["intervention_parameters"]
        == unsigned["intervention_parameters"]
        == classical["intervention_parameters"],
        "all_residual_invariants": all(
            row["residual_invariants"]["status"] == "pass" for row in results
        ),
        "deterministic_replay": all(row["deterministic_replay"] for row in results),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "status": "pass" if passed else "fail",
        "fresh_attention_geometry_benchmark_authorized": passed,
        "existing_task_benchmark_authorized": False,
        "five_seed_phase_authorized": False,
        "real_data_authorized": False,
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = choose_device(args.device or str(config["device"]))
    split = make_split(
        int(config["seed"]),
        int(config["dataset"]["size"]),
        device,
        weight_jitter=float(config["dataset"]["weight_jitter"]),
        nuisance_weight=float(config["dataset"]["nuisance_weight"]),
    )
    with torch.no_grad():
        baseline_logits = forward(None, split)[2]
    baseline_prediction = predictions(
        baseline_logits, float(config["mechanism"]["tie_tolerance"])
    )
    results = [
        evaluate(selector, split, baseline_prediction, config)
        for selector in config["selectors"]
    ]
    geometry = geometry_diagnostics(split, config)
    gate = promotion_gate(results, geometry, config)

    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / "seed7" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
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
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "dataset_identity": config["dataset"]["identity"],
        "seed": int(config["seed"]),
        "selectors": list(config["selectors"]),
        "geometry": geometry,
        "results": results,
        "gate": gate,
        "design_contract": {
            "fixed_walk_time": float(config["mechanism"]["walk_time"]),
            "parameter_sweep": False,
            "labels_passed_to_plugin": False,
            "targets_passed_to_plugin": False,
            "balanced_external_labels": True,
            "weighted_graphs": True,
            "random_gauge": True,
            "random_node_permutation": True,
            "symmetric_connected_nuisance_paths": True,
        },
        "limitations": [
            "This constructed suite audits geometry robustness, not structured-NLP task utility.",
            "Gauge and permutation checks are invariance tests, not evidence of quantum hardware advantage.",
            "A pass does not authorize an existing failed split, five seeds, real data, or collaborator work.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
