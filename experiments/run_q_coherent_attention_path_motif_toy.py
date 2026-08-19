#!/usr/bin/env python3
"""Score-hook canary for entity-anchored coherent attention paths."""

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


NODES = 5
HEADS = 1
HEAD_DIM = 4
ANCHOR = 0
SUBJECT = 1
OBJECT = 2
TARGET_NODES = (3, 4)
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 4))
EDGE_SIGNS = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0)
SELECTOR_TYPES = {
    "q_wap_signed": "quantum_signed",
    "q_wap_unsigned": "quantum_unsigned",
    "classical_wap_diffusion": "classical_diffusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_coherent_attention_path_motif_toy.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q-attention.coherent-attention-path-motif.v1":
        raise ValueError("unsupported Q-WAP motif config")
    if int(payload.get("seed", -1)) != 7:
        raise ValueError("Q-WAP motif canary requires fixed seed 7")
    if tuple(payload.get("selectors", ())) != (
        "disabled",
        "q_wap_signed",
        "q_wap_unsigned",
        "classical_wap_diffusion",
    ):
        raise ValueError("Q-WAP selectors must match the explicit allowlist")
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


def base_graph() -> torch.Tensor:
    graph = torch.zeros(NODES, NODES)
    for (source, target), sign in zip(EDGES, EDGE_SIGNS):
        graph[source, target] = sign
        graph[target, source] = sign
    return graph


def make_split(seed: int, size: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(size) % 2
    labels = labels[torch.randperm(size, generator=generator)]
    graph = base_graph()
    swap = torch.tensor([0, 1, 2, 4, 3])
    swapped = graph[swap][:, swap]
    scores = torch.where(labels[:, None, None].bool(), swapped, graph)
    scores = scores[:, None, :, :]

    query = torch.zeros(size, HEADS, NODES, HEAD_DIM)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    value[:, 0, TARGET_NODES[0], 0] = 1.0
    value[:, 0, TARGET_NODES[1], 1] = 1.0
    attention_mask = torch.ones(size, NODES, dtype=torch.bool)
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, SUBJECT] = True
    object_mask[:, OBJECT] = True
    return {
        "query": query.to(device),
        "key": key.to(device),
        "value": value.to(device),
        "scores": scores.to(device),
        "attention_mask": attention_mask.to(device),
        "subject_mask": subject_mask.to(device),
        "object_mask": object_mask.to(device),
        "labels": labels.to(device),
        "target_key": torch.where(
            labels == 0,
            torch.full_like(labels, TARGET_NODES[0]),
            torch.full_like(labels, TARGET_NODES[1]),
        ).to(device),
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
    logits = output[:, 0, ANCHOR, :2]
    return residual, attention, logits


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
    prediction = logits.argmax(dim=-1)
    labels = split["labels"]
    row = torch.arange(labels.shape[0], device=labels.device)
    target = split["target_key"]
    distractor = torch.where(target == TARGET_NODES[0], TARGET_NODES[1], TARGET_NODES[0])
    anchor_attention = attention[:, 0, ANCHOR, :]
    corrected = (~baseline_prediction.eq(labels)) & prediction.eq(labels)
    harmed = baseline_prediction.eq(labels) & (~prediction.eq(labels))
    return {
        "selector": selector,
        "accuracy": float(prediction.eq(labels).float().mean()),
        "corrected_examples": int(corrected.sum()),
        "harmed_correct_examples": int(harmed.sum()),
        "target_attention": float(anchor_attention[row, target].mean()),
        "distractor_attention": float(anchor_attention[row, distractor].mean()),
        "target_minus_distractor_attention": float(
            (anchor_attention[row, target] - anchor_attention[row, distractor]).mean()
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


def promotion_gate(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_selector = {row["selector"]: row for row in results}
    quantum = by_selector["q_wap_signed"]
    unsigned = by_selector["q_wap_unsigned"]
    classical = by_selector["classical_wap_diffusion"]
    disabled = by_selector["disabled"]
    gate = config["gate"]
    conditions = {
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
        "attention_score_hook_canary_authorized": passed,
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
        int(config["seed"]), int(config["dataset"]["size"]), device
    )
    with torch.no_grad():
        direct_baseline = forward(None, split)
    baseline_prediction = direct_baseline[2].argmax(dim=-1)
    results = [
        evaluate(selector, split, baseline_prediction, config)
        for selector in config["selectors"]
    ]
    gate = promotion_gate(results, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / "seed7" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "q-attention.coherent-attention-path-motif.v1",
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
        "results": results,
        "gate": gate,
        "design_contract": {
            "walk_time": float(config["mechanism"]["walk_time"]),
            "label_input_to_plugin": False,
            "target_input_to_plugin": False,
            "fixed_classifier": True,
            "parameter_sweep": False,
            "signed_quantum_vs_unsigned_and_classical_controls": True,
        },
        "limitations": [
            "The motif is constructed to isolate signed path interference.",
            "A pass is an interface canary, not task utility or label-free structured-NLP evidence.",
            "Exact matrix-exponential simulation does not establish quantum hardware speedup.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()

