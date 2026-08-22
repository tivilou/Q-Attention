#!/usr/bin/env python3
"""Held-out fixed-budget trainability canary for Q-WAP transport."""

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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_geometry_audit import (  # noqa: E402
    build_kernel,
    forward,
    predictions,
    residual_invariants,
)
from run_q_coherent_attention_path_qk_realizability_audit import (  # noqa: E402
    make_qk_split,
    realizability_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/q_coherent_attention_path_trainability_canary.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "q-attention.coherent-attention-path-trainability.v1"
    ):
        raise ValueError("unsupported Q-WAP trainability config")
    if int(payload.get("seed", -1)) != 7:
        raise ValueError("Q-WAP trainability canary requires fixed seed 7")
    if int(payload["dataset"]["nodes"]) != 7:
        raise ValueError("Q-WAP trainability canary requires seven-node attention")
    if abs(float(payload["mechanism"]["walk_time"]) - math.pi / 4) > 1e-12:
        raise ValueError("Q-WAP walk time is frozen at pi/4")
    if tuple(payload.get("selectors", ())) != (
        "disabled",
        "q_wap_signed",
        "q_wap_unsigned",
        "classical_wap_diffusion",
    ):
        raise ValueError("Q-WAP selectors must match the frozen allowlist")
    if payload["training"].get("optimizer") != "adam_full_batch":
        raise ValueError("Q-WAP trainability optimizer must remain fixed")
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


def make_splits(
    config: dict[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    dataset = config["dataset"]
    shared = {
        "device": device,
        "weight_jitter": float(dataset["weight_jitter"]),
        "nuisance_weight": float(dataset["nuisance_weight"]),
        "skew_score_scale": float(dataset["skew_score_scale"]),
    }
    train = make_qk_split(
        int(dataset["train_stream"]), int(dataset["train_size"]), **shared
    )
    valid = make_qk_split(
        int(dataset["valid_stream"]), int(dataset["valid_size"]), **shared
    )
    return train, valid


def split_diagnostics(
    train: dict[str, torch.Tensor], valid: dict[str, torch.Tensor]
) -> dict[str, Any]:
    train_flat = train["designed_scores"].detach().cpu().reshape(
        train["designed_scores"].shape[0], -1
    )
    valid_flat = valid["designed_scores"].detach().cpu().reshape(
        valid["designed_scores"].shape[0], -1
    )
    overlap = (
        train_flat[:, None, :].eq(valid_flat[None, :, :]).all(dim=-1).sum().item()
    )
    return {
        "train_examples": int(train["labels"].shape[0]),
        "valid_examples": int(valid["labels"].shape[0]),
        "train_positive_labels": int(train["labels"].sum()),
        "valid_positive_labels": int(valid["labels"].sum()),
        "exact_train_valid_score_overlap": int(overlap),
        "train_qk": realizability_diagnostics(train),
        "valid_qk": realizability_diagnostics(valid),
    }


def _metrics(
    kernel,
    split: dict[str, torch.Tensor],
    baseline_prediction: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    with torch.no_grad():
        residual, attention, logits = forward(kernel, split)
        replay = forward(kernel, split)
        loss = F.cross_entropy(logits, split["labels"])
    prediction = predictions(
        logits, float(config["mechanism"]["tie_tolerance"])
    )
    labels = split["labels"]
    row = torch.arange(labels.shape[0], device=labels.device)
    anchor_attention = attention[row, 0, split["anchor"], :]
    target_attention = anchor_attention[row, split["target_key"]]
    distractor_attention = anchor_attention[row, split["distractor_key"]]
    corrected = (~baseline_prediction.eq(labels)) & prediction.eq(labels)
    harmed = baseline_prediction.eq(labels) & (~prediction.eq(labels))
    return {
        "accuracy": float(prediction.eq(labels).float().mean()),
        "nll": float(loss),
        "corrected_examples": int(corrected.sum()),
        "harmed_correct_examples": int(harmed.sum()),
        "target_attention": float(target_attention.mean()),
        "distractor_attention": float(distractor_attention.mean()),
        "target_minus_distractor_attention": float(
            (target_attention - distractor_attention).mean()
        ),
        "deterministic_replay": bool(
            torch.equal(residual, replay[0])
            and torch.equal(attention, replay[1])
            and torch.equal(logits, replay[2])
        ),
        "residual_invariants": residual_invariants(
            residual, split, float(config["mechanism"]["max_transport"])
        ),
    }


def train_selector(
    selector: str,
    train: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    baseline_predictions: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, Any]:
    kernel = build_kernel(selector, config)
    if kernel is not None:
        kernel = kernel.to(train["scores"].device)
    initial_train = _metrics(
        kernel, train, baseline_predictions["train"], config
    )
    initial_valid = _metrics(
        kernel, valid, baseline_predictions["valid"], config
    )
    initial_transport = (
        0.0
        if kernel is None
        else float(kernel.transport_fractions(0).detach().mean())
    )
    losses = [initial_train["nll"]]
    if kernel is not None:
        optimizer = torch.optim.Adam(
            kernel.parameters(), lr=float(config["training"]["learning_rate"])
        )
        for _ in range(int(config["training"]["steps"])):
            optimizer.zero_grad(set_to_none=True)
            logits = forward(kernel, train)[2]
            loss = F.cross_entropy(logits, train["labels"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

    final_train = _metrics(kernel, train, baseline_predictions["train"], config)
    final_valid = _metrics(kernel, valid, baseline_predictions["valid"], config)
    final_transport = (
        0.0
        if kernel is None
        else float(kernel.transport_fractions(0).detach().mean())
    )
    parameters = 0 if kernel is None else sum(p.numel() for p in kernel.parameters())
    return {
        "selector": selector,
        "trainable_parameters": parameters,
        "initial_transport": initial_transport,
        "final_transport": final_transport,
        "transport_increase": final_transport - initial_transport,
        "training_curve": {
            "steps": 0 if kernel is None else int(config["training"]["steps"]),
            "initial_nll": losses[0],
            "last_step_nll": losses[-1],
            "minimum_nll": min(losses),
            "finite": all(math.isfinite(value) for value in losses),
        },
        "initial_train": initial_train,
        "initial_valid": initial_valid,
        "final_train": final_train,
        "final_valid": final_valid,
        "metadata": {"id": "disabled", "version": "1.0.0"}
        if kernel is None
        else kernel.metadata(),
    }


def promotion_gate(
    results: list[dict[str, Any]],
    splits: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_selector = {row["selector"]: row for row in results}
    disabled = by_selector["disabled"]
    quantum = by_selector["q_wap_signed"]
    unsigned = by_selector["q_wap_unsigned"]
    classical = by_selector["classical_wap_diffusion"]
    baseline_nll = disabled["final_valid"]["nll"]
    quantum_nll_gain = baseline_nll - quantum["final_valid"]["nll"]
    unsigned_nll_gain = baseline_nll - unsigned["final_valid"]["nll"]
    classical_nll_gain = baseline_nll - classical["final_valid"]["nll"]
    best_control_nll = min(
        unsigned["final_valid"]["nll"], classical["final_valid"]["nll"]
    )
    gate = config["gate"]
    conditions = {
        "disjoint_train_valid_scores": splits["exact_train_valid_score_overlap"] == 0,
        "balanced_train_labels": splits["train_positive_labels"] * 2
        == splits["train_examples"],
        "balanced_valid_labels": splits["valid_positive_labels"] * 2
        == splits["valid_examples"],
        "qk_contract": all(
            part["finite_query_key"]
            and part["maximum_qk_reconstruction_error"] <= 1e-5
            and part["maximum_hamiltonian_reconstruction_error"] <= 1e-5
            for part in (splits["train_qk"], splits["valid_qk"])
        ),
        "baseline_is_balanced_chance": abs(
            disabled["final_valid"]["accuracy"] - 0.5
        )
        <= 1e-8,
        "quantum_valid_accuracy": quantum["final_valid"]["accuracy"]
        >= float(gate["minimum_quantum_valid_accuracy"]),
        "unsigned_valid_accuracy": unsigned["final_valid"]["accuracy"]
        <= float(gate["maximum_control_valid_accuracy"]),
        "classical_valid_accuracy": classical["final_valid"]["accuracy"]
        <= float(gate["maximum_control_valid_accuracy"]),
        "quantum_valid_nll_gain": quantum_nll_gain
        >= float(gate["minimum_quantum_valid_nll_gain"]),
        "unsigned_valid_nll_bound": unsigned_nll_gain
        <= float(gate["maximum_control_valid_nll_gain"]),
        "classical_valid_nll_bound": classical_nll_gain
        <= float(gate["maximum_control_valid_nll_gain"]),
        "quantum_vs_control_nll_advantage": best_control_nll
        - quantum["final_valid"]["nll"]
        >= float(gate["minimum_quantum_vs_control_nll_advantage"]),
        "quantum_training_nll_reduction": quantum["initial_train"]["nll"]
        - quantum["final_train"]["nll"]
        >= float(gate["minimum_quantum_training_nll_reduction"]),
        "quantum_transport_learned": quantum["transport_increase"]
        >= float(gate["minimum_transport_increase"]),
        "quantum_train_valid_consistency": abs(
            quantum["final_train"]["nll"] - quantum["final_valid"]["nll"]
        )
        <= float(gate["maximum_train_valid_nll_gap"]),
        "minimum_corrected_errors": quantum["final_valid"]["corrected_examples"]
        >= int(gate["minimum_corrected_errors"]),
        "no_harmed_correct": quantum["final_valid"]["harmed_correct_examples"]
        <= int(gate["maximum_harmed_correct"]),
        "parameter_matching": quantum["trainable_parameters"]
        == unsigned["trainable_parameters"]
        == classical["trainable_parameters"]
        == 1,
        "finite_training": all(row["training_curve"]["finite"] for row in results),
        "all_residual_invariants": all(
            row[stage]["residual_invariants"]["status"] == "pass"
            for row in results
            for stage in ("initial_train", "initial_valid", "final_train", "final_valid")
        ),
        "deterministic_replay": all(
            row[stage]["deterministic_replay"]
            for row in results
            for stage in ("initial_train", "initial_valid", "final_train", "final_valid")
        ),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "status": "pass" if passed else "fail",
        "heldout_synthetic_attention_utility_authorized": passed,
        "existing_task_benchmark_authorized": False,
        "five_seed_phase_authorized": False,
        "real_data_authorized": False,
        "derived_metrics": {
            "quantum_valid_nll_gain": quantum_nll_gain,
            "unsigned_valid_nll_gain": unsigned_nll_gain,
            "classical_valid_nll_gain": classical_nll_gain,
            "quantum_vs_best_control_nll_advantage": best_control_nll
            - quantum["final_valid"]["nll"],
        },
    }


def run(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    train, valid = make_splits(config, device)
    tolerance = float(config["mechanism"]["tie_tolerance"])
    with torch.no_grad():
        baseline_predictions = {
            "train": predictions(forward(None, train)[2], tolerance),
            "valid": predictions(forward(None, valid)[2], tolerance),
        }
    results = [
        train_selector(
            selector, train, valid, baseline_predictions, config
        )
        for selector in config["selectors"]
    ]
    diagnostics = split_diagnostics(train, valid)
    return {
        "splits": diagnostics,
        "results": results,
        "gate": promotion_gate(results, diagnostics, config),
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = choose_device(args.device or str(config["device"]))
    outcome = run(config, device)

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
        "training": config["training"],
        **outcome,
        "design_contract": {
            "only_transport_parameter_trainable": True,
            "classifier_frozen": True,
            "query_key_value_frozen": True,
            "labels_used_only_by_external_loss": True,
            "targets_not_used_by_plugin_or_training": True,
            "fixed_train_valid_streams": [
                int(config["dataset"]["train_stream"]),
                int(config["dataset"]["valid_stream"]),
            ],
            "walk_time": float(config["mechanism"]["walk_time"]),
            "parameter_sweep": False,
        },
        "limitations": [
            "This is held-out synthetic attention utility, not structured-NLP utility.",
            "Q/K/V factors are constructed and frozen rather than learned from language tokens.",
            "A pass does not authorize old failed splits, five seeds, real data, or speedup claims.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "gate": outcome["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
