#!/usr/bin/env python3
"""Label-free full-position evidence-anchor prescreen for Q-WAP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import math
import platform
import sys
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_trained_baseline_gate import (  # noqa: E402
    _graph,
    choose_device,
    collect_scores,
    evaluate_baseline,
    git_revision,
    set_seed,
    tensor_batch,
)
from run_q_partial_evidence_triad_gate import (  # noqa: E402
    load_config as load_partial_config,
    make_splits,
)
from q_attention.models import RelationExtractionModel, RelationTransformerConfig  # noqa: E402


EVIDENCE_IDS = tuple(range(5, 11)) + tuple(range(13, 19))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_full_position_evidence_anchor_prescreen.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.q-full-position-evidence-anchor-prescreen.v1":
        raise ValueError("unsupported full-position evidence-anchor config")
    if int(config.get("seed", -1)) != 7:
        raise ValueError("full-position evidence-anchor prescreen requires seed 7")
    if abs(float(config["mechanism"]["walk_time"]) - math.pi / 4.0) > 1e-12:
        raise ValueError("walk time must remain pi/4")
    if config["mechanism"].get("anchor") != "baseline_context_attention":
        raise ValueError("prescreen requires the frozen baseline-context anchor")
    if config["training"].get("parameter_sweep") is not False:
        raise ValueError("parameter sweeps are forbidden")
    return config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_baseline(
    splits: dict[str, dict[str, Any]],
    source_config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> tuple[RelationExtractionModel, dict[str, Any]]:
    model_config = source_config["model"]
    vocab_size = max(
        int(tensor_batch(split)["input_ids"].max()) for split in splits.values()
    ) + 1
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=vocab_size,
            num_labels=2,
            dim=int(model_config["dim"]),
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            ff_dim=int(model_config["ff_dim"]),
            dropout=float(model_config["dropout"]),
            max_length=int(tensor_batch(next(iter(splits.values())))["input_ids"].shape[1]),
        )
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metrics = {}
    logits = {}
    for name in ("train", "valid", "test"):
        metrics[name], logits[name] = evaluate_baseline(model, splits[name])
        metrics[name].pop("predictions")
    return model, {"metrics": metrics, "logits": logits}


def _quantum_probabilities(graph: torch.Tensor, walk_time: float) -> torch.Tensor:
    complex_dtype = torch.complex128 if graph.dtype == torch.float64 else torch.complex64
    return torch.matrix_exp((-1j * walk_time) * graph.to(complex_dtype)).abs().square().to(graph.dtype)


def _classical_probabilities(graph: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    adjacency = graph.abs()
    degree = adjacency.sum(dim=-1, keepdim=True)
    transition = adjacency / degree.clamp_min(eps)
    isolated = degree.squeeze(-1) <= eps
    if isolated.any():
        identity = torch.eye(graph.shape[-1], device=graph.device, dtype=graph.dtype)
        identity = identity.view(1, 1, graph.shape[-1], graph.shape[-1])
        transition = torch.where(isolated[..., None], identity, transition)
    return transition @ transition


def _context_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["attention_mask"] & ~(batch["subject_mask"] | batch["object_mask"])


def _evidence_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    ids = batch["input_ids"]
    values = torch.tensor(EVIDENCE_IDS, device=ids.device, dtype=ids.dtype)
    return (ids[..., None] == values).any(dim=-1) & _context_mask(batch)


def _anchor_distribution(scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    key_mask = batch["attention_mask"][:, None, None, :]
    base = torch.softmax(scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min), dim=-1)
    context = _context_mask(batch)[:, None, None, :].to(scores.dtype)
    anchor = base * context
    return anchor / anchor.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _field_from_anchor(
    scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    walk_time: float,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph = _graph(scores, batch["attention_mask"])
    probabilities = (
        _quantum_probabilities(graph, walk_time)
        if kind == "quantum_signed"
        else _classical_probabilities(graph)
    )
    anchor = _anchor_distribution(scores, batch)
    field = torch.einsum("bhqj,bhjk->bhqk", anchor, probabilities)
    context = _context_mask(batch)[:, None, None, :].to(field.dtype)
    field = field * context
    field = field / field.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return field, anchor


def _candidate_delta(field: torch.Tensor) -> torch.Tensor:
    return field[..., 5] - field[..., 4]


def _orientation(delta: torch.Tensor, labels: torch.Tensor) -> int:
    direct = float((delta >= 0).eq(labels.bool()).float().mean())
    flipped = float((delta < 0).eq(labels.bool()).float().mean())
    return 1 if direct >= flipped else -1


def _field_metrics(
    field: torch.Tensor,
    anchor: torch.Tensor,
    batch: dict[str, torch.Tensor],
    orientation: int,
) -> dict[str, Any]:
    labels = batch["labels"].bool()
    delta = _candidate_delta(field).mean(dim=(1, 2))
    anchor_delta = _candidate_delta(anchor).mean(dim=(1, 2))
    evidence = _evidence_mask(batch)[:, None, None, :].to(field.dtype)
    marker_mass = (field * evidence).sum(dim=-1).mean(dim=(1, 2))
    anchor_marker_mass = (anchor * evidence).sum(dim=-1).mean(dim=(1, 2))
    signed_delta = orientation * delta
    return {
        "candidate_alignment": float((signed_delta >= 0).eq(labels).float().mean()),
        "candidate_margin": float(signed_delta.mean()),
        "candidate_margin_abs": float(signed_delta.abs().mean()),
        "marker_mass": float(marker_mass.mean()),
        "anchor_marker_mass": float(anchor_marker_mass.mean()),
        "marker_enrichment": float((marker_mass - anchor_marker_mass).mean()),
        "anchor_candidate_margin": float((orientation * anchor_delta).mean()),
        "field_anchor_l1": float((field - anchor).abs().mean()),
        "finite": bool(torch.isfinite(field).all() and torch.isfinite(anchor).all()),
        "field_sum_error": float((field.sum(dim=-1) - 1.0).abs().max()),
        "anchor_sum_error": float((anchor.sum(dim=-1) - 1.0).abs().max()),
    }


def _aggregate_layer_fields(
    captures: list[dict[str, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    walk_time: float,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fields = []
    anchors = []
    for capture in captures:
        field, anchor = _field_from_anchor(capture["scores"], batch, walk_time, kind)
        fields.append(field)
        anchors.append(anchor)
    return torch.stack(fields).mean(dim=0), torch.stack(anchors).mean(dim=0)


def _split_metrics(
    captures: list[dict[str, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    walk_time: float,
    orientation: dict[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("quantum_signed", "classical_diffusion"):
        field, anchor = _aggregate_layer_fields(captures, batch, walk_time, kind)
        result[kind] = _field_metrics(field, anchor, batch, orientation[kind])
    quantum = result["quantum_signed"]
    classical = result["classical_diffusion"]
    result["quantum_minus_classical_candidate_alignment"] = (
        quantum["candidate_alignment"] - classical["candidate_alignment"]
    )
    result["quantum_minus_classical_candidate_margin"] = (
        quantum["candidate_margin"] - classical["candidate_margin"]
    )
    result["quantum_minus_classical_marker_mass"] = (
        quantum["marker_mass"] - classical["marker_mass"]
    )
    result["quantum_minus_classical_marker_enrichment"] = (
        quantum["marker_enrichment"] - classical["marker_enrichment"]
    )
    return result


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    source_config_path = (ROOT / config["source_config"]).resolve()
    source_config = load_partial_config(source_config_path)
    checkpoint_path = (ROOT / config["source_checkpoint"]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen baseline checkpoint not found: {checkpoint_path}")
    device = choose_device(args.device or str(config["device"]))
    set_seed(int(config["seed"]))
    splits = make_splits(source_config, device)
    model, baseline = load_frozen_baseline(splits, source_config, checkpoint_path, device)
    expected_metrics = config["expected_baseline_metrics"]
    for name in ("valid", "test"):
        observed = baseline["metrics"][name]
        expected = expected_metrics[name]
        if abs(float(observed["accuracy"]) - float(expected["accuracy"])) > 1e-8:
            raise RuntimeError(f"frozen baseline accuracy mismatch on {name}")
        if abs(float(observed["nll"]) - float(expected["nll"])) > 1e-8:
            raise RuntimeError(f"frozen baseline NLL mismatch on {name}")

    captures_by_split: dict[str, list[dict[str, torch.Tensor]]] = {}
    logits_by_split: dict[str, torch.Tensor] = {}
    for name in ("train", "valid", "test"):
        captures_by_split[name], logits_by_split[name] = collect_scores(model, splits[name])

    orientation: dict[str, int] = {}
    train_metrics: dict[str, Any] = {}
    train_batch = tensor_batch(splits["train"])
    for kind in ("quantum_signed", "classical_diffusion"):
        field, _anchor = _aggregate_layer_fields(
            captures_by_split["train"], train_batch, float(config["mechanism"]["walk_time"]), kind
        )
        delta = _candidate_delta(field).mean(dim=(1, 2))
        orientation[kind] = _orientation(delta, train_batch["labels"])
        train_metrics[kind] = _field_metrics(field, _anchor, train_batch, orientation[kind])

    diagnostics: dict[str, Any] = {
        "orientation_from_train": orientation,
        "train": train_metrics,
        "valid": _split_metrics(
            captures_by_split["valid"], tensor_batch(splits["valid"]), float(config["mechanism"]["walk_time"]), orientation
        ),
        "test": _split_metrics(
            captures_by_split["test"], tensor_batch(splits["test"]), float(config["mechanism"]["walk_time"]), orientation
        ),
    }
    gate = config["stage_a_gate"]
    valid = diagnostics["valid"]
    test = diagnostics["test"]
    baseline_valid = baseline["metrics"]["valid"]["accuracy"] >= float(gate["minimum_baseline_accuracy"])
    baseline_test = baseline["metrics"]["test"]["accuracy"] >= float(gate["minimum_baseline_accuracy"])
    invariant = all(
        diagnostics[name][kind][key]
        for name in ("valid", "test")
        for kind in ("quantum_signed", "classical_diffusion")
        for key in ("finite",)
    ) and max(
        diagnostics[name][kind][key]
        for name in ("valid", "test")
        for kind in ("quantum_signed", "classical_diffusion")
        for key in ("field_sum_error", "anchor_sum_error")
    ) <= float(gate["maximum_normalization_error"])
    alignment = (
        valid["quantum_signed"]["candidate_alignment"] >= float(gate["minimum_candidate_alignment"])
        and test["quantum_signed"]["candidate_alignment"] >= float(gate["minimum_candidate_alignment"])
    )
    q_advantage = (
        valid["quantum_minus_classical_candidate_margin"] >= float(gate["minimum_quantum_margin_advantage"])
        and test["quantum_minus_classical_candidate_margin"] >= float(gate["minimum_quantum_margin_advantage"])
    )
    field_nontrivial = (
        valid["quantum_signed"]["field_anchor_l1"] >= float(gate["minimum_field_anchor_l1"])
        and test["quantum_signed"]["field_anchor_l1"] >= float(gate["minimum_field_anchor_l1"])
    )
    stage_a = {
        "status": "pass" if all((baseline_valid, baseline_test, invariant, alignment, q_advantage, field_nontrivial)) else "fail",
        "baseline_valid": baseline_valid,
        "baseline_test": baseline_test,
        "invariants": invariant,
        "candidate_alignment": alignment,
        "quantum_margin_advantage": q_advantage,
        "nontrivial_full_position_field": field_nontrivial,
        "stage_b_authorized": False,
        "multi_seed_authorized": False,
        "real_data_authorized": False,
        "hardware_claim_authorized": False,
    }
    if not baseline_valid or not baseline_test:
        stage_a["failure_reason"] = "baseline_invalid"
    elif not invariant:
        stage_a["failure_reason"] = "field_invariant_failure"
    elif not alignment:
        stage_a["failure_reason"] = "full_position_field_not_task_aligned"
    elif not q_advantage:
        stage_a["failure_reason"] = "quantum_not_better_than_classical_diffusion"
    elif not field_nontrivial:
        stage_a["failure_reason"] = "full_position_anchor_has_no_transport_effect"
    else:
        stage_a["failure_reason"] = None

    baseline.pop("logits")

    output_root = Path(args.output_root) if args.output_root else ROOT / config["output_root"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "seed7" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    run_summary = {
        "schema_version": config["schema_version"],
        "status": "complete",
        "experiment": config["experiment_name"],
        "seed": int(config["seed"]),
        "revision": git_revision(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "source_config_path": str(source_config_path.relative_to(ROOT)),
        "source_config_sha256": sha256(source_config_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "dataset_identity": source_config["dataset"]["identity"],
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "baseline": baseline,
        "design_contract": {
            "baseline_context_attention_is_only_intervention_anchor": True,
            "labels_and_audit_marker_masks_not_passed_to_field": True,
            "full_position_context_anchor": True,
            "quantum_and_classical_use_same_anchor_and_walk_time": True,
            "parameter_sweep": False,
            "single_seed_prescreen": True,
            "frozen_checkpoint_reused": True,
        },
        "diagnostics": diagnostics,
        "stage_a_gate": stage_a,
        "stage_b_gate": {
            "status": "not_run",
            "failure_reason": "prescreen_only",
            "multi_seed_authorized": False,
            "real_data_authorized": False,
            "hardware_claim_authorized": False,
        },
        "limitations": [
            "Synthetic fixed-role partial-evidence task, not natural language.",
            "Seed 7 is a pre-screen only; no selector or score intervention was trained.",
            "Exact matrix-exponential simulation does not establish hardware speedup.",
        ],
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(run_dir), "stage_a": stage_a, "diagnostics": {"valid": diagnostics["valid"], "test": diagnostics["test"]}}, sort_keys=True))


if __name__ == "__main__":
    main()
