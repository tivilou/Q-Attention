#!/usr/bin/env python3
"""Audit whether frozen, label-free signals identify useful score actions."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterator

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_trained_baseline_gate import (  # noqa: E402
    collect_scores,
    evaluate_baseline,
    fixed_score_residuals,
    git_revision,
    model_forward,
    set_seed,
    tensor_batch,
)
from run_q_full_position_evidence_anchor_prescreen import (  # noqa: E402
    load_frozen_baseline,
)
from run_q_partial_evidence_triad_gate import (  # noqa: E402
    load_config as load_partial_config,
    make_splits,
)


FAMILIES = (
    "action_identity",
    "attention",
    "value",
    "jacobian",
    "counterfactual",
    "combined",
    "shuffled_combined",
)


@dataclass(frozen=True)
class ActionSpec:
    layer: int
    query: int
    key: int
    sign: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_intervention_identifiability_audit.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.q-intervention-identifiability-audit.v1":
        raise ValueError("unsupported intervention-identifiability config")
    if int(config.get("seed", -1)) != 7:
        raise ValueError("identifiability audit requires fixed seed 7")
    if tuple(config.get("observable_families", ())) != FAMILIES:
        raise ValueError("observable families must match the frozen allowlist")
    if config["training"].get("parameter_sweep") is not False:
        raise ValueError("parameter sweeps are forbidden")
    if float(config["action_basis"]["max_delta"]) <= 0.0:
        raise ValueError("max_delta must be positive")
    return config


def context_positions(batch: dict[str, torch.Tensor]) -> tuple[int, ...]:
    context = batch["attention_mask"] & ~(
        batch["subject_mask"] | batch["object_mask"]
    )
    first = context[0]
    if not bool(context.eq(first).all()):
        raise ValueError("audit requires a shared fixed-position context mask")
    return tuple(int(index) for index in torch.where(first)[0].tolist())


def build_action_specs(
    num_layers: int,
    num_queries: int,
    keys: tuple[int, ...],
) -> tuple[ActionSpec, ...]:
    return tuple(
        ActionSpec(layer, query, key, sign)
        for layer in range(num_layers)
        for query in range(num_queries)
        for key in keys
        for sign in (-1, 1)
    )


def zero_sum_action_residual(
    batch: dict[str, torch.Tensor],
    spec: ActionSpec,
    num_heads: int,
    max_delta: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    context = batch["attention_mask"] & ~(
        batch["subject_mask"] | batch["object_mask"]
    )
    mask = context.to(dtype)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    target = torch.zeros_like(mask)
    target[:, spec.key] = 1.0
    basis = float(spec.sign) * max_delta * (target - mask / count) * mask
    residual = torch.zeros(
        batch["labels"].shape[0],
        num_heads,
        batch["input_ids"].shape[1],
        batch["input_ids"].shape[1],
        device=batch["labels"].device,
        dtype=dtype,
    )
    residual[:, :, spec.query, :] = basis[:, None, :]
    return residual


@contextmanager
def capture_predicted_margin_gradients(
    model: torch.nn.Module,
) -> Iterator[list[torch.Tensor | None]]:
    leaves: list[torch.Tensor | None] = [None] * len(model.encoder.layers)
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("score-hook output must be a tensor")
            leaf = output.detach().requires_grad_(True)
            leaves[layer_index] = leaf
            return leaf

        return hook

    try:
        for layer_index, layer in enumerate(model.encoder.layers):
            handles.append(
                layer.attn.score_intervention.register_forward_hook(
                    make_hook(layer_index)
                )
            )
        yield leaves
    finally:
        for handle in handles:
            handle.remove()


def predicted_margin_gradients(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with capture_predicted_margin_gradients(model) as leaves:
        logits = model_forward(model, batch)
        top2 = logits.topk(2, dim=-1)
        margin = top2.values[:, 0] - top2.values[:, 1]
        margin.sum().backward()
    gradients = []
    for leaf in leaves:
        if leaf is None or leaf.grad is None:
            raise RuntimeError("predicted-margin score gradient was not captured")
        gradients.append(leaf.grad.detach())
    return logits.detach(), gradients


def true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(labels.shape[0], device=labels.device)
    other = 1 - labels
    return logits[rows, labels] - logits[rows, other]


def predicted_margin(logits: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(predictions.shape[0], device=predictions.device)
    return logits[rows, predictions] - logits[rows, 1 - predictions]


def action_metadata(
    specs: tuple[ActionSpec, ...],
    num_layers: int,
    num_queries: int,
    num_keys: int,
    device: torch.device,
) -> torch.Tensor:
    width = num_layers + num_queries + num_keys + 1
    result = torch.zeros(len(specs), width, device=device)
    key_values = sorted({spec.key for spec in specs})
    key_map = {value: index for index, value in enumerate(key_values)}
    for row, spec in enumerate(specs):
        result[row, spec.layer] = 1.0
        result[row, num_layers + spec.query] = 1.0
        result[row, num_layers + num_queries + key_map[spec.key]] = 1.0
        result[row, -1] = float(spec.sign)
    return result


def action_observables(
    capture: dict[str, torch.Tensor],
    gradient: torch.Tensor,
    residual: torch.Tensor,
    spec: ActionSpec,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    scores = capture["scores"]
    query = capture["query"]
    value = capture["value"]
    context = batch["attention_mask"] & ~(
        batch["subject_mask"] | batch["object_mask"]
    )
    context_float = context[:, None, :].to(scores.dtype)
    count = context_float.sum(dim=-1).clamp_min(1.0)
    score_row = scores[:, :, spec.query, :]
    attention_row = torch.softmax(score_row, dim=-1)
    score_mean = (score_row * context_float).sum(dim=-1) / count
    attention_mean = (attention_row * context_float).sum(dim=-1) / count
    score_contrast = float(spec.sign) * (
        score_row[:, :, spec.key] - score_mean
    ).mean(dim=1)
    attention_contrast = float(spec.sign) * (
        attention_row[:, :, spec.key] - attention_mean
    ).mean(dim=1)

    value_mean = (
        value * context_float[..., None]
    ).sum(dim=-2) / count[..., None]
    centered_value = value[:, :, spec.key, :] - value_mean
    query_value = float(spec.sign) * (
        query[:, :, spec.query, :] * centered_value
    ).sum(dim=-1).div(math.sqrt(query.shape[-1])).mean(dim=1)
    value_norm = centered_value.norm(dim=-1).mean(dim=1)
    signed_value_norm = float(spec.sign) * value_norm
    directional_gradient = (
        gradient[:, :, spec.query, :] * residual[:, :, spec.query, :]
    ).sum(dim=(1, 2))
    return {
        "attention": torch.stack((score_contrast, attention_contrast), dim=-1),
        "value": torch.stack((query_value, signed_value_norm, value_norm), dim=-1),
        "jacobian": directional_gradient[:, None],
    }


def scan_split(
    model: torch.nn.Module,
    split: dict[str, Any],
    specs: tuple[ActionSpec, ...],
    max_delta: float,
) -> dict[str, Any]:
    batch = tensor_batch(split)
    captures, replay_logits = collect_scores(model, split)
    baseline_logits, gradients = predicted_margin_gradients(model, batch)
    replay_error = float((replay_logits - baseline_logits).abs().max())
    baseline_predictions = baseline_logits.argmax(dim=-1)
    base_true_margin = true_margin(baseline_logits, batch["labels"])
    base_predicted_margin = predicted_margin(baseline_logits, baseline_predictions)
    base_probabilities = torch.softmax(baseline_logits, dim=-1)
    base_entropy = -(base_probabilities * base_probabilities.clamp_min(1e-8).log()).sum(dim=-1)

    logits_by_action = []
    attention_features = []
    value_features = []
    jacobian_features = []
    counterfactual_features = []
    for spec in specs:
        residual = zero_sum_action_residual(
            batch,
            spec,
            model.config.num_heads,
            max_delta,
            captures[spec.layer]["scores"].dtype,
        )
        residuals = [torch.zeros_like(row["scores"]) for row in captures]
        residuals[spec.layer] = residual
        with torch.no_grad(), fixed_score_residuals(model, residuals):
            action_logits = model_forward(model, batch)
        logits_by_action.append(action_logits)
        observable = action_observables(
            captures[spec.layer], gradients[spec.layer], residual, spec, batch
        )
        attention_features.append(observable["attention"])
        value_features.append(observable["value"])
        jacobian_features.append(observable["jacobian"])
        action_predicted_margin = predicted_margin(
            action_logits, baseline_predictions
        )
        logit_l1 = (action_logits - baseline_logits).abs().sum(dim=-1)
        counterfactual_features.append(
            torch.stack(
                (
                    action_predicted_margin - base_predicted_margin,
                    logit_l1,
                ),
                dim=-1,
            )
        )

    action_logits = torch.stack(logits_by_action, dim=1)
    utility = true_margin(
        action_logits.reshape(-1, action_logits.shape[-1]),
        batch["labels"][:, None].expand(-1, len(specs)).reshape(-1),
    ).reshape(batch["labels"].shape[0], len(specs)) - base_true_margin[:, None]
    metadata = action_metadata(
        specs,
        model.config.num_layers,
        batch["input_ids"].shape[1],
        len(context_positions(batch)),
        batch["labels"].device,
    )
    global_features = torch.stack((base_predicted_margin, base_entropy), dim=-1)
    common = torch.cat(
        (
            metadata[None, :, :].expand(batch["labels"].shape[0], -1, -1),
            global_features[:, None, :].expand(-1, len(specs), -1),
        ),
        dim=-1,
    )
    attention = torch.stack(attention_features, dim=1)
    value = torch.stack(value_features, dim=1)
    jacobian = torch.stack(jacobian_features, dim=1)
    counterfactual = torch.stack(counterfactual_features, dim=1)
    features = {
        "action_identity": common,
        "attention": torch.cat((common, attention), dim=-1),
        "value": torch.cat((common, value), dim=-1),
        "jacobian": torch.cat((common, jacobian), dim=-1),
        "counterfactual": torch.cat((common, counterfactual), dim=-1),
        "combined": torch.cat(
            (common, attention, value, jacobian, counterfactual), dim=-1
        ),
    }
    return {
        "labels": batch["labels"],
        "baseline_logits": baseline_logits,
        "baseline_predictions": baseline_predictions,
        "action_logits": action_logits,
        "utility": utility,
        "features": features,
        "disabled_replay_error": replay_error,
        "maximum_residual": max_delta,
    }


def fit_ridge(
    features: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> dict[str, torch.Tensor]:
    flat = features.reshape(-1, features.shape[-1])
    y = target.reshape(-1, 1)
    mean = flat.mean(dim=0)
    scale = flat.std(dim=0).clamp_min(1e-6)
    normalized = ((flat - mean) / scale).to(torch.float64)
    y = y.to(torch.float64)
    design = torch.cat(
        (
            normalized,
            torch.ones(
                normalized.shape[0],
                1,
                device=normalized.device,
                dtype=normalized.dtype,
            ),
        ),
        dim=-1,
    )
    sample_count = float(design.shape[0])
    gram = (design.transpose(0, 1) @ design) / sample_count
    penalty = ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(
        gram + penalty, (design.transpose(0, 1) @ y) / sample_count
    ).squeeze(-1)
    return {"mean": mean, "scale": scale, "weights": weights}


def predict_ridge(model: dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    normalized = ((features - model["mean"]) / model["scale"]).to(
        model["weights"].dtype
    )
    return (
        normalized @ model["weights"][:-1] + model["weights"][-1]
    ).to(features.dtype)


def selector_metrics(
    row: dict[str, Any],
    predicted_utility: torch.Tensor,
    *,
    force_on_errors: bool = False,
) -> dict[str, Any]:
    labels = row["labels"]
    baseline = row["baseline_predictions"]
    best_score, best_action = predicted_utility.max(dim=-1)
    selected = baseline.ne(labels) if force_on_errors else best_score > 0.0
    indices = torch.arange(labels.shape[0], device=labels.device)
    selected_logits = row["action_logits"][indices, best_action]
    logits = torch.where(selected[:, None], selected_logits, row["baseline_logits"])
    prediction = logits.argmax(dim=-1)
    wrong = baseline.ne(labels)
    correct = ~wrong
    corrected = wrong & prediction.eq(labels)
    harmed = correct & prediction.ne(labels)
    chosen_actual_utility = row["utility"][indices, best_action]
    oracle_utility = row["utility"].max(dim=-1).values
    topk = {}
    error_mask = wrong[:, None]
    for k in (1, 5, 10):
        width = min(k, predicted_utility.shape[-1])
        top_indices = predicted_utility.topk(width, dim=-1).indices
        top_logits = row["action_logits"][
            indices[:, None], top_indices
        ]
        top_predictions = top_logits.argmax(dim=-1)
        topk[f"top_{k}_error_corrections"] = int(
            (error_mask & top_predictions.eq(labels[:, None])).any(dim=-1).sum()
        )
    return {
        "accuracy": float(prediction.eq(labels).float().mean()),
        "accuracy_delta": float(
            prediction.eq(labels).float().mean()
            - baseline.eq(labels).float().mean()
        ),
        "baseline_errors": int(wrong.sum()),
        "selected_examples": int(selected.sum()),
        "corrected_examples": int(corrected.sum()),
        "harmed_correct_examples": int(harmed.sum()),
        "correct_to_wrong_rate": float(harmed.float().sum() / correct.float().sum().clamp_min(1.0)),
        "positive_utility_selection_rate": float(
            ((chosen_actual_utility > 0.0) & selected).float().sum()
            / selected.float().sum().clamp_min(1.0)
        ),
        "mean_actual_utility_when_selected": float(
            (chosen_actual_utility * selected).sum()
            / selected.float().sum().clamp_min(1.0)
        ),
        "mean_oracle_regret": float(
            (oracle_utility - torch.where(selected, chosen_actual_utility, torch.zeros_like(chosen_actual_utility))).mean()
        ),
        "forced_error_corrections": int(
            (wrong & prediction.eq(labels)).sum()
        ) if force_on_errors else None,
        "forced_error_selected": int(selected.sum()) if force_on_errors else None,
        **topk,
    }


def oracle_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return selector_metrics(row, row["utility"])


def shuffled(features: torch.Tensor) -> torch.Tensor:
    if features.shape[0] < 2:
        raise ValueError("shuffled control requires at least two examples")
    return features.roll(1, dims=0)


def evaluate_families(
    scans: dict[str, dict[str, Any]],
    ridge: float,
) -> dict[str, Any]:
    models = {}
    train_features = scans["train"]["features"]
    for family in FAMILIES:
        base_family = "combined" if family == "shuffled_combined" else family
        features = train_features[base_family]
        if family == "shuffled_combined":
            features = shuffled(features)
        models[family] = fit_ridge(features, scans["train"]["utility"], ridge)

    result: dict[str, Any] = {}
    for split in ("valid", "test"):
        result[split] = {"oracle": oracle_metrics(scans[split])}
        for family in FAMILIES:
            base_family = "combined" if family == "shuffled_combined" else family
            features = scans[split]["features"][base_family]
            if family == "shuffled_combined":
                features = shuffled(features)
            scores = predict_ridge(models[family], features)
            result[split][family] = selector_metrics(scans[split], scores)
            result[split][family]["forced_error"] = selector_metrics(
                scans[split], scores, force_on_errors=True
            )
    return result


def audit_gate(results: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["gate"]
    single_controls = (
        "action_identity",
        "attention",
        "value",
        "jacobian",
        "counterfactual",
    )
    best_single = {
        split: max(results[split][name]["corrected_examples"] for name in single_controls)
        for split in ("valid", "test")
    }
    conditions = {
        "oracle_basis_headroom": all(
            results[split]["oracle"]["corrected_examples"]
            >= int(gate["minimum_oracle_corrections"])
            for split in ("valid", "test")
        ),
        "combined_minimum_corrections": all(
            results[split]["combined"]["corrected_examples"]
            >= int(gate["minimum_combined_corrections"])
            for split in ("valid", "test")
        ),
        "combined_beats_best_single": all(
            results[split]["combined"]["corrected_examples"] - best_single[split]
            >= int(gate["minimum_extra_corrections"])
            for split in ("valid", "test")
        ),
        "combined_beats_shuffled": all(
            results[split]["combined"]["corrected_examples"]
            - results[split]["shuffled_combined"]["corrected_examples"]
            >= int(gate["minimum_extra_corrections"])
            for split in ("valid", "test")
        ),
        "combined_low_harm": all(
            results[split]["combined"]["correct_to_wrong_rate"]
            <= float(gate["maximum_correct_to_wrong_rate"])
            for split in ("valid", "test")
        ),
    }
    passed = all(conditions.values())
    if not conditions["oracle_basis_headroom"]:
        failure_reason = "bounded_action_basis_has_insufficient_headroom"
    elif not conditions["combined_minimum_corrections"]:
        failure_reason = "label_free_observables_do_not_identify_corrections"
    elif not conditions["combined_beats_best_single"]:
        failure_reason = "combined_observables_add_no_identifiable_interaction"
    elif not conditions["combined_beats_shuffled"]:
        failure_reason = "observable_alignment_not_better_than_shuffled"
    elif not conditions["combined_low_harm"]:
        failure_reason = "identification_harms_correct_examples"
    else:
        failure_reason = None
    return {
        **conditions,
        "best_single_corrected_examples": best_single,
        "status": "pass" if passed else "fail",
        "failure_reason": failure_reason,
        "new_mechanism_ideation_authorized": passed,
        "quantum_estimator_authorized": False,
        "multi_seed_authorized": False,
        "real_data_authorized": False,
        "hardware_claim_authorized": False,
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    source_config_path = (ROOT / config["source_config"]).resolve()
    source_config = load_partial_config(source_config_path)
    checkpoint_path = (ROOT / config["source_checkpoint"]).resolve()
    if sha256(checkpoint_path) != config["source_checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash mismatch")
    device_name = args.device or str(config["device"])
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    set_seed(int(config["seed"]))
    splits = make_splits(source_config, device)
    model, baseline = load_frozen_baseline(
        splits, source_config, checkpoint_path, device
    )
    for split in ("valid", "test"):
        expected = config["expected_baseline_metrics"][split]
        observed = baseline["metrics"][split]
        if abs(float(observed["accuracy"]) - float(expected["accuracy"])) > 1e-8:
            raise RuntimeError(f"frozen baseline accuracy mismatch on {split}")
        if abs(float(observed["nll"]) - float(expected["nll"])) > 1e-8:
            raise RuntimeError(f"frozen baseline NLL mismatch on {split}")

    batch = tensor_batch(splits["train"])
    keys = context_positions(batch)
    specs = build_action_specs(
        model.config.num_layers, batch["input_ids"].shape[1], keys
    )
    expected_actions = (
        model.config.num_layers * batch["input_ids"].shape[1] * len(keys) * 2
    )
    if len(specs) != expected_actions:
        raise RuntimeError("action basis is incomplete")
    scans = {
        split: scan_split(
            model,
            splits[split],
            specs,
            float(config["action_basis"]["max_delta"]),
        )
        for split in ("train", "valid", "test")
    }
    results = evaluate_families(
        scans, float(config["training"]["ridge_penalty"])
    )
    gate = audit_gate(results, config)
    diagnostics = {
        "action_count": len(specs),
        "context_keys": list(keys),
        "disabled_replay_error": {
            split: scans[split]["disabled_replay_error"]
            for split in ("train", "valid", "test")
        },
        "maximum_residual": max(
            scans[split]["maximum_residual"] for split in scans
        ),
    }
    baseline.pop("logits")
    output_root = Path(args.output_root) if args.output_root else ROOT / config["output_root"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "seed7" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": config["schema_version"],
        "status": "complete",
        "experiment": config["experiment_name"],
        "revision": git_revision(),
        "seed": int(config["seed"]),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_config_path": source_config_path.relative_to(ROOT).as_posix(),
        "source_config_sha256": sha256(source_config_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "dataset_identity": source_config["dataset"]["identity"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "baseline": baseline,
        "diagnostics": diagnostics,
        "results": results,
        "gate": gate,
        "design_contract": {
            "bounded_complete_single_basis": True,
            "labels_used_only_for_training_targets_and_offline_evaluation": True,
            "heldout_observables_are_label_free": True,
            "frozen_checkpoint_reused": True,
            "no_audit_marker_masks_in_observables": True,
            "parameter_sweep": False,
            "single_seed_diagnostic": True,
            "quantum_module_implemented": False,
        },
        "limitations": [
            "Synthetic fixed-role partial-evidence task, not natural language.",
            "One fixed seed; this is an identifiability audit, not task utility.",
            "Counterfactual observables require exhaustive action evaluation and are not a deployable selector.",
            "A positive audit would not establish quantum advantage or authorize a quantum estimator."
        ],
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = {
        split: {
            family: {
                "corrected": row["corrected_examples"],
                "harmed": row["harmed_correct_examples"],
                "accuracy_delta": row["accuracy_delta"],
            }
            for family, row in results[split].items()
        }
        for split in ("valid", "test")
    }
    print(json.dumps({"output": str(run_dir), "gate": gate, "results": compact}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
