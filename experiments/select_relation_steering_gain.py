from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.experiments import (  # noqa: E402
    ANCHOR_CHOICES,
    EvaluationResult,
    choose_device,
    evaluate_relation_model,
    load_projector,
    load_relation_run,
    make_relation_loader,
    projector_shape_summary,
)
from q_attention.tasks.relation import (
    RelationRecord,
    load_relation_jsonl,
    sample_relation_records_proportional,
)  # noqa: E402

SELECTION_STRATEGIES = ("shared", "best_layer", "coordinate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select steering gain on validation and evaluate it once on held-out test data.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--projector_path", required=True)
    parser.add_argument("--validation_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gains", default="0.0,0.05,0.1,0.25,0.5")
    parser.add_argument("--selection_metric", default="macro_f1")
    parser.add_argument("--strategy", default="shared", choices=SELECTION_STRATEGIES)
    parser.add_argument("--coordinate_passes", type=int, default=1)
    parser.add_argument("--require_positive_ci", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--bootstrap_confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap_seed", type=int, default=13)
    parser.add_argument("--acceptance_fraction", type=float, default=0.0)
    parser.add_argument("--acceptance_seed", type=int, default=13)
    parser.add_argument("--anchor", default="subject_object", choices=ANCHOR_CHOICES)
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def parse_gain_values(value: str) -> list[float]:
    gains = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not gains:
        raise ValueError("at least one candidate gain is required")
    if any(not math.isfinite(gain) for gain in gains):
        raise ValueError("candidate gains must be finite")
    return list(dict.fromkeys(gains))


def metric_delta(steered: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    return {key: steered[key] - baseline[key] for key in steered.keys() & baseline.keys()}


def select_best_gain(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("gain selection requires at least one result")
    for row in rows:
        value = row.get("metrics", {}).get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"selection metric '{metric}' is missing or non-finite")
    best_index = max(
        range(len(rows)),
        key=lambda index: (float(rows[index]["metrics"][metric]), -abs(float(rows[index]["gain"])), -index),
    )
    return rows[best_index]


def zero_layer_gains(module_paths: list[str] | tuple[str, ...]) -> dict[str, float]:
    return {path: 0.0 for path in module_paths}


def set_layer_gain(gains: Mapping[str, float], module_path: str, value: float) -> dict[str, float]:
    if module_path not in gains:
        raise ValueError(f"unknown layer path '{module_path}'")
    updated = {path: float(gain) for path, gain in gains.items()}
    updated[module_path] = float(value)
    return updated


def split_validation_records(
    records: list[RelationRecord],
    *,
    acceptance_fraction: float,
    seed: int,
) -> tuple[list[RelationRecord], list[RelationRecord]]:
    if acceptance_fraction <= 0.0:
        return list(records), []
    if not 0.0 < acceptance_fraction < 1.0:
        raise ValueError("acceptance_fraction must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("validation splitting requires at least two records")
    acceptance_count = max(1, min(len(records) - 1, round(len(records) * acceptance_fraction)))
    acceptance = sample_relation_records_proportional(records, acceptance_count, seed=seed)
    acceptance_ids = {id(record) for record in acceptance}
    selection = [record for record in records if id(record) not in acceptance_ids]
    return selection, acceptance


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    confusion = confusion.to(dtype=torch.float64)
    true_positive = torch.diag(confusion)
    predicted = confusion.sum(dim=0)
    actual = confusion.sum(dim=1)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(true_positive))
    recall = torch.where(actual > 0, true_positive / actual, torch.zeros_like(true_positive))
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2.0 * precision * recall / denominator, torch.zeros_like(denominator))
    return float(f1.mean().item())


def paired_bootstrap_macro_f1_delta(
    baseline_predictions: list[int],
    steered_predictions: list[int],
    labels: list[int],
    *,
    num_labels: int,
    num_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    if len(baseline_predictions) != len(steered_predictions) or len(labels) != len(baseline_predictions):
        raise ValueError("paired bootstrap inputs must have the same length")
    if not labels:
        raise ValueError("paired bootstrap requires at least one example")
    if num_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must be between 0 and 1")

    baseline = torch.tensor(baseline_predictions, dtype=torch.long)
    steered = torch.tensor(steered_predictions, dtype=torch.long)
    target = torch.tensor(labels, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def macro_f1(predictions: torch.Tensor, gold: torch.Tensor) -> float:
        flat_indices = gold * num_labels + predictions
        confusion = torch.bincount(flat_indices, minlength=num_labels * num_labels).reshape(num_labels, num_labels)
        return macro_f1_from_confusion(confusion)

    observed_delta = macro_f1(steered, target) - macro_f1(baseline, target)
    deltas = torch.empty(num_samples, dtype=torch.float64)
    for sample_index in range(num_samples):
        indices = torch.randint(len(labels), (len(labels),), generator=generator)
        deltas[sample_index] = macro_f1(steered[indices], target[indices]) - macro_f1(
            baseline[indices], target[indices]
        )
    alpha = 1.0 - confidence
    lower = float(torch.quantile(deltas, alpha / 2.0).item())
    upper = float(torch.quantile(deltas, 1.0 - alpha / 2.0).item())
    return {
        "observed_delta": observed_delta,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
        "num_samples": num_samples,
        "seed": seed,
    }


def label_name(id_to_label: Mapping[int, str], label_id: int) -> str:
    return id_to_label.get(int(label_id), str(label_id))


def write_predictions(
    path: Path,
    records: list[RelationRecord],
    id_to_label: Mapping[int, str],
    steered: EvaluationResult,
    baseline: EvaluationResult,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            item = {
                "index": index,
                "tokens": list(record.tokens),
                "subject": list(record.subject),
                "object": list(record.object),
                "gold_id": int(steered.labels[index]),
                "gold_label": label_name(id_to_label, steered.labels[index]),
                "steered_prediction_id": int(steered.predictions[index]),
                "steered_prediction_label": label_name(id_to_label, steered.predictions[index]),
                "baseline_prediction_id": int(baseline.predictions[index]),
                "baseline_prediction_label": label_name(id_to_label, baseline.predictions[index]),
                "metadata": dict(record.metadata),
            }
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    validation_path = resolve_project_path(args.validation_path).resolve()
    test_path = resolve_project_path(args.test_path).resolve()
    if validation_path == test_path:
        raise ValueError("validation_path and test_path must be distinct for held-out gain selection")

    device = choose_device(args.device)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(model_dir, device)
    projector_path = Path(args.projector_path)
    projector, projector_metadata = load_projector(projector_path, device)

    validation_records = load_relation_jsonl(validation_path)
    selection_records, acceptance_records = split_validation_records(
        validation_records,
        acceptance_fraction=args.acceptance_fraction,
        seed=args.acceptance_seed,
    )
    if args.require_positive_ci and not acceptance_records:
        raise ValueError("require_positive_ci requires a positive acceptance_fraction")
    validation_loader = make_relation_loader(
        selection_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_baseline = evaluate_relation_model(
        artifacts.model, validation_loader, device, len(artifacts.label_to_id)
    )
    gain_values = parse_gain_values(args.gains)

    def validation_row(gain_config: float | Mapping[str, float], **metadata: Any) -> dict[str, Any]:
        result = evaluate_relation_model(
            artifacts.model,
            validation_loader,
            device,
            len(artifacts.label_to_id),
            projector=projector,
            key_module_paths=artifacts.key_module_paths,
            gain=gain_config,
            anchor=args.anchor,
        )
        return {
            **metadata,
            "metrics": result.metrics,
            "delta_vs_baseline": metric_delta(result.metrics, validation_baseline.metrics),
        }

    candidate_rows: list[dict[str, Any]] = []
    selected_gain: float | None = None
    selected_gains: dict[str, float] | None = None
    selected_layer: str | None = None
    coordinate_steps: list[dict[str, Any]] = []
    if args.strategy == "shared":
        candidate_rows = [validation_row(gain, gain=gain) for gain in gain_values]
        selected = select_best_gain(candidate_rows, args.selection_metric)
        selected_gain = float(selected["gain"])
    else:
        if not isinstance(projector, Mapping):
            raise ValueError(f"selection strategy '{args.strategy}' requires a layer-specific projector payload")
        layer_paths = list(artifacts.key_module_paths)
        if set(projector) != set(layer_paths):
            raise ValueError("layer-specific projector paths do not match the baseline key module paths")
        if args.strategy == "best_layer":
            for layer_path in layer_paths:
                for gain in gain_values:
                    layer_gains = set_layer_gain(zero_layer_gains(layer_paths), layer_path, gain)
                    candidate_rows.append(
                        validation_row(
                            layer_gains,
                            gain=gain,
                            layer_path=layer_path,
                            layer_gains=layer_gains,
                        )
                    )
            selected = select_best_gain(candidate_rows, args.selection_metric)
            selected_gain = float(selected["gain"])
            selected_gains = dict(selected["layer_gains"])
            selected_layer = str(selected["layer_path"]) if selected_gain != 0.0 else None
        else:
            if args.coordinate_passes <= 0:
                raise ValueError("coordinate_passes must be positive")
            selected_gains = zero_layer_gains(layer_paths)
            for pass_index in range(args.coordinate_passes):
                for layer_path in layer_paths:
                    step_rows = []
                    for gain in gain_values:
                        layer_gains = set_layer_gain(selected_gains, layer_path, gain)
                        row = validation_row(
                            layer_gains,
                            gain=gain,
                            layer_path=layer_path,
                            layer_gains=layer_gains,
                            pass_index=pass_index,
                        )
                        step_rows.append(row)
                        candidate_rows.append(row)
                    step_selected = select_best_gain(step_rows, args.selection_metric)
                    selected_gains = dict(step_selected["layer_gains"])
                    coordinate_steps.append(
                        {
                            "pass_index": pass_index,
                            "layer_path": layer_path,
                            "selected_gain": float(step_selected["gain"]),
                            "selected_metric": float(step_selected["metrics"][args.selection_metric]),
                            "selected_gains": selected_gains,
                        }
                    )
    test_gain: float | Mapping[str, float]
    test_gain = selected_gain if selected_gains is None else selected_gains
    if test_gain is None:
        raise RuntimeError("gain selection did not produce a test-time steering configuration")

    proposed_gain = selected_gain
    proposed_gains = None if selected_gains is None else dict(selected_gains)
    proposed_layer = selected_layer
    selected_validation_result = evaluate_relation_model(
        artifacts.model,
        validation_loader,
        device,
        len(artifacts.label_to_id),
        projector=projector,
        key_module_paths=artifacts.key_module_paths,
        gain=test_gain,
        anchor=args.anchor,
    )
    proposed_validation_metrics = selected_validation_result.metrics

    if acceptance_records:
        acceptance_loader = make_relation_loader(
            acceptance_records,
            artifacts.vocab,
            artifacts.label_to_id,
            batch_size=args.batch_size,
            shuffle=False,
        )
        acceptance_baseline = evaluate_relation_model(
            artifacts.model, acceptance_loader, device, len(artifacts.label_to_id)
        )
        proposed_acceptance_result = evaluate_relation_model(
            artifacts.model,
            acceptance_loader,
            device,
            len(artifacts.label_to_id),
            projector=projector,
            key_module_paths=artifacts.key_module_paths,
            gain=test_gain,
            anchor=args.anchor,
        )
    else:
        acceptance_baseline = validation_baseline
        proposed_acceptance_result = selected_validation_result

    confidence_diagnostics: dict[str, float | int] | None = None
    selection_accepted = True
    proposed_nonzero = (
        abs(float(proposed_gain)) > 0.0
        if proposed_gains is None
        else any(abs(float(gain)) > 0.0 for gain in proposed_gains.values())
    )
    if args.require_positive_ci and proposed_nonzero:
        if args.selection_metric != "macro_f1":
            raise ValueError("require_positive_ci currently supports selection_metric=macro_f1 only")
        confidence_diagnostics = paired_bootstrap_macro_f1_delta(
            acceptance_baseline.predictions,
            proposed_acceptance_result.predictions,
            proposed_acceptance_result.labels,
            num_labels=len(artifacts.label_to_id),
            num_samples=args.bootstrap_samples,
            confidence=args.bootstrap_confidence,
            seed=args.bootstrap_seed,
        )
        selection_accepted = float(confidence_diagnostics["lower"]) > 0.0
        if not selection_accepted:
            if selected_gains is None:
                selected_gain = 0.0
                test_gain = 0.0
            else:
                selected_gain = None
                selected_gains = zero_layer_gains(list(artifacts.key_module_paths))
                selected_layer = None
                test_gain = selected_gains
            selected_validation_result = validation_baseline
            selected_acceptance_result = acceptance_baseline
        else:
            selected_acceptance_result = proposed_acceptance_result
    else:
        selected_acceptance_result = proposed_acceptance_result

    test_records = load_relation_jsonl(test_path)
    test_loader = make_relation_loader(
        test_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_baseline = evaluate_relation_model(artifacts.model, test_loader, device, len(artifacts.label_to_id))
    test_steered = evaluate_relation_model(
        artifacts.model,
        test_loader,
        device,
        len(artifacts.label_to_id),
        projector=projector,
        key_module_paths=artifacts.key_module_paths,
        gain=test_gain,
        anchor=args.anchor,
    )

    selection_payload = {
        "selection_split": "validation",
        "selection_path": str(validation_path),
        "selection_metric": args.selection_metric,
        "selection_num_records": len(selection_records),
        "acceptance_num_records": len(acceptance_records),
        "acceptance_fraction": args.acceptance_fraction,
        "acceptance_seed": args.acceptance_seed,
        "selection_strategy": args.strategy,
        "selection_accepted": selection_accepted,
        "confidence_diagnostics": confidence_diagnostics,
        "proposed_gain": proposed_gain,
        "proposed_gains": proposed_gains,
        "proposed_layer": proposed_layer,
        "selected_gain": selected_gain,
        "selected_gains": selected_gains,
        "selected_layer": selected_layer,
        "coordinate_passes": args.coordinate_passes if args.strategy == "coordinate" else None,
        "coordinate_steps": coordinate_steps,
        "proposed_validation_metrics": proposed_validation_metrics,
        "selected_validation_metrics": selected_validation_result.metrics,
        "acceptance_baseline": acceptance_baseline.metrics,
        "proposed_acceptance_metrics": proposed_acceptance_result.metrics,
        "selected_acceptance_metrics": selected_acceptance_result.metrics,
        "tie_break": "higher_metric_then_smaller_absolute_gain_then_earlier_candidate",
        "validation_baseline": validation_baseline.metrics,
        "candidates": candidate_rows,
    }
    metrics_payload = {
        "evaluation_split": "test",
        "steered": test_steered.metrics,
        "baseline": test_baseline.metrics,
        "delta_vs_baseline": metric_delta(test_steered.metrics, test_baseline.metrics),
    }
    run_info = {
        "model_dir": str(model_dir),
        "projector_path": str(projector_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "selection_strategy": args.strategy,
        "selection_accepted": selection_accepted,
        "confidence_diagnostics": confidence_diagnostics,
        "selected_gain": selected_gain,
        "selected_gains": selected_gains,
        "selected_layer": selected_layer,
        "selection_metric": args.selection_metric,
        "anchor": args.anchor,
        "num_validation_records": len(validation_records),
        "num_selection_records": len(selection_records),
        "num_acceptance_records": len(acceptance_records),
        "num_test_records": len(test_records),
        "key_module_paths": list(artifacts.key_module_paths),
        "projector_shape": projector_shape_summary(projector),
        "projector_metadata": dict(projector_metadata),
    }
    (output_dir / "gain_selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")
    write_predictions(output_dir / "predictions.jsonl", test_records, artifacts.id_to_label, test_steered, test_baseline)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selection_strategy": args.strategy,
                "selection_accepted": selection_accepted,
                "selected_gain": selected_gain,
                "selected_gains": selected_gains,
                "selected_layer": selected_layer,
                "validation_metric": selected_validation_result.metrics[args.selection_metric],
                "test_metrics": test_steered.metrics,
                "delta_vs_baseline": metrics_payload["delta_vs_baseline"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
