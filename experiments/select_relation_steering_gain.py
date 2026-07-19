from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

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
from q_attention.tasks.relation import RelationRecord, load_relation_jsonl  # noqa: E402


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
    validation_loader = make_relation_loader(
        validation_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_baseline = evaluate_relation_model(
        artifacts.model, validation_loader, device, len(artifacts.label_to_id)
    )
    candidate_rows: list[dict[str, Any]] = []
    for gain in parse_gain_values(args.gains):
        result = evaluate_relation_model(
            artifacts.model,
            validation_loader,
            device,
            len(artifacts.label_to_id),
            projector=projector,
            key_module_paths=artifacts.key_module_paths,
            gain=gain,
            anchor=args.anchor,
        )
        candidate_rows.append(
            {
                "gain": gain,
                "metrics": result.metrics,
                "delta_vs_baseline": metric_delta(result.metrics, validation_baseline.metrics),
            }
        )
    selected = select_best_gain(candidate_rows, args.selection_metric)
    selected_gain = float(selected["gain"])

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
        gain=selected_gain,
        anchor=args.anchor,
    )

    selection_payload = {
        "selection_split": "validation",
        "selection_path": str(validation_path),
        "selection_metric": args.selection_metric,
        "selected_gain": selected_gain,
        "tie_break": "smaller_absolute_gain_then_earlier_candidate",
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
        "selected_gain": selected_gain,
        "selection_metric": args.selection_metric,
        "anchor": args.anchor,
        "num_validation_records": len(validation_records),
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
                "selected_gain": selected_gain,
                "validation_metric": selected["metrics"][args.selection_metric],
                "test_metrics": test_steered.metrics,
                "delta_vs_baseline": metrics_payload["delta_vs_baseline"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
