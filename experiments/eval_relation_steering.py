from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.experiments import (  # noqa: E402
    ANCHOR_CHOICES,
    choose_device,
    evaluate_relation_model,
    load_projector,
    load_relation_run,
    make_relation_loader,
    projector_shape_summary,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate relation extraction with spectral key steering.")
    parser.add_argument("--model_dir", required=True, help="Output directory produced by train_relation_baseline.py")
    parser.add_argument("--projector_path", default=None, help="Projector path; defaults to <model_dir>/relation_projector.pt")
    parser.add_argument("--data_path", default=None, help="Evaluation JSONL path; defaults to baseline valid_path")
    parser.add_argument("--output_dir", default=None, help="Directory for metrics and predictions")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--anchor", default="subject_object", choices=ANCHOR_CHOICES)
    parser.add_argument("--skip_baseline", action="store_true", help="Only run the steered model")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def metric_delta(steered: dict[str, float], baseline: dict[str, float] | None) -> dict[str, float]:
    if baseline is None:
        return {}
    return {key: steered[key] - baseline[key] for key in steered.keys() & baseline.keys()}


def label_name(id_to_label: dict[int, str], label_id: int | None) -> str | None:
    if label_id is None:
        return None
    return id_to_label.get(int(label_id), str(label_id))


def write_predictions(
    path: Path,
    records: list[Any],
    id_to_label: dict[int, str],
    steered_predictions: list[int],
    labels: list[int],
    baseline_predictions: list[int] | None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            baseline_id = baseline_predictions[idx] if baseline_predictions is not None else None
            item = {
                "index": idx,
                "tokens": list(record.tokens),
                "subject": list(record.subject),
                "object": list(record.object),
                "gold_id": int(labels[idx]),
                "gold_label": label_name(id_to_label, labels[idx]),
                "steered_prediction_id": int(steered_predictions[idx]),
                "steered_prediction_label": label_name(id_to_label, steered_predictions[idx]),
                "baseline_prediction_id": None if baseline_id is None else int(baseline_id),
                "baseline_prediction_label": label_name(id_to_label, baseline_id),
                "metadata": dict(record.metadata),
            }
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "relation_steering_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_relation_run(model_dir, device)
    projector_path = Path(args.projector_path) if args.projector_path else model_dir / "relation_projector.pt"
    projector, projector_metadata = load_projector(projector_path, device)

    data_path_value = args.data_path or artifacts.args.get("valid_path")
    if data_path_value is None:
        raise ValueError("provide --data_path or train with a metrics.json containing args.valid_path")
    data_path = resolve_project_path(data_path_value)

    records = load_relation_jsonl(data_path)
    loader = make_relation_loader(records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size, shuffle=False)

    baseline_result = None
    if not args.skip_baseline:
        baseline_result = evaluate_relation_model(artifacts.model, loader, device, len(artifacts.label_to_id))

    steered_result = evaluate_relation_model(
        artifacts.model,
        loader,
        device,
        len(artifacts.label_to_id),
        projector=projector,
        key_module_paths=artifacts.key_module_paths,
        gain=args.gain,
        anchor=args.anchor,
    )

    baseline_metrics = baseline_result.metrics if baseline_result is not None else None
    metrics_payload: dict[str, Any] = {
        "steered": steered_result.metrics,
        "baseline": baseline_metrics,
        "delta_vs_baseline": metric_delta(steered_result.metrics, baseline_metrics),
    }
    run_info: dict[str, Any] = {
        "model_dir": str(model_dir),
        "projector_path": str(projector_path),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "gain": args.gain,
        "anchor": args.anchor,
        "num_records": len(records),
        "key_module_paths": list(artifacts.key_module_paths),
        "projector_shape": projector_shape_summary(projector),
        "projector_metadata": dict(projector_metadata),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")
    write_predictions(
        output_dir / "predictions.jsonl",
        records,
        artifacts.id_to_label,
        steered_result.predictions,
        steered_result.labels,
        baseline_result.predictions if baseline_result is not None else None,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "steered": steered_result.metrics,
                "baseline": baseline_metrics,
                "delta_vs_baseline": metrics_payload["delta_vs_baseline"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
