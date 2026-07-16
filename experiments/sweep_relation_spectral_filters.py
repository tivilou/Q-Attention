from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.experiments import (  # noqa: E402
    ANCHOR_CHOICES,
    build_anchor_projector,
    choose_device,
    collect_anchor_key_vectors,
    evaluate_relation_model,
    load_relation_run,
    make_relation_loader,
)
from q_attention.projectors import SpectralProjectorConfig, cross_covariance, spectral_filter_diagnostics  # noqa: E402
from q_attention.quantum import QUANTUM_KERNEL_MODES, QuantumFeatureMapConfig, build_quantum_projector  # noqa: E402
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402

PROJECTOR_FAMILIES = ("classical", "quantum")
PROJECTOR_MODES = ("hard_topk", "high_pass", "band_pass", "soft_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep toy spectral filters for relation key steering.")
    parser.add_argument("--model_dir", required=True, help="Output directory produced by train_relation_baseline.py")
    parser.add_argument("--projector_data_path", default=None, help="JSONL data used to collect anchor keys; defaults to baseline train_path")
    parser.add_argument("--eval_path", default=None, help="Validation JSONL used only to select the best filter")
    parser.add_argument("--test_path", default=None, help="Optional held-out JSONL used once after validation selection")
    parser.add_argument("--output_dir", default=None, help="Sweep output directory; defaults to <model_dir>/spectral_filter_sweep")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--anchor", default="subject_object", choices=ANCHOR_CHOICES)
    parser.add_argument("--families", default="classical,quantum", help="Comma-separated projector families")
    parser.add_argument("--modes", default="hard_topk,high_pass,band_pass,soft_energy", help="Comma-separated filter modes")
    parser.add_argument("--ranks", default="2,4", help="Comma-separated ranks for hard_topk")
    parser.add_argument("--energy", type=float, default=0.9)
    parser.add_argument("--thresholds", default="0.5", help="Comma-separated thresholds for smooth filters")
    parser.add_argument("--sharpnesses", default="8.0", help="Comma-separated sharpness values for smooth filters")
    parser.add_argument("--gains", default="0.25", help="Comma-separated steering gains")
    parser.add_argument("--center", action="store_true", help="Mean-center anchor keys inside covariance/projector construction")
    parser.add_argument("--max_vectors", type=int, default=None, help="Optional deterministic subsample limit before projector construction")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed for max_vectors")
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--angle_scale", type=float, default=1.25)
    parser.add_argument("--feature_seed", type=int, default=17)
    parser.add_argument("--max_state_dim", type=int, default=1024)
    parser.add_argument("--kernel_mode", default="centered_fidelity", choices=QUANTUM_KERNEL_MODES)
    parser.add_argument("--kernel_temperature", type=float, default=1.0)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def metric_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] - baseline[key] for key in metrics.keys() & baseline.keys()}


def filter_grid(args: argparse.Namespace) -> Iterable[SpectralProjectorConfig]:
    modes = parse_csv(args.modes)
    invalid_modes = sorted(set(modes) - set(PROJECTOR_MODES))
    if invalid_modes:
        raise ValueError(f"unknown filter modes: {invalid_modes}")
    ranks = parse_int_csv(args.ranks)
    thresholds = parse_float_csv(args.thresholds)
    sharpnesses = parse_float_csv(args.sharpnesses)

    for mode in modes:
        if mode == "hard_topk":
            for rank in ranks:
                yield SpectralProjectorConfig(mode=mode, rank=rank, energy=args.energy)
        elif mode in {"high_pass", "band_pass"}:
            for threshold, sharpness in product(thresholds, sharpnesses):
                yield SpectralProjectorConfig(mode=mode, energy=args.energy, threshold=threshold, sharpness=sharpness)
        elif mode == "soft_energy":
            yield SpectralProjectorConfig(mode=mode, energy=args.energy)


def classical_singular_values(keys: torch.Tensor, *, center: bool) -> torch.Tensor:
    source = keys.float()
    if center:
        source = source - source.mean(dim=0, keepdim=True)
    omega = cross_covariance(source, source)
    return torch.linalg.svdvals(omega)


def build_sweep_projector(
    family: str,
    keys: torch.Tensor,
    config: SpectralProjectorConfig,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if family == "classical":
        singular_values = classical_singular_values(keys, center=args.center)
        projector = build_anchor_projector(keys, config, center=args.center)
        diagnostics = spectral_filter_diagnostics(singular_values, config)
        metadata = {
            "projector_family": "classical",
            "filter_diagnostics": diagnostics,
            "projector_config": asdict(config),
            "center": args.center,
        }
        return projector, metadata

    if family == "quantum":
        quantum_config = QuantumFeatureMapConfig(
            num_qubits=args.num_qubits,
            angle_scale=args.angle_scale,
            seed=args.feature_seed,
            max_state_dim=args.max_state_dim,
            kernel_mode=args.kernel_mode,
            kernel_temperature=args.kernel_temperature,
        )
        result = build_quantum_projector(
            keys,
            quantum_config=quantum_config,
            projector_config=config,
            center=args.center,
        )
        return result.projector, result.metadata

    raise ValueError(f"unknown projector family: {family}")


def projector_stats(projector: torch.Tensor) -> dict[str, float]:
    return {
        "fro_norm": float(torch.linalg.norm(projector).item()),
        "trace": float(torch.trace(projector).item()),
        "max_abs": float(projector.abs().max().item()),
    }


def select_best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=lambda row: row["metrics"].get("macro_f1", 0.0)) if rows else None


def write_test_predictions(
    path: Path,
    records: list[Any],
    labels: list[int],
    baseline_predictions: list[int],
    steered_predictions: list[int],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "gold_id": int(labels[index]),
                        "baseline_prediction_id": int(baseline_predictions[index]),
                        "steered_prediction_id": int(steered_predictions[index]),
                        "metadata": dict(record.metadata),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    families = parse_csv(args.families)
    invalid_families = sorted(set(families) - set(PROJECTOR_FAMILIES))
    if invalid_families:
        raise ValueError(f"unknown projector families: {invalid_families}")
    gains = parse_float_csv(args.gains)

    device = choose_device(args.device)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "spectral_filter_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_relation_run(model_dir, device)
    projector_data_value = args.projector_data_path or artifacts.args.get("train_path")
    eval_value = args.eval_path or artifacts.args.get("valid_path")
    if projector_data_value is None or eval_value is None:
        raise ValueError("provide --projector_data_path/--eval_path or train with metrics args paths")
    projector_data_path = resolve_project_path(projector_data_value)
    eval_path = resolve_project_path(eval_value)

    projector_records = load_relation_jsonl(projector_data_path)
    projector_loader = make_relation_loader(projector_records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size, shuffle=False)
    collection = collect_anchor_key_vectors(
        artifacts.model,
        projector_loader,
        device,
        artifacts.key_module_paths,
        anchor=args.anchor,
        max_vectors=args.max_vectors,
        seed=args.seed,
    )

    eval_records = load_relation_jsonl(eval_path)
    eval_loader = make_relation_loader(eval_records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size, shuffle=False)
    baseline = evaluate_relation_model(artifacts.model, eval_loader, device, len(artifacts.label_to_id))

    rows: list[dict[str, Any]] = []
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for family in families:
            for config in filter_grid(args):
                projector, metadata = build_sweep_projector(family, collection.keys, config, args)
                for gain in gains:
                    result = evaluate_relation_model(
                        artifacts.model,
                        eval_loader,
                        device,
                        len(artifacts.label_to_id),
                        projector=projector,
                        key_module_paths=artifacts.key_module_paths,
                        gain=gain,
                        anchor=args.anchor,
                    )
                    row: dict[str, Any] = {
                        "family": family,
                        "gain": gain,
                        "filter": asdict(config),
                        "metrics": result.metrics,
                        "delta_vs_baseline": metric_delta(result.metrics, baseline.metrics),
                        "projector_stats": projector_stats(projector),
                        "filter_diagnostics": metadata.get("filter_diagnostics", {}),
                        "quantum": {
                            "state_dim": metadata.get("state_dim"),
                            "kernel_mean": metadata.get("kernel_mean"),
                            "kernel_trace": metadata.get("kernel_trace"),
                            "quantum_config": metadata.get("quantum_config"),
                        },
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True))

    best = select_best_row(rows)
    best_on_test: dict[str, Any] | None = None
    test_baseline_metrics: dict[str, float] | None = None
    test_path: Path | None = None
    if args.test_path is not None:
        if best is None:
            raise ValueError("cannot evaluate a held-out test split because the sweep produced no candidates")
        test_path = resolve_project_path(args.test_path)
        test_records = load_relation_jsonl(test_path)
        test_loader = make_relation_loader(
            test_records,
            artifacts.vocab,
            artifacts.label_to_id,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_baseline = evaluate_relation_model(artifacts.model, test_loader, device, len(artifacts.label_to_id))
        selected_config = SpectralProjectorConfig(**best["filter"])
        selected_projector, _ = build_sweep_projector(best["family"], collection.keys, selected_config, args)
        test_result = evaluate_relation_model(
            artifacts.model,
            test_loader,
            device,
            len(artifacts.label_to_id),
            projector=selected_projector,
            key_module_paths=artifacts.key_module_paths,
            gain=float(best["gain"]),
            anchor=args.anchor,
        )
        test_baseline_metrics = test_baseline.metrics
        best_on_test = {
            "family": best["family"],
            "gain": best["gain"],
            "filter": best["filter"],
            "metrics": test_result.metrics,
            "delta_vs_baseline": metric_delta(test_result.metrics, test_baseline.metrics),
            "selected_on": str(eval_path),
            "selection_metrics": best["metrics"],
            "projector_stats": projector_stats(selected_projector),
            "filter_diagnostics": best.get("filter_diagnostics", {}),
            "quantum": best.get("quantum", {}),
        }
        write_test_predictions(
            output_dir / "test_predictions.jsonl",
            test_records,
            test_result.labels,
            test_baseline.predictions,
            test_result.predictions,
        )

    summary = {
        "model_dir": str(model_dir),
        "projector_data_path": str(projector_data_path),
        "eval_path": str(eval_path),
        "selection_path": str(eval_path),
        "test_path": None if test_path is None else str(test_path),
        "output_dir": str(output_dir),
        "baseline": baseline.metrics,
        "selection_baseline": baseline.metrics,
        "test_baseline": test_baseline_metrics,
        "num_rows": len(rows),
        "anchor": args.anchor,
        "key_collection": {
            "num_vectors": int(collection.keys.shape[0]),
            "sampled_from": collection.sampled_from,
            "layer_counts": collection.layer_counts,
            "num_batches": collection.num_batches,
        },
        "best_by_macro_f1": best,
        "best_on_test": best_on_test,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_rows": len(rows), "best_by_macro_f1": best}, sort_keys=True))


if __name__ == "__main__":
    main()
