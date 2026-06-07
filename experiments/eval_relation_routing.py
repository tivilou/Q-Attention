from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import EncoderKeySteeringAdapter, KeySteeringHookConfig  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    ANCHOR_CHOICES,
    anchor_mask_from_batch,
    build_anchor_projector,
    choose_device,
    collect_anchor_key_vectors,
    evaluate_relation_model,
    load_relation_run,
    make_relation_loader,
    move_batch,
)
from q_attention.metrics import classification_metrics  # noqa: E402
from q_attention.projectors import SpectralProjectorConfig  # noqa: E402
from q_attention.quantum import QUANTUM_KERNEL_MODES, QuantumFeatureMapConfig, build_quantum_projector  # noqa: E402
from q_attention.routing import (  # noqa: E402
    ROUTER_SCORE_MODES,
    ProjectorBank,
    RouterConfig,
    projector_prototype,
    route_projectors,
    stack_projector_bank,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate toy adaptive projector routing for relation key steering.")
    parser.add_argument("--model_dir", required=True, help="Output directory produced by train_relation_baseline.py")
    parser.add_argument("--projector_data_path", default=None, help="JSONL data used to collect anchor keys; defaults to baseline train_path")
    parser.add_argument("--eval_path", default=None, help="JSONL data used for evaluation; defaults to baseline valid_path")
    parser.add_argument("--output_dir", default=None, help="Routing output directory; defaults to <model_dir>/relation_routing_eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--anchor", default="subject_object", choices=ANCHOR_CHOICES)
    parser.add_argument("--gain", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--router_score_mode", default="hybrid", choices=ROUTER_SCORE_MODES)
    parser.add_argument("--router_prototype_weight", type=float, default=1.0)
    parser.add_argument("--router_energy_weight", type=float, default=1.0)
    parser.add_argument("--no_router_score_norm", action="store_true")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sharpness", type=float, default=8.0)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--max_vectors", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--angle_scale", type=float, default=1.25)
    parser.add_argument("--feature_seed", type=int, default=17)
    parser.add_argument("--max_state_dim", type=int, default=1024)
    parser.add_argument("--kernel_mode", default="centered_fidelity", choices=QUANTUM_KERNEL_MODES)
    parser.add_argument("--kernel_temperature", type=float, default=1.0)
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def metric_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] - baseline[key] for key in metrics.keys() & baseline.keys()}


def projector_stats(projector: torch.Tensor) -> dict[str, float]:
    return {
        "fro_norm": float(torch.linalg.norm(projector).item()),
        "trace": float(torch.trace(projector).item()),
        "max_abs": float(projector.abs().max().item()),
    }


def make_expert_bank(keys: torch.Tensor, args: argparse.Namespace) -> tuple[ProjectorBank, list[dict[str, Any]]]:
    quantum_config = QuantumFeatureMapConfig(
        num_qubits=args.num_qubits,
        angle_scale=args.angle_scale,
        seed=args.feature_seed,
        max_state_dim=args.max_state_dim,
        kernel_mode=args.kernel_mode,
        kernel_temperature=args.kernel_temperature,
    )
    expert_specs = [
        ("classical_hard_topk", "classical", SpectralProjectorConfig(mode="hard_topk", rank=args.rank)),
        (
            "classical_band_pass",
            "classical",
            SpectralProjectorConfig(mode="band_pass", threshold=args.threshold, sharpness=args.sharpness),
        ),
        ("quantum_hard_topk", "quantum", SpectralProjectorConfig(mode="hard_topk", rank=args.rank)),
        (
            "quantum_high_pass",
            "quantum",
            SpectralProjectorConfig(mode="high_pass", threshold=args.threshold, sharpness=args.sharpness),
        ),
    ]

    names: list[str] = []
    projectors: list[torch.Tensor] = []
    prototypes: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for name, family, config in expert_specs:
        if family == "classical":
            projector = build_anchor_projector(keys, config, center=args.center)
            expert_meta: dict[str, Any] = {"family": family, "filter": asdict(config)}
        else:
            result = build_quantum_projector(keys, quantum_config=quantum_config, projector_config=config, center=args.center)
            projector = result.projector
            expert_meta = {
                "family": family,
                "filter": asdict(config),
                "quantum_config": asdict(quantum_config),
                "state_dim": result.metadata["state_dim"],
                "kernel_mean": result.metadata["kernel_mean"],
                "kernel_trace": result.metadata["kernel_trace"],
            }
        names.append(name)
        projectors.append(projector.cpu())
        prototypes.append(projector_prototype(keys.cpu(), projector.cpu()))
        metadata.append({"name": name, **expert_meta, "projector_stats": projector_stats(projector.cpu())})

    return stack_projector_bank(names, projectors, prototypes), metadata


def batch_anchor_embeddings(model: torch.nn.Module, batch: dict[str, torch.Tensor], anchor: str) -> torch.Tensor:
    embeddings = model.encoder.token_embedding(batch["input_ids"])
    mask = anchor_mask_from_batch(batch, anchor).to(device=embeddings.device, dtype=embeddings.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.sum(embeddings * mask.unsqueeze(-1), dim=1) / denom


def summarize_routing(weights: torch.Tensor, entropy: torch.Tensor, names: tuple[str, ...]) -> dict[str, Any]:
    mean_weights = weights.mean(dim=0)
    dominant = torch.argmax(weights, dim=-1)
    counts = torch.bincount(dominant, minlength=len(names))
    return {
        "mean_entropy": float(entropy.mean().item()),
        "max_entropy": float(entropy.max().item()),
        "mean_weights": {name: float(mean_weights[idx].item()) for idx, name in enumerate(names)},
        "dominant_counts": {name: int(counts[idx].item()) for idx, name in enumerate(names)},
    }


def evaluate_routed_model(
    artifacts: Any,
    loader: Any,
    device: torch.device,
    bank: ProjectorBank,
    *,
    anchor: str,
    gain: float,
    router_config: RouterConfig,
) -> tuple[dict[str, float], list[int], list[int], list[dict[str, Any]], dict[str, Any]]:
    model = artifacts.model
    model.eval()
    adapter = EncoderKeySteeringAdapter(model, artifacts.key_module_paths)
    predictions: list[int] = []
    labels: list[int] = []
    routing_rows: list[dict[str, Any]] = []
    all_weights: list[torch.Tensor] = []
    all_entropy: list[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0
    item_index = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            anchors = batch_anchor_embeddings(model, batch, anchor)
            routed = route_projectors(anchors, bank, router_config)
            mask = anchor_mask_from_batch(batch, anchor)
            hook_config = KeySteeringHookConfig(projector=routed.projectors, mask=mask, gain=gain)
            with adapter.steering(hook_config):
                logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])

            loss = F.cross_entropy(logits, batch["labels"])
            batch_size = int(batch["labels"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            batch_predictions = torch.argmax(logits, dim=-1).detach().cpu()
            batch_labels = batch["labels"].detach().cpu()
            predictions.extend(batch_predictions.tolist())
            labels.extend(batch_labels.tolist())

            weights_cpu = routed.weights.detach().cpu()
            entropy_cpu = routed.entropy.detach().cpu()
            all_weights.append(weights_cpu)
            all_entropy.append(entropy_cpu)
            dominant = torch.argmax(weights_cpu, dim=-1)
            for row in range(batch_size):
                routing_rows.append(
                    {
                        "index": item_index,
                        "prediction_id": int(batch_predictions[row].item()),
                        "gold_id": int(batch_labels[row].item()),
                        "entropy": float(entropy_cpu[row].item()),
                        "dominant_expert": bank.names[int(dominant[row].item())],
                        "weights": {name: float(weights_cpu[row, idx].item()) for idx, name in enumerate(bank.names)},
                    }
                )
                item_index += 1

    metrics = classification_metrics(predictions, labels, len(artifacts.label_to_id))
    metrics["loss"] = total_loss / max(total_items, 1)
    weights = torch.cat(all_weights, dim=0)
    entropy = torch.cat(all_entropy, dim=0)
    routing_summary = summarize_routing(weights, entropy, bank.names)
    return metrics, predictions, labels, routing_rows, routing_summary


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "relation_routing_eval"
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
    bank, expert_metadata = make_expert_bank(collection.keys, args)
    router_config = RouterConfig(
        temperature=args.temperature,
        score_mode=args.router_score_mode,
        prototype_weight=args.router_prototype_weight,
        energy_weight=args.router_energy_weight,
        normalize_scores=not args.no_router_score_norm,
    )

    eval_records = load_relation_jsonl(eval_path)
    eval_loader = make_relation_loader(eval_records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size, shuffle=False)
    baseline = evaluate_relation_model(artifacts.model, eval_loader, device, len(artifacts.label_to_id))
    routed_metrics, predictions, labels, routing_rows, routing_summary = evaluate_routed_model(
        artifacts,
        eval_loader,
        device,
        bank,
        anchor=args.anchor,
        gain=args.gain,
        router_config=router_config,
    )

    metrics_payload = {
        "baseline": baseline.metrics,
        "routed": routed_metrics,
        "delta_vs_baseline": metric_delta(routed_metrics, baseline.metrics),
        "routing_summary": routing_summary,
    }
    run_info = {
        "model_dir": str(model_dir),
        "projector_data_path": str(projector_data_path),
        "eval_path": str(eval_path),
        "output_dir": str(output_dir),
        "anchor": args.anchor,
        "gain": args.gain,
        "temperature": args.temperature,
        "router_config": asdict(router_config),
        "key_collection": {
            "num_vectors": int(collection.keys.shape[0]),
            "sampled_from": collection.sampled_from,
            "layer_counts": collection.layer_counts,
            "num_batches": collection.num_batches,
        },
        "experts": expert_metadata,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "routing.jsonl").open("w", encoding="utf-8") as handle:
        for row in routing_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(eval_records):
            item = {
                "index": idx,
                "tokens": list(record.tokens),
                "subject": list(record.subject),
                "object": list(record.object),
                "gold_id": int(labels[idx]),
                "gold_label": artifacts.id_to_label[int(labels[idx])],
                "prediction_id": int(predictions[idx]),
                "prediction_label": artifacts.id_to_label[int(predictions[idx])],
                "metadata": dict(record.metadata),
            }
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    print(json.dumps({"output_dir": str(output_dir), **metrics_payload}, sort_keys=True))


if __name__ == "__main__":
    main()
