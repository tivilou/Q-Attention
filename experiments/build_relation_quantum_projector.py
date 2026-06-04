from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.experiments import (  # noqa: E402
    ANCHOR_CHOICES,
    choose_device,
    collect_anchor_key_vectors,
    load_relation_run,
    make_relation_loader,
)
from q_attention.projectors import SpectralProjectorConfig  # noqa: E402
from q_attention.quantum import QuantumFeatureMapConfig, build_quantum_projector  # noqa: E402
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402

PROJECTOR_MODES = ("hard_topk", "high_pass", "band_pass", "soft_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a toy quantum-inspired projector from relation anchor keys.")
    parser.add_argument("--model_dir", required=True, help="Output directory produced by train_relation_baseline.py")
    parser.add_argument("--data_path", default=None, help="JSONL data used to collect anchor keys; defaults to baseline train_path")
    parser.add_argument("--output_path", default=None, help="Projector .pt path; defaults to <model_dir>/relation_quantum_projector.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--anchor", default="subject_object", choices=ANCHOR_CHOICES)
    parser.add_argument("--mode", default="hard_topk", choices=PROJECTOR_MODES)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--energy", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sharpness", type=float, default=8.0)
    parser.add_argument("--center", action="store_true", help="Mean-center anchor keys inside the quantum-weighted covariance")
    parser.add_argument("--max_vectors", type=int, default=None, help="Optional deterministic subsample limit before kernel construction")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed for max_vectors")
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--angle_scale", type=float, default=1.0)
    parser.add_argument("--feature_seed", type=int, default=17)
    parser.add_argument("--max_state_dim", type=int, default=1024)
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_metadata.json")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path) if args.output_path else model_dir / "relation_quantum_projector.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = load_relation_run(model_dir, device)
    data_path_value = args.data_path or artifacts.args.get("train_path")
    if data_path_value is None:
        raise ValueError("provide --data_path or train with a metrics.json containing args.train_path")
    data_path = resolve_project_path(data_path_value)

    records = load_relation_jsonl(data_path)
    loader = make_relation_loader(records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size, shuffle=False)
    collection = collect_anchor_key_vectors(
        artifacts.model,
        loader,
        device,
        artifacts.key_module_paths,
        anchor=args.anchor,
        max_vectors=args.max_vectors,
        seed=args.seed,
    )

    projector_config = SpectralProjectorConfig(
        mode=args.mode,
        rank=args.rank,
        energy=args.energy,
        threshold=args.threshold,
        sharpness=args.sharpness,
    )
    quantum_config = QuantumFeatureMapConfig(
        num_qubits=args.num_qubits,
        angle_scale=args.angle_scale,
        seed=args.feature_seed,
        max_state_dim=args.max_state_dim,
    )
    result = build_quantum_projector(
        collection.keys,
        quantum_config=quantum_config,
        projector_config=projector_config,
        center=args.center,
    )

    metadata: dict[str, Any] = {
        **result.metadata,
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "anchor": args.anchor,
        "key_module_paths": list(artifacts.key_module_paths),
        "key_collection": {
            "num_vectors": int(collection.keys.shape[0]),
            "sampled_from": collection.sampled_from,
            "layer_counts": collection.layer_counts,
            "num_batches": collection.num_batches,
        },
    }

    torch.save({"projector": result.projector.cpu(), "metadata": metadata}, output_path)
    metadata_path = metadata_path_for(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "projector_path": str(output_path),
                "metadata_path": str(metadata_path),
                "projector_shape": list(result.projector.shape),
                "num_key_vectors": int(collection.keys.shape[0]),
                "state_dim": int(result.features.shape[1]),
                "kernel_mean": metadata["kernel_mean"],
                "device": str(device),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()