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
    choose_device,
    collect_relation_key_samples,
    load_relation_run,
    make_relation_loader,
)
from q_attention.projectors import SpectralProjectorConfig  # noqa: E402
from q_attention.quantum import (  # noqa: E402
    QUANTUM_KERNEL_MODES,
    SupervisedQuantumProjectorConfig,
    build_supervised_quantum_projector,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402

PROJECTOR_MODES = ("hard_topk", "high_pass", "band_pass", "soft_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone label-aligned quantum projector for relation extraction.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", default=None, help="Training JSONL used for supervised quantum projector learning")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default="hard_topk", choices=PROJECTOR_MODES)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--energy", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sharpness", type=float, default=8.0)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--max_vectors", type=int, default=512)
    parser.add_argument("--collection_max_vectors", type=int, default=None, help="Pre-balance relation sample pool; defaults to 4x max_vectors")
    parser.add_argument("--seed", type=int, default=13, help="Relation-sample collection seed")
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--angle_scale", type=float, default=1.0)
    parser.add_argument("--feature_seed", type=int, default=17, help="Quantum circuit and encoding seed")
    parser.add_argument("--max_state_dim", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--training_steps", type=int, default=80)
    parser.add_argument("--kernel_mode", default="centered_fidelity", choices=QUANTUM_KERNEL_MODES)
    parser.add_argument("--kernel_temperature", type=float, default=1.0)
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
    output_path = Path(args.output_path) if args.output_path else model_dir / "relation_supervised_quantum_projector.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = load_relation_run(model_dir, device)
    data_path_value = args.data_path or artifacts.args.get("train_path")
    if data_path_value is None:
        raise ValueError("provide --data_path or train with args.train_path")
    data_path = resolve_project_path(data_path_value)
    records = load_relation_jsonl(data_path)
    loader = make_relation_loader(records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size, shuffle=False)
    collection_limit = args.collection_max_vectors or args.max_vectors * 4
    collection = collect_relation_key_samples(
        artifacts.model,
        loader,
        device,
        artifacts.key_module_paths,
        max_vectors=collection_limit,
        seed=args.seed,
    )

    projector_config = SpectralProjectorConfig(
        mode=args.mode,
        rank=args.rank,
        energy=args.energy,
        threshold=args.threshold,
        sharpness=args.sharpness,
    )
    quantum_config = SupervisedQuantumProjectorConfig(
        num_qubits=args.num_qubits,
        depth=args.depth,
        angle_scale=args.angle_scale,
        seed=args.feature_seed,
        max_state_dim=args.max_state_dim,
        max_train_samples=args.max_train_samples,
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        kernel_mode=args.kernel_mode,
        kernel_temperature=args.kernel_temperature,
    )
    result = build_supervised_quantum_projector(
        collection.keys,
        collection.relation_features,
        collection.labels,
        quantum_config=quantum_config,
        projector_config=projector_config,
        center=args.center,
        max_vectors=args.max_vectors,
    )

    metadata: dict[str, Any] = {
        **result.metadata,
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "key_module_paths": list(artifacts.key_module_paths),
        "key_collection": {
            "num_vectors": int(collection.keys.shape[0]),
            "collection_limit": collection_limit,
            "sampled_from": collection.sampled_from,
            "layer_counts": collection.layer_counts,
            "num_batches": collection.num_batches,
        },
    }
    torch.save(
        {
            "projector": result.projector.cpu(),
            "parameters": result.parameters,
            "metadata": metadata,
        },
        output_path,
    )
    metadata_path = metadata_path_for(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "projector_path": str(output_path),
                "metadata_path": str(metadata_path),
                "projector_shape": list(result.projector.shape),
                "num_key_vectors": int(collection.keys.shape[0]),
                "state_dim": int(result.states.shape[1]),
                "initial_alignment": metadata["training"]["initial_alignment"],
                "final_alignment": metadata["training"]["final_alignment"],
                "device": str(device),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
