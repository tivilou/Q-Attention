from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import QuantumPluginSteeringAdapter  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    choose_device,
    evaluate_relation_quantum_plugins,
    load_relation_run,
    make_relation_loader,
)
from q_attention.plugins import load_quantum_steering_checkpoint  # noqa: E402
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen relation model with a quantum plugin checkpoint."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def metric_delta(steered: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: steered[key] - baseline[key] for key in steered.keys() & baseline.keys()}


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(Path(args.model_dir), device)
    steering, checkpoint_metadata = load_quantum_steering_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    steering.to(device)
    expected_paths = checkpoint_metadata.get("key_module_paths")
    if expected_paths is not None and tuple(expected_paths) != tuple(artifacts.key_module_paths):
        raise ValueError("plugin checkpoint key-module paths do not match the base model")
    steering_anchor = str(checkpoint_metadata.get("steering_anchor", "all_tokens"))
    loader = make_relation_loader(
        load_relation_jsonl(Path(args.data_path)),
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )
    adapter = QuantumPluginSteeringAdapter(
        artifacts.model,
        artifacts.key_module_paths,
        steering,
    )
    baseline = evaluate_relation_quantum_plugins(
        artifacts.model,
        loader,
        device,
        len(artifacts.label_to_id),
        adapter=None,
        steering_anchor=steering_anchor,
    )
    steered = evaluate_relation_quantum_plugins(
        artifacts.model,
        loader,
        device,
        len(artifacts.label_to_id),
        adapter=adapter,
        steering_anchor=steering_anchor,
    )
    payload = {
        "baseline": baseline,
        "steered": steered,
        "delta_vs_baseline": metric_delta(steered, baseline),
        "plugin_metadata": steering.metadata(),
        "checkpoint_metadata": checkpoint_metadata,
        "data_path": args.data_path,
        "device": str(device),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
