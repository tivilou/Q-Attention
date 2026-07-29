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

from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    choose_device,
    diagnose_relation_expert_direction_alignment,
    load_relation_run,
    make_relation_loader,
)
from q_attention.plugins import load_relation_attention_score_kernel_checkpoint  # noqa: E402
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose task alignment of routed attention-score experts."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--valid_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _mean(payload: dict[str, Any], name: str) -> float | None:
    value = payload[name]["mean"]
    return None if value is None else float(value)


def _render_report(payload: dict[str, Any]) -> str:
    rows = [
        "# Expert Direction Alignment",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Routing conditioning: `{payload['kernel_metadata']['expert_router']['config']['routing_conditioning']}`",
        "",
        "| Split | Layer | Head | Role | Routed | Uniform | Gain | Oracle | Regret | Top match |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, diagnostics in payload["splits"].items():
        for layer in diagnostics["layers"]:
            for head in layer["heads"]:
                for role, role_payload in head["query_roles"].items():
                    values = [
                        _mean(role_payload, "routed_alignment"),
                        _mean(role_payload, "uniform_alignment"),
                        _mean(role_payload, "routing_gain_over_uniform"),
                        _mean(role_payload, "oracle_alignment"),
                        _mean(role_payload, "oracle_regret"),
                        _mean(role_payload, "top_expert_match"),
                    ]
                    rendered = ["n/a" if value is None else f"{value:.6f}" for value in values]
                    rows.append(
                        f"| {split} | {layer['layer_index']} | {head['head_index']} | {role} | "
                        + " | ".join(rendered)
                        + " |"
                    )
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(Path(args.model_dir), device)
    kernel, checkpoint_metadata = load_relation_attention_score_kernel_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    if kernel.expert_router is None:
        raise ValueError("checkpoint must contain an expert router")
    expected_paths = checkpoint_metadata.get("score_module_paths")
    if expected_paths is not None and tuple(expected_paths) != tuple(
        artifacts.model.score_module_paths
    ):
        raise ValueError("checkpoint module paths do not match the base model")
    kernel.to(device).eval()
    adapter = AttentionScoreKernelAdapter(
        artifacts.model,
        artifacts.model.score_module_paths,
        kernel,
    )
    splits: dict[str, Any] = {}
    for split, data_path in {
        "train": Path(args.train_path),
        "valid": Path(args.valid_path),
    }.items():
        loader = make_relation_loader(
            load_relation_jsonl(data_path),
            artifacts.vocab,
            artifacts.label_to_id,
            batch_size=args.batch_size,
            shuffle=False,
        )
        splits[split] = diagnose_relation_expert_direction_alignment(
            artifacts.model,
            loader,
            device,
            adapter=adapter,
        )
    payload = {
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_metadata": checkpoint_metadata,
        "kernel_metadata": kernel.metadata(),
        "splits": splits,
    }
    (output_dir / "expert_direction_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "expert_direction_diagnostics.md").write_text(
        _render_report(payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "splits": list(splits),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
