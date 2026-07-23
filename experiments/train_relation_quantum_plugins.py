from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import QuantumPluginSteeringAdapter  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    ANCHOR_CHOICES,
    choose_device,
    evaluate_relation_quantum_plugins,
    load_relation_run,
    make_relation_loader,
    move_batch,
    quantum_plugin_hook_config,
)
from q_attention.plugins import (  # noqa: E402
    PLUGIN_NAMES,
    build_quantum_steering,
    normalize_plugin_names,
    save_quantum_steering_checkpoint,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train composable quantum steering plugins on a frozen relation model."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--valid_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--plugins",
        default="headwise_projector",
        help=f"Comma-separated subset of: {','.join(PLUGIN_NAMES)}",
    )
    parser.add_argument("--steering_anchor", default="all_tokens", choices=ANCHOR_CHOICES)
    parser.add_argument("--operator_reduction", default="mean", choices=["sum", "mean"])
    parser.add_argument("--identity_gain", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_path(value: str | None, fallback: Any, name: str) -> Path:
    selected = value or fallback
    if selected is None:
        raise ValueError(f"{name} must be provided or recorded by the baseline")
    path = Path(str(selected))
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    plugin_names = normalize_plugin_names(args.plugins)
    if not plugin_names:
        raise ValueError("training requires at least one quantum plugin")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")

    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(Path(args.model_dir), device)
    train_path = resolve_data_path(args.train_path, artifacts.args.get("train_path"), "train_path")
    valid_path = resolve_data_path(args.valid_path, artifacts.args.get("valid_path"), "valid_path")
    train_loader = make_relation_loader(
        load_relation_jsonl(train_path),
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = make_relation_loader(
        load_relation_jsonl(valid_path),
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = artifacts.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_config = model.config
    steering = build_quantum_steering(
        plugin_names,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        head_dim=model_config.dim // model_config.num_heads,
        operator_reduction=args.operator_reduction,
        identity_gain=args.identity_gain,
    ).to(device)
    adapter = QuantumPluginSteeringAdapter(model, artifacts.key_module_paths, steering)
    optimizer = torch.optim.AdamW(steering.parameters(), lr=args.lr)

    baseline_valid = evaluate_relation_quantum_plugins(
        model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        adapter=None,
        steering_anchor=args.steering_anchor,
    )
    checkpoint_path = output_dir / "quantum_plugins.pt"
    history: list[dict[str, Any]] = []
    best_valid: dict[str, float] | None = None
    best_score = (float("-inf"), float("-inf"))
    for epoch in range(1, args.epochs + 1):
        steering.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with adapter.steering(
                quantum_plugin_hook_config(batch, args.steering_anchor)
            ):
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["subject_mask"],
                    batch["object_mask"],
                )
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * batch["labels"].shape[0]
            total_items += batch["labels"].shape[0]

        valid_metrics = evaluate_relation_quantum_plugins(
            model,
            valid_loader,
            device,
            len(artifacts.label_to_id),
            adapter=adapter,
            steering_anchor=args.steering_anchor,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            "valid": valid_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        score = (valid_metrics["macro_f1"], -valid_metrics["loss"])
        if score > best_score:
            best_score = score
            best_valid = valid_metrics
            save_quantum_steering_checkpoint(
                checkpoint_path,
                steering,
                extra_metadata={
                    "base_model_dir": str(Path(args.model_dir)),
                    "key_module_paths": list(artifacts.key_module_paths),
                    "steering_anchor": args.steering_anchor,
                    "train_path": str(train_path),
                    "valid_path": str(valid_path),
                    "seed": args.seed,
                },
            )

    payload = {
        "args": vars(args),
        "plugin_metadata": steering.metadata(),
        "trainable_parameters": sum(
            parameter.numel() for parameter in steering.parameters() if parameter.requires_grad
        ),
        "base_model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "baseline_valid": baseline_valid,
        "best_valid": best_valid,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint": str(checkpoint_path),
                "baseline_valid": baseline_valid,
                "best_valid": best_valid,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
