from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    GradientNormTracker,
    choose_device,
    diagnose_relation_expert_routing,
    diagnose_relation_routing_task_alignment,
    evaluate_relation_attention_score_kernel,
    expert_routing_objective,
    load_relation_run,
    make_relation_loader,
    move_batch,
)
from q_attention.plugins import (  # noqa: E402
    EXPERT_DIRECTION_MODES,
    EXPERT_ROUTER_TYPES,
    ROUTER_CONDITIONING_CHOICES,
    ROUTER_RESIDUAL_REFERENCES,
    RelationExpertRouterConfig,
    build_relation_expert_router,
    load_relation_attention_score_kernel_checkpoint,
    save_relation_attention_score_kernel_checkpoint,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train identifiable relation-conditioned observable-expert routing."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--core_checkpoint", required=True)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--valid_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--router_type",
        choices=EXPERT_ROUTER_TYPES,
        default="quantum",
    )
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--router_qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--angle_scale", type=float, default=1.0)
    parser.add_argument("--max_gain", type=float, default=0.5)
    parser.add_argument("--initial_gain", type=float, default=0.05)
    parser.add_argument(
        "--residual_reference",
        choices=ROUTER_RESIDUAL_REFERENCES,
        default="core",
    )
    parser.add_argument("--normalize_routed_energy", action="store_true")
    parser.add_argument(
        "--routing_conditioning",
        choices=ROUTER_CONDITIONING_CHOICES,
        default="relation",
    )
    parser.add_argument("--trainable_router_projection", action="store_true")
    parser.add_argument("--information_weight", type=float, default=0.1)
    parser.add_argument("--utility_alignment_weight", type=float, default=0.0)
    parser.add_argument(
        "--direction_mode",
        choices=EXPERT_DIRECTION_MODES,
        default="fixed",
    )
    parser.add_argument("--direction_diversity_weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
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
        raise ValueError(f"{name} must be provided or recorded by the base run")
    path = Path(str(selected))
    return path if path.is_absolute() else ROOT / path


def routing_selection_score(
    learned: dict[str, float],
    uniform: dict[str, float],
    diagnostics: dict[str, Any],
) -> tuple[int, int, float, float]:
    margin = diagnostics["learned_vs_uniform"]["correct_margin_delta"]["mean"]
    margin_value = float(margin) if margin is not None else float("-inf")
    task_pass = learned["loss"] < uniform["loss"] and margin_value > 0.0
    return (
        int(diagnostics["mechanism_pass"]),
        int(task_pass),
        uniform["loss"] - learned["loss"],
        margin_value,
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    if args.information_weight < 0:
        raise ValueError("information_weight must be non-negative")
    if args.direction_diversity_weight < 0:
        raise ValueError("direction_diversity_weight must be non-negative")
    if args.utility_alignment_weight < 0:
        raise ValueError("utility_alignment_weight must be non-negative")
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(Path(args.model_dir), device)
    train_path = resolve_data_path(
        args.train_path,
        artifacts.args.get("train_path"),
        "train_path",
    )
    valid_path = resolve_data_path(
        args.valid_path,
        artifacts.args.get("valid_path"),
        "valid_path",
    )
    train_records = load_relation_jsonl(train_path)
    valid_records = load_relation_jsonl(valid_path)
    train_loader = make_relation_loader(
        train_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = make_relation_loader(
        valid_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = artifacts.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel, core_metadata = load_relation_attention_score_kernel_checkpoint(
        args.core_checkpoint,
        map_location=device,
    )
    if kernel.expert_router is not None:
        raise ValueError("core_checkpoint must not already contain an expert router")
    if kernel.config.score_readout != "observable":
        raise ValueError("expert routing requires an observable score core")
    if tuple(core_metadata.get("score_module_paths", ())) not in {
        (),
        tuple(model.score_module_paths),
    }:
        raise ValueError("core checkpoint module paths do not match the base model")
    kernel.to(device).eval()
    core_adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    core_valid = evaluate_relation_attention_score_kernel(
        model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        adapter=core_adapter,
    )

    router = build_relation_expert_router(
        args.router_type,
        RelationExpertRouterConfig(
            num_layers=kernel.config.num_layers,
            num_heads=kernel.config.num_heads,
            head_dim=kernel.config.head_dim,
            num_observables=2 * kernel.config.num_qubits,
            num_experts=args.num_experts,
            router_qubits=args.router_qubits,
            depth=args.depth,
            angle_scale=args.angle_scale,
            max_gain=args.max_gain,
            initial_gain=args.initial_gain,
            residual_reference=args.residual_reference,
            normalize_routed_energy=args.normalize_routed_energy,
            routing_conditioning=args.routing_conditioning,
            trainable_projection=args.trainable_router_projection,
            direction_mode=args.direction_mode,
            seed=args.seed + 6007,
        ),
    ).to(device)
    kernel.attach_expert_router(router)
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)
    gradient_tracker = GradientNormTracker(router.named_parameters())

    initial_learned = evaluate_relation_attention_score_kernel(
        model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        adapter=adapter,
        routing_mode="learned",
    )
    initial_uniform = evaluate_relation_attention_score_kernel(
        model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        adapter=adapter,
        routing_mode="uniform",
    )
    initial_diagnostics = diagnose_relation_expert_routing(
        model,
        valid_loader,
        device,
        adapter=adapter,
    )

    checkpoint = output_dir / "expert_router.pt"
    final_checkpoint = output_dir / "expert_router_final.pt"
    checkpoint_metadata = {
        "base_model_dir": str(Path(args.model_dir)),
        "core_checkpoint": str(Path(args.core_checkpoint)),
        "core_checkpoint_metadata": core_metadata,
        "score_module_paths": list(model.score_module_paths),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "seed": args.seed,
        "residual_reference": args.residual_reference,
        "normalize_routed_energy": args.normalize_routed_energy,
        "routing_conditioning": args.routing_conditioning,
        "trainable_router_projection": args.trainable_router_projection,
        "direction_mode": args.direction_mode,
        "direction_diversity_weight": args.direction_diversity_weight,
        "utility_alignment_weight": args.utility_alignment_weight,
    }
    history: list[dict[str, Any]] = []
    best_score = (-1, -1, float("-inf"), float("-inf"))
    best_epoch: int | None = None
    best_learned: dict[str, float] | None = None
    best_uniform: dict[str, float] | None = None
    best_diagnostics: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        router.train()
        totals: dict[str, float] = {"objective": 0.0}
        total_items = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            objective, components = expert_routing_objective(
                model,
                batch,
                adapter,
                information_weight=args.information_weight,
                direction_diversity_weight=args.direction_diversity_weight,
                utility_alignment_weight=args.utility_alignment_weight,
            )
            objective.backward()
            gradient_tracker.update()
            optimizer.step()
            items = batch["labels"].shape[0]
            totals["objective"] += float(objective.detach().item()) * items
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * items
            total_items += items

        router.eval()
        learned = evaluate_relation_attention_score_kernel(
            model,
            valid_loader,
            device,
            len(artifacts.label_to_id),
            adapter=adapter,
            routing_mode="learned",
        )
        uniform = evaluate_relation_attention_score_kernel(
            model,
            valid_loader,
            device,
            len(artifacts.label_to_id),
            adapter=adapter,
            routing_mode="uniform",
        )
        diagnostics = diagnose_relation_expert_routing(
            model,
            valid_loader,
            device,
            adapter=adapter,
        )
        score = routing_selection_score(learned, uniform, diagnostics)
        row = {
            "epoch": epoch,
            "train": {
                name: value / max(total_items, 1)
                for name, value in totals.items()
            },
            "learned": learned,
            "uniform": uniform,
            "diagnostics": diagnostics,
            "routing_pass": bool(score[0] and score[1]),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_learned = learned
            best_uniform = uniform
            best_diagnostics = diagnostics
            save_relation_attention_score_kernel_checkpoint(
                checkpoint,
                kernel,
                extra_metadata={**checkpoint_metadata, "epoch": epoch, "kind": "best"},
            )

    save_relation_attention_score_kernel_checkpoint(
        final_checkpoint,
        kernel,
        extra_metadata={
            **checkpoint_metadata,
            "epoch": args.epochs,
            "kind": "final",
        },
    )
    best_kernel, _ = load_relation_attention_score_kernel_checkpoint(
        checkpoint,
        map_location=device,
    )
    best_kernel.to(device).eval()
    best_adapter = AttentionScoreKernelAdapter(
        model,
        model.score_module_paths,
        best_kernel,
    )
    best_alignment = diagnose_relation_routing_task_alignment(
        model,
        valid_loader,
        device,
        adapter=best_adapter,
    )
    best_margin = (
        best_diagnostics["learned_vs_uniform"]["correct_margin_delta"]["mean"]
        if best_diagnostics is not None
        else None
    )
    routing_pass = bool(
        best_diagnostics is not None
        and best_diagnostics["mechanism_pass"]
        and best_learned is not None
        and best_uniform is not None
        and best_learned["loss"] < best_uniform["loss"]
        and best_margin is not None
        and best_margin > 0.0
    )
    payload = {
        "args": vars(args),
        "kernel_metadata": best_kernel.metadata(),
        "core_valid": core_valid,
        "initial_learned": initial_learned,
        "initial_uniform": initial_uniform,
        "initial_diagnostics": initial_diagnostics,
        "best_epoch": best_epoch,
        "best_learned": best_learned,
        "best_uniform": best_uniform,
        "best_diagnostics": best_diagnostics,
        "best_alignment": best_alignment,
        "routing_pass": routing_pass,
        "history": history,
        "router_trainable_parameters": sum(
            parameter.numel() for parameter in router.parameters()
        ),
        "direction_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in router.named_parameters()
            if name in {"expert_direction_parameters", "expert_measurement_angles"}
        ),
        "core_frozen": all(
            not parameter.requires_grad
            for name, parameter in kernel.named_parameters()
            if not name.startswith("expert_router.")
        ),
        "gradients": gradient_tracker.summary(),
        "checkpoint": str(checkpoint),
        "final_checkpoint": str(final_checkpoint),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "initial": initial_diagnostics,
                "best": best_diagnostics,
                "best_alignment": best_alignment,
                "gradients": gradient_tracker.summary(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint": str(checkpoint),
                "best_epoch": best_epoch,
                "best_learned": best_learned,
                "best_uniform": best_uniform,
                "routing_pass": routing_pass,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
