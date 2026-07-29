from __future__ import annotations

import argparse
import json
from itertools import islice
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

from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    RELATION_SELECTION_CHOICES,
    GradientNormTracker,
    attention_score_hook_config,
    choose_device,
    diagnose_relation_attention_score_kernel,
    diagnose_relation_attention_score_task_alignment,
    evaluate_relation_attention_score_kernel,
    load_relation_run,
    make_relation_loader,
    move_batch,
    relation_selection_score,
)
from q_attention.plugins import (  # noqa: E402
    SCORE_INPUT_ENCODING_CHOICES,
    SCORE_QUERY_SCOPE_CHOICES,
    SCORE_READOUT_CHOICES,
    RelationScoreKernelConfig,
    build_relation_attention_score_kernel,
    load_relation_attention_score_kernel_checkpoint,
    save_relation_attention_score_kernel_checkpoint,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a relation-conditioned attention-score kernel."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--valid_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kernel_type", choices=["quantum", "classical"], default="quantum")
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--angle_scale", type=float, default=1.0)
    parser.add_argument("--max_gain", type=float, default=0.5)
    parser.add_argument("--initial_gain", type=float, default=0.02)
    parser.add_argument("--normalize_readout_energy", action="store_true")
    parser.add_argument(
        "--score_readout",
        choices=SCORE_READOUT_CHOICES,
        default="fidelity",
    )
    parser.add_argument(
        "--input_encoding",
        choices=SCORE_INPUT_ENCODING_CHOICES,
        default="joint",
    )
    parser.add_argument(
        "--query_scope",
        choices=SCORE_QUERY_SCOPE_CHOICES,
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diversity_weight", type=float, default=0.0)
    parser.add_argument("--diagnostic_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--selection_metric",
        choices=RELATION_SELECTION_CHOICES,
        default="macro_f1_then_loss",
    )
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


def diagnostic_loader(loader: Any, max_batches: int) -> Any:
    return loader if max_batches <= 0 else islice(loader, max_batches)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    if args.diversity_weight < 0:
        raise ValueError("diversity_weight must be non-negative")
    if args.diagnostic_batches < 0:
        raise ValueError("diagnostic_batches must be non-negative")
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(Path(args.model_dir), device)
    train_path = resolve_data_path(args.train_path, artifacts.args.get("train_path"), "train_path")
    valid_path = resolve_data_path(args.valid_path, artifacts.args.get("valid_path"), "valid_path")
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
    train_diagnostic_loader = make_relation_loader(
        train_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = artifacts.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config
    kernel = build_relation_attention_score_kernel(
        args.kernel_type,
        RelationScoreKernelConfig(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.dim // config.num_heads,
            num_qubits=args.num_qubits,
            depth=args.depth,
            angle_scale=args.angle_scale,
            max_gain=args.max_gain,
            initial_gain=args.initial_gain,
            normalize_readout_energy=args.normalize_readout_energy,
            score_readout=args.score_readout,
            input_encoding=args.input_encoding,
            query_scope=args.query_scope,
            seed=args.seed,
        ),
    ).to(device)
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    optimizer = torch.optim.AdamW(kernel.parameters(), lr=args.lr)
    gradient_tracker = GradientNormTracker(kernel.named_parameters())
    baseline_valid = evaluate_relation_attention_score_kernel(
        model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        adapter=None,
    )
    initial_diagnostics = diagnose_relation_attention_score_kernel(
        model,
        diagnostic_loader(valid_loader, args.diagnostic_batches),
        device,
        score_module_paths=model.score_module_paths,
        score_kernel=kernel,
    )
    initial_alignment = diagnose_relation_attention_score_task_alignment(
        model,
        diagnostic_loader(valid_loader, args.diagnostic_batches),
        device,
        score_module_paths=model.score_module_paths,
        score_kernel=kernel,
    )
    checkpoint = output_dir / "attention_score_kernel.pt"
    final_checkpoint = output_dir / "attention_score_kernel_final.pt"
    checkpoint_metadata = {
        "base_model_dir": str(Path(args.model_dir)),
        "score_module_paths": list(model.score_module_paths),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "diversity_weight": args.diversity_weight,
        "normalize_readout_energy": args.normalize_readout_energy,
    }
    history: list[dict[str, Any]] = []
    best_valid: dict[str, float] | None = None
    best_epoch: int | None = None
    best_score = (float("-inf"), float("-inf"))
    for epoch in range(1, args.epochs + 1):
        kernel.train()
        total_loss = 0.0
        total_objective = 0.0
        total_diversity = 0.0
        total_items = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if args.diversity_weight > 0.0:
                with kernel.capture_centered_kernels():
                    with adapter.steering(attention_score_hook_config(batch)):
                        logits = model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["subject_mask"],
                            batch["object_mask"],
                        )
                    task_loss = F.cross_entropy(logits, batch["labels"])
                    diversity_loss = kernel.functional_diversity_loss()
                    objective = task_loss + args.diversity_weight * diversity_loss
                    objective.backward()
            else:
                with adapter.steering(attention_score_hook_config(batch)):
                    logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["subject_mask"],
                        batch["object_mask"],
                    )
                task_loss = F.cross_entropy(logits, batch["labels"])
                diversity_loss = task_loss.detach() * 0.0
                objective = task_loss
                objective.backward()
            gradient_tracker.update()
            optimizer.step()
            total_loss += float(task_loss.detach().item()) * batch["labels"].shape[0]
            total_objective += float(objective.detach().item()) * batch["labels"].shape[0]
            total_diversity += float(diversity_loss.detach().item()) * batch["labels"].shape[0]
            total_items += batch["labels"].shape[0]

        valid = evaluate_relation_attention_score_kernel(
            model,
            valid_loader,
            device,
            len(artifacts.label_to_id),
            adapter=adapter,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            "train_objective": total_objective / max(total_items, 1),
            "diversity_loss": total_diversity / max(total_items, 1),
            "valid": valid,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        score = relation_selection_score(valid, args.selection_metric)
        if score > best_score:
            best_score = score
            best_valid = valid
            best_epoch = epoch
            save_relation_attention_score_kernel_checkpoint(
                checkpoint,
                kernel,
                extra_metadata={**checkpoint_metadata, "epoch": epoch, "kind": "best"},
            )

    save_relation_attention_score_kernel_checkpoint(
        final_checkpoint,
        kernel,
        extra_metadata={**checkpoint_metadata, "epoch": args.epochs, "kind": "final"},
    )
    final_diagnostics = diagnose_relation_attention_score_kernel(
        model,
        diagnostic_loader(valid_loader, args.diagnostic_batches),
        device,
        score_module_paths=model.score_module_paths,
        score_kernel=kernel,
    )
    final_alignment = {
        "train": diagnose_relation_attention_score_task_alignment(
            model,
            diagnostic_loader(train_diagnostic_loader, args.diagnostic_batches),
            device,
            score_module_paths=model.score_module_paths,
            score_kernel=kernel,
        ),
        "valid": diagnose_relation_attention_score_task_alignment(
            model,
            valid_loader,
            device,
            score_module_paths=model.score_module_paths,
            score_kernel=kernel,
        ),
    }
    best_kernel, _ = load_relation_attention_score_kernel_checkpoint(
        checkpoint,
        map_location=device,
    )
    best_kernel.to(device)
    best_diagnostics = diagnose_relation_attention_score_kernel(
        model,
        diagnostic_loader(valid_loader, args.diagnostic_batches),
        device,
        score_module_paths=model.score_module_paths,
        score_kernel=best_kernel,
    )
    best_alignment = {
        "train": diagnose_relation_attention_score_task_alignment(
            model,
            diagnostic_loader(train_diagnostic_loader, args.diagnostic_batches),
            device,
            score_module_paths=model.score_module_paths,
            score_kernel=best_kernel,
        ),
        "valid": diagnose_relation_attention_score_task_alignment(
            model,
            valid_loader,
            device,
            score_module_paths=model.score_module_paths,
            score_kernel=best_kernel,
        ),
    }
    diagnostics = {
        "initial": initial_diagnostics,
        "best": best_diagnostics,
        "final": final_diagnostics,
        "gradients": gradient_tracker.summary(),
        "task_alignment": {
            "initial_valid": initial_alignment,
            "best": best_alignment,
            "final": final_alignment,
        },
    }
    payload = {
        "args": vars(args),
        "kernel_metadata": kernel.metadata(),
        "trainable_parameters": sum(parameter.numel() for parameter in kernel.parameters()),
        "base_model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "baseline_valid": baseline_valid,
        "best_valid": best_valid,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "history": history,
        "checkpoint": str(checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "diagnostics": diagnostics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint": str(checkpoint),
                "baseline_valid": baseline_valid,
                "best_valid": best_valid,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
