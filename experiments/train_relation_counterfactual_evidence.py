from __future__ import annotations

import argparse
import json
from itertools import islice
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
    COUNTERFACTUAL_OBJECTIVE_CHOICES,
    GradientNormTracker,
    choose_device,
    counterfactual_evidence_objective,
    diagnose_relation_counterfactual_evidence,
    diagnose_relation_evidence_task_alignment,
    evaluate_relation_attention_score_kernel,
    load_relation_run,
    make_relation_loader,
    move_batch,
)
from q_attention.plugins import (  # noqa: E402
    EVIDENCE_CORRELATION_MODES,
    EVIDENCE_DIRECT_BIAS_MODES,
    EVIDENCE_GATE_CALIBRATION_MODES,
    EVIDENCE_INTERVENTION_MODES,
    EVIDENCE_MEASUREMENT_MODES,
    EVIDENCE_READOUT_CHOICES,
    EVIDENCE_SELECTOR_TYPES,
    EVIDENCE_TASK_READOUT_MODES,
    EVIDENCE_VIEW_SCORE_MODES,
    EVIDENCE_WEIGHT_MODES,
    RelationEvidenceSelectorConfig,
    RELATION_EVIDENCE_ANCHOR_MODES,
    build_relation_evidence_selector,
    load_relation_attention_score_kernel_checkpoint,
    save_relation_attention_score_kernel_checkpoint,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train counterfactual evidence on a fixed attention-score kernel."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--core_checkpoint", required=True)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--valid_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--evidence_type",
        choices=EVIDENCE_SELECTOR_TYPES,
        default="quantum",
    )
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--angle_scale", type=float, default=1.0)
    parser.add_argument("--mask_floor", type=float, default=0.05)
    parser.add_argument(
        "--evidence_gate_calibration",
        choices=EVIDENCE_GATE_CALIBRATION_MODES,
        default="none",
    )
    parser.add_argument(
        "--evidence_view_score_mode",
        choices=EVIDENCE_VIEW_SCORE_MODES,
        default="positive",
    )
    parser.add_argument(
        "--evidence_task_readout",
        choices=EVIDENCE_TASK_READOUT_MODES,
        default="shared",
    )
    parser.add_argument("--initial_sharpness", type=float, default=2.0)
    parser.add_argument(
        "--evidence_readout",
        choices=EVIDENCE_READOUT_CHOICES,
        default="factorized_observable",
    )
    parser.add_argument(
        "--evidence_correlation_mode",
        choices=EVIDENCE_CORRELATION_MODES,
        default="connected",
    )
    parser.add_argument(
        "--evidence_weight_mode",
        choices=EVIDENCE_WEIGHT_MODES,
        default="positive_simplex",
    )
    parser.add_argument(
        "--relation_anchor_mode",
        choices=RELATION_EVIDENCE_ANCHOR_MODES,
        default="entity_pair",
    )
    parser.add_argument(
        "--evidence_measurement_mode",
        choices=EVIDENCE_MEASUREMENT_MODES,
        default="fixed",
    )
    parser.add_argument("--max_conditioning_gain", type=float, default=2.0)
    parser.add_argument("--initial_conditioning_gain", type=float, default=0.5)
    parser.add_argument("--relation_frame_scale", type=float, default=1.0)
    parser.add_argument("--max_frame_fusion_gain", type=float, default=2.0)
    parser.add_argument("--initial_frame_fusion_gain", type=float, default=1.0)
    parser.add_argument(
        "--intervention_mode",
        choices=EVIDENCE_INTERVENTION_MODES,
        default="kernel_scale",
    )
    parser.add_argument(
        "--direct_bias_mode",
        choices=EVIDENCE_DIRECT_BIAS_MODES,
        default="centered",
    )
    parser.add_argument("--max_direct_gain", type=float, default=1.0)
    parser.add_argument("--initial_direct_gain", type=float, default=0.1)
    parser.add_argument(
        "--disable_cross_entanglement",
        action="store_true",
        help="Disable relation-token entangling gates for the separable Q-RES control.",
    )
    parser.add_argument("--evidence_budget", type=float, default=0.35)
    parser.add_argument("--counterfactual_weight", type=float, default=1.0)
    parser.add_argument("--keep_weight", type=float, default=1.0)
    parser.add_argument("--drop_weight", type=float, default=1.0)
    parser.add_argument("--budget_weight", type=float, default=0.2)
    parser.add_argument("--rank_margin", type=float, default=0.01)
    parser.add_argument("--task_alignment_weight", type=float, default=0.0)
    parser.add_argument(
        "--objective_mode",
        choices=COUNTERFACTUAL_OBJECTIVE_CHOICES,
        default="detached_margin",
    )
    parser.add_argument("--random_repeats", type=int, default=4)
    parser.add_argument("--diagnostic_batches", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
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
        raise ValueError(f"{name} must be provided or recorded by the baseline")
    path = Path(str(selected))
    return path if path.is_absolute() else ROOT / path


def diagnostic_loader(loader: Any, max_batches: int) -> Any:
    return loader if max_batches <= 0 else islice(loader, max_batches)


def _diagnostic_mean(diagnostics: dict[str, Any], name: str) -> float:
    value = diagnostics["metrics"][name]["mean"]
    return float(value) if value is not None else float("-inf")


def evidence_selection_score(
    metrics: dict[str, float],
    diagnostics: dict[str, Any],
) -> tuple[int, float, float, float]:
    keep_advantage = _diagnostic_mean(diagnostics, "keep_advantage")
    drop_advantage = _diagnostic_mean(diagnostics, "drop_advantage")
    return (
        int(diagnostics["selectivity_pass"]),
        min(keep_advantage, drop_advantage),
        keep_advantage + drop_advantage,
        -metrics["loss"],
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    if args.random_repeats <= 0:
        raise ValueError("random_repeats must be positive")
    if args.diagnostic_batches < 0:
        raise ValueError("diagnostic_batches must be non-negative")
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
    if kernel.evidence_selector is not None:
        raise ValueError("core_checkpoint must not already contain an evidence selector")
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

    selector = build_relation_evidence_selector(
        args.evidence_type,
        RelationEvidenceSelectorConfig(
            num_layers=kernel.config.num_layers,
            num_heads=kernel.config.num_heads,
            head_dim=kernel.config.head_dim,
            num_qubits=args.num_qubits,
            depth=args.depth,
            angle_scale=args.angle_scale,
            mask_floor=args.mask_floor,
            evidence_gate_calibration=args.evidence_gate_calibration,
            evidence_budget=args.evidence_budget,
            evidence_view_score_mode=args.evidence_view_score_mode,
            evidence_task_readout=args.evidence_task_readout,
            initial_sharpness=args.initial_sharpness,
            evidence_readout=args.evidence_readout,
            evidence_correlation_mode=args.evidence_correlation_mode,
            relation_anchor_mode=args.relation_anchor_mode,
            evidence_weight_mode=args.evidence_weight_mode,
            evidence_measurement_mode=args.evidence_measurement_mode,
            max_conditioning_gain=args.max_conditioning_gain,
            initial_conditioning_gain=args.initial_conditioning_gain,
            relation_frame_scale=args.relation_frame_scale,
            max_frame_fusion_gain=args.max_frame_fusion_gain,
            initial_frame_fusion_gain=args.initial_frame_fusion_gain,
            intervention_mode=args.intervention_mode,
            direct_bias_mode=args.direct_bias_mode,
            max_direct_gain=args.max_direct_gain,
            initial_direct_gain=args.initial_direct_gain,
            cross_entanglement=not args.disable_cross_entanglement,
            seed=args.seed + 4001,
        ),
    ).to(device)
    kernel.attach_evidence_selector(selector)
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    for parameter in selector.parameters():
        parameter.requires_grad_(True)
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr)
    gradient_tracker = GradientNormTracker(selector.named_parameters())
    initial_valid = evaluate_relation_attention_score_kernel(
        model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        adapter=adapter,
    )
    initial_diagnostics = diagnose_relation_counterfactual_evidence(
        model,
        diagnostic_loader(valid_loader, args.diagnostic_batches),
        device,
        adapter=adapter,
        random_repeats=args.random_repeats,
        random_seed=args.seed + 8009,
    )

    checkpoint = output_dir / "counterfactual_evidence.pt"
    task_checkpoint = output_dir / "counterfactual_evidence_task.pt"
    final_checkpoint = output_dir / "counterfactual_evidence_final.pt"
    checkpoint_metadata = {
        "base_model_dir": str(Path(args.model_dir)),
        "core_checkpoint": str(Path(args.core_checkpoint)),
        "core_checkpoint_metadata": core_metadata,
        "score_module_paths": list(model.score_module_paths),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "seed": args.seed,
    }
    history: list[dict[str, Any]] = []
    best_score = evidence_selection_score(initial_valid, initial_diagnostics)
    best_epoch: int | None = 0
    best_valid: dict[str, float] | None = initial_valid
    best_selectivity: dict[str, Any] | None = initial_diagnostics
    best_task_epoch: int | None = None
    best_task_valid: dict[str, float] | None = None
    best_task_selectivity: dict[str, Any] | None = None
    save_relation_attention_score_kernel_checkpoint(
        checkpoint,
        kernel,
        extra_metadata={**checkpoint_metadata, "epoch": 0, "kind": "initial"},
    )
    if initial_diagnostics["selectivity_pass"]:
        best_task_epoch = 0
        best_task_valid = initial_valid
        best_task_selectivity = initial_diagnostics
        save_relation_attention_score_kernel_checkpoint(
            task_checkpoint,
            kernel,
            extra_metadata={
                **checkpoint_metadata,
                "epoch": 0,
                "kind": "task_best",
            },
        )

    for epoch in range(1, args.epochs + 1):
        selector.train()
        totals: dict[str, float] = {"objective": 0.0}
        total_items = 0
        for batch_index, raw_batch in enumerate(train_loader):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            objective, components = counterfactual_evidence_objective(
                model,
                batch,
                adapter,
                counterfactual_weight=args.counterfactual_weight,
                keep_weight=args.keep_weight,
                drop_weight=args.drop_weight,
                budget_weight=args.budget_weight,
                evidence_budget=args.evidence_budget,
                rank_margin=args.rank_margin,
                random_seed=args.seed + 100003 * epoch + batch_index,
                objective_mode=args.objective_mode,
                task_alignment_weight=args.task_alignment_weight,
            )
            objective.backward()
            gradient_tracker.update()
            optimizer.step()
            items = batch["labels"].shape[0]
            totals["objective"] += float(objective.detach().item()) * items
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * items
            total_items += items

        selector.eval()
        valid = evaluate_relation_attention_score_kernel(
            model,
            valid_loader,
            device,
            len(artifacts.label_to_id),
            adapter=adapter,
        )
        selectivity = diagnose_relation_counterfactual_evidence(
            model,
            valid_loader,
            device,
            adapter=adapter,
            random_repeats=args.random_repeats,
            random_seed=args.seed + 8009,
        )
        row = {
            "epoch": epoch,
            "train": {
                name: value / max(total_items, 1)
                for name, value in totals.items()
            },
            "valid": valid,
            "selectivity": selectivity,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        score = evidence_selection_score(valid, selectivity)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_valid = valid
            best_selectivity = selectivity
            save_relation_attention_score_kernel_checkpoint(
                checkpoint,
                kernel,
                extra_metadata={**checkpoint_metadata, "epoch": epoch, "kind": "best"},
            )
        if selectivity["selectivity_pass"] and (
            best_task_valid is None or valid["loss"] < best_task_valid["loss"]
        ):
            best_task_epoch = epoch
            best_task_valid = valid
            best_task_selectivity = selectivity
            save_relation_attention_score_kernel_checkpoint(
                task_checkpoint,
                kernel,
                extra_metadata={
                    **checkpoint_metadata,
                    "epoch": epoch,
                    "kind": "task_best",
                },
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
    best_alignment = diagnose_relation_evidence_task_alignment(
        model,
        diagnostic_loader(valid_loader, args.diagnostic_batches),
        device,
        adapter=best_adapter,
    )
    best_task_alignment: dict[str, Any] | None = None
    task_kernel_metadata: dict[str, Any] | None = None
    if best_task_epoch is not None:
        task_kernel, _ = load_relation_attention_score_kernel_checkpoint(
            task_checkpoint,
            map_location=device,
        )
        task_kernel.to(device).eval()
        task_kernel_metadata = task_kernel.metadata()
        task_adapter = AttentionScoreKernelAdapter(
            model,
            model.score_module_paths,
            task_kernel,
        )
        best_task_alignment = diagnose_relation_evidence_task_alignment(
            model,
            valid_loader,
            device,
            adapter=task_adapter,
        )
    payload = {
        "args": vars(args),
        "kernel_metadata": best_kernel.metadata(),
        "core_valid": core_valid,
        "initial_valid": initial_valid,
        "initial_selectivity": initial_diagnostics,
        "best_epoch": best_epoch,
        "best_valid": best_valid,
        "best_selectivity": best_selectivity,
        "best_alignment": best_alignment,
        "best_task_epoch": best_task_epoch,
        "best_task_valid": best_task_valid,
        "best_task_selectivity": best_task_selectivity,
        "best_task_alignment": best_task_alignment,
        "task_kernel_metadata": task_kernel_metadata,
        "history": history,
        "selector_trainable_parameters": sum(
            parameter.numel() for parameter in selector.parameters()
        ),
        "core_frozen": all(
            not parameter.requires_grad
            for name, parameter in kernel.named_parameters()
            if not name.startswith("evidence_selector.")
        ),
        "gradients": gradient_tracker.summary(),
        "checkpoint": str(checkpoint),
        "task_checkpoint": (
            str(task_checkpoint) if best_task_epoch is not None else None
        ),
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
                "best": best_selectivity,
                "best_alignment": best_alignment,
                "task_best": best_task_selectivity,
                "task_best_alignment": best_task_alignment,
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
                "best_valid": best_valid,
                "task_checkpoint": (
                    str(task_checkpoint) if best_task_epoch is not None else None
                ),
                "best_task_epoch": best_task_epoch,
                "best_task_valid": best_task_valid,
                "selectivity_pass": (
                    best_selectivity["selectivity_pass"]
                    if best_selectivity is not None
                    else False
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
