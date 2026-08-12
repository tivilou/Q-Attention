#!/usr/bin/env python3
"""Run the five-seed Q-NESS mechanism gate on synthetic relation evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins import (
    RelationEvidenceSelectorConfig,
    build_relation_evidence_selector,
)


SELECTORS = (
    "qness",
    "qness_commuting",
    "qness_separable",
    "qness_phase_scrambled",
    "qness_dephased",
    "qness_classical",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--output_root", default="runs/qness_toy_gate")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fail_on_gate", action="store_true")
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list without duplicates")
    return seeds


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def synthetic_batch(
    seed: int,
    batch_size: int,
    tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if tokens < 7:
        raise ValueError("Q-NESS toy gate requires at least seven tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    key = torch.randn(batch_size, 2, tokens, 4, generator=generator)
    token_code = torch.linspace(-1.0, 1.0, tokens).view(1, 1, tokens, 1)
    key = key + torch.cat(
        (token_code, token_code.square(), token_code.sin(), token_code.cos()),
        dim=-1,
    )
    attention = torch.ones(batch_size, tokens, dtype=torch.bool)
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 0] = True
    object_[:, 1] = True
    necessity = torch.full((batch_size, 2, tokens), 0.15)
    sufficiency = torch.full((batch_size, 2, tokens), 0.15)
    necessity[:, :, (2, 3, 4)] = 0.85
    sufficiency[:, :, (4, 5, 6)] = 0.85
    return {
        "key": key.to(device),
        "attention_mask": attention.to(device),
        "subject_mask": subject.to(device),
        "object_mask": object_.to(device),
        "necessity_target": necessity.to(device),
        "sufficiency_target": sufficiency.to(device),
    }


def selector_config(seed: int) -> RelationEvidenceSelectorConfig:
    return RelationEvidenceSelectorConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=4,
        depth=2,
        evidence_readout="connected_relation_token",
        evidence_task_readout="dual",
        evidence_weight_mode="signed_centered_l1",
        evidence_measurement_mode="fixed",
        evidence_gate_calibration="none",
        quantum_diagnostic_limit=0,
        seed=seed + 4001,
    )


def context_values(values: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    context = (
        batch["attention_mask"]
        & ~(batch["subject_mask"] | batch["object_mask"])
    )[:, None, :]
    return values.masked_select(context.expand_as(values))


def resource_means(
    captured: tuple[tuple[int, int, dict[str, torch.Tensor]], ...],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    context = batch["attention_mask"] & ~(
        batch["subject_mask"] | batch["object_mask"]
    )
    values: dict[str, list[torch.Tensor]] = {}
    for _layer, _head, diagnostics in captured:
        for name, tensor in diagnostics.items():
            values.setdefault(name, []).append(tensor.masked_select(context))
    return {
        name: float(torch.cat(tensors).double().mean().item())
        for name, tensors in values.items()
    }


def run_selector(
    selector_name: str,
    seed: int,
    batch: dict[str, torch.Tensor],
    *,
    steps: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selector = build_relation_evidence_selector(
        selector_name,
        selector_config(seed),
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr)
    context = (
        batch["attention_mask"]
        & ~(batch["subject_mask"] | batch["object_mask"])
    )[:, None, :].to(dtype=batch["key"].dtype)
    started = time.perf_counter()
    finite_gradients = True
    final_loss = float("nan")
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        necessity, sufficiency = selector.token_readouts(
            batch["key"],
            layer_index=0,
            attention_mask=batch["attention_mask"],
            subject_mask=batch["subject_mask"],
            object_mask=batch["object_mask"],
        )
        necessity_loss = (
            (necessity - batch["necessity_target"]).square() * context
        ).sum() / context.sum().clamp_min(1.0)
        sufficiency_loss = (
            (sufficiency - batch["sufficiency_target"]).square() * context
        ).sum() / context.sum().clamp_min(1.0)
        loss = necessity_loss + sufficiency_loss
        loss.backward()
        finite_gradients = finite_gradients and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in selector.parameters()
        )
        optimizer.step()
        final_loss = float(loss.detach().item())

    selector.eval()
    with torch.no_grad(), selector.capture_token_scores():
        necessity, sufficiency = selector.token_readouts(
            batch["key"],
            layer_index=0,
            attention_mask=batch["attention_mask"],
            subject_mask=batch["subject_mask"],
            object_mask=batch["object_mask"],
        )
        resources = resource_means(selector.captured_quantum_diagnostics(), batch)
    necessity_values = context_values(necessity, batch)
    sufficiency_values = context_values(sufficiency, batch)
    necessity_target = context_values(batch["necessity_target"], batch)
    sufficiency_target = context_values(batch["sufficiency_target"], batch)
    necessity_centered = necessity_values - necessity_values.mean()
    sufficiency_centered = sufficiency_values - sufficiency_values.mean()
    overlap = ((necessity_values > 0.5) & (sufficiency_values > 0.5)).sum()
    union = ((necessity_values > 0.5) | (sufficiency_values > 0.5)).sum()
    return {
        "selector": selector_name,
        "seed": seed,
        "steps": steps,
        "runtime_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in selector.parameters()),
        "finite_gradients": finite_gradients,
        "final_training_loss": final_loss,
        "necessity_mse": float(
            F.mse_loss(necessity_values, necessity_target).item()
        ),
        "sufficiency_mse": float(
            F.mse_loss(sufficiency_values, sufficiency_target).item()
        ),
        "complement_error": float(
            (necessity_values + sufficiency_values - 1.0).abs().mean().item()
        ),
        "necessity_sufficiency_cosine": float(
            F.cosine_similarity(
                necessity_centered.unsqueeze(0),
                sufficiency_centered.unsqueeze(0),
                dim=-1,
            ).item()
        ),
        "mask_overlap_iou": float((overlap / union.clamp_min(1)).item()),
        "resources": resources,
    }


def seed_gate(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qness = rows["qness"]
    commuting = rows["qness_commuting"]
    separable = rows["qness_separable"]
    scrambled = rows["qness_phase_scrambled"]
    dephased = rows["qness_dephased"]
    classical = rows["qness_classical"]
    qness_joint_mse = qness["necessity_mse"] + qness["sufficiency_mse"]
    classical_joint_mse = (
        classical["necessity_mse"] + classical["sufficiency_mse"]
    )
    mechanism_pass = bool(
        qness["finite_gradients"]
        and qness_joint_mse < 0.20
        and qness["complement_error"] > 0.05
        and abs(qness["necessity_sufficiency_cosine"]) < 0.99
    )
    controls_pass = bool(
        qness["resources"]["observable_commutator_norm"] > 1.0
        and commuting["resources"]["observable_commutator_norm"] < 1e-8
        and separable["resources"]["mutual_information"] < 1e-5
        and dephased["resources"]["off_diagonal_density_norm"] < 1e-8
        and scrambled["finite_gradients"]
    )
    return {
        "mechanism_pass": mechanism_pass,
        "controls_pass": controls_pass,
        "gate_pass": mechanism_pass and controls_pass,
        "qness_joint_mse": qness_joint_mse,
        "classical_joint_mse": classical_joint_mse,
        "classical_equivalence_gap": classical_joint_mse - qness_joint_mse,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Q-NESS Toy Mechanism Gate",
        "",
        f"- revision: `{summary['revision']}`",
        f"- device: `{summary['device']}`",
        f"- seeds: `{summary['seeds']}`",
        f"- overall gate: `{summary['gate_pass']}`",
        "- task metrics: not applicable at the mechanism-only toy stage",
        "",
        "| seed | gate | Q-NESS MSE | classical MSE | complement error | MI | runtime (s) |",
        "| ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed_row in summary["results"]:
        qness = seed_row["selectors"]["qness"]
        gate = seed_row["gate"]
        lines.append(
            "| {seed} | {passed} | {q_mse:.6f} | {c_mse:.6f} | "
            "{complement:.6f} | {mi:.6f} | {runtime:.2f} |".format(
                seed=seed_row["seed"],
                passed=gate["gate_pass"],
                q_mse=gate["qness_joint_mse"],
                c_mse=gate["classical_joint_mse"],
                complement=qness["complement_error"],
                mi=qness["resources"]["mutual_information"],
                runtime=sum(
                    row["runtime_seconds"] for row in seed_row["selectors"].values()
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.lr <= 0.0:
        raise ValueError("steps, batch_size, and lr must be positive")
    seeds = parse_seeds(args.seeds)
    device = choose_device(args.device)
    output_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results = []
    for seed in seeds:
        batch = synthetic_batch(seed, args.batch_size, args.tokens, device)
        selector_rows = {
            name: run_selector(
                name,
                seed,
                batch,
                steps=args.steps,
                lr=args.lr,
                device=device,
            )
            for name in SELECTORS
        }
        results.append(
            {
                "seed": seed,
                "selectors": selector_rows,
                "gate": seed_gate(selector_rows),
            }
        )
    summary = {
        "schema_version": "qness-toy-gate.v1",
        "revision": git_revision(),
        "device": str(device),
        "seeds": seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "learning_rate": args.lr,
        "runtime_seconds": time.perf_counter() - started,
        "gate_pass": all(row["gate"]["gate_pass"] for row in results),
        "results": results,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, output_dir / "run_summary.md")
    print(json.dumps({"output_dir": str(output_dir), "gate_pass": summary["gate_pass"]}))
    if args.fail_on_gate and not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
