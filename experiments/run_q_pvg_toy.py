#!/usr/bin/env python3
"""Run the bounded Q-PVG toy screen and emit semantic Case Study traces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins.q_pvg import QPVGConfig, build_q_pvg  # noqa: E402


TOKENS = 6
HEADS = 1
HEAD_DIM = 4
QUERY_POSITIONS = (2, 3)
CONTEXT_POSITIONS = (4, 5)
SELECTORS = (
    "disabled",
    "q_pvg_phase_value",
    "q_pvg_real_only",
    "q_pvg_scalar_value",
    "q_pvg_classical_complex",
    "q_pvg_random_phase",
    "q_pvg_score_value",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--valid-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--output-root", default="runs/q_pvg_toy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--case-study-per-split", type=int, default=1)
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
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
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def make_split(seed: int, size: int, device: torch.device) -> dict[str, Any]:
    """Create paired relation queries with query-local target values."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    relation_sign = torch.randint(0, 2, (size,), generator=generator).float().mul(2.0).sub(1.0)
    query = torch.randn(size, HEADS, TOKENS, HEAD_DIM, generator=generator) * 0.01
    key = torch.randn(size, HEADS, TOKENS, HEAD_DIM, generator=generator) * 0.01
    query_type = torch.tensor([1.0, -1.0])[None, :].expand(size, -1)
    query[:, 0, QUERY_POSITIONS[0], 0] = 1.0
    query[:, 0, QUERY_POSITIONS[1], 0] = -1.0
    key[:, 0, 0, 1] = relation_sign
    key[:, 0, 1, 1] = relation_sign
    key[:, 0, CONTEXT_POSITIONS[0], 1] = relation_sign
    key[:, 0, CONTEXT_POSITIONS[1], 1] = -relation_sign
    labels = (query_type * relation_sign[:, None] < 0).long()
    target_key = torch.where(
        labels == 0,
        torch.full_like(labels, CONTEXT_POSITIONS[0]),
        torch.full_like(labels, CONTEXT_POSITIONS[1]),
    )
    value = torch.zeros(size, HEADS, TOKENS, HEAD_DIM)
    value[:, 0, CONTEXT_POSITIONS[0], 0] = 1.0
    value[:, 0, CONTEXT_POSITIONS[1], 1] = 1.0
    attention_mask = torch.ones(size, TOKENS, dtype=torch.bool)
    attention_mask[:, -1] = False
    query_mask = torch.ones(size, TOKENS, dtype=torch.bool)
    query_mask[:, -1] = False
    scores = torch.randn(size, HEADS, TOKENS, TOKENS, generator=generator) * 0.02
    scores = scores.masked_fill(~attention_mask[:, None, None, :], 0.0)
    return {
        "query": query.to(device),
        "key": key.to(device),
        "value": value.to(device),
        "labels": labels.to(device),
        "query_type": query_type.to(device),
        "relation_sign": relation_sign.to(device),
        "target_key": target_key.to(device),
        "attention_mask": attention_mask.to(device),
        "query_mask": query_mask.to(device),
        "scores": scores.to(device),
        "tokens": ["[CLS]", "Alice", "works", "query", "Acme", "[SEP]"],
        "sentence": "Alice works at Acme.",
        "entity_pair": {"subject": "Alice", "object": "Acme"},
        "entity_spans": {"subject": [1, 1], "object": [4, 4]},
    }


def batches(split: dict[str, Any], batch_size: int):
    size = split["labels"].shape[0]
    tensor_names = (
        "query", "key", "value", "labels", "query_type", "relation_sign",
        "target_key", "attention_mask", "query_mask", "scores",
    )
    for start in range(0, size, batch_size):
        batch = {name: split[name][start : start + batch_size] for name in tensor_names}
        yield batch


def build_selector(selector: str, seed: int) -> nn.Module | None:
    if selector == "disabled":
        return None
    phase_mode = "complex"
    route_mode = "branch_interpolation"
    score_mode = "value_only"
    readout_mode = "quantum"
    if selector == "q_pvg_real_only":
        phase_mode = "real_only"
    elif selector == "q_pvg_scalar_value":
        route_mode = "scalar"
    elif selector == "q_pvg_classical_complex":
        readout_mode = "classical"
    elif selector == "q_pvg_random_phase":
        phase_mode = "random"
    elif selector == "q_pvg_score_value":
        score_mode = "score_value"
    elif selector != "q_pvg_phase_value":
        raise ValueError(f"unknown selector: {selector}")
    return build_q_pvg(
        QPVGConfig(
            num_layers=1,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            register_qubits=2,
            depth=2,
            angle_scale=1.0,
            gate_temperature=1.0,
            score_gain=0.25,
            value_route_mode=route_mode,
            score_mode=score_mode,
            readout_mode=readout_mode,
            phase_mode=phase_mode,
            seed=seed + 3000,
        )
    )


def base_forward(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    key_mask = batch["attention_mask"][:, None, None, :]
    attention = torch.softmax(
        batch["scores"].masked_fill(~key_mask, torch.finfo(batch["scores"].dtype).min), dim=-1
    )
    attention = attention * key_mask.to(dtype=attention.dtype)
    output = torch.einsum("bhqk,bhkd->bhqd", attention, batch["value"])
    trace = {
        "query_state_real": batch["query"],
        "query_state_imag": torch.zeros_like(batch["query"]),
        "key_state_real": batch["key"],
        "key_state_imag": torch.zeros_like(batch["key"]),
        "alignment_real": torch.einsum("bhqd,bhkd->bhqk", batch["query"], batch["key"]),
        "alignment_imag": torch.zeros_like(batch["scores"]),
        "gate": torch.ones_like(batch["scores"]) * key_mask.to(dtype=batch["scores"].dtype),
        "base_attention": attention,
        "score_adjustment": torch.zeros_like(batch["scores"]),
        "attention": attention,
        "value_branch0": batch["value"],
        "value_branch1": batch["value"],
        "routed_values": batch["value"][:, :, None, :, :].expand(-1, -1, batch["query"].shape[2], -1, -1),
        "output": output,
    }
    return output, trace


def forward(
    kernel: nn.Module | None,
    classifier: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    return_trace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    if kernel is None:
        output, trace = base_forward(batch)
    else:
        output, trace = kernel(
            batch["query"],
            batch["key"],
            batch["value"],
            scores=batch["scores"],
            layer_index=0,
            attention_mask=batch["attention_mask"],
            query_mask=batch["query_mask"],
            return_trace=True,
        )
    query_output = output[:, :, QUERY_POSITIONS, :].reshape(-1, HEAD_DIM)
    logits = classifier(query_output)
    return output, logits, trace["attention"], trace if return_trace else None


def _target_stats(
    attention: torch.Tensor,
    base_attention: torch.Tensor,
    target_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_attention = attention[:, :, QUERY_POSITIONS, :]
    base_query_attention = base_attention[:, :, QUERY_POSITIONS, :]
    targets = target_key[:, None, :, None].expand(-1, attention.shape[1], -1, -1)
    target_mass = query_attention.gather(-1, targets).squeeze(-1).mean(dim=1)
    base_mass = base_query_attention.gather(-1, targets).squeeze(-1).mean(dim=1)
    return target_mass, base_mass


def evaluate(
    kernel: nn.Module | None,
    classifier: nn.Module,
    split: dict[str, Any],
    batch_size: int,
) -> dict[str, float]:
    classifier.eval()
    if kernel is not None:
        kernel.eval()
    total = 0
    correct = 0
    masses = []
    base_masses = []
    losses = []
    with torch.no_grad():
        for batch in batches(split, batch_size):
            _, logits, attention, trace = forward(kernel, classifier, batch, return_trace=True)
            labels = batch["labels"].reshape(-1)
            losses.append(F.cross_entropy(logits, labels).detach())
            correct += int((logits.argmax(dim=-1) == labels).sum())
            total += labels.numel()
            mass, base_mass = _target_stats(attention, trace["base_attention"], batch["target_key"])
            masses.append(mass)
            base_masses.append(base_mass)
    target_mass = torch.cat(masses).mean().item()
    base_mass = torch.cat(base_masses).mean().item()
    return {
        "query_accuracy": correct / max(total, 1),
        "loss": torch.stack(losses).mean().item(),
        "target_mass": target_mass,
        "baseline_target_mass": base_mass,
        "target_mass_gain": target_mass - base_mass,
    }


def tensor_json(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "values": detached.tolist(),
    }


def make_case_trace(
    selector: str,
    seed: int,
    split_name: str,
    kernel: nn.Module | None,
    classifier: nn.Module,
    split: dict[str, Any],
) -> dict[str, Any]:
    batch = {name: value[:1] for name, value in split.items() if isinstance(value, torch.Tensor)}
    _, logits, _, trace = forward(kernel, classifier, batch, return_trace=True)
    labels = batch["labels"].reshape(-1)
    prediction = logits.argmax(dim=-1).reshape(1, -1)
    semantic = {
        "sentence": split["sentence"],
        "tokens": split["tokens"],
        "entity_pair": split["entity_pair"],
        "entity_spans": split["entity_spans"],
        "target_relation": labels.tolist(),
        "predicted_relation": prediction.tolist(),
        "query_positions": list(QUERY_POSITIONS),
        "target_value_keys": batch["target_key"].detach().cpu().tolist(),
    }
    return {
        "schema": "sample-trace.v1",
        "selector": selector,
        "seed": seed,
        "split": split_name,
        "sample_index": 0,
        "semantic": semantic,
        "intermediate": {name: tensor_json(value) for name, value in trace.items()},
        "metadata": {
            "base_attention_is_input_to_value_route": True,
            "phase_sensitive": selector in {"q_pvg_phase_value", "q_pvg_score_value"},
            "full_tensor_trace": True,
        },
    }


def run_selector(
    selector: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    kernel = build_selector(selector, seed)
    if kernel is not None:
        kernel.to(device)
    classifier = nn.Linear(HEAD_DIM, 2).to(device)
    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + ([] if kernel is None else list(kernel.parameters())),
        lr=args.lr,
    )
    train = make_split(seed + 101, args.train_size, device)
    valid = make_split(seed + 211, args.valid_size, device)
    test = make_split(seed + 307, args.valid_size, device)
    for step in range(args.steps):
        if step % max(1, args.train_size // args.batch_size) == 0:
            permutation = torch.randperm(args.train_size, device=device)
        start = (step * args.batch_size) % args.train_size
        indices = permutation[start : start + args.batch_size]
        if indices.numel() == 0:
            indices = permutation[: args.batch_size]
        batch = {
            name: value[indices]
            for name, value in train.items()
            if isinstance(value, torch.Tensor)
        }
        classifier.train()
        if kernel is not None:
            kernel.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits, _, _ = forward(kernel, classifier, batch)
        loss = F.cross_entropy(logits, batch["labels"].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(classifier.parameters()) + ([] if kernel is None else list(kernel.parameters())),
            max_norm=5.0,
        )
        optimizer.step()
    metrics = {
        "train": evaluate(kernel, classifier, train, args.batch_size),
        "valid": evaluate(kernel, classifier, valid, args.batch_size),
        "test": evaluate(kernel, classifier, test, args.batch_size),
    }
    traces = {
        "train": [make_case_trace(selector, seed, "train", kernel, classifier, train)],
        "valid": [make_case_trace(selector, seed, "valid", kernel, classifier, valid)],
        "test": [make_case_trace(selector, seed, "test", kernel, classifier, test)],
    }
    return {
        "selector": selector,
        "seed": seed,
        "metrics": metrics,
        "case_study": traces,
        "parameter_count": sum(param.numel() for param in classifier.parameters())
        + (0 if kernel is None else sum(param.numel() for param in kernel.parameters())),
        "metadata": None if kernel is None else kernel.metadata(),
    }


def mechanism_diagnostics(seed: int, device: torch.device) -> dict[str, Any]:
    """Measure phase sensitivity and control separation independently of task fit."""
    split = make_split(seed + 701, 8, device)
    batch = {name: value[:4] for name, value in split.items() if isinstance(value, torch.Tensor)}
    phase = build_selector("q_pvg_phase_value", seed)
    real_only = build_selector("q_pvg_real_only", seed)
    classical = build_selector("q_pvg_classical_complex", seed)
    random_phase = build_selector("q_pvg_random_phase", seed)
    for kernel in (phase, real_only, classical, random_phase):
        assert kernel is not None
        kernel.to(device)
        kernel.eval()
    with torch.no_grad():
        _, _, _, phase_plus = forward(phase, nn.Linear(HEAD_DIM, 2).to(device), batch, return_trace=True)
        # Recompute the reversed-phase trace through the kernel's explicit diagnostic input.
        _, minus_trace = phase(
            batch["query"],
            batch["key"],
            batch["value"],
            scores=batch["scores"],
            layer_index=0,
            attention_mask=batch["attention_mask"],
            query_mask=batch["query_mask"],
            phase_sign=-1.0,
            return_trace=True,
        )
        _, _, _, real_trace = forward(real_only, nn.Linear(HEAD_DIM, 2).to(device), batch, return_trace=True)
        _, _, _, classical_trace = forward(classical, nn.Linear(HEAD_DIM, 2).to(device), batch, return_trace=True)
        _, _, _, random_trace = forward(random_phase, nn.Linear(HEAD_DIM, 2).to(device), batch, return_trace=True)
    finite = all(
        torch.isfinite(trace[name]).all()
        for trace in (phase_plus, minus_trace, real_trace, classical_trace, random_trace)
        for name in ("gate", "alignment_real", "alignment_imag", "attention", "output")
    )
    masked = torch.zeros_like(phase_plus["attention"][..., -1])
    mask_ok = torch.allclose(phase_plus["attention"][..., -1], masked, atol=1e-7)
    query_condition_delta = (
        phase_plus["gate"][:, :, QUERY_POSITIONS[0], :]
        - phase_plus["gate"][:, :, QUERY_POSITIONS[1], :]
    ).abs().mean().item()

    gradient_kernel = build_selector("q_pvg_phase_value", seed + 9000)
    gradient_kernel.to(device)
    gradient_classifier = nn.Linear(HEAD_DIM, 2).to(device)
    gradient_kernel.train()
    gradient_classifier.train()
    _, _, _, gradient_trace = forward(
        gradient_kernel, gradient_classifier, batch, return_trace=True
    )
    gradient_loss = (
        gradient_trace["output"].square().mean()
        + gradient_trace["alignment_real"].square().mean()
        + gradient_trace["alignment_imag"].square().mean()
    )
    gradient_loss.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in gradient_kernel.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = (
        torch.cat([gradient.reshape(-1) for gradient in gradients]).norm().item()
        if gradients
        else 0.0
    )
    gradient_finite = bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)
    return {
        "finite": bool(finite),
        "mask_ok": bool(mask_ok),
        "phase_reversal_gate_delta": (phase_plus["gate"] - minus_trace["gate"]).abs().mean().item(),
        "complex_vs_real_gate_delta": (phase_plus["gate"] - real_trace["gate"]).abs().mean().item(),
        "complex_vs_classical_gate_delta": (phase_plus["gate"] - classical_trace["gate"]).abs().mean().item(),
        "complex_vs_random_gate_delta": (phase_plus["gate"] - random_trace["gate"]).abs().mean().item(),
        "query_condition_gate_delta": query_condition_delta,
        "gradient_finite": gradient_finite,
        "gradient_nonzero": bool(gradient_norm > 0.0),
        "gradient_norm": gradient_norm,
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.train_size <= 0 or args.valid_size <= 0:
        raise ValueError("steps and split sizes must be positive")
    seeds = parse_seeds(args.seeds)
    device = choose_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        for selector in SELECTORS:
            result = run_selector(selector, seed, args, device)
            results.append(result)
            print(
                f"seed={seed:>3} selector={selector:<24} "
                f"valid_acc={result['metrics']['valid']['query_accuracy']:.3f} "
                f"valid_gain={result['metrics']['valid']['target_mass_gain']:+.4f}",
                flush=True,
            )
    diagnostics = [mechanism_diagnostics(seed, device) for seed in seeds]
    toy_gates = {
        "all_finite": all(item["finite"] for item in diagnostics),
        "all_masks_respected": all(item["mask_ok"] for item in diagnostics),
        "phase_reversal_detected": all(item["phase_reversal_gate_delta"] > 1e-5 for item in diagnostics),
        "query_condition_detected": all(item["query_condition_gate_delta"] > 1e-5 for item in diagnostics),
        "control_distances_present": all(
            item["complex_vs_real_gate_delta"] > 1e-5
            and item["complex_vs_classical_gate_delta"] > 1e-5
            and item["complex_vs_random_gate_delta"] > 1e-5
            for item in diagnostics
        ),
        "gradient_health": all(
            item["gradient_finite"] and item["gradient_nonzero"]
            for item in diagnostics
        ),
    }
    payload = {
        "schema": "q-pvg-toy.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "device": str(device),
        "config": vars(args),
        "selectors": list(SELECTORS),
        "results": results,
        "mechanism_diagnostics": diagnostics,
        "toy_gates": toy_gates,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    case_study = {
        "schema": "sample-trace.v1",
        "experiment": "q_pvg_toy",
        "selectors": list(SELECTORS),
        "traces": [
            {
                "seed": item["seed"],
                "selector": item["selector"],
                "split": split,
                "samples": samples,
            }
            for item in results
            for split, samples in item["case_study"].items()
        ],
    }
    (output_root / "case_study.json").write_text(
        json.dumps(case_study, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {output_root / 'summary.json'}")
    print(f"Wrote {output_root / 'case_study.json'}")


if __name__ == "__main__":
    main()
