#!/usr/bin/env python3
"""Gate Q-WAP on a fixed-role signed-triad relation task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_trained_baseline_gate import (  # noqa: E402
    SELECTORS,
    _graph,
    choose_device,
    collect_scores,
    geometry_diagnostics,
    git_revision,
    oracle_action_headroom,
    set_seed,
    split_diagnostics,
    stage_a_gate,
    stage_b_gate,
    tensor_batch,
    train_baseline,
    train_selector,
)


TOKEN_IDS = {
    "anchor": 1,
    "subject_positive": 2,
    "subject_negative": 3,
    "object_positive": 4,
    "object_negative": 5,
    "bridge_positive": 6,
    "bridge_negative": 7,
    "candidate_zero": 8,
    "candidate_one": 9,
}
NUISANCE_START = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_fixed_signed_triad_gate.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.q-fixed-signed-triad-gate.v1":
        raise ValueError("unsupported fixed signed-triad config")
    if int(config.get("seed", -1)) != 7:
        raise ValueError("fixed signed-triad gate requires fixed seed 7")
    if tuple(config.get("selectors", ())) != SELECTORS:
        raise ValueError("selectors must match the frozen trained-baseline allowlist")
    streams = [int(config["dataset"][f"{name}_stream"]) for name in ("train", "valid", "test")]
    if len(set(streams)) != 3:
        raise ValueError("train, valid, and test streams must be distinct")
    return config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_split(
    stream: int,
    size: int,
    nuisance_tokens: int,
    device: torch.device,
    seen: set[tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    if size <= 0 or size % 2:
        raise ValueError("split size must be positive and even")
    if nuisance_tokens < 16:
        raise ValueError("at least sixteen nuisance tokens are required")
    generator = torch.Generator(device="cpu").manual_seed(stream)
    target_labels = torch.arange(size, dtype=torch.long) % 2
    target_labels = target_labels[torch.randperm(size, generator=generator)]
    seen = seen if seen is not None else set()
    rows = []
    fingerprints = []
    polarity_bits = []
    for target in target_labels.tolist():
        for _attempt in range(20000):
            polarities = torch.randint(0, 2, (3,), generator=generator)
            label = int(int(polarities.sum()) % 2 == 1)
            nuisance = NUISANCE_START + torch.randint(
                nuisance_tokens, (2,), generator=generator
            )
            row = torch.tensor(
                [
                    TOKEN_IDS["anchor"],
                    TOKEN_IDS["subject_negative" if polarities[0] else "subject_positive"],
                    TOKEN_IDS["object_negative" if polarities[1] else "object_positive"],
                    TOKEN_IDS["bridge_negative" if polarities[2] else "bridge_positive"],
                    TOKEN_IDS["candidate_zero"],
                    TOKEN_IDS["candidate_one"],
                    int(nuisance[0]),
                    int(nuisance[1]),
                ],
                dtype=torch.long,
            )
            key = tuple(int(value) for value in row)
            if label != target or key in seen:
                continue
            seen.add(key)
            rows.append(row)
            polarity_bits.append(polarities)
            fingerprints.append(hashlib.sha256(bytes(key)).hexdigest())
            break
        else:
            raise RuntimeError("could not construct a unique fixed signed-triad example")
    input_ids = torch.stack(rows).to(device)
    masks = {}
    for name, index in (
        ("subject_mask", 1),
        ("object_mask", 2),
        ("bridge_mask", 3),
        ("candidate_zero_mask", 4),
        ("candidate_one_mask", 5),
    ):
        mask = torch.zeros(size, 8, dtype=torch.bool)
        mask[:, index] = True
        masks[name] = mask.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        **masks,
        "labels": target_labels.to(device),
        "polarity_bits": torch.stack(polarity_bits).to(device),
        "fingerprints": fingerprints,
    }


def make_splits(config: dict[str, Any], device: torch.device) -> dict[str, dict[str, Any]]:
    dataset = config["dataset"]
    seen: set[tuple[int, ...]] = set()
    return {
        name: make_split(
            int(dataset[f"{name}_stream"]),
            int(dataset[f"{name}_size"]),
            int(dataset["nuisance_tokens"]),
            device,
            seen,
        )
        for name in ("train", "valid", "test")
    }


def cycle_alignment(
    captures: list[dict[str, torch.Tensor]], batch: dict[str, torch.Tensor]
) -> dict[str, Any]:
    statistics = []
    for capture in captures:
        graph = _graph(capture["scores"], batch["attention_mask"])
        cycle = graph[..., 1, 2] * graph[..., 2, 3] * graph[..., 3, 1]
        statistics.append(cycle.mean(dim=1))
    statistic = torch.stack(statistics).mean(dim=0)
    labels = batch["labels"].bool()
    direct = float((statistic < 0).eq(labels).float().mean())
    flipped = float((statistic >= 0).eq(labels).float().mean())
    return {
        "orientation_invariant_accuracy": max(direct, flipped),
        "preferred_orientation": "negative_is_one" if direct >= flipped else "positive_is_one",
        "mean_absolute_cycle_product": float(statistic.abs().mean()),
        "cycle_product_std": float(statistic.std()),
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = choose_device(args.device or str(config["device"]))
    set_seed(int(config["seed"]))
    splits = make_splits(config, device)
    model, baseline = train_baseline(splits, config, device)
    diagnostics = split_diagnostics(splits)
    diagnostics["geometry"] = {}
    diagnostics["cycle_alignment"] = {}
    replay_logits = {}
    for name in ("valid", "test"):
        captures, replay_logits[name] = collect_scores(model, splits[name])
        batch = tensor_batch(splits[name])
        diagnostics["geometry"][name] = geometry_diagnostics(
            captures, batch, float(config["mechanism"]["walk_time"])
        )
        diagnostics["cycle_alignment"][name] = cycle_alignment(captures, batch)
    diagnostics["maximum_disabled_logit_difference"] = max(
        float((baseline["logits"][name] - replay_logits[name]).abs().max())
        for name in replay_logits
    )
    diagnostics["action_headroom"] = {
        name: oracle_action_headroom(
            model, splits[name], float(config["stage_a_gate"]["oracle_residual_step"])
        )
        for name in ("valid", "test")
    }
    stage_a = stage_a_gate(baseline, diagnostics, config)
    minimum_alignment = float(config["stage_a_gate"]["minimum_cycle_alignment_accuracy"])
    stage_a["valid_cycle_alignment"] = (
        diagnostics["cycle_alignment"]["valid"]["orientation_invariant_accuracy"]
        >= minimum_alignment
    )
    stage_a["test_cycle_alignment"] = (
        diagnostics["cycle_alignment"]["test"]["orientation_invariant_accuracy"]
        >= minimum_alignment
    )
    if not stage_a["valid_cycle_alignment"] or not stage_a["test_cycle_alignment"]:
        stage_a["status"] = "fail"
        stage_a["failure_reason"] = "learned_cycle_sign_not_task_aligned"
        stage_a["stage_b_authorized"] = False
    baseline_predictions = {
        name: baseline["logits"][name].argmax(dim=-1)
        for name in ("train", "valid", "test")
    }
    results = []
    stage_b: dict[str, Any] = {
        "status": "not_run",
        "failure_reason": "stage_a_failed",
        "multi_seed_authorized": False,
        "real_data_authorized": False,
        "hardware_claim_authorized": False,
    }
    if stage_a["stage_b_authorized"]:
        results = [
            train_selector(
                selector,
                model,
                splits,
                baseline_predictions,
                config,
                device,
            )
            for selector in SELECTORS
        ]
        stage_b = stage_b_gate(results, config)
    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / "seed7" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "baseline_model.pt"
    torch.save(model.state_dict(), checkpoint)
    baseline.pop("logits")
    summary = {
        "schema_version": config["schema_version"],
        "status": "complete",
        "revision": git_revision(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "baseline_checkpoint_sha256": sha256(checkpoint),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "seed": int(config["seed"]),
        "dataset_identity": config["dataset"]["identity"],
        "baseline": baseline,
        "diagnostics": diagnostics,
        "stage_a_gate": stage_a,
        "results": results,
        "stage_b_gate": stage_b,
        "design_contract": {
            "labels_defined_only_by_input_signed_triad_parity": True,
            "fixed_role_positions": True,
            "exact_examples_disjoint_with_shared_nuisance_vocabulary": True,
            "complete_relation_model_trained_once_then_frozen": True,
            "scores_captured_from_trained_query_key_projections": True,
            "bridge_and_candidate_masks_used_for_audit_only": True,
            "labels_or_audit_masks_passed_to_intervention": False,
            "parameter_sweep": False,
        },
        "limitations": [
            "This is a fixed-role synthetic structural relation task, not natural language.",
            "Seed 7 is a prequalification gate only.",
            "Exact matrix-exponential simulation does not establish hardware speedup.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "baseline": baseline["metrics"],
                "alignment": diagnostics["cycle_alignment"],
                "stage_a": stage_a,
                "stage_b": stage_b,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
