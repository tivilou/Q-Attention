#!/usr/bin/env python3
"""Stage-A gate for signed-triad relation extraction with relocated evidence."""

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


ROLE_NAMES = ("subject", "object", "bridge")
ROLE_POSITIONS = {"subject": 1, "object": 2, "bridge": 3}
ROLE_TOKEN_IDS = {
    "subject_generic": 2,
    "object_generic": 3,
    "bridge_generic": 4,
    "subject_positive": 5,
    "subject_negative": 6,
    "object_positive": 7,
    "object_negative": 8,
    "bridge_positive": 9,
    "bridge_negative": 10,
}
ANCHOR_ID = 1
CANDIDATE_ZERO_ID = 11
CANDIDATE_ONE_ID = 12
MOVED_START = 13
NUISANCE_START = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/q_partial_evidence_triad_gate.json"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.q-partial-evidence-triad-gate.v1":
        raise ValueError("unsupported partial-evidence config")
    if int(config.get("seed", -1)) != 7:
        raise ValueError("partial-evidence gate requires fixed seed 7")
    if tuple(config.get("selectors", ())) != SELECTORS:
        raise ValueError("selectors must match the frozen trained-baseline allowlist")
    streams = [
        int(config["dataset"][f"{name}_stream"])
        for name in ("train", "valid", "test")
    ]
    if len(set(streams)) != 3:
        raise ValueError("train, valid, and test streams must be distinct")
    if int(config["dataset"]["nodes"]) != 16:
        raise ValueError("partial-evidence task requires sixteen positions")
    return config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role_token(role: str, polarity: int) -> int:
    sign = "negative" if polarity else "positive"
    return ROLE_TOKEN_IDS[f"{role}_{sign}"]


def _moved_token(role_index: int, polarity: int) -> int:
    return MOVED_START + 2 * role_index + int(polarity)


def make_split(
    stream: int,
    size: int,
    nuisance_tokens: int,
    relocation_probability: float,
    device: torch.device,
    seen: set[tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    if size <= 0 or size % 2:
        raise ValueError("split size must be positive and even")
    if nuisance_tokens < 16:
        raise ValueError("at least sixteen nuisance tokens are required")
    if not 0.0 < relocation_probability < 1.0:
        raise ValueError("relocation probability must lie in (0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(stream)
    labels = torch.arange(size, dtype=torch.long) % 2
    labels = labels[torch.randperm(size, generator=generator)]
    seen = seen if seen is not None else set()
    rows: list[torch.Tensor] = []
    cycle_labels: list[int] = []
    relocated_masks: list[torch.Tensor] = []
    fingerprints: list[str] = []
    for target in labels.tolist():
        for _attempt in range(100000):
            polarities = torch.randint(0, 2, (3,), generator=generator)
            cycle = int(int(polarities.sum()) % 2 == 1)
            relocated = torch.rand(3, generator=generator) < relocation_probability
            if not bool(relocated.any()):
                relocated[0] = True
            row = torch.empty(16, dtype=torch.long)
            row[0] = ANCHOR_ID
            row[1] = ROLE_TOKEN_IDS["subject_generic"]
            row[2] = ROLE_TOKEN_IDS["object_generic"]
            row[3] = ROLE_TOKEN_IDS["bridge_generic"]
            for role_index, role in enumerate(ROLE_NAMES):
                if not bool(relocated[role_index]):
                    row[ROLE_POSITIONS[role]] = _role_token(role, int(polarities[role_index]))
            row[4] = CANDIDATE_ZERO_ID
            row[5] = CANDIDATE_ONE_ID
            available_positions = list(range(6, 16))
            marker_positions = []
            for role_index, role in enumerate(ROLE_NAMES):
                if bool(relocated[role_index]):
                    position_index = int(
                        torch.randint(len(available_positions), (1,), generator=generator)
                    )
                    position = available_positions.pop(position_index)
                    marker_positions.append(position)
                    row[position] = _moved_token(role_index, int(polarities[role_index]))
            for position in available_positions:
                row[position] = NUISANCE_START + int(
                    torch.randint(nuisance_tokens, (1,), generator=generator)
                )
            key = tuple(int(value) for value in row)
            if cycle != target or key in seen:
                continue
            seen.add(key)
            rows.append(row)
            cycle_labels.append(cycle)
            relocated_masks.append(relocated)
            fingerprints.append(hashlib.sha256(bytes(key)).hexdigest())
            break
        else:
            raise RuntimeError("could not construct a unique partial-evidence example")
    input_ids = torch.stack(rows).to(device)
    masks: dict[str, torch.Tensor] = {}
    for name, index in (
        ("subject_mask", 1),
        ("object_mask", 2),
        ("bridge_mask", 3),
        ("candidate_zero_mask", 4),
        ("candidate_one_mask", 5),
    ):
        mask = torch.zeros(size, 16, dtype=torch.bool)
        mask[:, index] = True
        masks[name] = mask.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        **masks,
        "labels": labels.to(device),
        "cycle_labels": torch.tensor(cycle_labels, dtype=torch.long, device=device),
        "relocated_roles": torch.stack(relocated_masks).to(device),
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
            float(dataset["relocation_probability"]),
            device,
            seen,
        )
        for name in ("train", "valid", "test")
    }


def cycle_alignment(
    captures: list[dict[str, torch.Tensor]], batch: dict[str, torch.Tensor]
) -> dict[str, Any]:
    products = []
    for capture in captures:
        graph = _graph(capture["scores"], batch["attention_mask"])
        product = graph[..., 1, 2] * graph[..., 2, 3] * graph[..., 3, 1]
        products.append(product.mean(dim=1))
    statistic = torch.stack(products).mean(dim=0)
    labels = batch["cycle_labels"].bool()
    direct = float((statistic < 0).eq(labels).float().mean())
    flipped = float((statistic >= 0).eq(labels).float().mean())
    relocated = batch["relocated_roles"].any(dim=1)
    clear = ~relocated
    return {
        "orientation_invariant_accuracy": max(direct, flipped),
        "clear_example_accuracy": max(
            float((statistic[clear] < 0).eq(labels[clear]).float().mean())
            if clear.any()
            else 0.0,
            float((statistic[clear] >= 0).eq(labels[clear]).float().mean())
            if clear.any()
            else 0.0,
        ),
        "relocated_example_fraction": float(relocated.float().mean()),
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
            "labels_are_signed_triad_cycle_parity": True,
            "relocated_role_markers_are_input_evidence_not_label_noise": True,
            "cycle_alignment_audits_latent_cycle_bit": True,
            "fixed_entity_role_positions": True,
            "complete_relation_model_trained_once_then_frozen": True,
            "scores_captured_from_trained_query_key_projections": True,
            "audit_masks_not_passed_to_intervention": True,
            "train_valid_test_streams_distinct": True,
            "parameter_sweep": False,
        },
        "limitations": [
            "This is a synthetic fixed-role structural relation task, not natural language.",
            "Seed 7 is a Stage-A prequalification only.",
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
                "headroom": diagnostics["action_headroom"],
                "stage_a": stage_a,
                "stage_b": stage_b,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
