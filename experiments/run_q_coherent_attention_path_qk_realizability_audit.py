#!/usr/bin/env python3
"""Audit whether robust Q-WAP geometry is realizable as QK attention scores."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_geometry_audit import (  # noqa: E402
    NODES,
    evaluate,
    forward,
    geometry_diagnostics,
    make_split as make_geometry_split,
    predictions,
    promotion_gate as geometry_promotion_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/q_coherent_attention_path_qk_realizability_audit.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "q-attention.coherent-attention-path-qk-realizability.v1"
    ):
        raise ValueError("unsupported Q-WAP QK-realizability config")
    if int(payload.get("seed", -1)) != 7:
        raise ValueError("Q-WAP QK-realizability audit requires fixed seed 7")
    if int(payload["dataset"]["nodes"]) != NODES:
        raise ValueError(f"Q-WAP QK-realizability audit requires {NODES} nodes")
    if abs(float(payload["mechanism"]["walk_time"]) - math.pi / 4) > 1e-12:
        raise ValueError("Q-WAP walk time is frozen at pi/4")
    if tuple(payload.get("selectors", ())) != (
        "disabled",
        "q_wap_signed",
        "q_wap_unsigned",
        "classical_wap_diffusion",
    ):
        raise ValueError("Q-WAP selectors must match the frozen allowlist")
    return payload


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _skew_score_perturbation(
    anchors: torch.Tensor,
    *,
    seed: int,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 17000)
    random_scores = (
        2.0 * torch.rand(anchors.shape[0], NODES, NODES, generator=generator) - 1.0
    )
    skew = 0.5 * (random_scores - random_scores.transpose(-1, -2)) * scale
    row = torch.arange(anchors.shape[0])
    cpu_anchors = anchors.detach().cpu()
    skew[row, cpu_anchors, :] = 0.0
    skew[row, :, cpu_anchors] = 0.0
    return skew.to(dtype=dtype)


def factorize_attention_scores(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Q and K such that QK^T/sqrt(d) reconstructs scores."""
    if scores.shape[-2:] != (NODES, NODES):
        raise ValueError("QK factorization requires square seven-node scores")
    u, singular_values, vh = torch.linalg.svd(scores)
    root = singular_values.clamp_min(0.0).sqrt()
    attention_scale = NODES ** 0.25
    query = u * root.unsqueeze(-2) * attention_scale
    key = vh.transpose(-1, -2) * root.unsqueeze(-2) * attention_scale
    return query, key


def make_qk_split(
    seed: int,
    size: int,
    device: torch.device,
    *,
    weight_jitter: float,
    nuisance_weight: float,
    skew_score_scale: float,
) -> dict[str, torch.Tensor]:
    split = make_geometry_split(
        seed,
        size,
        torch.device("cpu"),
        weight_jitter=weight_jitter,
        nuisance_weight=nuisance_weight,
    )
    intended_hamiltonian = split["scores"].clone()
    skew = _skew_score_perturbation(
        split["anchor"],
        seed=seed,
        scale=skew_score_scale,
        dtype=intended_hamiltonian.dtype,
    ).unsqueeze(1)
    designed_scores = intended_hamiltonian + skew
    query, key = factorize_attention_scores(designed_scores)
    reconstructed_scores = torch.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(
        NODES
    )

    padded_value = torch.zeros(
        size, 1, NODES, NODES, dtype=split["value"].dtype
    )
    padded_value[..., : split["value"].shape[-1]] = split["value"]
    split.update(
        {
            "query": query,
            "key": key,
            "value": padded_value,
            "scores": reconstructed_scores,
            "intended_hamiltonian": intended_hamiltonian,
            "designed_scores": designed_scores,
            "skew_score_perturbation": skew,
        }
    )
    return {
        key_name: value.to(device) if isinstance(value, torch.Tensor) else value
        for key_name, value in split.items()
    }


def realizability_diagnostics(split: dict[str, torch.Tensor]) -> dict[str, Any]:
    reconstructed = torch.einsum(
        "bhqd,bhkd->bhqk", split["query"], split["key"]
    ) / math.sqrt(split["query"].shape[-1])
    realized_hamiltonian = 0.5 * (
        split["scores"] + split["scores"].transpose(-1, -2)
    )
    asymmetry = split["scores"] - split["scores"].transpose(-1, -2)
    row = torch.arange(split["scores"].shape[0], device=split["scores"].device)
    anchor_rows = split["scores"][row, 0, split["anchor"], :]
    intended_anchor_rows = split["intended_hamiltonian"][
        row, 0, split["anchor"], :
    ]
    ranks = torch.linalg.matrix_rank(split["scores"][:, 0])
    return {
        "attention_head_dim": int(split["query"].shape[-1]),
        "minimum_raw_score_rank": int(ranks.min()),
        "maximum_qk_reconstruction_error": float(
            (reconstructed - split["designed_scores"]).abs().max()
        ),
        "maximum_hamiltonian_reconstruction_error": float(
            (realized_hamiltonian - split["intended_hamiltonian"]).abs().max()
        ),
        "maximum_raw_score_asymmetry": float(asymmetry.abs().max()),
        "maximum_anchor_row_perturbation": float(
            (anchor_rows - intended_anchor_rows).abs().max()
        ),
        "finite_query_key": bool(
            torch.isfinite(split["query"]).all()
            and torch.isfinite(split["key"]).all()
        ),
    }


def result_replay_error(
    symmetric_results: list[dict[str, Any]],
    qk_results: list[dict[str, Any]],
) -> float:
    symmetric = {row["selector"]: row for row in symmetric_results}
    qk = {row["selector"]: row for row in qk_results}
    fields = (
        "accuracy",
        "target_attention",
        "distractor_attention",
        "target_minus_distractor_attention",
    )
    return max(
        abs(float(qk[selector][field]) - float(symmetric[selector][field]))
        for selector in qk
        for field in fields
    )


def promotion_gate(
    geometry_gate: dict[str, Any],
    realizability: dict[str, Any],
    symmetric_qk_metric_error: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    inherited = {
        key: value
        for key, value in geometry_gate.items()
        if key
        not in {
            "status",
            "fresh_attention_geometry_benchmark_authorized",
            "existing_task_benchmark_authorized",
            "five_seed_phase_authorized",
            "real_data_authorized",
        }
    }
    conditions = {
        **inherited,
        "finite_query_key": realizability["finite_query_key"],
        "full_attention_score_rank": realizability["minimum_raw_score_rank"]
        == NODES,
        "qk_reconstruction": realizability["maximum_qk_reconstruction_error"]
        <= float(gate["maximum_qk_reconstruction_error"]),
        "hamiltonian_reconstruction": realizability[
            "maximum_hamiltonian_reconstruction_error"
        ]
        <= float(gate["maximum_hamiltonian_reconstruction_error"]),
        "nontrivial_score_asymmetry": realizability["maximum_raw_score_asymmetry"]
        >= float(gate["minimum_raw_score_asymmetry"]),
        "anchor_row_is_unperturbed": realizability["maximum_anchor_row_perturbation"]
        <= float(gate["maximum_anchor_row_perturbation"]),
        "symmetric_qk_metric_replay": symmetric_qk_metric_error
        <= float(gate["maximum_symmetric_qk_metric_error"]),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "status": "pass" if passed else "fail",
        "qk_derived_attention_benchmark_authorized": passed,
        "existing_task_benchmark_authorized": False,
        "five_seed_phase_authorized": False,
        "real_data_authorized": False,
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    device = choose_device(args.device or str(config["device"]))
    dataset = config["dataset"]
    qk_split = make_qk_split(
        int(config["seed"]),
        int(dataset["size"]),
        device,
        weight_jitter=float(dataset["weight_jitter"]),
        nuisance_weight=float(dataset["nuisance_weight"]),
        skew_score_scale=float(dataset["skew_score_scale"]),
    )
    symmetric_split = make_geometry_split(
        int(config["seed"]),
        int(dataset["size"]),
        device,
        weight_jitter=float(dataset["weight_jitter"]),
        nuisance_weight=float(dataset["nuisance_weight"]),
    )
    tolerance = float(config["mechanism"]["tie_tolerance"])
    with torch.no_grad():
        qk_baseline = predictions(forward(None, qk_split)[2], tolerance)
        symmetric_baseline = predictions(forward(None, symmetric_split)[2], tolerance)
    qk_results = [
        evaluate(selector, qk_split, qk_baseline, config)
        for selector in config["selectors"]
    ]
    symmetric_results = [
        evaluate(selector, symmetric_split, symmetric_baseline, config)
        for selector in config["selectors"]
    ]
    geometry = geometry_diagnostics(qk_split, config)
    geometry_gate = geometry_promotion_gate(qk_results, geometry, config)
    realizability = realizability_diagnostics(qk_split)
    metric_error = result_replay_error(symmetric_results, qk_results)
    gate = promotion_gate(geometry_gate, realizability, metric_error, config)

    output_root = Path(args.output_root or str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output = output_root / "seed7" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": config["schema_version"],
        "status": "complete",
        "revision": git_revision(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "dataset_identity": dataset["identity"],
        "seed": int(config["seed"]),
        "geometry": geometry,
        "realizability": realizability,
        "symmetric_qk_metric_replay_error": metric_error,
        "results": qk_results,
        "symmetric_reference_results": symmetric_results,
        "gate": gate,
        "design_contract": {
            "scores_recomputed_from_qk": True,
            "attention_scale": f"1/sqrt({NODES})",
            "qkv_head_dimensions_match": True,
            "fixed_walk_time": float(config["mechanism"]["walk_time"]),
            "skew_score_component_cancels_under_hermitian_symmetrization": True,
            "label_input_to_plugin_or_factorization": False,
            "target_input_to_plugin_or_factorization": False,
            "parameter_sweep": False,
        },
        "limitations": [
            "The Q and K factors are constructed algebraically rather than learned from language inputs.",
            "Exact QK realizability establishes an attention interface, not structured-NLP task utility.",
            "A pass does not authorize old failed splits, five seeds, real data, or hardware-speedup claims.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
