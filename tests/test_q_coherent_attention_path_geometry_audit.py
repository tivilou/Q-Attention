from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_q_coherent_attention_path_geometry_audit import (  # noqa: E402
    evaluate,
    forward,
    geometry_diagnostics,
    make_split,
    predictions,
    promotion_gate,
)


def config() -> dict:
    return json.loads(
        (
            ROOT / "configs" / "q_coherent_attention_path_geometry_audit.json"
        ).read_text(encoding="utf-8")
    )


def split(size: int = 96) -> dict[str, torch.Tensor]:
    cfg = config()
    return make_split(
        7,
        size,
        torch.device("cpu"),
        weight_jitter=float(cfg["dataset"]["weight_jitter"]),
        nuisance_weight=float(cfg["dataset"]["nuisance_weight"]),
    )


def test_geometry_suite_is_balanced_diverse_and_reproducible() -> None:
    first = split()
    second = split()
    assert torch.equal(first["scores"], second["scores"])
    assert int(first["labels"].sum()) == 48
    assert torch.unique(first["scores"].reshape(96, -1), dim=0).shape[0] >= 90
    assert torch.unique(first["permutations"], dim=0).shape[0] >= 80


def test_geometry_preserves_signed_margin_and_control_symmetry() -> None:
    cfg = config()
    diagnostics = geometry_diagnostics(split(), cfg)
    gate = cfg["gate"]
    assert diagnostics["minimum_signed_target_gap"] >= gate[
        "minimum_signed_path_gap"
    ]
    assert diagnostics["maximum_absolute_unsigned_target_gap"] <= gate[
        "maximum_control_path_gap"
    ]
    assert diagnostics["maximum_absolute_classical_target_gap"] <= gate[
        "maximum_control_path_gap"
    ]
    assert diagnostics["maximum_gauge_probability_error"] <= gate[
        "maximum_gauge_error"
    ]
    assert diagnostics["maximum_permutation_residual_error"] <= gate[
        "maximum_permutation_error"
    ]


def test_geometry_audit_passes_only_signed_q_wap() -> None:
    cfg = config()
    dataset = split()
    baseline_logits = forward(None, dataset)[2]
    baseline_prediction = predictions(
        baseline_logits, float(cfg["mechanism"]["tie_tolerance"])
    )
    results = [
        evaluate(selector, dataset, baseline_prediction, cfg)
        for selector in cfg["selectors"]
    ]
    geometry = geometry_diagnostics(dataset, cfg)
    gate = promotion_gate(results, geometry, cfg)
    by_selector = {row["selector"]: row for row in results}
    assert by_selector["disabled"]["accuracy"] == 0.5
    assert by_selector["q_wap_signed"]["accuracy"] >= 0.95
    assert by_selector["q_wap_unsigned"]["accuracy"] <= 0.55
    assert by_selector["classical_wap_diffusion"]["accuracy"] <= 0.55
    assert gate["status"] == "pass"
    assert gate["fresh_attention_geometry_benchmark_authorized"] is True
    assert gate["existing_task_benchmark_authorized"] is False
