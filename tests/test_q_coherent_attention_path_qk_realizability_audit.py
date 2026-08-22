from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_coherent_attention_path_geometry_audit import (  # noqa: E402
    evaluate,
    forward,
    geometry_diagnostics,
    make_split as make_geometry_split,
    predictions,
    promotion_gate as geometry_promotion_gate,
)
from run_q_coherent_attention_path_qk_realizability_audit import (  # noqa: E402
    make_qk_split,
    promotion_gate,
    realizability_diagnostics,
    result_replay_error,
)


def config() -> dict:
    return json.loads(
        (
            ROOT
            / "configs"
            / "q_coherent_attention_path_qk_realizability_audit.json"
        ).read_text(encoding="utf-8")
    )


def splits(size: int = 96) -> tuple[dict, dict]:
    cfg = config()
    dataset = cfg["dataset"]
    qk = make_qk_split(
        7,
        size,
        torch.device("cpu"),
        weight_jitter=float(dataset["weight_jitter"]),
        nuisance_weight=float(dataset["nuisance_weight"]),
        skew_score_scale=float(dataset["skew_score_scale"]),
    )
    symmetric = make_geometry_split(
        7,
        size,
        torch.device("cpu"),
        weight_jitter=float(dataset["weight_jitter"]),
        nuisance_weight=float(dataset["nuisance_weight"]),
    )
    return qk, symmetric


def test_qk_factorization_reconstructs_nontrivial_asymmetric_scores() -> None:
    cfg = config()
    qk, _ = splits(16)
    diagnostics = realizability_diagnostics(qk)
    gate = cfg["gate"]
    assert qk["query"].shape == qk["key"].shape == (16, 1, 7, 7)
    assert qk["value"].shape == (16, 1, 7, 7)
    assert diagnostics["maximum_qk_reconstruction_error"] <= gate[
        "maximum_qk_reconstruction_error"
    ]
    assert diagnostics["maximum_hamiltonian_reconstruction_error"] <= gate[
        "maximum_hamiltonian_reconstruction_error"
    ]
    assert diagnostics["maximum_raw_score_asymmetry"] >= gate[
        "minimum_raw_score_asymmetry"
    ]
    assert diagnostics["maximum_anchor_row_perturbation"] <= gate[
        "maximum_anchor_row_perturbation"
    ]


def test_qk_scores_preserve_geometry_results() -> None:
    cfg = config()
    qk, symmetric = splits()
    tolerance = float(cfg["mechanism"]["tie_tolerance"])
    qk_baseline = predictions(forward(None, qk)[2], tolerance)
    symmetric_baseline = predictions(forward(None, symmetric)[2], tolerance)
    qk_results = [
        evaluate(selector, qk, qk_baseline, cfg) for selector in cfg["selectors"]
    ]
    symmetric_results = [
        evaluate(selector, symmetric, symmetric_baseline, cfg)
        for selector in cfg["selectors"]
    ]
    assert result_replay_error(symmetric_results, qk_results) <= cfg["gate"][
        "maximum_symmetric_qk_metric_error"
    ]
    by_selector = {row["selector"]: row for row in qk_results}
    assert by_selector["disabled"]["accuracy"] == 0.5
    assert by_selector["q_wap_signed"]["accuracy"] >= 0.95
    assert by_selector["q_wap_unsigned"]["accuracy"] <= 0.55
    assert by_selector["classical_wap_diffusion"]["accuracy"] <= 0.55


def test_qk_realizability_gate_does_not_authorize_existing_tasks() -> None:
    cfg = config()
    qk, symmetric = splits()
    tolerance = float(cfg["mechanism"]["tie_tolerance"])
    qk_baseline = predictions(forward(None, qk)[2], tolerance)
    symmetric_baseline = predictions(forward(None, symmetric)[2], tolerance)
    qk_results = [
        evaluate(selector, qk, qk_baseline, cfg) for selector in cfg["selectors"]
    ]
    symmetric_results = [
        evaluate(selector, symmetric, symmetric_baseline, cfg)
        for selector in cfg["selectors"]
    ]
    geometry = geometry_diagnostics(qk, cfg)
    geometry_gate = geometry_promotion_gate(qk_results, geometry, cfg)
    gate = promotion_gate(
        geometry_gate,
        realizability_diagnostics(qk),
        result_replay_error(symmetric_results, qk_results),
        cfg,
    )
    assert gate["status"] == "pass"
    assert gate["qk_derived_attention_benchmark_authorized"] is True
    assert gate["existing_task_benchmark_authorized"] is False
    assert gate["five_seed_phase_authorized"] is False
