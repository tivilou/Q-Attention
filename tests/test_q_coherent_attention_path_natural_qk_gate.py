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

from run_q_coherent_attention_path_natural_qk_gate import (  # noqa: E402
    evaluate,
    forward,
    geometry,
    make_split,
    predictions,
    promotion_gate,
)


def config() -> dict:
    return json.loads(
        (ROOT / "configs" / "q_coherent_attention_path_natural_qk_gate.json").read_text(
            encoding="utf-8"
        )
    )


def test_fixed_qk_encoder_is_balanced_diverse_and_reproducible() -> None:
    cfg = config()
    first = make_split(7, 96, torch.device("cpu"), cfg)
    second = make_split(7, 96, torch.device("cpu"), cfg)
    assert torch.equal(first["scores"], second["scores"])
    assert torch.equal(first["query"], second["query"])
    assert int(first["labels"].sum()) == 48
    assert geometry(first)["unique_scores"] >= 90
    assert geometry(first)["minimum_score_rank"] >= 4
    assert geometry(first)["maximum_qk_reconstruction_error"] <= 1e-5


def test_natural_qk_controls_tie_and_signed_path_separates() -> None:
    cfg = config()
    split = make_split(7, 96, torch.device("cpu"), cfg)
    baseline = predictions(
        forward(None, split)[2], float(cfg["mechanism"]["tie_tolerance"])
    )
    results = [evaluate(selector, split, baseline, cfg) for selector in cfg["selectors"]]
    by = {row["selector"]: row for row in results}
    assert by["disabled"]["accuracy"] == 0.5
    assert by["q_wap_signed"]["accuracy"] >= 0.95
    assert by["q_wap_unsigned"]["accuracy"] <= 0.55
    assert by["classical_wap_diffusion"]["accuracy"] <= 0.55
    assert abs(by["q_wap_unsigned"]["target_minus_distractor_attention"]) <= 1e-5
    assert abs(by["classical_wap_diffusion"]["target_minus_distractor_attention"]) <= 1e-5


def test_natural_qk_gate_keeps_task_and_hardware_claims_closed() -> None:
    cfg = config()
    split = make_split(7, 96, torch.device("cpu"), cfg)
    baseline = predictions(
        forward(None, split)[2], float(cfg["mechanism"]["tie_tolerance"])
    )
    results = [evaluate(selector, split, baseline, cfg) for selector in cfg["selectors"]]
    gate = promotion_gate(results, geometry(split), cfg)
    assert gate["status"] == "pass"
    assert gate["natural_qk_attention_benchmark_authorized"] is True
    assert gate["existing_task_benchmark_authorized"] is False
    assert gate["five_seed_phase_authorized"] is False
    assert gate["real_data_authorized"] is False
