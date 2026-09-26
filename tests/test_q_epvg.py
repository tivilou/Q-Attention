from __future__ import annotations

import json
from pathlib import Path
import sys
import runpy

import torch

from q_attention.plugins.q_epvg import (
    EPVG_CONTROLS,
    EPVG_OBSERVABLES,
    EPVG_PATHS,
    QEPVGConfig,
    build_q_epvg,
)


def inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(19)
    query = torch.randn(2, 1, 5, 4, generator=generator)
    key = torch.randn(2, 1, 5, 4, generator=generator)
    value = torch.randn(2, 1, 5, 4, generator=generator)
    scores = torch.randn(2, 1, 5, 5, generator=generator)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, -1] = False
    return {"query": query, "key": key, "value": value, "scores": scores, "attention_mask": mask, "query_mask": mask.clone()}


def test_matrix_choices_and_fixed_maps() -> None:
    for observable in EPVG_OBSERVABLES:
        for path in EPVG_PATHS:
            model = build_q_epvg(QEPVGConfig(observable=observable, path=path))
            assert model.reducer.requires_grad is False
            assert model.linear_map.requires_grad is False
            output, trace = model(**inputs(), return_trace=True)
            assert output.shape == (2, 1, 5, 4)
            assert torch.isfinite(output).all()
            assert torch.isfinite(trace["observable"]).all()
            assert torch.all(trace["observable"].abs() <= 1.0 + 1e-6)
            assert torch.allclose(trace["theta"], model.config.gate_scale * trace["observable"])


def test_controls_have_equal_parameter_budget_and_masks_hold() -> None:
    models = [build_q_epvg(QEPVGConfig()) for _ in EPVG_CONTROLS]
    assert len({sum(p.numel() for p in model.parameters()) for model in models}) == 1
    for control, model in zip(EPVG_CONTROLS, models):
        _, trace = model(**inputs(), control=control, return_trace=True)
        assert torch.allclose(trace["attention"][..., -1], torch.zeros_like(trace["attention"][..., -1]), atol=1e-7)
        assert torch.allclose(trace["attention"].sum(dim=-1), torch.ones_like(trace["attention"].sum(dim=-1)), atol=1e-5)


def test_gradients_are_finite() -> None:
    model = build_q_epvg(QEPVGConfig(observable="trainable_pauli_mix", path="score_value"))
    output, trace = model(**inputs(), return_trace=True)
    (output.square().mean() + trace["observable"].square().mean()).backward()
    assert model.pauli_logits.grad is not None
    assert torch.isfinite(model.pauli_logits.grad).all()


def test_runner_emits_complete_matrix_and_sample_trace(tmp_path: Path) -> None:
    runner = Path(__file__).resolve().parents[1] / "experiments" / "run_q_epvg_toy.py"
    old_argv = sys.argv
    try:
        sys.argv = [str(runner), "--seeds", "13", "--output-root", str(tmp_path / "epvg")]
        runpy.run_path(str(runner), run_name="__main__")
    finally:
        sys.argv = old_argv
    summary = json.loads((tmp_path / "epvg" / "summary.json").read_text(encoding="utf-8"))
    trace = json.loads((tmp_path / "epvg" / "sample-trace.json").read_text(encoding="utf-8"))
    assert summary["toy_gates"] == {"all_finite": True, "all_masks_respected": True, "all_row_sums_unit": True, "matrix_complete": True, "control_outputs_present": True}
    assert len(summary["results"]) == 27
    assert trace["schema"] == "sample-trace.v1"
    assert {item["split"] for item in trace["traces"]} == {"train", "valid", "test"}
    assert {item["checkpoint"] for item in trace["traces"]} == {"initial", "best", "final"}
    required = {"query", "key", "value", "zz", "xx", "observable", "theta", "gate", "base_attention", "score_adjustment", "attention", "routed_values", "query_update", "output"}
    assert required.issubset(trace["traces"][0]["intermediate"])
