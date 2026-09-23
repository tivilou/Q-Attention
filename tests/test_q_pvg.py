from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from q_attention.plugins.q_pvg import QPVGConfig, build_q_pvg


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(17)
    scores = torch.randn(2, 1, 5, 5, generator=generator)
    query = torch.randn(2, 1, 5, 4, generator=generator)
    key = torch.randn(2, 1, 5, 4, generator=generator)
    value = torch.randn(2, 1, 5, 4, generator=generator)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    attention_mask[:, -1] = False
    query_mask = torch.ones(2, 5, dtype=torch.bool)
    query_mask[:, -1] = False
    return scores, query, key, value, attention_mask, query_mask


def _build(**overrides):
    config = QPVGConfig(
        num_layers=1,
        num_heads=1,
        head_dim=4,
        register_qubits=2,
        depth=2,
        seed=23,
        **overrides,
    )
    return build_q_pvg(config)


def _run(kernel, inputs, **kwargs):
    scores, query, key, value, attention_mask, query_mask = inputs
    return kernel(
        query,
        key,
        value,
        scores=scores,
        layer_index=0,
        attention_mask=attention_mask,
        query_mask=query_mask,
        return_trace=True,
        **kwargs,
    )


def test_quantum_and_classical_controls_are_parameter_matched() -> None:
    quantum = _build(readout_mode="quantum")
    classical = _build(readout_mode="classical")
    assert sum(p.numel() for p in quantum.parameters()) == sum(
        p.numel() for p in classical.parameters()
    )


def test_quantum_and_classical_share_the_same_prepared_states() -> None:
    inputs = _inputs()
    quantum = _build(readout_mode="quantum")
    classical = _build(readout_mode="classical")
    _, quantum_trace = _run(quantum, inputs)
    _, classical_trace = _run(classical, inputs)
    for name in ("query_state_real", "query_state_imag", "key_state_real", "key_state_imag"):
        assert torch.allclose(quantum_trace[name], classical_trace[name], atol=1e-6)
    assert torch.allclose(
        classical_trace["gate_imag"], torch.zeros_like(classical_trace["gate_imag"])
    )


def test_explicit_state_preparation_is_normalized() -> None:
    _, trace = _run(_build(), _inputs())
    query_norm = trace["query_state_real"].square() + trace["query_state_imag"].square()
    key_norm = trace["key_state_real"].square() + trace["key_state_imag"].square()
    assert torch.allclose(query_norm.sum(dim=-1), torch.ones_like(query_norm[..., 0]), atol=1e-5)
    assert torch.allclose(key_norm.sum(dim=-1), torch.ones_like(key_norm[..., 0]), atol=1e-5)


def test_phase_reversal_changes_complex_alignment() -> None:
    inputs = _inputs()
    kernel = _build()
    _, plus = _run(kernel, inputs, phase_sign=1.0)
    _, minus = _run(kernel, inputs, phase_sign=-1.0)
    assert torch.isfinite(plus["alignment_imag"]).all()
    assert not torch.allclose(plus["alignment_imag"], minus["alignment_imag"])
    assert not torch.allclose(plus["gate"], minus["gate"])


def test_real_only_control_removes_imaginary_gate_channel() -> None:
    inputs = _inputs()
    complex_kernel = _build(phase_mode="complex")
    real_kernel = _build(phase_mode="real_only")
    _, complex_trace = _run(complex_kernel, inputs)
    _, real_trace = _run(real_kernel, inputs)
    assert torch.allclose(real_trace["gate_imag"], torch.zeros_like(real_trace["gate_imag"]))
    assert not torch.allclose(complex_trace["gate"], real_trace["gate"])


def test_value_route_is_value_sensitive_and_fixed_attention_is_preserved() -> None:
    inputs = list(_inputs())
    kernel = _build(score_mode="value_only")
    output, trace = _run(kernel, tuple(inputs))
    assert output.shape == (2, 1, 5, 4)
    assert torch.allclose(trace["attention"], trace["base_attention"], atol=1e-6)
    inputs[3] = inputs[3].clone()
    inputs[3][:, :, 2, 0] += 2.0
    changed, _ = _run(kernel, tuple(inputs))
    assert not torch.allclose(output, changed)


def test_mask_is_respected_and_outputs_are_finite() -> None:
    inputs = _inputs()
    kernel = _build(score_mode="score_value")
    output, trace = _run(kernel, inputs)
    assert torch.isfinite(output).all()
    assert torch.isfinite(trace["gate"]).all()
    assert torch.allclose(trace["attention"][..., -1], torch.zeros_like(trace["attention"][..., -1]))
    assert torch.allclose(output[..., -1, :], torch.zeros_like(output[..., -1, :]))


def test_backward_gradients_are_finite() -> None:
    kernel = _build()
    output, trace = _run(kernel, _inputs())
    loss = output.square().mean() + trace["alignment_real"].square().mean()
    loss = loss + trace["alignment_imag"].square().mean()
    loss.backward()
    populated = 0
    for parameter in kernel.parameters():
        if parameter.grad is not None:
            populated += 1
            assert torch.isfinite(parameter.grad).all()
    assert populated >= 6


@pytest.mark.parametrize("route_mode", ["branch_interpolation", "scalar"])
def test_both_value_route_variants_run(route_mode: str) -> None:
    output, trace = _run(_build(value_route_mode=route_mode), _inputs())
    assert output.shape[-1] == 4
    assert trace["routed_values"].shape[-1] == 4


def test_toy_runner_emits_summary_and_complete_case_trace(tmp_path: Path) -> None:
    runner = Path(__file__).resolve().parents[1] / "experiments" / "run_q_pvg_toy.py"
    output_root = tmp_path / "q_pvg_toy"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--seeds",
            "7",
            "--steps",
            "3",
            "--train-size",
            "8",
            "--valid-size",
            "4",
            "--batch-size",
            "4",
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    case_study = json.loads((output_root / "case_study.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "q-pvg-toy.v1"
    assert set(summary["selectors"]) >= {"q_pvg_phase_value", "q_pvg_real_only"}
    assert summary["toy_gates"]["all_finite"]
    assert summary["toy_gates"]["all_masks_respected"]
    assert summary["toy_gates"]["phase_reversal_detected"]
    assert summary["toy_gates"]["control_distances_present"]
    assert case_study["schema"] == "sample-trace.v1"
    assert len(case_study["traces"]) == len(summary["selectors"]) * 3
    assert {item["split"] for item in case_study["traces"]} == {"train", "valid", "test"}
    sample = case_study["traces"][1]["samples"][0]
    assert sample["semantic"]["sentence"]
    assert "alignment_real" in sample["intermediate"]
    assert "routed_values" in sample["intermediate"]
