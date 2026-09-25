from __future__ import annotations

import torch

from experiments.run_qceasc_counterfactual_preflight import (
    _expected_cost,
    _run_stability_probes,
)


def test_expected_cost_exposes_quadratic_leave_one_out_budget() -> None:
    cost = _expected_cost(
        batch_size=2,
        heads=3,
        context_size=16,
        chunk_size=4,
        leave_one_out_chunk_size=4,
        head_dim=8,
    )
    assert cost["active_context_evaluations_per_query_row"] == 17
    assert cost["base_evaluations"] == 2 * 3 * 16 * 17
    assert cost["query_chunks_per_head"] == 4
    assert cost["peak_influence_elements_per_constructor_call"] == 2 * 4 * 16 * 8


def test_query_chunk_does_not_hide_total_base_evaluation_count() -> None:
    small = _expected_cost(
        batch_size=1,
        heads=1,
        context_size=64,
        chunk_size=1,
        leave_one_out_chunk_size=1,
        head_dim=16,
    )
    large = _expected_cost(
        batch_size=1,
        heads=1,
        context_size=64,
        chunk_size=64,
        leave_one_out_chunk_size=64,
        head_dim=16,
    )
    assert small["base_evaluations"] == large["base_evaluations"] == 64 * 65
    assert small["peak_influence_elements_per_constructor_call"] < large["peak_influence_elements_per_constructor_call"]


def test_stability_probe_uses_direct_constructor_query_shape() -> None:
    raw = {
        "head_dim": 4,
        "auxiliary_qubits": 2,
        "depth": 1,
        "support_width": 1,
        "action_rank": 2,
        "angle_scale": 1.0,
        "max_gain": 0.25,
        "initial_gain": 0.05,
        "span_rcond": 1e-6,
        "max_context_size": 8,
        "stability_context_size": 8,
        "gradient_context_size": 4,
        "seed": 37,
    }
    probes = _run_stability_probes(raw, device=torch.device("cpu"))
    assert probes["mask_zero_norm_float32"]["finite"]
    assert probes["mask_zero_norm_float64"]["finite"]
    assert probes["backward_float32"]["status"] == "ok"
    assert probes["backward_float32"]["finite"]


def test_score_kernel_exposes_production_model_dimensions() -> None:
    from q_attention.plugins.q_ceasc_counterfactual import (
        QCEASCCounterfactualScoreKernelConfig,
        build_qceasc_counterfactual_score_kernel,
    )

    kernel = build_qceasc_counterfactual_score_kernel(
        "q_ceasc_counterfactual",
        QCEASCCounterfactualScoreKernelConfig(
            num_layers=2,
            num_heads=3,
            head_dim=4,
            auxiliary_qubits=2,
            support_width=1,
            action_rank=2,
        ),
    )
    assert kernel.model_dimensions == (2, 3, 4)
