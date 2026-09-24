from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from q_attention.adapters.attention_context import (
    AttentionContextHookConfig,
    AttentionContextKernelAdapter,
)
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_pvg import QPVGConfig, build_q_pvg

_TRAINING_RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "run_q_pvg_transfer_base.py"
_TRAINING_SPEC = importlib.util.spec_from_file_location("q_pvg_training_base", _TRAINING_RUNNER)
assert _TRAINING_SPEC is not None and _TRAINING_SPEC.loader is not None
_TRAINING_MODULE = importlib.util.module_from_spec(_TRAINING_SPEC)
_TRAINING_SPEC.loader.exec_module(_TRAINING_MODULE)


def _model() -> RelationExtractionModel:
    torch.manual_seed(17)
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=20,
            num_labels=3,
            dim=8,
            num_layers=2,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
            max_length=8,
        )
    )
    model.eval()
    return model


def _batch() -> tuple[torch.Tensor, ...]:
    input_ids = torch.tensor([[1, 2, 3, 4, 0], [4, 3, 2, 1, 0]])
    attention_mask = input_ids.ne(0)
    subject_mask = torch.tensor(
        [[False, True, False, False, False], [False, False, False, True, False]]
    )
    object_mask = torch.tensor(
        [[False, False, False, True, False], [False, True, False, False, False]]
    )
    return input_ids, attention_mask, subject_mask, object_mask


def _kernel() -> torch.nn.Module:
    return build_q_pvg(
        QPVGConfig(
            num_layers=2,
            num_heads=2,
            head_dim=4,
            register_qubits=2,
            depth=2,
            seed=29,
        )
    )


def _attached_logits(
    model: RelationExtractionModel,
    kernel: torch.nn.Module,
    *,
    query_chunk_size: int | None,
    capture_trace: bool = False,
) -> tuple[torch.Tensor, AttentionContextKernelAdapter]:
    input_ids, attention_mask, subject_mask, object_mask = _batch()
    adapter = AttentionContextKernelAdapter(
        model,
        model.context_module_paths,
        kernel,
        query_chunk_size=query_chunk_size,
    )
    adapter.attach(
        AttentionContextHookConfig(
            attention_mask=attention_mask,
            query_mask=attention_mask,
            capture_trace=capture_trace,
        )
    )
    try:
        logits = model(input_ids, attention_mask, subject_mask, object_mask)
    finally:
        adapter.remove()
    return logits, adapter


def test_default_context_path_is_a_passthrough() -> None:
    model = _model()
    args = _batch()
    first = model(*args)
    second = model(*args)
    assert torch.equal(first, second)
    for layer in model.encoder.layers:
        assert layer.attn.context_intervention(*(
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
        )) is None


def test_q_pvg_full_and_query_chunked_contexts_match() -> None:
    model = _model()
    kernel = _kernel()
    full, _ = _attached_logits(model, kernel, query_chunk_size=None)
    chunked, _ = _attached_logits(model, kernel, query_chunk_size=2)
    assert torch.allclose(full, chunked, atol=1e-6, rtol=1e-5)


def test_q_pvg_context_adapter_captures_trace_and_finite_gradients() -> None:
    model = _model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel = _kernel()
    logits, adapter = _attached_logits(
        model, kernel, query_chunk_size=2, capture_trace=True
    )
    loss = F.cross_entropy(logits, torch.tensor([0, 2]))
    loss.backward()
    assert set(adapter.last_traces) == {0, 1}
    assert "alignment_imag" in adapter.last_traces[0]
    assert "routed_values" in adapter.last_traces[0]
    gradients = [parameter.grad for parameter in kernel.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "selector",
    [
        "q_pvg_phase_value",
        "q_pvg_real_only",
        "q_pvg_classical_complex",
        "q_pvg_random_phase",
        "q_pvg_score_value",
    ],
)
def test_formal_selector_first_batch_satisfies_gradient_contract(selector: str) -> None:
    model = _model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    kernel = _TRAINING_MODULE.build_kernel(
        selector,
        model,
        13,
        SimpleNamespace(
            register_qubits=2,
            depth=2,
            angle_scale=1.0,
            gate_temperature=1.0,
            score_gain=0.25,
            value_route_mode="branch_interpolation",
        ),
    )
    assert kernel is not None
    logits, _ = _attached_logits(model, kernel, query_chunk_size=2)
    F.cross_entropy(logits, torch.tensor([0, 2])).backward()

    _TRAINING_MODULE._validate_kernel_gradients(
        kernel,
        selector=selector,
        epoch=1,
    )


@pytest.mark.parametrize(
    ("route_mode", "phase_mode", "readout_mode", "active_names", "inactive_names"),
    [
        (
            "branch_interpolation", "complex", "quantum",
            {"value_branch0", "value_branch1", "gate_imag_weight"},
            {"scalar_value_logit"},
        ),
        (
            "scalar", "complex", "quantum",
            {"scalar_value_logit", "gate_imag_weight"},
            {"value_branch0", "value_branch1"},
        ),
        (
            "branch_interpolation", "real_only", "quantum",
            {"value_branch0", "value_branch1"},
            {"scalar_value_logit", "gate_imag_weight"},
        ),
        (
            "branch_interpolation", "complex", "classical",
            {"value_branch0", "value_branch1"},
            {"scalar_value_logit", "gate_imag_weight"},
        ),
    ],
)
def test_formal_gradient_expectations_follow_route_and_readout(
    route_mode: str,
    phase_mode: str,
    readout_mode: str,
    active_names: set[str],
    inactive_names: set[str],
) -> None:
    kernel = build_q_pvg(
        QPVGConfig(
            num_layers=2,
            num_heads=2,
            head_dim=4,
            value_route_mode=route_mode,
            phase_mode=phase_mode,
            readout_mode=readout_mode,
        )
    )
    expected = _TRAINING_MODULE._expected_gradient_parameter_names(kernel)
    assert expected is not None
    assert active_names <= expected
    assert not (inactive_names & expected)
