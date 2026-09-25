from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


STAGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING_ROOT / "src"))

from q_attention.plugins.q_query_key_coherent_transport import (  # noqa: E402
    ClassicalQueryKeyCoherentTransportKernel,
    QueryKeyCoherentTransportConfig,
    QuantumQueryKeyCoherentTransportKernel,
)


def _config(**overrides: object) -> QueryKeyCoherentTransportConfig:
    values: dict[str, object] = {
        "num_layers": 1,
        "num_heads": 1,
        "head_dim": 2,
        "register_qubits": 1,
        "depth": 1,
        "pair_chunk_size": 7,
        "seed": 19,
    }
    values.update(overrides)
    return QueryKeyCoherentTransportConfig(**values)


def _set_angles(
    kernel: QuantumQueryKeyCoherentTransportKernel,
    *,
    phase: float,
    post_q: float,
    post_k: float,
    depth: int = 0,
) -> None:
    with torch.no_grad():
        kernel.raw_phase[0, 0, depth, 0] = math.atanh(phase / kernel.config.max_phase)
        kernel.raw_post_rotation[0, 0, depth, 0, 0] = math.atanh(
            post_q / kernel.config.max_post_rotation
        )
        kernel.raw_post_rotation[0, 0, depth, 1, 0] = math.atanh(
            post_k / kernel.config.max_post_rotation
        )


def _ry(angle: torch.Tensor) -> torch.Tensor:
    half = angle / 2.0
    return torch.stack(
        (
            torch.stack((torch.cos(half), -torch.sin(half))),
            torch.stack((torch.sin(half), torch.cos(half))),
        )
    )


def _analytic_score(quantum: torch.Tensor, key: torch.Tensor, phase: torch.Tensor, post_q: torch.Tensor, post_k: torch.Tensor) -> torch.Tensor:
    dtype = torch.complex128
    q_state = torch.stack((torch.cos(quantum / 2), torch.sin(quantum / 2))).to(dtype)
    k_state = torch.stack((torch.cos(key / 2), torch.sin(key / 2))).to(dtype)
    state = torch.kron(q_state, k_state)
    controlled_phase = torch.diag(
        torch.cat(
            (
                torch.ones(3, dtype=dtype),
                torch.exp(1j * phase).reshape(1).to(dtype),
            )
        )
    )
    observable = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0],
         [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
        dtype=dtype,
    )
    state = torch.kron(_ry(post_q), _ry(post_k)).to(dtype) @ (controlled_phase @ state)
    return (state.conj() @ (observable @ state)).real


def _forward(
    kernel: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor,
    subject_mask: torch.Tensor,
    object_mask: torch.Tensor,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return kernel(
        query,
        key,
        scores=torch.zeros(
            query.shape[0], query.shape[1], query.shape[2], key.shape[2],
            dtype=query.dtype,
        ),
        layer_index=0,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
        query_mask=query_mask,
    )


def test_quantum_score_matches_independent_4x4_oracle() -> None:
    kernel = QuantumQueryKeyCoherentTransportKernel(_config(head_dim=1)).double()
    phase, post_q, post_k = 0.73, -0.31, 0.44
    _set_angles(kernel, phase=phase, post_q=post_q, post_k=post_k)
    q = torch.tensor([0.27, -0.81], dtype=torch.float64)
    k = torch.tensor([-0.52, 0.63], dtype=torch.float64)
    actual = kernel._single_qubit_quantum_score(q, k, layer_index=0, head_index=0, qubit=0)
    expected = torch.stack(
        [
            _analytic_score(q[index], k[index], torch.tensor(phase), torch.tensor(post_q), torch.tensor(post_k))
            for index in range(q.numel())
        ]
    )
    torch.testing.assert_close(actual.contiguous(), expected.contiguous(), rtol=1e-7, atol=1e-8)


def test_controlled_phase_is_input_dependent_and_not_replayed_by_control() -> None:
    config = _config(head_dim=1)
    quantum = QuantumQueryKeyCoherentTransportKernel(config).double()
    classical = ClassicalQueryKeyCoherentTransportKernel(config).double()
    classical.load_state_dict(quantum.state_dict(), strict=False)
    _set_angles(quantum, phase=1.1, post_q=0.28, post_k=-0.37)
    _set_angles(classical, phase=1.1, post_q=0.28, post_k=-0.37)
    q_angles = torch.tensor([-0.71, 0.46], dtype=torch.float64)
    k_angles = torch.tensor([-0.22, 0.89], dtype=torch.float64)
    q_grid = q_angles.repeat_interleave(2)
    k_grid = k_angles.repeat(2)
    score_on = quantum._single_qubit_quantum_score(q_grid, k_grid, layer_index=0, head_index=0, qubit=0)
    with torch.no_grad():
        quantum.raw_phase.zero_()
    score_off = quantum._single_qubit_quantum_score(q_grid, k_grid, layer_index=0, head_index=0, qubit=0)
    delta = (score_on - score_off).reshape(2, 2)
    assert float(delta.abs().max()) > 1e-4
    assert float((delta - delta.mean()).abs().max()) > 1e-5
    control = classical._single_qubit_classical_score(q_grid, k_grid, layer_index=0, head_index=0, qubit=0)
    assert float((score_on - control).abs().mean()) > 1e-4


def test_depth_parameter_changes_the_quantum_circuit() -> None:
    kernel = QuantumQueryKeyCoherentTransportKernel(_config(head_dim=1, depth=2)).double()
    q = torch.tensor([0.2, -0.7], dtype=torch.float64)
    k = torch.tensor([0.3, 0.8], dtype=torch.float64)
    with torch.no_grad():
        kernel.raw_phase[0, 0, 1, 0] = 0.0
        kernel.raw_post_rotation[0, 0, 1].zero_()
    first = kernel._single_qubit_quantum_score(q, k, layer_index=0, head_index=0, qubit=0)
    with torch.no_grad():
        kernel.raw_phase[0, 0, 1, 0] = 0.41
        kernel.raw_post_rotation[0, 0, 1].fill_(0.27)
    second = kernel._single_qubit_quantum_score(q, k, layer_index=0, head_index=0, qubit=0)
    assert not torch.allclose(first, second)


def test_mask_shapes_query_mask_and_zero_sum_contract() -> None:
    kernel = QuantumQueryKeyCoherentTransportKernel(_config(num_heads=2, head_dim=2)).double()
    generator = torch.Generator().manual_seed(4)
    query = torch.randn(2, 2, 3, 2, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 2, 4, 2, generator=generator, dtype=torch.float64)
    key_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool)
    subject = torch.zeros(2, 4, dtype=torch.bool)
    object_mask = torch.zeros(2, 4, dtype=torch.bool)
    object_mask[:, 1] = True
    query_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    output_key = _forward(kernel, query, key, key_mask, subject, object_mask, query_mask)
    pair_mask = key_mask[:, None, :].expand(2, 3, 4).clone()
    output_pair = _forward(kernel, query, key, pair_mask, subject[:, None, :].expand(2, 3, 4), object_mask[:, None, :].expand(2, 3, 4), query_mask)
    torch.testing.assert_close(output_key, output_pair)
    context = pair_mask & ~(object_mask[:, None, :].expand_as(pair_mask))
    assert torch.allclose(output_key.masked_fill(~context[:, None], 0.0).sum(dim=-1), torch.zeros(2, 2, 3, dtype=torch.float64), atol=1e-9)
    assert torch.count_nonzero(output_key[:, :, :, 1]).item() == 0
    assert torch.count_nonzero(output_key[0, :, 2, :]).item() == 0
    all_invalid = _forward(kernel, query, key, torch.zeros_like(key_mask), subject, object_mask, query_mask)
    assert torch.count_nonzero(all_invalid).item() == 0
    assert torch.isfinite(all_invalid).all()


def test_permutation_equivariance_and_parameter_match() -> None:
    config = _config(num_heads=2)
    quantum = QuantumQueryKeyCoherentTransportKernel(config)
    classical = ClassicalQueryKeyCoherentTransportKernel(config)
    assert quantum.parameter_count == classical.parameter_count
    generator = torch.Generator().manual_seed(10)
    query = torch.randn(2, 2, 3, 2, generator=generator)
    key = torch.randn(2, 2, 4, 2, generator=generator)
    masks = torch.ones(2, 4, dtype=torch.bool)
    subject = torch.zeros_like(masks)
    object_mask = torch.zeros_like(masks)
    base = _forward(quantum, query, key, masks, subject, object_mask)
    torch.testing.assert_close(_forward(quantum, query, key, masks, subject, object_mask), base)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = _forward(quantum, query, key[:, :, permutation], masks[:, permutation], subject[:, permutation], object_mask[:, permutation])
    torch.testing.assert_close(permuted, base[:, :, :, permutation], rtol=1e-6, atol=1e-6)


def test_gradients_are_finite_for_all_trainable_parameters() -> None:
    kernel = QuantumQueryKeyCoherentTransportKernel(_config()).double()
    generator = torch.Generator().manual_seed(22)
    query = torch.randn(2, 1, 2, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    key = torch.randn(2, 1, 3, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    masks = torch.ones(2, 3, dtype=torch.bool)
    output = _forward(kernel, query, key, masks, torch.zeros_like(masks), torch.zeros_like(masks))
    target = torch.randn_like(output)
    loss = (output - target).square().mean()
    loss.backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()
    for parameter in kernel.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().max()) > 1e-12


def test_complex_autograd_phase_matches_finite_difference() -> None:
    kernel = QuantumQueryKeyCoherentTransportKernel(_config(head_dim=1)).double()
    _set_angles(kernel, phase=0.83, post_q=0.21, post_k=-0.17)
    q = torch.tensor([0.24, -0.55], dtype=torch.float64)
    k = torch.tensor([-0.43, 0.72], dtype=torch.float64)
    kernel.zero_grad(set_to_none=True)
    analytic = kernel._single_qubit_quantum_score(q, k, layer_index=0, head_index=0, qubit=0).sum()
    analytic.backward()
    gradient = float(kernel.raw_phase.grad[0, 0, 0, 0])
    with torch.no_grad():
        original = float(kernel.raw_phase[0, 0, 0, 0])
        step = 1e-6
        kernel.raw_phase[0, 0, 0, 0] = original + step
        plus = float(kernel._single_qubit_quantum_score(q, k, layer_index=0, head_index=0, qubit=0).sum())
        kernel.raw_phase[0, 0, 0, 0] = original - step
        minus = float(kernel._single_qubit_quantum_score(q, k, layer_index=0, head_index=0, qubit=0).sum())
        kernel.raw_phase[0, 0, 0, 0] = original
    finite_difference = (plus - minus) / (2.0 * step)
    assert math.isclose(gradient, finite_difference, rel_tol=2e-5, abs_tol=2e-6)


def test_input_validation_rejects_wrong_head_count() -> None:
    kernel = QuantumQueryKeyCoherentTransportKernel(_config(num_heads=2))
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="head count"):
        _forward(kernel, query, key, mask, mask, mask)
