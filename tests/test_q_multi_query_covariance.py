from __future__ import annotations

import torch

from q_attention.plugins.q_multi_query_covariance import (
    ClassicalMultiQueryCovarianceKernel,
    MultiQueryCovarianceConfig,
    QuantumMultiQueryCovarianceKernel,
)


def _config(**overrides: object) -> MultiQueryCovarianceConfig:
    values: dict[str, object] = {
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 3,
        "num_qubits": 2,
        "depth": 1,
        "pair_chunk_size": 3,
        "seed": 19,
    }
    values.update(overrides)
    return MultiQueryCovarianceConfig(**values)


def _fixture(*, num_heads: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(2, num_heads, 5, 3, generator=generator)
    key = torch.randn(2, num_heads, 5, 3, generator=generator)
    attention = torch.ones(2, 5, dtype=torch.bool)
    subject = torch.zeros(2, 5, dtype=torch.bool)
    object_ = torch.zeros(2, 5, dtype=torch.bool)
    subject[:, 0] = True
    object_[:, 1] = True
    query_mask = torch.ones(2, 5, dtype=torch.bool)
    return query, key, attention, subject, object_, query_mask


def _forward(kernel: torch.nn.Module, fixture: tuple[torch.Tensor, ...]) -> torch.Tensor:
    query, key, attention, subject, object_, query_mask = fixture
    return kernel(
        query,
        key,
        scores=torch.zeros(query.shape[0], query.shape[1], query.shape[2], key.shape[2]),
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        query_mask=query_mask,
    )


def test_quantum_contract_masks_centers_and_exposes_connected_covariance() -> None:
    kernel = QuantumMultiQueryCovarianceKernel(_config())
    output = _forward(kernel, _fixture())
    assert output.shape == (2, 2, 5, 5)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, :, :, 0]).item() == 0
    assert torch.count_nonzero(output[:, :, :, 1]).item() == 0
    assert torch.count_nonzero(output[:, :, :, 2:]).item() > 0
    assert torch.allclose(output[:, :, :2].sum(dim=-1), torch.zeros(2, 2, 2), atol=1e-6)
    connected = kernel.last_connected_covariance
    independent = kernel.last_independent_product
    assert connected is not None and independent is not None
    assert connected.shape == (2, 2, 5)
    assert float((connected - independent).abs().max()) > 1e-7


def test_quantum_and_classical_are_parameter_matched_but_not_replayed() -> None:
    config = _config(num_heads=1)
    quantum = QuantumMultiQueryCovarianceKernel(config)
    classical = ClassicalMultiQueryCovarianceKernel(config)
    classical.load_state_dict(quantum.state_dict(), strict=True)
    assert quantum.parameter_count == classical.parameter_count
    quantum_output = _forward(quantum, _fixture(num_heads=1))
    classical_output = _forward(classical, _fixture(num_heads=1))
    assert torch.isfinite(classical_output).all()
    assert float((quantum_output - classical_output).abs().mean()) > 1e-7


def test_query_role_swap_changes_only_fixed_role_sign() -> None:
    kernel = QuantumMultiQueryCovarianceKernel(_config(num_heads=1))
    fixture = _fixture(num_heads=1)
    base = _forward(kernel, fixture)
    swapped = (fixture[0], fixture[1], fixture[2], fixture[4], fixture[3], fixture[5])
    swapped_output = _forward(kernel, swapped)
    torch.testing.assert_close(swapped_output, -base, rtol=1e-5, atol=2e-5)


def test_key_permutation_equivariance_and_chunk_replay() -> None:
    kernel = QuantumMultiQueryCovarianceKernel(_config(num_heads=1, pair_chunk_size=1))
    generator = torch.Generator().manual_seed(47)
    query = torch.randn(2, 1, 2, 3, generator=generator)
    key = torch.randn(2, 1, 5, 3, generator=generator)
    attention = torch.ones(2, 5, dtype=torch.bool)
    subject = torch.zeros(2, 2, dtype=torch.bool)
    object_ = torch.zeros(2, 2, dtype=torch.bool)
    subject[:, 0] = True
    object_[:, 1] = True
    fixture = (query, key, attention, subject, object_, torch.ones(2, 2, dtype=torch.bool))
    base = _forward(kernel, fixture)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    permuted = (
        fixture[0],
        fixture[1][:, :, permutation],
        fixture[2][:, permutation],
        fixture[3],
        fixture[4],
        fixture[5],
    )
    permuted_output = _forward(kernel, permuted)
    torch.testing.assert_close(permuted_output, base[:, :, :, permutation], rtol=1e-5, atol=1e-6)


def test_all_trainable_parameters_receive_finite_gradient() -> None:
    kernel = QuantumMultiQueryCovarianceKernel(_config(num_heads=1)).double()
    fixture = tuple(value.double() if value.is_floating_point() else value for value in _fixture(num_heads=1))
    output = _forward(kernel, fixture)
    loss = output.square().mean() + kernel.last_connected_covariance.square().mean()  # type: ignore[union-attr]
    loss.backward()
    for parameter in kernel.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().max()) > 1e-12


def test_invalid_role_mask_is_rejected() -> None:
    kernel = QuantumMultiQueryCovarianceKernel(_config(num_heads=1))
    fixture = _fixture(num_heads=1)
    with torch.no_grad():
        bad_subject = torch.zeros(2, 4, dtype=torch.bool)
    bad = (fixture[0], fixture[1], fixture[2], bad_subject, fixture[4], fixture[5])
    try:
        _forward(kernel, bad)
    except ValueError as error:
        assert "role mask" in str(error)
    else:
        raise AssertionError("invalid role mask was accepted")
