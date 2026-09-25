import torch

from q_attention.plugins.q_ceasc_score import (
    QCEASCScoreKernelConfig,
    build_qceasc_score_kernel,
)


def _inputs():
    query = torch.randn(2, 2, 5, 4)
    key = torch.randn(2, 2, 5, 4)
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    subject = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    object_ = torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.bool)
    return query, key, attention, subject, object_


def test_qceasc_score_shape_masks_and_zero_sum():
    config = QCEASCScoreKernelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        auxiliary_qubits=2,
        support_width=2,
        action_rank=2,
        query_chunk_size=2,
    )
    kernel = build_qceasc_score_kernel("q_ceasc", config)
    query, key, attention, subject, object_ = _inputs()
    residual = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    assert residual.shape == (2, 2, 5, 5)
    assert torch.isfinite(residual).all()
    assert torch.allclose(residual[:, :, :, -1], torch.zeros(2, 2, 5), atol=1e-7)
    assert torch.allclose(residual.sum(dim=-1), torch.zeros(2, 2, 5), atol=1e-6)
    assert torch.allclose(
        residual[:, :, -1, :], torch.zeros(2, 2, 5), atol=1e-7
    )


def test_qceasc_score_matches_classical_parameter_budget():
    config = QCEASCScoreKernelConfig(
        num_layers=2,
        num_heads=2,
        head_dim=4,
        auxiliary_qubits=2,
        support_width=2,
        action_rank=2,
    )
    quantum = build_qceasc_score_kernel("q_ceasc", config)
    classical = build_qceasc_score_kernel("classical_span", config)
    assert quantum.parameter_count == classical.parameter_count
