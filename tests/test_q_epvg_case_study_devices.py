from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
from run_q_epvg_selector_worker import (
    _case_study_scores,
    _case_study_score_views,
    _context_to_head_layout,
    _model_attention_from_scores,
    _q_epvg_model_scores,
    _recompute_case_study_context,
    _sample_context_to_head_layout,
    write_case_study,
)
from q_attention.adapters.q_epvg_attention import QEPVGAttentionAdapter
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_epvg import QEPVGConfig, build_q_epvg
from q_attention.tasks.relation import RelationRecord, build_vocab


def _cross_device_reference() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("meta")


def test_case_study_trace_tensor_moves_to_score_device_and_dtype() -> None:
    device = _cross_device_reference()
    reference = torch.zeros(2, 1, 3, 3, dtype=torch.float32, device=device)
    captured = torch.ones(2, 1, 3, 3, device="cpu", dtype=torch.float64)

    adjustment, steered_scores = _case_study_scores(captured, reference)

    assert adjustment.device == reference.device
    assert adjustment.dtype == reference.dtype
    assert steered_scores.device == reference.device
    assert steered_scores.dtype == reference.dtype
    assert steered_scores.shape == reference.shape
    if device.type != "meta":
        assert torch.allclose(steered_scores, reference + 1.0)


def test_cpu_case_study_alignment_preserves_values() -> None:
    reference = torch.zeros(2, 1, 3, 3, dtype=torch.float32)
    captured = torch.ones(2, 1, 3, 3, dtype=torch.float64)

    adjustment, steered_scores = _case_study_scores(captured, reference)

    assert adjustment.device == reference.device
    assert adjustment.dtype == reference.dtype
    assert torch.allclose(steered_scores, reference + 1.0)


def test_steered_score_projection_preserves_legacy_view_and_actual_model_scores() -> None:
    pre_intervention = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    query_update_scores = torch.tensor([[[[0.5, 0.0], [0.0, -0.5]]]])
    model_scores = pre_intervention + query_update_scores
    adjustment_source = torch.full_like(model_scores, 0.25)

    adjustment, saved_pre_intervention, actual_model = _case_study_score_views(
        adjustment_source, pre_intervention, model_scores
    )

    assert torch.equal(adjustment, adjustment_source)
    assert torch.equal(saved_pre_intervention, pre_intervention + adjustment_source)
    assert torch.equal(actual_model, model_scores + adjustment_source)
    assert not torch.equal(saved_pre_intervention, actual_model)


@pytest.mark.parametrize("path", ["value_only", "score_value", "query"])
def test_q_epvg_paths_emit_rank_five_routed_values(path: str) -> None:
    generator = torch.Generator(device="cpu").manual_seed(191)
    query = torch.randn(2, 2, 5, 4, generator=generator)
    key = torch.randn(2, 2, 5, 4, generator=generator)
    value = torch.randn(2, 2, 5, 4, generator=generator)
    scores = torch.matmul(query, key.transpose(-1, -2)) / (4**0.5)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    attention_mask[:, -1] = False
    kernel = build_q_epvg(QEPVGConfig(num_layers=1, num_heads=2, head_dim=4, path=path))

    output, trace = kernel(
        query,
        key,
        value,
        scores=scores,
        attention_mask=attention_mask,
        query_mask=attention_mask,
        return_trace=True,
    )

    assert output.shape == query.shape
    assert trace["routed_values"].shape == (2, 2, 5, 5, 4)
    assert trace["query_update"].shape == query.shape


@pytest.mark.parametrize("path", ["value_only", "score_value", "query"])
def test_relation_transformer_context_matches_model_attention_contraction(path: str) -> None:
    torch.manual_seed(73)
    num_heads = 2
    head_dim = 4
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=24,
            num_labels=3,
            dim=num_heads * head_dim,
            num_layers=1,
            num_heads=num_heads,
            ff_dim=16,
            dropout=0.0,
            max_length=6,
        )
    ).eval()
    kernel = build_q_epvg(
        QEPVGConfig(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            path=path,
            gate_scale=3.0,
            score_gain=1.0,
        )
    )
    adapter = QEPVGAttentionAdapter(model, [kernel])
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
            captured[name] = output.detach()
        return hook

    query_handle = model.encoder.layers[0].attn.query_proj.register_forward_hook(capture("query"))
    key_handle = model.encoder.layers[0].attn.key_proj.register_forward_hook(capture("key"))
    context_handle = model.encoder.layers[0].attn.out_proj.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("context", inputs[0].detach())
    )
    input_ids = torch.tensor([[2, 4, 6, 8, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    subject_mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    object_mask = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool)
    try:
        adapter.attach()
        with torch.no_grad():
            model(input_ids, attention_mask, subject_mask, object_mask)
        trace = adapter.traces[0]
    finally:
        query_handle.remove()
        key_handle.remove()
        context_handle.remove()
        adapter.remove()

    projected_query = _context_to_head_layout(captured["query"], num_heads)
    projected_key = _context_to_head_layout(captured["key"], num_heads)
    query = trace["query"]
    query_update = trace["query_update"]
    key = trace["key"]
    adjustment = trace["score_adjustment"]
    if path == "query":
        assert torch.count_nonzero(query_update) > 0
    if path == "score_value":
        assert torch.count_nonzero(adjustment) > 0
    torch.testing.assert_close(query - query_update, projected_query)
    torch.testing.assert_close(key, projected_key)

    pre_intervention_scores = torch.matmul(
        projected_query, key.transpose(-1, -2)
    ) / (head_dim**0.5)
    model_scores = torch.matmul(query, key.transpose(-1, -2)) / (head_dim**0.5)
    aligned_adjustment, saved_pre_intervention_scores, actual_model_scores = _case_study_score_views(
        adjustment, pre_intervention_scores, model_scores
    )
    witness_scores = torch.matmul(query - query_update, key.transpose(-1, -2)) / (head_dim**0.5)
    torch.testing.assert_close(saved_pre_intervention_scores, witness_scores + aligned_adjustment)
    torch.testing.assert_close(actual_model_scores, model_scores + aligned_adjustment)

    model_attention = _model_attention_from_scores(
        actual_model_scores[0], attention_mask[0]
    )
    torch.testing.assert_close(trace["attention"][0], model_attention)
    if path == "query":
        legacy_double_update_scores = actual_model_scores[0] + (
            torch.matmul(query_update[0], key[0].transpose(-1, -2)) / (head_dim**0.5)
        )
        legacy_double_update_attention = _model_attention_from_scores(
            legacy_double_update_scores, attention_mask[0]
        )
        assert not torch.allclose(trace["attention"][0], legacy_double_update_attention)
    routed_values = trace["routed_values"][0]
    expected_context = torch.einsum("hqk,hqkd->hqd", model_attention, routed_values)
    captured_context = _context_to_head_layout(captured["context"], num_heads)[0]
    torch.testing.assert_close(captured_context, expected_context, rtol=1e-4, atol=1e-5)


def test_q_epvg_model_score_capture_fails_closed_when_query_or_key_is_missing() -> None:
    with pytest.raises(RuntimeError, match="refusing to substitute baseline scores"):
        _q_epvg_model_scores(
            {"query": torch.ones(1, 1, 2, 4)},
            torch.zeros(1, 1, 2, 2),
            head_dim=4,
            layer_index=0,
        )


def test_missing_case_study_trace_tensor_uses_reference_zeros() -> None:
    device = _cross_device_reference()
    reference = torch.zeros(2, 1, 3, 3, dtype=torch.float32, device=device)

    adjustment, steered_scores = _case_study_scores(None, reference)

    assert adjustment.device == reference.device
    assert adjustment.dtype == reference.dtype
    assert steered_scores.device == reference.device
    assert steered_scores.dtype == reference.dtype
    assert steered_scores.shape == reference.shape
    if device.type != "meta":
        assert torch.count_nonzero(adjustment) == 0
        assert torch.allclose(steered_scores, reference)


def test_captured_batch_context_selects_sample_after_head_layout_conversion() -> None:
    context = torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12)

    result = _context_to_head_layout(context, num_heads=3)
    sample = _sample_context_to_head_layout(context, sample_index=1, num_heads=3)

    assert result.shape == (2, 3, 5, 4)
    assert torch.equal(result.transpose(1, 2).reshape_as(context), context)
    assert sample.shape == (3, 5, 4)
    assert torch.equal(sample, result[1])
    assert torch.equal(sample.transpose(0, 1).reshape_as(context[1]), context[1])


def test_context_layout_rejects_hidden_size_not_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible hidden size"):
        _context_to_head_layout(torch.zeros(1, 5, 10), num_heads=3)


@pytest.mark.parametrize(
    ("attention_dtype", "routed_dtype", "captured_dtype", "expected_dtype"),
    [
        (torch.float32, torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.bfloat16, torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ],
)
def test_context_evidence_contraction_promotes_mixed_dtypes_without_mutating_captures(
    attention_dtype: torch.dtype,
    routed_dtype: torch.dtype,
    captured_dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    model_attention = torch.full((1, 1, 2, 2), 0.5, dtype=attention_dtype)
    routed_values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2).to(routed_dtype)
    expected_context = torch.tensor([[[[1.0, 2.0], [5.0, 6.0]]]])
    captured_context = expected_context.to(dtype=captured_dtype)
    original_dtypes = (model_attention.dtype, routed_values.dtype, captured_context.dtype)

    recomputed, comparable = _recompute_case_study_context(
        model_attention, routed_values, captured_context
    )

    assert recomputed.dtype == expected_dtype
    assert comparable.dtype == expected_dtype
    assert torch.allclose(recomputed, expected_context.to(dtype=expected_dtype))
    assert (model_attention.dtype, routed_values.dtype, captured_context.dtype) == original_dtypes


def test_write_case_study_emits_linked_observed_selection_and_context_stages(tmp_path: Path) -> None:
    record = RelationRecord(
        tokens=("Acme", "acquired", "Beta"),
        subject=(0, 1),
        object=(2, 3),
        label="org:acquired",
    )
    vocab = build_vocab([record])
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=len(vocab),
            num_labels=1,
            dim=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
            max_length=8,
        )
    ).eval()
    class TinyKernelStack(torch.nn.ModuleList):
        def metadata(self) -> dict[str, str]:
            return {"selector": "q_epvg_test"}

    kernel = TinyKernelStack(
        [build_q_epvg(QEPVGConfig(num_layers=1, num_heads=2, head_dim=4, path="query"))]
    )
    artifacts = SimpleNamespace(
        vocab=vocab,
        label_to_id={"org:acquired": 0},
        id_to_label={0: "org:acquired"},
    )
    config = {
        "seed": 13,
        "kernel": {"epochs": 1},
        "case_study": {
            "records": {"train": [0], "valid": [0], "test": [0]},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "selector-output"
    output_dir.mkdir()

    write_case_study(
        model=model,
        kernel=kernel,
        records={"train": [record], "valid": [record], "test": [record]},
        artifacts=artifacts,
        device=torch.device("cpu"),
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        selector="q_epvg_test",
    )

    case_study = json.loads((output_dir / "case_study.json").read_text(encoding="utf-8"))
    sample_trace = json.loads((output_dir / "sample_trace.json").read_text(encoding="utf-8"))
    case = case_study["cases"][0]
    stages = {stage["stage"]: stage for stage in case["stages"]}
    selection = stages["selection"]
    intervention = stages["attention_intervention"]
    context = stages["context"]

    assert selection["status"] == intervention["status"] == context["status"] == "observed"
    selection_inputs = {item["id"]: item for item in selection["inputs"]["representations"]}
    selection_outputs = {item["id"]: item for item in selection["outputs"]["representations"]}
    context_inputs = {item["id"]: item for item in context["inputs"]["representations"]}
    intervention_outputs = {item["id"]: item for item in intervention["outputs"]["representations"]}
    for rep_id in ("q_epvg_query", "q_epvg_key", "q_epvg_score_adjustment"):
        assert rep_id in selection_inputs
    assert "q_epvg_model_scores" in selection_outputs
    assert "q_epvg_model_attention" in selection_outputs
    assert context_inputs["q_epvg_model_attention"]["manifest_id"] == selection_outputs[
        "q_epvg_model_attention"
    ]["manifest_id"]
    assert context_inputs["q_epvg_routed_values"]["manifest_id"] == intervention_outputs[
        "q_epvg_routed_values"
    ]["manifest_id"]
    assert "q_epvg_output" in {item["id"]: item for item in context["outputs"]["representations"]}
    trace_sample = sample_trace["samples"][0]
    trace_stages = {stage["stage"]: stage for stage in trace_sample["stages"]}
    assert trace_stages["selection"]["status"] == "observed"
    assert trace_stages["context"]["status"] == "observed"


def test_model_attention_uses_post_intervention_scores_and_masks_only_keys() -> None:
    query_after_intervention = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    scores = torch.matmul(query_after_intervention, key.transpose(-1, -2)) / (2**0.5)

    weights = _model_attention_from_scores(scores, torch.tensor([1, 1, 0]))
    expected = torch.softmax(
        scores.masked_fill(
            ~torch.tensor([1, 1, 0], dtype=torch.bool).view(1, 1, -1),
            torch.finfo(scores.dtype).min,
        ),
        dim=-1,
    )

    assert torch.equal(weights, expected)
    assert torch.count_nonzero(weights[..., 2]) == 0
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
