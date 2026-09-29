from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch


def load_repairer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "repair_q_epvg_case_study_stages.py"
    spec = importlib.util.spec_from_file_location("q_epvg_case_study_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_padded_token_ids_are_trimmed_only_with_mask_witness() -> None:
    module = load_repairer()
    case = {
        "case_id": "selector_a:test:0:final",
        "tokens": ["Alice", "works"],
        "token_ids": [7, 8, 0],
        "attention_mask": [1, 1, 0],
    }
    sample = {
        "stages": [
            {
                "stage": "preprocess",
                "outputs": {"token_ids": [7, 8, 0]},
            }
        ]
    }
    repair = module._repair_token_alignment(case, sample, where="fixture")
    assert repair == {
        "case_id": "selector_a:test:0:final",
        "original_token_id_count": 3,
        "trimmed_token_id_count": 2,
        "padding_count": 1,
        "witness": "attention_mask_prefix_and_original_token_count",
    }
    assert case["token_ids"] == [7, 8]
    assert sample["stages"][0]["outputs"]["token_ids"] == [7, 8]


def test_token_alignment_repair_rejects_unproven_mismatch() -> None:
    module = load_repairer()
    case = {
        "case_id": "selector_a:test:0:final",
        "tokens": ["Alice", "works"],
        "token_ids": [7, 8, 9],
        "attention_mask": [1, 1, 1],
    }
    sample = {"stages": [{"stage": "preprocess", "outputs": {"token_ids": [7, 8, 9]}}]}
    with pytest.raises(ValueError, match="not producer-proven batch padding"):
        module._repair_token_alignment(case, sample, where="fixture")


def test_stage_merge_allows_only_repair_owned_differences() -> None:
    module = load_repairer()
    original = [
        {"stage": "data", "status": "observed"},
        {"stage": "context", "status": "failed"},
    ]
    existing = [
        {"stage": "data", "status": "observed"},
        {"stage": "context", "status": "observed", "capture_mode": "reconstructed"},
    ]
    repaired = [
        {"stage": "data", "status": "observed"},
        {"stage": "context", "status": "observed", "capture_mode": "reconstructed"},
    ]
    assert module._merge_repair_owned_stage_updates(existing, original, repaired, where="fixture") == repaired

    bad_existing = [
        {"stage": "data", "status": "changed"},
        original[1],
    ]
    with pytest.raises(ValueError, match="outside the repair-owned boundary"):
        module._merge_repair_owned_stage_updates(bad_existing, original, repaired, where="fixture")


def _make_group(
    tmp_path: Path,
    *,
    mismatch: bool = False,
    existing_failed: bool = False,
    trace_failed: bool = False,
) -> Path:
    group = tmp_path / "group"
    selector = group / "seed_13" / "selectors" / "selector_a"
    selector.mkdir(parents=True)
    (group / "multi_seed_run_summary.json").write_text(
        json.dumps({"success": True, "tasks": []}), encoding="utf-8"
    )
    stage_names = [
        "data", "preprocess", "embedding", "encoder", "training",
        "attention_baseline", "scoring", "selection", "attention_intervention", "context", "classifier",
        "evaluation", "diagnosis",
    ]
    stages = [
        {"stage": name, "status": "observed", "input_refs": [], "output_refs": []}
        for name in stage_names
    ]
    if trace_failed:
        stages[6]["status"] = "failed"
    case_id = "selector_a:test:0:final"
    case = {
        "schema_version": "q-attention.Q-EPVG-case-study.v2",
        "selector": "selector_a",
        "cases": [{"case_id": case_id, "representations": {}}],
    }
    if existing_failed:
        case["cases"][0]["stages"] = [
            {"stage": "scoring", "status": "failed", "input_refs": [], "output_refs": []}
        ]
    sample = {"sample_id": case_id, "stages": stages}
    if mismatch:
        sample["sample_id"] = "other"
    trace = {"schema_version": "sample-trace.v1", "samples": [sample]}
    (selector / "case_study.json").write_text(json.dumps(case), encoding="utf-8")
    (selector / "sample_trace.json").write_text(json.dumps(trace), encoding="utf-8")
    return group


def _tensor_manifest(selector: Path, manifest_id: str, rep_id: str, producer: str, tensor: torch.Tensor, axes: list[str]) -> dict:
    buffer = io.BytesIO()
    torch.save(tensor.contiguous(), buffer, _use_new_zipfile_serialization=False)
    raw = buffer.getvalue()
    relative = f"case_study_tensors/{manifest_id}.pt"
    target = selector / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    values = tensor.detach().float()
    return {
        "id": rep_id,
        "manifest_id": manifest_id,
        "producer_stage": producer,
        "path": relative,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "axis_semantics": axes,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "preview": {
            "values_first_32": tensor.flatten()[:32].tolist(),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "l2_norm": float(values.norm().item()),
        },
    }


def _replace_rep_tensor(
    selector: Path,
    case_payload: dict,
    rep_id: str,
    tensor: torch.Tensor,
) -> dict:
    case = case_payload["cases"][0]
    previous = case["representations"][rep_id]
    replacement = _tensor_manifest(
        selector,
        previous["manifest_id"],
        rep_id,
        previous["producer_stage"],
        tensor,
        previous["axis_semantics"],
    )
    case["representations"][rep_id] = replacement
    case_payload["tensor_manifest"] = [
        replacement if item.get("manifest_id") == previous["manifest_id"] else item
        for item in case_payload["tensor_manifest"]
    ]
    return replacement


def _make_context_reconstruction_group(
    tmp_path: Path,
    *,
    zero_query_update: bool = False,
    legacy_only_attention_trace: bool = False,
) -> tuple[Path, torch.Tensor, torch.Tensor]:
    group = _make_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    case = case_payload["cases"][0]
    case.update({"split": "test", "record_index": 0, "checkpoint": "final"})
    case["attention_mask"] = [1, 1, 0]
    query = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)
    query_update = (
        torch.zeros_like(query)
        if zero_query_update
        else torch.tensor([[[[0.0, 2.0], [1.0, 0.0]]]], dtype=torch.float32)
    )
    score_adjustment = torch.tensor(
        [[[[0.25, -0.15, 0.4], [-0.2, 0.3, 0.1]]]], dtype=torch.float32
    )
    scores = torch.matmul(query, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)
    key_mask = torch.tensor(case["attention_mask"], dtype=torch.bool).view(1, 1, 1, -1)
    model_attention = torch.softmax(
        (scores + score_adjustment).masked_fill(~key_mask, torch.finfo(scores.dtype).min), dim=-1
    )
    legacy_trace_scores = (
        scores
        + score_adjustment
        + torch.matmul(query_update, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)
    )
    legacy_trace_attention = torch.softmax(
        legacy_trace_scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min), dim=-1
    )
    pre_intervention_scores = (
        torch.matmul(query - query_update, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)
    ) + score_adjustment
    routed_values = torch.arange(1, 13, dtype=torch.float32).reshape(1, 1, 2, 3, 2)
    source_tensors = (
        ("q_epvg_query", "scoring", query, ["layers", "heads", "query_tokens", "head_dim"]),
        ("q_epvg_key", "scoring", key, ["layers", "heads", "key_tokens", "head_dim"]),
        ("q_epvg_query_update", "attention_intervention", query_update, ["layers", "heads", "query_tokens", "head_dim"]),
        ("q_epvg_score_adjustment", "scoring", score_adjustment, ["layers", "heads", "query_tokens", "key_tokens"]),
        (
            "q_epvg_attention",
            "selection",
            legacy_trace_attention if legacy_only_attention_trace else model_attention,
            ["layers", "heads", "query_tokens", "key_tokens"],
        ),
        ("steered_attention_scores", "selection", pre_intervention_scores, ["layers", "heads", "query_tokens", "key_tokens"]),
        ("q_epvg_routed_values", "attention_intervention", routed_values, ["layers", "heads", "query_tokens", "key_tokens", "value_dim"]),
    )
    source_manifests = {}
    for rep_id, producer, tensor, axes in source_tensors:
        source_manifests[rep_id] = _tensor_manifest(
            selector,
            f"test_0_final__{rep_id}",
            rep_id,
            producer,
            tensor,
            axes,
        )
    case["representations"] = source_manifests
    case_payload["tensor_manifest"] = list(source_manifests.values())
    case_payload["representation_inventory"] = sorted(source_manifests)
    case_payload["manifest_inventory"] = sorted(
        manifest["manifest_id"] for manifest in source_manifests.values()
    )
    sample = trace_payload["samples"][0]
    preprocess = next(stage for stage in sample["stages"] if stage["stage"] == "preprocess")
    preprocess.setdefault("outputs", {})["attention_mask"] = list(case["attention_mask"])
    selection = next(stage for stage in sample["stages"] if stage["stage"] == "selection")
    selection["output_refs"] = [
        {"ref": source_manifests["q_epvg_attention"]["manifest_id"], "kind": "representation", "producer_stage": "selection"},
        {"ref": source_manifests["steered_attention_scores"]["manifest_id"], "kind": "representation", "producer_stage": "selection"},
    ]
    intervention = next(stage for stage in sample["stages"] if stage["stage"] == "attention_intervention")
    intervention["input_refs"] = [
        {"ref": source_manifests["q_epvg_attention"]["manifest_id"], "kind": "representation", "producer_stage": "selection"}
    ]
    context = next(stage for stage in sample["stages"] if stage["stage"] == "context")
    context.update({
        "status": "failed",
        "reason": "Q-EPVG context output was not captured",
        "input_refs": [
            {"ref": source_manifests["q_epvg_attention"]["manifest_id"], "kind": "representation", "producer_stage": "selection"},
            {"ref": source_manifests["q_epvg_routed_values"]["manifest_id"], "kind": "representation", "producer_stage": "attention_intervention"},
        ],
        "inputs": {"representations": [
            {key: value for key, value in manifest.items() if key in {"id", "manifest_id", "shape", "dtype", "axis_semantics", "producer_stage"}}
            for manifest in (source_manifests["q_epvg_attention"], source_manifests["q_epvg_routed_values"])
        ]},
        "output_refs": [],
        "outputs": {"representations": []},
    })
    case["stages"] = json.loads(json.dumps(sample["stages"]))
    trace_payload["semantic_contract"] = {
        "version": "q-attention.case-study-trace-contract.v3",
        "lineage": "producer_owned_stage_input_output",
        "representation_inventory": list(case_payload["representation_inventory"]),
        "manifest_inventory": list(case_payload["manifest_inventory"]),
    }
    trace_payload["coverage"] = {
        "data": "observed",
        "selection": "observed",
        "context": "failed",
        "future_stage_extension": "preserved",
    }
    (selector / "case_study.json").write_text(json.dumps(case_payload), encoding="utf-8")
    (selector / "sample_trace.json").write_text(json.dumps(trace_payload), encoding="utf-8")
    expected_context = torch.einsum("lhqk,lhqkd->lhqd", model_attention, routed_values)
    assert torch.count_nonzero(score_adjustment) > 0
    if not zero_query_update:
        assert not torch.allclose(model_attention, legacy_trace_attention)
    return group, model_attention, expected_context


def test_repair_copies_stages_and_keeps_backup(tmp_path: Path) -> None:
    module = load_repairer()
    group = _make_group(tmp_path)

    manifest = module.repair_group(group, apply=True, root=group)

    case_path = group / "seed_13/selectors/selector_a/case_study.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    assert case["cases"][0]["stages"][0]["stage"] == "data"
    assert (case_path.parent / "case_study.json.pre_stage_repair").is_file()
    assert manifest["updated_case_count"] == 1
    assert (group / "case_study_stage_repair_manifest.json").is_file()


def test_repair_rejects_sample_id_mismatch_without_writing(tmp_path: Path) -> None:
    module = load_repairer()
    group = _make_group(tmp_path, mismatch=True)
    case_path = group / "seed_13/selectors/selector_a/case_study.json"
    before = case_path.read_bytes()

    with pytest.raises(ValueError, match="no matching sample_trace"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before
    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("split", "train"), ("record_index", 1), ("checkpoint", "best_valid_or_declared_selection_checkpoint")],
)
def test_repair_rejects_case_identity_fields_that_disagree_with_sample_id(
    tmp_path: Path, field: str, value: object
) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["cases"][0][field] = value
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")
    before_case = case_path.read_bytes()
    before_trace = trace_path.read_bytes()

    with pytest.raises(ValueError, match="case/sample ID components disagree"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (selector / "case_study.json.pre_stage_repair").exists()
    assert not (selector / "sample_trace.json.pre_stage_repair").exists()


def test_repair_replaces_existing_but_semantically_incomplete_stages(tmp_path: Path) -> None:
    module = load_repairer()
    group = _make_group(tmp_path, existing_failed=True)

    manifest = module.repair_group(group, apply=True, root=group)

    case_path = group / "seed_13/selectors/selector_a/case_study.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    statuses = {stage["stage"]: stage["status"] for stage in case["cases"][0]["stages"]}
    assert statuses["scoring"] == "observed"
    assert manifest["updated_case_count"] == 1
    assert manifest["replaced_case_count"] == 1
    assert manifest["selectors"][0]["replaced_cases"] == 1
    backup = case_path.parent / "case_study.json.pre_stage_repair"
    assert backup.is_file()
    old = json.loads(backup.read_text(encoding="utf-8"))
    assert old["cases"][0]["stages"][0]["status"] == "failed"


def test_repair_rejects_incomplete_source_trace_without_writing(tmp_path: Path) -> None:
    module = load_repairer()
    group = _make_group(tmp_path, trace_failed=True)
    case_path = group / "seed_13/selectors/selector_a/case_study.json"
    before = case_path.read_bytes()

    with pytest.raises(ValueError, match="required stage 'scoring'.*failed"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before
    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_reconstructs_context_from_verified_saved_tensors(tmp_path: Path) -> None:
    module = load_repairer()
    group, expected_attention, expected_context = _make_context_reconstruction_group(tmp_path)

    manifest = module.repair_group(group, apply=True, root=group)

    selector = group / "seed_13/selectors/selector_a"
    case_payload = json.loads((selector / "case_study.json").read_text(encoding="utf-8"))
    trace_payload = json.loads((selector / "sample_trace.json").read_text(encoding="utf-8"))
    case = case_payload["cases"][0]
    context = next(stage for stage in case["stages"] if stage["stage"] == "context")
    trace_context = next(stage for stage in trace_payload["samples"][0]["stages"] if stage["stage"] == "context")
    selection = next(stage for stage in trace_payload["samples"][0]["stages"] if stage["stage"] == "selection")
    intervention = next(stage for stage in trace_payload["samples"][0]["stages"] if stage["stage"] == "attention_intervention")
    attention_manifest = case["representations"]["q_epvg_model_attention"]
    output_manifest = case["representations"]["q_epvg_output"]
    attention_path = selector / attention_manifest["path"]
    output_path = selector / output_manifest["path"]
    actual_attention = torch.load(attention_path, map_location="cpu", weights_only=True)
    actual = torch.load(output_path, map_location="cpu", weights_only=True)

    assert torch.equal(actual_attention, expected_attention)
    assert torch.equal(actual, expected_context)
    assert context["status"] == trace_context["status"] == "observed"
    assert context["capture_mode"] == "reconstructed_from_checksum_verified_tensors"
    assert output_manifest["producer_stage"] == "context"
    assert attention_manifest["producer_stage"] == "selection"
    assert context["reconstruction"]["legacy_attention_trace_comparison"] == "model_attention_aligned"
    attention_derivation = attention_manifest["derivation"]
    representations = case["representations"]
    assert attention_derivation["source_manifest_ids"] == [
        representations[name]["manifest_id"]
        for name in ("q_epvg_query", "q_epvg_key", "q_epvg_score_adjustment")
    ]
    assert attention_derivation["source_sha256"] == [
        representations[name]["sha256"]
        for name in ("q_epvg_query", "q_epvg_key", "q_epvg_score_adjustment")
    ]
    assert attention_derivation["query_semantics_witness"]["source_manifest_ids"] == [
        representations[name]["manifest_id"]
        for name in ("q_epvg_query_update", "steered_attention_scores")
    ]
    assert attention_derivation["attention_trace_comparison"]["manifest_id"] == representations[
        "q_epvg_attention"
    ]["manifest_id"]
    assert context["reconstruction"]["source_manifest_ids"] == [
        attention_manifest["manifest_id"],
        representations["q_epvg_routed_values"]["manifest_id"],
    ]
    assert representations["q_epvg_attention"]["manifest_id"] not in context["reconstruction"]["source_manifest_ids"]
    assert representations["steered_attention_scores"]["manifest_id"] not in attention_derivation[
        "source_manifest_ids"
    ]
    assert any(item["ref"] == attention_manifest["manifest_id"] for item in context["input_refs"])
    assert any(item["ref"] == attention_manifest["manifest_id"] for item in selection["output_refs"])
    direct_selection_inputs = [
        representations[name]["manifest_id"]
        for name in ("q_epvg_query", "q_epvg_key", "q_epvg_score_adjustment")
    ]
    assert all(
        any(item["ref"] == manifest_id for item in selection["input_refs"])
        for manifest_id in direct_selection_inputs
    )
    assert all(
        any(item["manifest_id"] == manifest_id for item in selection["inputs"]["representations"])
        for manifest_id in direct_selection_inputs
    )
    assert selection["lineage_reconstruction"]["added_input_refs"] == direct_selection_inputs
    assert not any(item.get("ref", "").endswith("__q_epvg_attention") for item in intervention["input_refs"])
    assert selection["lineage_reconstruction"]["capture_mode"] == "mixed_producer_observed_and_reconstructed"
    assert intervention["lineage_reconciliation"]["capture_mode"] == "producer_observed_with_reconciled_inputs"
    assert "q_epvg_model_attention" in case_payload["representation_inventory"]
    assert manifest["reconstructed_context_count"] == 1
    assert manifest["selectors"][0]["attention_trace_roles"][case["case_id"]] == "model_attention_aligned"
    assert manifest["selectors"][0]["stage_lineage_updates"][0]["selection_added_output_refs"] == [attention_manifest["manifest_id"]]
    assert trace_payload["coverage"] == {
        "data": "observed",
        "selection": "observed",
        "context": "observed",
        "future_stage_extension": "preserved",
    }
    for artifact in manifest["selectors"][0]["tensor_artifacts"]:
        artifact_bytes = (selector / artifact["relative_path"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
        assert artifact["byte_count"] == len(artifact_bytes)
    assert (selector / "case_study.json.pre_stage_repair").is_file()
    assert (selector / "sample_trace.json.pre_stage_repair").is_file()


def test_repeated_apply_preserves_original_repair_manifest_and_backups(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)

    first = module.repair_group(group, apply=True, root=group)

    manifest_path = group / "case_study_stage_repair_manifest.json"
    case_path = group / "seed_13/selectors/selector_a/case_study.json"
    trace_path = case_path.with_name("sample_trace.json")
    case_backup = case_path.with_name("case_study.json.pre_stage_repair")
    trace_backup = trace_path.with_name("sample_trace.json.pre_stage_repair")
    manifest_before = manifest_path.read_bytes()
    case_backup_before = case_backup.read_bytes()
    trace_backup_before = trace_backup.read_bytes()

    second = module.repair_group(group, apply=True, root=group)

    assert first["reconstructed_context_count"] == 1
    assert second["no_op"] is True
    assert second["existing_manifest_preserved"] is True
    assert manifest_path.read_bytes() == manifest_before
    assert case_backup.read_bytes() == case_backup_before
    assert trace_backup.read_bytes() == trace_backup_before


def test_repair_aborts_if_source_changes_after_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    original_trace = trace_path.read_bytes()
    prepare_one = module._prepare_one

    def prepare_then_external_write(path: Path, *, staged_tensor_dir: Path) -> dict[str, object]:
        plan = prepare_one(path, staged_tensor_dir=staged_tensor_dir)
        if path == selector:
            with case_path.open("ab") as handle:
                handle.write(b"\n")
        return plan

    monkeypatch.setattr(module, "_prepare_one", prepare_then_external_write)
    with pytest.raises(ValueError, match="source changed after repair preflight"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes().endswith(b"\n")
    assert trace_path.read_bytes() == original_trace
    assert not case_path.with_name("case_study.json.pre_stage_repair").exists()
    assert not trace_path.with_name("sample_trace.json.pre_stage_repair").exists()
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_output.pt").exists()


def test_repair_accepts_model_aligned_attention_trace(tmp_path: Path) -> None:
    module = load_repairer()
    group, expected_attention, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    _replace_rep_tensor(selector, case_payload, "q_epvg_attention", expected_attention)
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")

    manifest = module.repair_group(group, apply=True, root=group)

    assert manifest["selectors"][0]["attention_trace_roles"]["selector_a:test:0:final"] == "model_attention_aligned"


def test_repair_rejects_legacy_only_attention_trace(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path, legacy_only_attention_trace=True)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    before_case = case_path.read_bytes()
    before_trace = trace_path.read_bytes()

    with pytest.raises(ValueError, match="matches only the legacy query-trace formula"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()
    assert not (case_path.parent / "case_study_tensors/test_0_final__q_epvg_model_attention.pt").exists()


def test_repair_marks_attention_formulas_aligned_when_query_update_is_zero(tmp_path: Path) -> None:
    module = load_repairer()
    group, expected_attention, _ = _make_context_reconstruction_group(
        tmp_path, zero_query_update=True
    )

    manifest = module.repair_group(group, apply=True, root=group)

    selector = group / "seed_13/selectors/selector_a"
    case_payload = json.loads((selector / "case_study.json").read_text(encoding="utf-8"))
    context = next(
        stage for stage in case_payload["cases"][0]["stages"] if stage["stage"] == "context"
    )
    role = "model_and_legacy_formulas_aligned"
    assert manifest["selectors"][0]["attention_trace_roles"]["selector_a:test:0:final"] == role
    assert context["reconstruction"]["legacy_attention_trace_comparison"] == role


@pytest.mark.parametrize(
    "selector_name",
    [
        "q_epvg_zz_value_only_quantum",
        "q_epvg_trainable_pauli_mix_score_value_classical",
    ],
)
def test_repair_non_query_path_ignores_zero_malformed_query_update_witness(
    tmp_path: Path, selector_name: str
) -> None:
    module = load_repairer()
    group, expected_attention, expected_context = _make_context_reconstruction_group(
        tmp_path, zero_query_update=True
    )
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    case = case_payload["cases"][0]
    old_id = case["case_id"]
    new_id = f"{selector_name}:test:0:final"
    case_payload["selector"] = selector_name
    case["case_id"] = new_id
    trace_payload["samples"][0]["sample_id"] = new_id
    query_update = torch.zeros((1, 1, 3, 2), dtype=torch.float32)
    _replace_rep_tensor(selector, case_payload, "q_epvg_query_update", query_update)
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")

    manifest = module.repair_group(group, apply=True, root=group)

    repaired_case = json.loads(case_path.read_text(encoding="utf-8"))["cases"][0]
    context = next(stage for stage in repaired_case["stages"] if stage["stage"] == "context")
    output_manifest = repaired_case["representations"]["q_epvg_output"]
    actual = torch.load(selector / output_manifest["path"], map_location="cpu", weights_only=True)
    assert torch.equal(actual, expected_context)
    assert context["status"] == "observed"
    assert context["capture_mode"] == "reconstructed_from_checksum_verified_tensors"
    witness = context["reconstruction"]["query_update_witness"]
    assert witness["policy"] == "ignored_non_query_zero_witness_shape_mismatch"
    assert witness["shape"] == [1, 1, 3, 2]
    assert witness["expected_shape"] == [1, 1, 2, 2]
    assert manifest["reconstructed_context_count"] == 1
    actual_attention = torch.load(
        selector / repaired_case["representations"]["q_epvg_model_attention"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(actual_attention, expected_attention)


def test_repair_rejects_attention_trace_matching_neither_formula(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    before_trace = trace_path.read_bytes()
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    _replace_rep_tensor(selector, case_payload, "q_epvg_attention", torch.zeros(1, 1, 2, 3))
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")
    before_case = case_path.read_bytes()

    with pytest.raises(ValueError, match="matches neither the model's masked softmax"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_rejects_unverified_post_intervention_query_semantics(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    _replace_rep_tensor(selector, case_payload, "steered_attention_scores", torch.zeros(1, 1, 2, 3))
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not confirm that q_epvg_query is the post-intervention query"):
        module.repair_group(group, apply=True, root=group)

    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_rejects_a_nonzero_but_numerically_indistinguishable_query_update(
    tmp_path: Path,
) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case = case_payload["cases"][0]

    query_manifest = case["representations"]["q_epvg_query"]
    key_manifest = case["representations"]["q_epvg_key"]
    adjustment_manifest = case["representations"]["q_epvg_score_adjustment"]
    query = torch.load(selector / query_manifest["path"], map_location="cpu", weights_only=True)
    key = torch.load(selector / key_manifest["path"], map_location="cpu", weights_only=True)
    adjustment = torch.load(
        selector / adjustment_manifest["path"], map_location="cpu", weights_only=True
    )
    tiny_update = torch.full_like(query, 1e-4)
    _replace_rep_tensor(selector, case_payload, "q_epvg_query_update", tiny_update)
    pre_scores = torch.matmul(query - tiny_update, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)
    pre_scores = pre_scores + adjustment
    _replace_rep_tensor(selector, case_payload, "steered_attention_scores", pre_scores)
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not distinguish the pre-intervention"):
        module.repair_group(group, apply=True, root=group)

    assert not case_path.with_name("case_study.json.pre_stage_repair").exists()
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_model_attention.pt").exists()


def test_repair_rejects_manifest_path_bound_to_another_tensor_id(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case = case_payload["cases"][0]
    original = case["representations"]["q_epvg_query"]
    replacement = {**original, "path": "case_study_tensors/test_0_final__q_epvg_key.pt"}
    case["representations"]["q_epvg_query"] = replacement
    case_payload["tensor_manifest"] = [
        replacement if item["manifest_id"] == original["manifest_id"] else item
        for item in case_payload["tensor_manifest"]
    ]
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="tensor path does not match its manifest_id"):
        module.repair_group(group, apply=True, root=group)

    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_rejects_unexpected_source_producer_stage(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    original = case_payload["cases"][0]["representations"]["steered_attention_scores"]
    replacement = {**original, "producer_stage": "context"}
    case_payload["cases"][0]["representations"]["steered_attention_scores"] = replacement
    case_payload["tensor_manifest"] = [
        replacement if item["manifest_id"] == original["manifest_id"] else item
        for item in case_payload["tensor_manifest"]
    ]
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="steered_attention_scores has unexpected producer_stage"):
        module.repair_group(group, apply=True, root=group)

    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_rejects_duplicate_tensor_manifest_ids(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["tensor_manifest"].append(json.loads(json.dumps(case_payload["tensor_manifest"][0])))
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate tensor manifest_id"):
        module.repair_group(group, apply=True, root=group)

    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_rejects_all_masked_sample(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    case_payload["cases"][0]["attention_mask"] = [0, 0, 0]
    preprocess = next(stage for stage in trace_payload["samples"][0]["stages"] if stage["stage"] == "preprocess")
    preprocess["outputs"]["attention_mask"] = [0, 0, 0]
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="attention_mask has no valid key tokens"):
        module.repair_group(group, apply=True, root=group)

    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()
    assert not (trace_path.parent / "sample_trace.json.pre_stage_repair").exists()


def test_repair_rejects_context_source_hash_mismatch_without_writing(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    before_case = case_path.read_bytes()
    trace_path = selector / "sample_trace.json"
    before_trace = trace_path.read_bytes()
    source = selector / "case_study_tensors/test_0_final__q_epvg_attention.pt"
    source.write_bytes(source.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="source tensor size/hash mismatch"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_output.pt").exists()
    assert not (selector / "case_study.json.pre_stage_repair").exists()


def test_repair_rejects_query_attention_mask_mismatch_without_writing(tmp_path: Path) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["cases"][0]["attention_mask"] = [1, 0, 1]
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")
    before_case = case_path.read_bytes()
    before_trace = trace_path.read_bytes()

    with pytest.raises(ValueError, match="attention masks disagree"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_model_attention.pt").exists()
    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_rejects_duplicate_case_id_without_writing(tmp_path: Path) -> None:
    module = load_repairer()
    group = _make_group(tmp_path)
    case_path = group / "seed_13/selectors/selector_a/case_study.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["cases"].append(json.loads(json.dumps(payload["cases"][0])))
    case_path.write_text(json.dumps(payload), encoding="utf-8")
    before = case_path.read_bytes()

    with pytest.raises(ValueError, match="duplicate case_id"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before
    assert not (case_path.parent / "case_study.json.pre_stage_repair").exists()


def test_repair_preflights_all_backup_conflicts_before_writing(tmp_path: Path) -> None:
    module = load_repairer()
    group = _make_group(tmp_path)
    second = group / "seed_13/selectors/selector_b"
    second.mkdir(parents=True)
    first_selector = group / "seed_13/selectors/selector_a"
    first_case = json.loads((first_selector / "case_study.json").read_text(encoding="utf-8"))
    first_trace = json.loads((first_selector / "sample_trace.json").read_text(encoding="utf-8"))
    first_case["selector"] = "selector_b"
    first_case["cases"][0]["case_id"] = "selector_b:test:0:final"
    first_trace["samples"][0]["sample_id"] = "selector_b:test:0:final"
    (second / "case_study.json").write_text(json.dumps(first_case), encoding="utf-8")
    (second / "sample_trace.json").write_text(json.dumps(first_trace), encoding="utf-8")
    first_case_path = first_selector / "case_study.json"
    first_before = first_case_path.read_bytes()
    (second / "case_study.json.pre_stage_repair").write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="backup already exists"):
        module.repair_group(group, apply=True, root=group)

    assert first_case_path.read_bytes() == first_before
    assert not (first_selector / "case_study.json.pre_stage_repair").exists()


def test_rollback_never_restores_a_partial_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    before_case = case_path.read_bytes()
    before_trace = trace_path.read_bytes()

    def copy_then_fail(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"partial backup")
        raise OSError("simulated backup ENOSPC")

    monkeypatch.setattr(module.shutil, "copy2", copy_then_fail)
    with pytest.raises(OSError, match="simulated backup ENOSPC"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (selector / "case_study.json.pre_stage_repair").exists()
    assert not (selector / "sample_trace.json.pre_stage_repair").exists()
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_output.pt").exists()


def test_repair_detects_and_removes_a_silently_truncated_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    before_case = case_path.read_bytes()
    before_trace = trace_path.read_bytes()

    def copy_truncated(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"truncated")

    monkeypatch.setattr(module.shutil, "copy2", copy_truncated)
    with pytest.raises(OSError, match="backup verification failed"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (selector / "case_study.json.pre_stage_repair").exists()
    assert not (selector / "sample_trace.json.pre_stage_repair").exists()


def test_atomic_writer_preserves_file_mode_and_cleans_temporary_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_repairer()
    target = tmp_path / "shared.json"
    target.write_bytes(b"before")
    target.chmod(0o640)

    module._write_bytes_atomic(target, b"after")

    assert target.read_bytes() == b"after"
    assert target.stat().st_mode & 0o777 == 0o640
    assert not list(tmp_path.glob(".shared.json.*.tmp"))

    def fail_replace(_source: Path, _target: Path) -> None:
        raise PermissionError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="simulated replace failure"):
        module._write_bytes_atomic(target, b"must-not-appear")
    assert target.read_bytes() == b"after"
    assert not list(tmp_path.glob(".shared.json.*.tmp"))


def test_keyboard_interrupt_rolls_back_json_and_reconstructed_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    before_case = case_path.read_bytes()
    before_trace = trace_path.read_bytes()
    write_json_atomic = module._write_json_atomic

    def interrupt_trace(path: Path, payload: object) -> None:
        if path.name == "sample_trace.json":
            raise KeyboardInterrupt()
        write_json_atomic(path, payload)

    monkeypatch.setattr(module, "_write_json_atomic", interrupt_trace)
    with pytest.raises(KeyboardInterrupt):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == before_case
    assert trace_path.read_bytes() == before_trace
    assert not (group / "case_study_stage_repair_manifest.json").exists()
    assert not (selector / "case_study.json.pre_stage_repair").exists()
    assert not (selector / "sample_trace.json.pre_stage_repair").exists()
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_model_attention.pt").exists()
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_output.pt").exists()


def test_interrupt_after_tensor_replace_rolls_back_registered_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    original_write = module._write_bytes_atomic

    def write_then_interrupt(path: Path, payload: bytes) -> None:
        original_write(path, payload)
        if path.name.endswith("__q_epvg_model_attention.pt"):
            raise KeyboardInterrupt()

    monkeypatch.setattr(module, "_write_bytes_atomic", write_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.repair_group(group, apply=True, root=group)

    assert not (selector / "case_study_tensors/test_0_final__q_epvg_model_attention.pt").exists()
    assert not (selector / "case_study_tensors/test_0_final__q_epvg_output.pt").exists()
    assert not (group / "case_study_stage_repair_manifest.json").exists()
    assert not (selector / "case_study.json.pre_stage_repair").exists()
    assert not (selector / "sample_trace.json.pre_stage_repair").exists()


def test_rollback_reports_a_source_that_fails_post_restore_hash_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_repairer()
    group, _, _ = _make_context_reconstruction_group(tmp_path)
    selector = group / "seed_13/selectors/selector_a"
    case_path = selector / "case_study.json"
    trace_path = selector / "sample_trace.json"
    before_case = case_path.read_bytes()
    before_trace = trace_path.read_bytes()
    original_write_json = module._write_json_atomic
    original_write_bytes = module._write_bytes_atomic

    def interrupt_trace(path: Path, payload: object) -> None:
        if path == trace_path:
            raise OSError("simulated trace write failure")
        original_write_json(path, payload)

    def corrupt_restored_case(path: Path, payload: bytes) -> None:
        original_write_bytes(path, payload)
        if path == case_path and payload == before_case:
            original_write_bytes(path, b"corrupt restored source")

    monkeypatch.setattr(module, "_write_json_atomic", interrupt_trace)
    monkeypatch.setattr(module, "_write_bytes_atomic", corrupt_restored_case)
    with pytest.raises(RuntimeError, match="rollback incomplete.*restored source checksum mismatch"):
        module.repair_group(group, apply=True, root=group)

    assert case_path.read_bytes() == b"corrupt restored source"
    assert trace_path.read_bytes() == before_trace
    assert (selector / "case_study.json.pre_stage_repair").read_bytes() == before_case
    assert (selector / "sample_trace.json.pre_stage_repair").is_file()
