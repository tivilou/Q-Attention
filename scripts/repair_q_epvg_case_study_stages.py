#!/usr/bin/env python3
"""Repair Q-EPVG Case Study stage projections and a missing context tensor.

The repair never retrains or changes metrics. It requires matching sample IDs,
verifies every source tensor's identity and checksum, reconstructs the model's
actual masked attention from the post-intervention query/key and score
adjustment, and only then contracts it with the recorded routed values. A
pre-existing stage list is complete only when every required stage is present,
ordered, and marked ``observed``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


CASE_SCHEMA = "q-attention.Q-EPVG-case-study.v2"
TRACE_SCHEMA = "sample-trace.v1"
MANIFEST_SCHEMA = "q-attention.q-epvg-case-study-stage-repair.v3"

REQUIRED_STAGES = (
    "data",
    "preprocess",
    "embedding",
    "encoder",
    "training",
    "attention_baseline",
    "scoring",
    "selection",
    "context",
    "classifier",
    "evaluation",
    "diagnosis",
)
OPTIONAL_STAGES = {"retrieval", "generation", "attention_intervention"}
STAGE_ORDER = (
    "data",
    "preprocess",
    "embedding",
    "encoder",
    "training",
    "retrieval",
    "attention_baseline",
    "scoring",
    "selection",
    "attention_intervention",
    "context",
    "classifier",
    "generation",
    "evaluation",
    "diagnosis",
)
ALLOWED_STATUSES = {"observed", "not_applicable", "failed"}


def _selector_intervention_path(selector: Any) -> str | None:
    """Extract the declared Q-EPVG intervention path from a selector id.

    The selector id is producer-owned metadata in the uploaded Case Study.  The
    path controls only the query-update witness policy; attention and context
    are still recomputed and verified below.
    """

    if not isinstance(selector, str):
        return None
    for path in ("value_only", "score_value", "query"):
        if f"_{path}_" in f"_{selector}_":
            return path
    return None


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_atomic(path, data)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, target_mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _selector_dirs(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in group_dir.glob("seed_*/selectors/*")
        if path.is_dir()
        and (path / "case_study.json").is_file()
        and (path / "sample_trace.json").is_file()
    )


def _stage_completeness_error(stages: object, *, where: str) -> str | None:
    """Return a precise error when a stage projection is not export-complete.

    This deliberately checks the same structural contract that matters for
    repair. The full exporter remains responsible for tensor references and
    manifest checks; this function prevents the repairer from treating a
    merely present, but failed or incomplete, stage list as complete.
    """

    if not isinstance(stages, list) or not stages:
        return "stages must be a non-empty list"

    names: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or not isinstance(stage.get("stage"), str) or not stage["stage"]:
            return f"stages[{index}] is missing a non-empty stage name"
        status = stage.get("status")
        if status not in ALLOWED_STATUSES:
            return f"stages[{index}] ({stage['stage']!r}) has invalid status {status!r}"
        names.append(stage["stage"])

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return f"duplicate stage names: {', '.join(duplicates)}"

    missing = [name for name in REQUIRED_STAGES if name not in names]
    if missing:
        return f"missing required stages: {', '.join(missing)}"

    unknown = [
        name
        for name in names
        if name not in REQUIRED_STAGES and name not in OPTIONAL_STAGES
    ]
    if unknown:
        return f"unknown stages: {', '.join(unknown)}"

    positions = [STAGE_ORDER.index(name) for name in names]
    if positions != sorted(positions):
        return "stages are not in producer order"

    by_name = {stage["stage"]: stage for stage in stages}
    for name in REQUIRED_STAGES:
        status = by_name[name]["status"]
        if status != "observed":
            index = names.index(name)
            return f"required stage {name!r} at index {index} is not fully observed ({status})"
    return None


def _one_stage(stages: object, name: str, *, where: str) -> dict[str, Any]:
    if not isinstance(stages, list):
        raise ValueError(f"{where}: stages must be a list")
    matches = [stage for stage in stages if isinstance(stage, dict) and stage.get("stage") == name]
    if len(matches) != 1:
        raise ValueError(f"{where}: expected exactly one {name!r} stage, found {len(matches)}")
    return matches[0]


def _repair_token_alignment(
    case: dict[str, Any],
    trace_sample: dict[str, Any],
    *,
    where: str,
) -> dict[str, Any] | None:
    """Remove only producer-proven batch padding from legacy token IDs.

    Older workers serialized the complete padded batch row as ``token_ids``
    while ``tokens`` represented the unpadded source record.  The attention
    mask is the only accepted witness for this compatibility repair: it must
    be a prefix of ones, its active length must equal ``len(tokens)``, and the
    trace's preprocess projection must contain the same padded IDs.  Any other
    mismatch remains a hard error.
    """

    tokens = case.get("tokens")
    token_ids = case.get("token_ids")
    attention_mask = case.get("attention_mask")
    # Minimal reconstruction fixtures may omit public semantic fields; leave
    # those untouched because this helper only owns legacy alignment repair.
    if tokens is None and token_ids is None:
        return None
    if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens):
        raise ValueError(f"{where}: tokens must be a string list")
    if not isinstance(token_ids, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in token_ids):
        raise ValueError(f"{where}: token_ids must be an integer list")
    if len(token_ids) == len(tokens):
        return None
    if len(token_ids) < len(tokens):
        raise ValueError(f"{where}: token_ids are shorter than tokens and cannot be reconstructed")
    if not isinstance(attention_mask, list) or len(attention_mask) != len(token_ids):
        raise ValueError(f"{where}: token_ids/token attention mask lengths are incompatible")
    if not all(isinstance(item, (bool, int)) and item in (0, 1) for item in attention_mask):
        raise ValueError(f"{where}: attention_mask is not a binary list")
    active_length = sum(int(item) for item in attention_mask)
    if active_length != len(tokens) or attention_mask[:active_length] != [1] * active_length or any(attention_mask[active_length:]):
        raise ValueError(
            f"{where}: token_ids mismatch is not producer-proven batch padding "
            f"(tokens={len(tokens)}, ids={len(token_ids)}, active={active_length})"
        )

    preprocess = _one_stage(trace_sample.get("stages"), "preprocess", where=where)
    outputs = preprocess.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("token_ids") != token_ids:
        raise ValueError(f"{where}: preprocess token_ids do not match the Case Study token_ids witness")

    trimmed = token_ids[:active_length]
    case["token_ids"] = trimmed
    outputs["token_ids"] = list(trimmed)
    return {
        "case_id": case.get("case_id"),
        "original_token_id_count": len(token_ids),
        "trimmed_token_id_count": len(trimmed),
        "padding_count": len(token_ids) - len(trimmed),
        "witness": "attention_mask_prefix_and_original_token_count",
    }


def _load_verified_tensor(
    selector_dir: Path,
    manifest: dict[str, Any],
    *,
    where: str,
) -> torch.Tensor:
    relative = Path(str(manifest.get("path", "")))
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError(f"{where}: tensor manifest_id must be a non-empty string")
    if relative.as_posix() != f"case_study_tensors/{manifest_id}.pt":
        raise ValueError(f"{where}: tensor path does not match its manifest_id")
    if relative.is_absolute() or ".." in relative.parts or not relative.as_posix().startswith("case_study_tensors/"):
        raise ValueError(f"{where}: unsafe tensor path {str(relative)!r}")
    selector_root = selector_dir.resolve()
    source = (selector_root / relative).resolve()
    try:
        source.relative_to(selector_root)
    except ValueError as exc:
        raise ValueError(f"{where}: tensor path escapes selector directory") from exc
    if not source.is_file():
        raise ValueError(f"{where}: missing source tensor {relative}")
    raw = source.read_bytes()
    if len(raw) != manifest.get("byte_count") or hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
        raise ValueError(f"{where}: source tensor size/hash mismatch")
    try:
        tensor = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ValueError(
            f"{where}: installed PyTorch does not support safe weights_only tensor loading"
        ) from exc
    except Exception as exc:
        raise ValueError(f"{where}: cannot safely load source tensor: {exc}") from exc
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{where}: source artifact did not contain a tensor")
    if list(tensor.shape) != manifest.get("shape") or str(tensor.dtype) != manifest.get("dtype"):
        raise ValueError(f"{where}: source tensor shape/dtype disagrees with its manifest")
    if not tensor.is_floating_point():
        raise ValueError(f"{where}: source tensor must have a floating dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{where}: source tensor contains non-finite values")
    return tensor.contiguous()


def _context_manifest_and_bytes(
    output: torch.Tensor,
    *,
    manifest_id: str,
    relative_path: str,
) -> tuple[dict[str, Any], bytes]:
    value = output.detach().to(device="cpu").contiguous()
    if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
        raise ValueError("reconstructed tensor must be finite and floating point")
    payload = io.BytesIO()
    torch.save(value, payload, _use_new_zipfile_serialization=False)
    raw = payload.getvalue()
    preview_value = value.real if value.is_complex() else value
    preview_float = preview_value.float()
    manifest = {
        "id": "q_epvg_output",
        "manifest_id": manifest_id,
        "producer_stage": "context",
        "path": relative_path,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "axis_semantics": ["layers", "heads", "query_tokens", "value_dim"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "preview": {
            "values_first_32": preview_value.flatten()[:32].tolist(),
            "min": float(preview_float.min().item()) if preview_float.numel() else 0.0,
            "max": float(preview_float.max().item()) if preview_float.numel() else 0.0,
            "mean": float(preview_float.mean().item()) if preview_float.numel() else 0.0,
            "l2_norm": float(preview_float.norm().item()),
        },
    }
    return manifest, raw


def _safe_representation_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    preview = manifest["preview"]
    safe_preview = {
        key: preview[key]
        for key in ("min", "max", "mean", "l2_norm")
        if key in preview
    }
    if isinstance(preview.get("values_first_32"), list):
        safe_preview["values_first_8"] = preview["values_first_32"][:8]
    projection = {
        "id": manifest["id"],
        "manifest_id": manifest["manifest_id"],
        "shape": manifest["shape"],
        "dtype": manifest["dtype"],
        "axis_semantics": manifest["axis_semantics"],
        "producer_stage": manifest["producer_stage"],
        "preview": safe_preview,
    }
    if isinstance(manifest.get("semantic_role"), str):
        projection["semantic_role"] = manifest["semantic_role"]
    return projection


def _reconstruct_context_output(
    selector_dir: Path,
    case_payload: dict[str, Any],
    trace_payload: dict[str, Any],
    case: dict[str, Any],
    trace_sample: dict[str, Any],
    *,
    staged_tensor_dir: Path,
) -> dict[str, Any] | None:
    trace_context = _one_stage(
        trace_sample.get("stages"), "context", where=f"sample {trace_sample.get('sample_id')}"
    )
    if trace_context.get("status") != "failed":
        return None
    expected_reason = "Q-EPVG context output was not captured"
    if trace_context.get("reason") != expected_reason:
        raise ValueError(
            f"sample {trace_sample.get('sample_id')}: context failed for an unrecognized reason; "
            "refusing to infer or rewrite it"
        )

    case_stages = case.get("stages")
    if isinstance(case_stages, list):
        case_context = _one_stage(case_stages, "context", where=f"case {case.get('case_id')}")
        if case_context.get("status") != "failed" or case_context.get("reason") != expected_reason:
            raise ValueError(
                f"case {case.get('case_id')}: Case Study and source trace disagree about the failed context stage"
            )

    representations = case.get("representations")
    manifests = case_payload.get("tensor_manifest")
    if not isinstance(representations, dict) or not isinstance(manifests, list):
        raise ValueError(f"case {case.get('case_id')}: tensor representations/manifests are missing")
    manifest_ids: list[str] = []
    for index, item in enumerate(manifests):
        if not isinstance(item, dict) or not isinstance(item.get("manifest_id"), str) or not item["manifest_id"]:
            raise ValueError(f"case {case.get('case_id')}: tensor_manifest[{index}] has no valid manifest_id")
        manifest_ids.append(item["manifest_id"])
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError(f"case {case.get('case_id')}: duplicate tensor manifest_id")
    manifest_by_id = {
        item.get("manifest_id"): item
        for item in manifests
        if isinstance(item, dict) and isinstance(item.get("manifest_id"), str)
    }
    source_rep_ids = (
        "q_epvg_query",
        "q_epvg_key",
        "q_epvg_query_update",
        "q_epvg_score_adjustment",
        "q_epvg_attention",
        "steered_attention_scores",
        "q_epvg_routed_values",
    )
    source_manifests = {rep_id: representations.get(rep_id) for rep_id in source_rep_ids}
    missing = [rep_id for rep_id, manifest in source_manifests.items() if not isinstance(manifest, dict)]
    if missing:
        raise ValueError(
            f"case {case.get('case_id')}: exact context recovery requires saved representations: {', '.join(missing)}"
        )
    typed_manifests = {rep_id: manifest for rep_id, manifest in source_manifests.items() if isinstance(manifest, dict)}
    for rep_id, manifest in typed_manifests.items():
        manifest_id = manifest.get("manifest_id")
        if not isinstance(manifest_id, str) or manifest_by_id.get(manifest_id) != manifest:
            raise ValueError(f"case {case.get('case_id')}: {rep_id} disagrees with tensor_manifest")
    expected_producer_stages = {
        "q_epvg_query": "scoring",
        "q_epvg_key": "scoring",
        "q_epvg_query_update": "attention_intervention",
        "q_epvg_score_adjustment": "scoring",
        "q_epvg_attention": "selection",
        "steered_attention_scores": "selection",
        "q_epvg_routed_values": "attention_intervention",
    }
    for rep_id, expected_stage in expected_producer_stages.items():
        if typed_manifests[rep_id].get("producer_stage") != expected_stage:
            raise ValueError(
                f"case {case.get('case_id')}: {rep_id} has unexpected producer_stage; "
                f"expected {expected_stage!r}"
            )
    checkpoint_slugs = {
        "initial_or_pre_training": "initial",
        "best_valid_or_declared_selection_checkpoint": "best",
        "final": "final",
    }
    case_selector = case_payload.get("selector")
    intervention_path = _selector_intervention_path(case_selector)
    # Legacy synthetic fixtures and older exports may not encode the path in
    # the selector id. Keep the historical strict query-update behavior for
    # those records instead of silently treating them as non-query paths.
    query_intervention_path = intervention_path in {None, "query"}
    non_query_path = intervention_path in {"value_only", "score_value"}
    case_split = case.get("split")
    record_index = case.get("record_index")
    checkpoint_slug = checkpoint_slugs.get(str(case.get("checkpoint")))
    sample_id = trace_sample.get("sample_id")
    if (
        not isinstance(case_selector, str)
        or not case_selector
        or case_split not in {"train", "valid", "test"}
        or not isinstance(record_index, int)
        or isinstance(record_index, bool)
        or record_index < 0
        or checkpoint_slug is None
    ):
        raise ValueError(f"case {case.get('case_id')}: split/record/checkpoint identity fields are invalid")
    expected_case_id = f"{case_selector}:{case_split}:{record_index}:{checkpoint_slug}"
    if case.get("case_id") != expected_case_id or sample_id != expected_case_id:
        raise ValueError(
            f"case {case.get('case_id')}: case/sample ID components disagree with split, record, or checkpoint"
        )
    query_manifest = typed_manifests["q_epvg_query"]
    query_id = query_manifest.get("manifest_id")
    suffix = "__q_epvg_query"
    if not isinstance(query_id, str) or not query_id.endswith(suffix):
        raise ValueError(f"case {case.get('case_id')}: unexpected Q-EPVG query manifest identity")
    prefix = query_id[: -len(suffix)]
    expected_prefix = f"{case_split}_{record_index}_{checkpoint_slug}"
    if prefix != expected_prefix:
        raise ValueError(f"case {case.get('case_id')}: tensor capture identity does not match the sample")
    expected_suffixes = {
        "q_epvg_key": "__q_epvg_key",
        "q_epvg_query_update": "__q_epvg_query_update",
        "q_epvg_score_adjustment": "__q_epvg_score_adjustment",
        "q_epvg_attention": "__q_epvg_attention",
        "steered_attention_scores": "__steered_attention_scores",
        "q_epvg_routed_values": "__q_epvg_routed_values",
    }
    for rep_id, expected_suffix in expected_suffixes.items():
        if typed_manifests[rep_id].get("manifest_id") != f"{prefix}{expected_suffix}":
            raise ValueError(f"case {case.get('case_id')}: {rep_id} comes from a different sample/checkpoint capture")

    expected_axes = {
        "q_epvg_query": ["layers", "heads", "query_tokens", "head_dim"],
        "q_epvg_key": ["layers", "heads", "key_tokens", "head_dim"],
        "q_epvg_query_update": ["layers", "heads", "query_tokens", "head_dim"],
        "q_epvg_score_adjustment": ["layers", "heads", "query_tokens", "key_tokens"],
        "q_epvg_attention": ["layers", "heads", "query_tokens", "key_tokens"],
        "steered_attention_scores": ["layers", "heads", "query_tokens", "key_tokens"],
        "q_epvg_routed_values": ["layers", "heads", "query_tokens", "key_tokens", "value_dim"],
    }
    for rep_id, axes in expected_axes.items():
        if typed_manifests[rep_id].get("axis_semantics") != axes:
            raise ValueError(f"case {case.get('case_id')}: {rep_id} axes are not the declared Q-EPVG layout")

    original_context_refs = trace_context.get("input_refs")
    if not isinstance(original_context_refs, list):
        raise ValueError(f"case {case.get('case_id')}: context input_refs must be a list")
    context_input_ids = {
        item.get("ref") for item in original_context_refs
        if isinstance(item, dict) and item.get("kind") == "representation"
    }
    if {
        typed_manifests["q_epvg_attention"]["manifest_id"],
        typed_manifests["q_epvg_routed_values"]["manifest_id"],
    } - context_input_ids:
        raise ValueError(f"case {case.get('case_id')}: failed context stage does not reference its recorded source tensors")

    case_mask = case.get("attention_mask")
    if (
        not isinstance(case_mask, list)
        or not all(isinstance(value, (bool, int)) and value in (0, 1) for value in case_mask)
    ):
        raise ValueError(f"case {case.get('case_id')}: attention_mask must be a one-dimensional binary list")
    preprocess_stage = _one_stage(
        trace_sample.get("stages"), "preprocess", where=f"sample {trace_sample.get('sample_id')}"
    )
    preprocess_outputs = preprocess_stage.get("outputs")
    if not isinstance(preprocess_outputs, dict) or preprocess_outputs.get("attention_mask") != case_mask:
        raise ValueError(f"case {case.get('case_id')}: Case Study and preprocessing attention masks disagree")

    tensors = {
        rep_id: _load_verified_tensor(
            selector_dir, manifest, where=f"case {case.get('case_id')} {rep_id}"
        )
        for rep_id, manifest in typed_manifests.items()
    }
    query = tensors["q_epvg_query"]
    key = tensors["q_epvg_key"]
    query_update = tensors["q_epvg_query_update"]
    score_adjustment = tensors["q_epvg_score_adjustment"]
    trace_attention = tensors["q_epvg_attention"]
    saved_pre_intervention_scores = tensors["steered_attention_scores"]
    routed_values = tensors["q_epvg_routed_values"]
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(f"case {case.get('case_id')}: query/key/update tensors have unexpected ranks or shapes")
    query_update_is_zero = not bool(torch.count_nonzero(query_update).item())
    if query_intervention_path and query_update.shape != query.shape:
        raise ValueError(f"case {case.get('case_id')}: query/key/update tensors have unexpected ranks or shapes")
    if non_query_path and not query_update_is_zero:
        raise ValueError(
            f"case {case.get('case_id')}: value_only query_update witness is nonzero; "
            "refusing to infer query semantics"
        )
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"case {case.get('case_id')}: query/key layer, head, or feature dimensions differ")
    expected_score_shape = (*query.shape[:3], key.shape[-2])
    if (
        tuple(score_adjustment.shape) != expected_score_shape
        or tuple(trace_attention.shape) != expected_score_shape
        or tuple(saved_pre_intervention_scores.shape) != expected_score_shape
    ):
        raise ValueError(f"case {case.get('case_id')}: score/attention tensor shapes disagree with query/key")
    if tuple(routed_values.shape[:-1]) != expected_score_shape:
        raise ValueError(f"case {case.get('case_id')}: routed values are incompatible with query/key positions")
    if len(case_mask) != key.shape[-2]:
        raise ValueError(f"case {case.get('case_id')}: attention_mask length does not match key tokens")
    if not any(case_mask):
        raise ValueError(f"case {case.get('case_id')}: attention_mask has no valid key tokens")
    if any(tensor.dtype != query.dtype for tensor in tensors.values()):
        raise ValueError(f"case {case.get('case_id')}: source tensor dtypes differ")

    compare_rtol, compare_atol = (
        (2e-2, 2e-3) if query.dtype in {torch.bfloat16, torch.float16} else (2e-3, 2e-4)
    )
    model_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    model_scores = model_scores + score_adjustment
    if non_query_path:
        # Non-query paths never change Q. The malformed zero witness is
        # retained for provenance but must not be broadcast or subtracted.
        expected_pre_intervention_scores = model_scores
    else:
        pre_intervention_query = query - query_update
        expected_pre_intervention_scores = (
            torch.matmul(pre_intervention_query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        ) + score_adjustment
    if not torch.allclose(
        saved_pre_intervention_scores.float(),
        expected_pre_intervention_scores.float(),
        rtol=compare_rtol,
        atol=compare_atol,
    ):
        raise ValueError(
            f"case {case.get('case_id')}: saved steered_attention_scores do not confirm that "
            "q_epvg_query is the post-intervention query"
        )

    saved_scores_match_post_query = torch.allclose(
        saved_pre_intervention_scores.float(),
        model_scores.float(),
        rtol=compare_rtol,
        atol=compare_atol,
    )
    witness_separation = model_scores - expected_pre_intervention_scores
    if saved_scores_match_post_query and bool(torch.count_nonzero(witness_separation).item()):
        raise ValueError(
            f"case {case.get('case_id')}: saved steered_attention_scores do not distinguish "
            "the pre-intervention and post-intervention query hypotheses within the declared tolerance"
        )
    key_mask = torch.tensor(case_mask, dtype=torch.bool).view(1, 1, 1, -1)
    model_attention = torch.softmax(
        model_scores.masked_fill(~key_mask, torch.finfo(model_scores.dtype).min), dim=-1
    )
    if non_query_path:
        # A malformed non-query witness cannot be used to form a legacy
        # query-intervention score tensor.  Keep this comparison explicit so
        # a shape error cannot be hidden by broadcasting.
        legacy_trace_attention = model_attention
    else:
        legacy_trace_scores = model_scores + (
            torch.matmul(query_update, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        )
        legacy_trace_attention = torch.softmax(
            legacy_trace_scores.masked_fill(~key_mask, torch.finfo(legacy_trace_scores.dtype).min), dim=-1
        )
    trace_matches_model = torch.allclose(
        trace_attention.float(), model_attention.float(), rtol=compare_rtol, atol=compare_atol
    )
    trace_matches_legacy_plugin = torch.allclose(
        trace_attention.float(), legacy_trace_attention.float(), rtol=compare_rtol, atol=compare_atol
    )
    if trace_matches_legacy_plugin and not trace_matches_model:
        raise ValueError(
            f"case {case.get('case_id')}: recorded q_epvg_attention matches only the legacy query-trace "
            "formula, contradicting the verified model attention semantics"
        )
    if not trace_matches_model and not trace_matches_legacy_plugin:
        raise ValueError(
            f"case {case.get('case_id')}: recorded q_epvg_attention matches neither the model's "
            "masked softmax nor the legacy query-trace formula"
        )
    trace_attention_role = (
        (
            "model_attention_aligned_value_only_zero_malformed_query_update"
            if intervention_path == "value_only"
            else "model_attention_aligned_score_value_zero_malformed_query_update"
        )
        if non_query_path and trace_matches_model
        else
        "model_and_legacy_formulas_aligned"
        if trace_matches_model and trace_matches_legacy_plugin
        else "model_attention_aligned"
        if trace_matches_model
        else "legacy_query_trace_applied_query_update_again"
    )

    model_attention_id = f"{prefix}__q_epvg_model_attention"
    output_manifest_id = f"{prefix}__q_epvg_output"
    for candidate_id in (model_attention_id, output_manifest_id):
        if candidate_id in manifest_by_id:
            raise ValueError(f"case {case.get('case_id')}: derived tensor manifest already exists: {candidate_id}")
    for rep_id in ("q_epvg_model_attention", "q_epvg_output"):
        if representations.get(rep_id) is not None:
            raise ValueError(f"case {case.get('case_id')}: derived representation already exists: {rep_id}")

    artifacts: list[dict[str, Any]] = []
    mask_sha256 = hashlib.sha256(
        json.dumps(case_mask, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    attention_source_rep_ids = ("q_epvg_query", "q_epvg_key", "q_epvg_score_adjustment")
    query_witness_rep_ids = ("q_epvg_query_update", "steered_attention_scores")
    attention_source_ids = [typed_manifests[name]["manifest_id"] for name in attention_source_rep_ids]
    attention_source_hashes = [typed_manifests[name]["sha256"] for name in attention_source_rep_ids]
    query_witness_ids = [typed_manifests[name]["manifest_id"] for name in query_witness_rep_ids]
    query_witness_hashes = [typed_manifests[name]["sha256"] for name in query_witness_rep_ids]
    attention_comparison = typed_manifests["q_epvg_attention"]
    attention_relpath = f"case_study_tensors/{model_attention_id}.pt"
    attention_target = (selector_dir / attention_relpath).resolve()
    for target in (attention_target, (selector_dir / f"case_study_tensors/{output_manifest_id}.pt").resolve()):
        try:
            target.relative_to(selector_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"case {case.get('case_id')}: reconstructed tensor path escapes selector directory") from exc
        if target.exists():
            raise ValueError(f"refusing to overwrite existing reconstructed tensor: {target}")

    model_attention_manifest, model_attention_raw = _context_manifest_and_bytes(
        model_attention,
        manifest_id=model_attention_id,
        relative_path=attention_relpath,
    )
    model_attention_manifest.update({
        "id": "q_epvg_model_attention",
        "producer_stage": "selection",
        "semantic_role": "model_attention_weights_used_for_context",
        "axis_semantics": ["layers", "heads", "query_tokens", "key_tokens"],
        "derivation": {
            "method": "attention_from_post_intervention_query_and_key",
            "equation": "softmax(masked(query @ key.T / sqrt(head_dim) + score_adjustment))",
            "source_manifest_ids": attention_source_ids,
            "source_sha256": attention_source_hashes,
            "attention_mask_sha256": mask_sha256,
            "query_semantics_witness": {
                "source_manifest_ids": query_witness_ids,
                "source_sha256": query_witness_hashes,
                "equation": (
                    "steered_attention_scores ~= q_epvg_query @ q_epvg_key.T / sqrt(head_dim) + q_epvg_score_adjustment"
                    if non_query_path
                    else "steered_attention_scores ~= (q_epvg_query - q_epvg_query_update) @ q_epvg_key.T / sqrt(head_dim) + q_epvg_score_adjustment"
                ),
                "query_update_policy": (
                    "ignored_non_query_zero_witness_shape_mismatch"
                    if non_query_path
                    else "applied_shape_compatible_query_update"
                ),
                "query_update_shape": list(query_update.shape),
                "query_shape": list(query.shape),
                "query_update_is_zero": query_update_is_zero,
            },
            "attention_trace_comparison": {
                "manifest_id": attention_comparison["manifest_id"],
                "sha256": attention_comparison["sha256"],
                "classification": trace_attention_role,
            },
            "capture_mode": "reconstructed_from_checksum_verified_tensors",
        },
    })
    attention_stage_path = staged_tensor_dir / f"{hashlib.sha256(str(attention_target).encode()).hexdigest()}.pt"
    attention_stage_path.parent.mkdir(parents=True, exist_ok=True)
    attention_stage_path.write_bytes(model_attention_raw)
    artifacts.append({
        "case_id": case.get("case_id"),
        "manifest_id": model_attention_id,
        "relative_path": attention_relpath,
        "target_path": str(attention_target),
        "staged_path": str(attention_stage_path),
        "manifest": model_attention_manifest,
        "projection": _safe_representation_projection(model_attention_manifest),
    })

    output = torch.einsum("lhqk,lhqkd->lhqd", model_attention, routed_values)
    output_relpath = f"case_study_tensors/{output_manifest_id}.pt"
    output_target = (selector_dir / output_relpath).resolve()
    output_manifest, output_raw = _context_manifest_and_bytes(
        output,
        manifest_id=output_manifest_id,
        relative_path=output_relpath,
    )
    output_manifest["derivation"] = {
        "method": "verified_model_attention_weighted_routed_value_contraction",
        "equation": "context[layer,head,query,dim] = sum_key(q_epvg_model_attention * q_epvg_routed_values)",
        "source_manifest_ids": [model_attention_id, typed_manifests["q_epvg_routed_values"]["manifest_id"]],
        "source_sha256": [
            model_attention_manifest["sha256"],
            typed_manifests["q_epvg_routed_values"]["sha256"],
        ],
        "capture_mode": "reconstructed_from_checksum_verified_tensors",
    }
    output_manifest["semantic_role"] = "model_context_reconstructed_from_verified_attention_and_values"
    output_stage_path = staged_tensor_dir / f"{hashlib.sha256(str(output_target).encode()).hexdigest()}.pt"
    output_stage_path.parent.mkdir(parents=True, exist_ok=True)
    output_stage_path.write_bytes(output_raw)
    artifacts.append({
        "case_id": case.get("case_id"),
        "manifest_id": output_manifest_id,
        "relative_path": output_relpath,
        "target_path": str(output_target),
        "staged_path": str(output_stage_path),
        "manifest": output_manifest,
        "projection": _safe_representation_projection(output_manifest),
    })

    representations["q_epvg_model_attention"] = model_attention_manifest
    representations["q_epvg_output"] = output_manifest
    manifests.extend((model_attention_manifest, output_manifest))
    case_payload["manifest_inventory"] = sorted(
        {item["manifest_id"] for item in manifests if isinstance(item, dict) and isinstance(item.get("manifest_id"), str)}
    )
    case_payload["representation_inventory"] = sorted(
        {item["id"] for item in manifests if isinstance(item, dict) and isinstance(item.get("id"), str)}
    )
    semantic_contract = trace_payload.get("semantic_contract")
    if not isinstance(semantic_contract, dict):
        raise ValueError(f"{selector_dir}: sample trace semantic_contract is missing")
    semantic_contract["representation_inventory"] = list(case_payload["representation_inventory"])
    semantic_contract["manifest_inventory"] = list(case_payload["manifest_inventory"])

    trace_stages = trace_sample.get("stages")
    selection_stage = _one_stage(trace_stages, "selection", where=f"sample {trace_sample.get('sample_id')}")
    if selection_stage.get("status") != "observed":
        raise ValueError(f"case {case.get('case_id')}: selection stage is not observed")
    selection_input_refs = selection_stage.get("input_refs")
    if not isinstance(selection_input_refs, list):
        raise ValueError(f"case {case.get('case_id')}: selection input_refs must be a list")
    added_selection_input_refs: list[str] = []
    for rep_id in attention_source_rep_ids:
        source_manifest = typed_manifests[rep_id]
        source_manifest_id = source_manifest["manifest_id"]
        if not any(
            isinstance(item, dict) and item.get("ref") == source_manifest_id
            for item in selection_input_refs
        ):
            selection_input_refs.append({
                "ref": source_manifest_id,
                "kind": "representation",
                "producer_stage": source_manifest["producer_stage"],
            })
            added_selection_input_refs.append(source_manifest_id)
    selection_inputs = selection_stage.get("inputs")
    if not isinstance(selection_inputs, dict):
        selection_inputs = {}
        selection_stage["inputs"] = selection_inputs
    selection_input_representations = selection_inputs.get("representations")
    if not isinstance(selection_input_representations, list):
        selection_input_representations = []
        selection_inputs["representations"] = selection_input_representations
    for rep_id in attention_source_rep_ids:
        source_manifest = typed_manifests[rep_id]
        if not any(
            isinstance(item, dict) and item.get("manifest_id") == source_manifest["manifest_id"]
            for item in selection_input_representations
        ):
            selection_input_representations.append(_safe_representation_projection(source_manifest))
    selection_stage["output_refs"] = [
        item for item in selection_stage.get("output_refs", [])
        if not (isinstance(item, dict) and item.get("ref") == model_attention_id)
    ] + [{"ref": model_attention_id, "kind": "representation", "producer_stage": "selection"}]
    selection_outputs = selection_stage.get("outputs")
    if not isinstance(selection_outputs, dict):
        selection_outputs = {}
        selection_stage["outputs"] = selection_outputs
    selection_representations = selection_outputs.get("representations")
    if not isinstance(selection_representations, list):
        selection_representations = []
        selection_outputs["representations"] = selection_representations
    selection_representations[:] = [
        item for item in selection_representations
        if not (isinstance(item, dict) and item.get("manifest_id") == model_attention_id)
    ] + [_safe_representation_projection(model_attention_manifest)]
    selection_stage["lineage_reconstruction"] = {
        "capture_mode": "mixed_producer_observed_and_reconstructed",
        "added_input_refs": added_selection_input_refs,
        "added_output_refs": [model_attention_id],
        "source_manifest_ids": attention_source_ids,
        "semantic_witness_manifest_ids": query_witness_ids,
        "comparison_manifest_ids": [attention_comparison["manifest_id"]],
        "note": "q_epvg_model_attention is reconstructed from checksum-verified post-intervention query/key and score adjustment; producer-captured selection outputs remain intact.",
    }
    selection_stage["representation_roles"] = {
        **(
            selection_stage.get("representation_roles", {})
            if isinstance(selection_stage.get("representation_roles"), dict)
            else {}
        ),
        "q_epvg_attention": "plugin_trace_for_comparison_only",
        "q_epvg_model_attention": "actual_model_weights_used_for_context",
    }

    intervention_stage = _one_stage(
        trace_stages, "attention_intervention", where=f"sample {trace_sample.get('sample_id')}"
    )
    legacy_attention_ref = typed_manifests["q_epvg_attention"]["manifest_id"]
    removed_intervention_refs = [
        item for item in intervention_stage.get("input_refs", [])
        if isinstance(item, dict) and item.get("ref") == legacy_attention_ref
    ]
    intervention_stage["input_refs"] = [
        item for item in intervention_stage.get("input_refs", [])
        if not (isinstance(item, dict) and item.get("ref") == legacy_attention_ref)
    ]
    intervention_inputs = intervention_stage.get("inputs")
    removed_intervention_representations: list[dict[str, Any]] = []
    if isinstance(intervention_inputs, dict) and isinstance(intervention_inputs.get("representations"), list):
        removed_intervention_representations = [
            item for item in intervention_inputs["representations"]
            if isinstance(item, dict) and item.get("manifest_id") == legacy_attention_ref
        ]
        intervention_inputs["representations"] = [
            item for item in intervention_inputs["representations"]
            if not (
                isinstance(item, dict)
                and item.get("manifest_id") == legacy_attention_ref
            )
        ]
    intervention_stage["lineage_reconciliation"] = {
        "capture_mode": "producer_observed_with_reconciled_inputs",
        "removed_input_refs": [item.get("ref") for item in removed_intervention_refs],
        "removed_input_representations": [item.get("manifest_id") for item in removed_intervention_representations],
        "note": "The plugin attention trace is comparison evidence, not an input to value routing.",
    }

    repaired_context = json.loads(json.dumps(trace_context))
    repaired_context["status"] = "observed"
    repaired_context["capture_mode"] = "reconstructed_from_checksum_verified_tensors"
    repaired_context["reconstruction"] = {
        "method": "verified_model_attention_weighted_routed_value_contraction",
        "equation": "context = sum_key(softmax(masked(query @ key.T / sqrt(head_dim) + score_adjustment)) * routed_values)",
        "query_capture_semantics": (
            "q_epvg_query is the query actually used by the model; this path has no query intervention"
            if non_query_path
            else "q_epvg_query is the query actually used by the model after any query intervention"
        ),
        "query_update_witness": {
            "policy": (
                "ignored_non_query_zero_witness_shape_mismatch"
                if non_query_path
                else "applied_shape_compatible_query_update"
            ),
            "shape": list(query_update.shape),
            "expected_shape": list(query.shape),
            "is_zero": query_update_is_zero,
            "note": (
                "The producer-saved query_update is retained as a legacy trace witness; "
                "it is not used to reconstruct non-query attention."
                if non_query_path
                else "The producer-saved query_update participates in the query-intervention witness."
            ),
        },
        "legacy_attention_trace_comparison": trace_attention_role,
        "source_manifest_ids": [
            model_attention_id,
            typed_manifests["q_epvg_routed_values"]["manifest_id"],
        ],
        "source_sha256": [
            model_attention_manifest["sha256"],
            typed_manifests["q_epvg_routed_values"]["sha256"],
        ],
        "attention_derivation": {
            "source_manifest_ids": attention_source_ids,
            "source_sha256": attention_source_hashes,
            "attention_mask_sha256": mask_sha256,
            "query_semantics_witness_manifest_ids": query_witness_ids,
            "query_semantics_witness_sha256": query_witness_hashes,
            "comparison_manifest_id": attention_comparison["manifest_id"],
            "comparison_sha256": attention_comparison["sha256"],
            "comparison_classification": trace_attention_role,
        },
        "stage_lineage_updates": {
            "selection_added_output_refs": [model_attention_id],
            "attention_intervention_removed_input_refs": [item.get("ref") for item in removed_intervention_refs],
        },
    }
    repaired_context.pop("reason", None)
    repaired_context["input_refs"] = [
        {"ref": model_attention_id, "kind": "representation", "producer_stage": "selection"},
        {
            "ref": typed_manifests["q_epvg_routed_values"]["manifest_id"],
            "kind": "representation",
            "producer_stage": "attention_intervention",
        },
    ]
    repaired_context["inputs"] = {
        "representations": [
            _safe_representation_projection(model_attention_manifest),
            _safe_representation_projection(typed_manifests["q_epvg_routed_values"]),
        ]
    }
    repaired_context["output_refs"] = [
        {"ref": output_manifest_id, "kind": "representation", "producer_stage": "context"}
    ]
    repaired_context["outputs"] = {"representations": [_safe_representation_projection(output_manifest)]}
    trace_stage_index = next(
        index for index, item in enumerate(trace_stages)
        if isinstance(item, dict) and item.get("stage") == "context"
    )
    trace_stages[trace_stage_index] = repaired_context

    return {
        "case_id": case.get("case_id"),
        "attention_trace_role": trace_attention_role,
        "stage_lineage_updates": {
            "selection_added_output_refs": [model_attention_id],
            "attention_intervention_removed_input_refs": [item.get("ref") for item in removed_intervention_refs],
        },
        "model_attention_manifest_id": model_attention_id,
        "manifest_id": output_manifest_id,
        "tensor_artifacts": artifacts,
    }


def _prepare_one(selector_dir: Path, *, staged_tensor_dir: Path) -> dict[str, Any]:
    case_path = selector_dir / "case_study.json"
    trace_path = selector_dir / "sample_trace.json"
    case_before_sha256 = _sha256(case_path)
    trace_before_sha256 = _sha256(trace_path)
    source_case_payload = _load(case_path)
    source_trace_payload = _load(trace_path)
    if not isinstance(source_case_payload, dict):
        raise ValueError(f"{case_path}: expected JSON object")
    if not isinstance(source_trace_payload, dict):
        raise ValueError(f"{trace_path}: expected JSON object")
    if source_case_payload.get("schema_version") != CASE_SCHEMA:
        raise ValueError(f"{case_path}: unsupported Case Study schema")
    if source_trace_payload.get("schema_version") != TRACE_SCHEMA:
        raise ValueError(f"{trace_path}: unsupported sample trace schema")
    case_payload = json.loads(json.dumps(source_case_payload))
    trace_payload = json.loads(json.dumps(source_trace_payload))
    cases = case_payload.get("cases")
    samples = trace_payload.get("samples")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{case_path}: cases must be a non-empty list")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{trace_path}: samples must be a non-empty list")
    sample_by_id: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
            raise ValueError(f"{trace_path}: samples[{index}] lacks sample_id")
        sample_id = sample["sample_id"]
        if sample_id in sample_by_id:
            raise ValueError(f"{trace_path}: duplicate sample_id {sample_id!r}")
        stages = sample.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"{trace_path}: samples[{index}].stages must be a non-empty list")
        sample_by_id[sample_id] = sample
    updated = 0
    replaced = 0
    replaced_case_ids: list[str] = []
    already_complete = 0
    reconstructed_context_cases: list[str] = []
    attention_trace_roles: dict[str, str] = {}
    stage_lineage_updates: list[dict[str, Any]] = []
    tensor_artifacts: list[dict[str, Any]] = []
    token_alignment_repairs: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    repaired_cases = case_payload["cases"]
    for index, case in enumerate(repaired_cases):
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError(f"{case_path}: cases[{index}] lacks case_id")
        case_id = case["case_id"]
        if case_id in case_ids:
            raise ValueError(f"{case_path}: duplicate case_id {case_id!r}")
        case_ids.add(case_id)
        sample = sample_by_id.get(case_id)
        if sample is None:
            raise ValueError(f"{case_path}: no matching sample_trace entry for {case_id!r}")
        token_alignment = _repair_token_alignment(
            case,
            sample,
            where=f"{case_path}: {case_id}",
        )
        if token_alignment is not None:
            token_alignment_repairs.append(token_alignment)
        reconstruction = _reconstruct_context_output(
            selector_dir,
            case_payload,
            trace_payload,
            case,
            sample,
            staged_tensor_dir=staged_tensor_dir,
        )
        if reconstruction is not None:
            reconstructed_context_cases.append(case_id)
            attention_trace_roles[case_id] = reconstruction["attention_trace_role"]
            stage_lineage_updates.append({
                "case_id": case_id,
                **reconstruction["stage_lineage_updates"],
            })
            tensor_artifacts.extend(reconstruction["tensor_artifacts"])
        trace_stages = sample["stages"]
        stage_error = _stage_completeness_error(
            trace_stages,
            where=f"{trace_path}: sample {case_id}",
        )
        if stage_error is not None:
            raise ValueError(f"{trace_path}: sample {case_id}: {stage_error}")
        existing = case.get("stages")
        if existing is None:
            case["stages"] = json.loads(json.dumps(trace_stages))
            updated += 1
        else:
            existing_error = _stage_completeness_error(
                existing,
                where=f"{case_path}: {case_id}",
            )
            if existing_error is None:
                if existing != trace_stages:
                    raise ValueError(
                        f"{case_path}: {case_id}.stages is semantically complete "
                        "but disagrees with sample_trace.json"
                    )
                already_complete += 1
            else:
                case["stages"] = json.loads(json.dumps(trace_stages))
                replaced += 1
                replaced_case_ids.append(case_id)

    if len(cases) != len(samples):
        raise ValueError(f"{selector_dir}: case/sample count mismatch ({len(cases)} != {len(samples)})")

    if case_ids != set(sample_by_id):
        missing = sorted(set(sample_by_id) - case_ids)
        extra = sorted(case_ids - set(sample_by_id))
        raise ValueError(
            f"{selector_dir}: case/sample IDs differ; missing_cases={missing!r}, "
            f"extra_cases={extra!r}"
        )

    if reconstructed_context_cases:
        coverage = trace_payload.get("coverage")
        if not isinstance(coverage, dict) or "context" not in coverage:
            raise ValueError(f"{trace_path}: coverage.context is missing or invalid")
        context_statuses: set[str] = set()
        for sample in samples:
            for stage in sample["stages"]:
                if isinstance(stage, dict) and stage.get("stage") == "context":
                    context_statuses.add(str(stage.get("status")))
        coverage["context"] = (
            "failed" if "failed" in context_statuses
            else "observed" if "observed" in context_statuses
            else "not_applicable"
        )

    case_changed = case_payload != source_case_payload
    trace_changed = trace_payload != source_trace_payload
    if _sha256(case_path) != case_before_sha256 or _sha256(trace_path) != trace_before_sha256:
        raise ValueError(f"{selector_dir}: Case Study or source trace changed while the repair plan was being prepared")

    return {
        "selector_dir": str(selector_dir),
        "selector": str(case_payload.get("selector", selector_dir.name)),
        "case_path": str(case_path),
        "trace_path": str(trace_path),
        "case_count": len(cases),
        "updated_cases": updated + replaced,
        "added_cases": updated,
        "replaced_cases": replaced,
        "replaced_case_ids": replaced_case_ids,
        "already_complete_cases": already_complete,
        "reconstructed_context_cases": reconstructed_context_cases,
        "attention_trace_roles": attention_trace_roles,
        "stage_lineage_updates": stage_lineage_updates,
        "tensor_artifacts": tensor_artifacts,
        "token_alignment_repairs": token_alignment_repairs,
        "case_changed": case_changed,
        "trace_changed": trace_changed,
        "before_sha256": case_before_sha256,
        "trace_before_sha256": trace_before_sha256,
        "after_payload": case_payload,
        "after_trace_payload": trace_payload,
    }


def repair_group(group_dir: Path, *, apply: bool, root: Path) -> dict[str, Any]:
    group_dir = group_dir.resolve()
    if not group_dir.is_dir():
        raise ValueError(f"group directory does not exist: {group_dir}")
    state_path = group_dir / "multi_seed_run_summary.json"
    if not state_path.is_file():
        raise ValueError(f"missing completed scheduler state: {state_path}")
    state = _load(state_path)
    if not isinstance(state, dict) or state.get("success") is not True:
        raise ValueError("refusing repair: multi_seed_run_summary.json does not declare success")
    selector_dirs = _selector_dirs(group_dir)
    if not selector_dirs:
        raise ValueError(f"no selector Case Study pairs found under {group_dir}")

    with tempfile.TemporaryDirectory(prefix="q-epvg-context-repair-") as temporary_dir:
        staged_root = Path(temporary_dir)
        plans = [
            _prepare_one(path, staged_tensor_dir=staged_root / str(index))
            for index, path in enumerate(selector_dirs)
        ]
        changes = [
            plan for plan in plans
            if plan["case_changed"] or plan["trace_changed"] or plan["tensor_artifacts"]
        ]
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "group_dir": str(group_dir),
            "source": "checksum-verified q_epvg_query/key/score_adjustment + attention_mask -> q_epvg_model_attention; q_epvg_model_attention × q_epvg_routed_values -> q_epvg_output; sample_trace -> case_study stages",
            "git_revision": _git_revision(root),
            "apply": bool(apply),
            "selector_count": len(plans),
            "case_count": sum(int(item["case_count"]) for item in plans),
            "updated_case_count": sum(int(item["updated_cases"]) for item in plans),
            "replaced_case_count": sum(int(item["replaced_cases"]) for item in plans),
            "already_complete_case_count": sum(int(item["already_complete_cases"]) for item in plans),
            "reconstructed_context_count": sum(len(item["reconstructed_context_cases"]) for item in plans),
            "token_alignment_repair_count": sum(len(item["token_alignment_repairs"]) for item in plans),
            "reconstructed_context_cases": [
                case_id
                for plan in plans
                for case_id in plan["reconstructed_context_cases"]
            ],
            "selectors": [],
        }

        def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in plan.items()
                if key not in {"after_payload", "after_trace_payload", "tensor_artifacts"}
            } | {
                "tensor_artifacts": [
                    {
                        key: value
                        for key, value in artifact.items()
                        if key not in {"staged_path", "manifest", "projection"}
                    } | {
                        "sha256": artifact["manifest"]["sha256"],
                        "byte_count": artifact["manifest"]["byte_count"],
                    }
                    for artifact in plan["tensor_artifacts"]
                ]
            }

        manifest["selectors"] = [public_plan(plan) for plan in plans]
        manifest["no_op"] = False
        manifest_path = group_dir / "case_study_stage_repair_manifest.json"
        if not apply:
            return manifest
        if not changes:
            if manifest_path.is_file():
                existing_manifest = _load(manifest_path)
                if not isinstance(existing_manifest, dict):
                    raise ValueError(f"{manifest_path}: existing repair manifest is not a JSON object")
                return {**existing_manifest, "no_op": True, "existing_manifest_preserved": True}
            return {**manifest, "no_op": True}

        backup_pairs: list[tuple[Path, Path, str]] = []
        for plan in changes:
            for path_key, digest_key in (
                ("case_path", "before_sha256"),
                ("trace_path", "trace_before_sha256"),
            ):
                source = Path(plan[path_key])
                if _sha256(source) != plan[digest_key]:
                    raise ValueError(f"source changed after repair preflight; refusing to apply: {source}")
            for path_key, changed_key in (("case_path", "case_changed"), ("trace_path", "trace_changed")):
                if not plan[changed_key]:
                    continue
                source = Path(plan[path_key])
                backup = source.with_name(source.name + ".pre_stage_repair")
                if backup.exists():
                    raise ValueError(f"refusing overwrite because backup already exists: {backup}")
                digest_key = "before_sha256" if path_key == "case_path" else "trace_before_sha256"
                backup_pairs.append((source, backup, plan[digest_key]))
            for artifact in plan["tensor_artifacts"]:
                target = Path(artifact["target_path"])
                if target.exists():
                    raise ValueError(f"refusing overwrite because reconstructed tensor already exists: {target}")

        old_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
        if old_manifest is not None:
            manifest_backup = manifest_path.with_name(manifest_path.name + ".pre_stage_repair")
            if manifest_backup.exists():
                raise ValueError(f"refusing overwrite because backup already exists: {manifest_backup}")
            backup_pairs.append((manifest_path, manifest_backup, hashlib.sha256(old_manifest).hexdigest()))
        created_backups: list[tuple[Path, Path, str]] = []
        written_sources: set[Path] = set()
        created_tensors: list[Path] = []
        try:
            for source, backup, expected_digest in backup_pairs:
                shutil.copy2(source, backup)
                if backup.stat().st_size != source.stat().st_size or _sha256(backup) != expected_digest:
                    raise OSError(f"backup verification failed: {backup}")
                created_backups.append((source, backup, expected_digest))
            for source, backup, expected_digest in created_backups:
                if _sha256(source) != expected_digest or _sha256(backup) != expected_digest:
                    raise ValueError(f"source changed while its repair backup was being prepared: {source}")
            for plan in changes:
                for artifact in plan["tensor_artifacts"]:
                    staged = Path(artifact["staged_path"])
                    raw = staged.read_bytes()
                    if (
                        len(raw) != artifact["manifest"]["byte_count"]
                        or hashlib.sha256(raw).hexdigest() != artifact["manifest"]["sha256"]
                    ):
                        raise ValueError(f"staged tensor checksum mismatch: {staged}")
                    target = Path(artifact["target_path"])
                    created_tensors.append(target)
                    _write_bytes_atomic(target, raw)
            for plan in changes:
                if plan["case_changed"]:
                    source = Path(plan["case_path"])
                    if _sha256(source) != plan["before_sha256"]:
                        raise ValueError(f"source changed after repair backup; refusing to write: {source}")
                    written_sources.add(source)
                    _write_json_atomic(source, plan["after_payload"])
                    plan["after_sha256"] = _sha256(source)
                if plan["trace_changed"]:
                    source = Path(plan["trace_path"])
                    if _sha256(source) != plan["trace_before_sha256"]:
                        raise ValueError(f"source changed after repair backup; refusing to write: {source}")
                    written_sources.add(source)
                    _write_json_atomic(source, plan["after_trace_payload"])
                    plan["trace_after_sha256"] = _sha256(source)
            manifest["selectors"] = [public_plan(plan) for plan in plans]
            written_sources.add(manifest_path)
            _write_json_atomic(manifest_path, manifest)
        except BaseException as exc:
            rollback_errors: list[str] = []
            completed_pairs = set(created_backups)
            for source, backup, expected_digest in reversed(backup_pairs):
                if (source, backup, expected_digest) not in completed_pairs:
                    if backup.exists():
                        try:
                            backup.unlink()
                        except OSError as rollback_exc:
                            rollback_errors.append(f"partial backup {backup}: {rollback_exc}")
                    continue
                if source not in written_sources:
                    continue
                try:
                    restored_bytes = backup.read_bytes()
                    if hashlib.sha256(restored_bytes).hexdigest() != expected_digest:
                        rollback_errors.append(f"{source}: backup checksum mismatch; source was not restored")
                        continue
                    _write_bytes_atomic(source, restored_bytes)
                    if _sha256(source) != expected_digest:
                        rollback_errors.append(f"{source}: restored source checksum mismatch")
                except OSError as rollback_exc:
                    rollback_errors.append(f"{source}: {rollback_exc}")
            for target in reversed(created_tensors):
                try:
                    target.unlink(missing_ok=True)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{target}: {rollback_exc}")
            try:
                if old_manifest is None:
                    manifest_path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{manifest_path}: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError(
                    f"repair failed ({exc}); rollback incomplete, preserve backups: "
                    + "; ".join(rollback_errors)
                ) from exc
            for _source, backup, _expected_digest in created_backups:
                backup.unlink(missing_ok=True)
            raise
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="write backups and repaired Case Study files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        manifest = repair_group(args.group_dir, apply=args.apply, root=root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print("status=" + ("APPLIED" if args.apply else "DRY_RUN"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
