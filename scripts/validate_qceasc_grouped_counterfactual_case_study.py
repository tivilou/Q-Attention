#!/usr/bin/env python3
"""Validate the semantic Q-CEASC grouped counterfactual Case Study contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REQUIRED_SPLITS = {"train", "valid", "test"}
REQUIRED_CHECKPOINTS = {
    "initial_or_pre_training",
    "best_valid_or_declared_selection_checkpoint",
    "final",
}
REQUIRED_REPRESENTATIONS = {
    "token_embeddings",
    "encoder_hidden_states",
    "subject_object_pooled_states",
    "attention_qkv",
    "baseline_attention_scores",
    "q_ceasc_auxiliary_state",
    "q_ceasc_observable_coefficients",
    "q_ceasc_projected_residual",
    "q_ceasc_group_manifest",
    "q_ceasc_group_masked_supports",
    "q_ceasc_group_influence_vectors",
    "q_ceasc_group_scores",
    "q_ceasc_member_scores",
    "steered_attention_scores",
    "classifier_logits_probabilities",
}


def _read(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid: {exc}"
    return value if isinstance(value, dict) else None, "expected an object"


def validate(selector_dir: Path, *, config_sha256: str | None = None) -> list[str]:
    errors: list[str] = []
    case, case_error = _read(selector_dir / "case_study.json")
    trace, trace_error = _read(selector_dir / "sample_trace.json")
    if case is None:
        return [f"case_study.json: {case_error}"]
    if trace is None:
        return [f"sample_trace.json: {trace_error}"]
    if case.get("schema_version") != "q-attention.q-ceasc-case-study.v2":
        errors.append("case_study schema must be q-attention.q-ceasc-case-study.v2")
    if set(case.get("required_splits", [])) != REQUIRED_SPLITS:
        errors.append("Case Study required_splits must be train/valid/test")
    if set(case.get("checkpoint_policy", [])) != REQUIRED_CHECKPOINTS:
        errors.append("Case Study checkpoint_policy is incomplete")
    inventory = set(case.get("representation_inventory", []))
    if not REQUIRED_REPRESENTATIONS.issubset(inventory):
        errors.append("Case Study representation_inventory is incomplete")
    cases = case.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("Case Study cases must be a non-empty list")
        cases = []
    else:
        if {item.get("split") for item in cases if isinstance(item, dict)} != REQUIRED_SPLITS:
            errors.append("Case Study cases do not cover train/valid/test")
        if {item.get("checkpoint") for item in cases if isinstance(item, dict)} != REQUIRED_CHECKPOINTS:
            errors.append("Case Study cases do not cover all checkpoints")
        for index, item in enumerate(cases):
            if not isinstance(item, dict):
                errors.append(f"cases[{index}] is not an object")
                continue
            required_fields = (
                "sentence", "tokens", "token_ids", "attention_mask", "subject",
                "object", "gold_relation", "baseline_prediction", "selector_prediction",
                "baseline_logits", "selector_logits", "baseline_probabilities",
                "selector_probabilities", "representations",
            )
            for field in required_fields:
                if field not in item:
                    errors.append(f"cases[{index}] is missing {field}")
            for entity_name in ("subject", "object"):
                entity = item.get(entity_name)
                if not isinstance(entity, dict) or not {"text", "span", "token_positions", "entity_type"}.issubset(entity):
                    errors.append(f"cases[{index}].{entity_name} is missing text/span/token_positions/entity_type")
    if trace.get("schema_version") != "sample-trace.v1":
        errors.append("sample trace schema must be sample-trace.v1")
    semantic = trace.get("semantic_contract")
    if not isinstance(semantic, dict):
        errors.append("sample trace is missing semantic_contract")
    else:
        if set(semantic.get("required_splits", [])) != REQUIRED_SPLITS:
            errors.append("sample trace semantic split contract is incomplete")
        if set(semantic.get("required_checkpoints", [])) != REQUIRED_CHECKPOINTS:
            errors.append("sample trace semantic checkpoint contract is incomplete")
        if not REQUIRED_REPRESENTATIONS.issubset(set(semantic.get("representation_inventory", []))):
            errors.append("sample trace semantic representation inventory is incomplete")
    selection = trace.get("sample_selection")
    if not isinstance(selection, dict) or selection.get("selected_count") != len(cases):
        errors.append("sample trace selected_count does not match Case Study cases")
    experiment = trace.get("experiment")
    if config_sha256 is not None and (
        not isinstance(experiment, dict) or experiment.get("config_sha256") != config_sha256
    ):
        errors.append("sample trace config hash does not match the frozen config")
    manifests = case.get("tensor_manifest")
    if not isinstance(manifests, list) or not manifests:
        errors.append("tensor_manifest must be a non-empty list")
        manifests = []
    for index, entry in enumerate(manifests):
        if not isinstance(entry, dict):
            errors.append(f"tensor_manifest[{index}] is not an object")
            continue
        for field in ("id", "path", "shape", "dtype", "axis_semantics", "sha256", "byte_count"):
            if field not in entry:
                errors.append(f"tensor_manifest[{index}] is missing {field}")
        target = selector_dir / str(entry.get("path", ""))
        if not target.is_file():
            errors.append(f"missing tensor capture: {target}")
        else:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != entry.get("sha256"):
                errors.append(f"tensor hash mismatch: {target.name}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_dir", type=Path)
    parser.add_argument("--config-sha256")
    args = parser.parse_args(argv)
    errors = validate(args.selector_dir, config_sha256=args.config_sha256)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {args.selector_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
