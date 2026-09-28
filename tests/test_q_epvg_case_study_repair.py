from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_repairer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "repair_q_epvg_case_study_stages.py"
    spec = importlib.util.spec_from_file_location("q_epvg_case_study_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "attention_baseline", "scoring", "selection", "context", "classifier",
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
