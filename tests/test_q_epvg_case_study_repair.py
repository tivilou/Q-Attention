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


def _make_group(tmp_path: Path, *, mismatch: bool = False) -> Path:
    group = tmp_path / "group"
    selector = group / "seed_13" / "selectors" / "selector_a"
    selector.mkdir(parents=True)
    (group / "multi_seed_run_summary.json").write_text(
        json.dumps({"success": True, "tasks": []}), encoding="utf-8"
    )
    stages = [{"stage": "data", "status": "observed", "input_refs": [], "output_refs": []}]
    case_id = "selector_a:test:0:final"
    case = {
        "schema_version": "q-attention.Q-EPVG-case-study.v2",
        "selector": "selector_a",
        "cases": [{"case_id": case_id, "representations": {}}],
    }
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
