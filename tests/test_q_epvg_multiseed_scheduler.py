from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_scheduler():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_retacred_q_epvg_formal_multi_seed.py"
    spec = importlib.util.spec_from_file_location("q_epvg_multiseed_scheduler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_queue_is_seed_round_robin() -> None:
    scheduler = load_scheduler()
    scheduler.SELECTOR_TASKS = ("selector_a", "selector_b")
    tasks = scheduler.make_tasks([13, 29, 53])

    order = scheduler.selector_queue_order(tasks, [13, 29, 53])

    assert order == [
        "selector:13:selector_a",
        "selector:29:selector_a",
        "selector:53:selector_a",
        "selector:13:selector_b",
        "selector:29:selector_b",
        "selector:53:selector_b",
    ]


def test_runtime_wrapper_exports_shared_hardware_contract() -> None:
    scheduler = load_scheduler()
    runtime = scheduler._runtime_module()

    assert callable(runtime.choose_hardware_profile)
    assert callable(runtime._adaptive_profile_at)

    profile = runtime.choose_hardware_profile("adaptive", {}, [0], [])
    assert profile["name"] == "adaptive"
    assert profile["adaptive"] is True
    assert runtime._adaptive_profile_at(profile, 0)["name"] == "adaptive_full_batch"


def test_selector_queue_skips_completed_tasks_without_reordering_remaining_seeds() -> None:
    scheduler = load_scheduler()
    scheduler.SELECTOR_TASKS = ("selector_a", "selector_b")
    tasks = scheduler.make_tasks([13, 29, 53])
    tasks["selector:29:selector_a"]["status"] = "complete"
    tasks["selector:13:selector_b"]["status"] = "complete"

    assert scheduler.selector_queue_order(tasks, [13, 29, 53]) == [
        "selector:13:selector_a",
        "selector:53:selector_a",
        "selector:29:selector_b",
        "selector:53:selector_b",
    ]


def test_seed13_import_promotes_embedded_provenance_to_standard_file(tmp_path: Path) -> None:
    scheduler = load_scheduler()
    scheduler.SELECTOR_TASKS = ()
    source = tmp_path / "seed13-report"
    source.mkdir()
    provenance = {"git_dirty": False, "git_revision": "source-revision"}
    files = {
        "RUN_COMPLETE": "done\n",
        "run_config.json": "{}\n",
        "run_summary.json": json.dumps({"provenance": provenance}),
        "run_summary.data": "summary\n",
        "run_summary.md": "summary\n",
        "data.sha256": "a" * 64 + "  train.jsonl\n",
        "data_counts.txt": "1 train\n",
        "export_manifest.json": "{}\n",
        "metrics/baseline.json": "{}\n",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    group = tmp_path / "multi-seed"
    metadata = {
        "seed": 13,
        "provenance_source": "run_summary.json:provenance",
    }
    scheduler.import_seed13_report(
        source,
        group,
        config_path=tmp_path / "config.json",
        config={},
        current_commit="current-revision",
        metadata=metadata,
    )

    imported_provenance = json.loads(
        (group / "seed_13/provenance.json").read_text(encoding="utf-8")
    )
    imported_metadata = json.loads(
        (group / "seed_13/imported_report.json").read_text(encoding="utf-8")
    )
    assert imported_provenance == provenance
    assert imported_metadata["provenance_source"] == "run_summary.json:provenance"
    assert imported_metadata["provenance_sha256"] == scheduler._provenance_sha256(provenance)
    assert imported_metadata["provenance_projection_schema"] == "q-attention.provenance-projection.v1"


def test_seed13_provenance_rejects_conflicting_projection(tmp_path: Path) -> None:
    scheduler = load_scheduler()
    source = tmp_path / "seed13-report"
    source.mkdir()
    (source / "provenance.json").write_text(
        json.dumps({"git_dirty": False, "git_revision": "other"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="differs from run_summary.json"):
        scheduler._seed13_provenance(
            source,
            {"provenance": {"git_dirty": False, "git_revision": "canonical"}},
        )


def test_seed13_provenance_rejects_missing_sources(tmp_path: Path) -> None:
    scheduler = load_scheduler()

    with pytest.raises(ValueError, match="lacks provenance.json"):
        scheduler._seed13_provenance(tmp_path, {})


def test_seed13_provenance_accepts_matching_projection(tmp_path: Path) -> None:
    scheduler = load_scheduler()
    provenance = {"git_dirty": False, "git_revision": "canonical"}
    (tmp_path / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )

    loaded, source = scheduler._seed13_provenance(
        tmp_path, {"provenance": {"git_revision": "canonical", "git_dirty": False}}
    )

    assert loaded == provenance
    assert source == "provenance.json"
