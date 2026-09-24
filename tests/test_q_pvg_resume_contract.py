from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_selector_resume_compatibility_wrapper_delegates_to_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiments_dir = str(Path(__file__).resolve().parents[1] / "experiments")
    monkeypatch.syspath_prepend(experiments_dir)

    scheduler = importlib.import_module("run_q_pvg_scheduler_base")
    original_compatibility = scheduler.selector_resume_contract_compatible
    runner = importlib.import_module("run_q_pvg_formal_single_seed")
    previous = {"source": {"git_revision": "old", "files": {}}}
    current = {"source": {"git_revision": "new", "files": {}}}

    try:
        assert runner._base_selector_resume_contract_compatible is original_compatibility
        assert (
            scheduler.selector_resume_contract_compatible
            is runner.selector_resume_contract_compatible
        )
        assert runner.selector_resume_contract_compatible(previous, current) is True
    finally:
        scheduler.selector_resume_contract_compatible = original_compatibility


def test_elastic_resume_accepts_multigpu_to_single_gpu_but_keeps_code_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiments_dir = str(Path(__file__).resolve().parents[1] / "experiments")
    monkeypatch.syspath_prepend(experiments_dir)
    scheduler = importlib.import_module("run_q_pvg_scheduler_base")
    persisted = {
        "training_semantics": {
            "parallel_mode": "selector_or_serial",
            "model_parallel_gpu_ids": [],
            "selector_gpu_ids": [0, 1, 2],
            "seed": 13,
        },
        "source": {
            "git_revision": "old",
            "files": {"runner": {"sha256": "old"}, "q_pvg": {"sha256": "same"}},
        },
    }
    current = {
        "training_semantics": {
            "parallel_mode": "selector_or_serial",
            "model_parallel_gpu_ids": [],
            "selector_gpu_ids": [0],
            "seed": 13,
        },
        "source": {
            "git_revision": "new",
            "files": {"runner": {"sha256": "new"}, "q_pvg": {"sha256": "same"}},
        },
    }

    assert not scheduler._elastic_run_contract_compatible(persisted, current)
    assert scheduler._elastic_run_contract_compatible(
        persisted,
        {
            **current,
            "source": {
                "git_revision": "old",
                "files": {
                    "runner": {"sha256": "old"},
                    "q_pvg": {"sha256": "same"},
                },
            },
        },
    )
    assert scheduler._combined_code_and_topology_contract_compatible(
        persisted, current
    )


def test_elastic_resume_keeps_scientific_contract_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiments_dir = str(Path(__file__).resolve().parents[1] / "experiments")
    monkeypatch.syspath_prepend(experiments_dir)
    scheduler = importlib.import_module("run_q_pvg_scheduler_base")
    persisted = {
        "training_semantics": {
            "parallel_mode": "selector_or_serial",
            "model_parallel_gpu_ids": [],
            "selector_gpu_ids": [0, 1],
            "seed": 13,
            "batch_size": 256,
        },
        "source": {"git_revision": "same", "files": {"q_pvg": {"sha256": "same"}}},
    }
    topology_only = {
        **persisted,
        "training_semantics": {**persisted["training_semantics"], "selector_gpu_ids": [0]},
    }
    changed_science = {
        **topology_only,
        "training_semantics": {**topology_only["training_semantics"], "batch_size": 128},
    }

    assert scheduler._elastic_run_contract_compatible(persisted, topology_only)
    assert not scheduler._elastic_run_contract_compatible(persisted, changed_science)


def test_manifest_resume_requires_both_flags_for_combined_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments_dir = str(Path(__file__).resolve().parents[1] / "experiments")
    monkeypatch.syspath_prepend(experiments_dir)
    scheduler = importlib.import_module("run_q_pvg_scheduler_base")
    previous = {
        "training_semantics": {
            "parallel_mode": "selector_or_serial",
            "model_parallel_gpu_ids": [],
            "selector_gpu_ids": [0, 1],
            "seed": 13,
        },
        "source": {
            "git_revision": "old",
            "files": {"runner": {"sha256": "old"}, "q_pvg": {"sha256": "same"}},
        },
    }
    current = {
        "training_semantics": {
            "parallel_mode": "selector_or_serial",
            "model_parallel_gpu_ids": [],
            "selector_gpu_ids": [0],
            "seed": 13,
        },
        "source": {
            "git_revision": "new",
            "files": {"runner": {"sha256": "new"}, "q_pvg": {"sha256": "same"}},
        },
    }
    run_dir = tmp_path / "resume"
    run_dir.mkdir()
    manifest = {
        "schema_version": scheduler.RUN_MANIFEST_SCHEMA,
        "contract_fingerprint": scheduler.fingerprint(previous),
        "contract": previous,
        "started_at_utc": "20260924T003315Z",
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(scheduler.ResumeCompatibilityError):
        scheduler._validate_or_create_run_manifest(
            run_dir,
            current,
            resume=True,
            started_at_utc="20260924T003315Z",
            allow_gpu_topology_change=True,
        )
    with pytest.raises(scheduler.ResumeCompatibilityError):
        scheduler._validate_or_create_run_manifest(
            run_dir,
            current,
            resume=True,
            started_at_utc="20260924T003315Z",
            allow_code_update=True,
        )

    migrated = scheduler._validate_or_create_run_manifest(
        run_dir,
        current,
        resume=True,
        started_at_utc="20260924T003315Z",
        allow_gpu_topology_change=True,
        allow_code_update=True,
    )

    assert migrated["contract"] == current
    assert migrated["resume_migrations"][-1]["event"] == "code_and_gpu_topology_resume"
