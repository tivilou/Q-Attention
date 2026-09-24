from __future__ import annotations

import importlib
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
