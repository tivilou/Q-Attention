from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _scheduler_module():
    path = Path(__file__).parents[1] / "scripts/run_retacred_dual_qres_multi_seed.py"
    spec = importlib.util.spec_from_file_location("multi_gpu_scheduler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_integer_list_preserves_order_and_removes_duplicates() -> None:
    scheduler = _scheduler_module()

    assert scheduler.parse_integer_list("7,11,7,13", label="seeds") == [7, 11, 13]


def test_parse_integer_list_rejects_invalid_values() -> None:
    scheduler = _scheduler_module()

    with pytest.raises(ValueError, match="comma-separated"):
        scheduler.parse_integer_list("7,bad", label="seeds")


def test_build_run_command_pins_each_seed_to_one_gpu(tmp_path: Path) -> None:
    scheduler = _scheduler_module()
    args = argparse.Namespace(
        progress_format="both",
        log_every_batches=25,
        stale_timeout_minutes=45,
        stage_timeout_hours=36,
        skip_canary=False,
        canary_only=False,
        dry_run=False,
    )

    command = scheduler.build_run_command(
        args,
        seed=17,
        gpu_id=1,
        run_dir=tmp_path / "seed_17",
    )

    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--gpus") + 1] == "1"
    assert command[command.index("--parallel-mode") + 1] == "serial"
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "seed_17")
    assert "--skip-preflight" in command
    assert "--no-latest-pointer" in command


def test_resolve_group_dir_rejects_paths_outside_runs(tmp_path: Path) -> None:
    scheduler = _scheduler_module()

    with pytest.raises(ValueError, match="inside the repository runs"):
        scheduler.resolve_group_dir(str(tmp_path / "outside"), stamp="20260807_120000")
