from __future__ import annotations

import importlib.util
from pathlib import Path


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
