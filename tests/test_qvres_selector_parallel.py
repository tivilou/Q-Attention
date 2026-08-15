from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_qvres_relation_transfer_selector_parallel import (  # noqa: E402
    TRAINABLE_SELECTORS,
    aggregate_selector_parallel_run,
)
from run_qvres_relation_transfer_selector_parallel import (  # noqa: E402
    render_selector_dashboard,
    selector_dashboard_snapshot,
)
import run_qvres_relation_transfer_multi_seed as multi_seed_runner  # noqa: E402


def metrics(macro_f1: float) -> dict[str, float]:
    return {
        "accuracy": macro_f1 + 0.1,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "loss": 1.0 - macro_f1,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_stage(
    stage_dir: Path,
    *,
    selector: str,
    baseline_model_dir: Path,
    selector_valid: float,
    baseline_valid: float = 0.2,
) -> None:
    baseline_test = metrics(0.19)
    selector_test = metrics(selector_valid - 0.01)
    row = {
        "selector": selector,
        "seed": 13,
        "valid": {"metrics": metrics(selector_valid)},
        "test": {
            "metrics": selector_test,
            "delta_vs_baseline": {
                "delta_macro_f1": selector_test["macro_f1"] - baseline_test["macro_f1"]
            },
        },
        "trainable_parameters": 0 if selector == "disabled" else 72,
        "finite": True,
    }
    provenance = {
        "git_commit": "abc123",
        "git_branch": "main",
        "git_dirty": False,
        "config_sha256": "config-sha",
        "cuda_device": "test-gpu",
        "train": {"source_sha256": "train", "records": 10},
        "valid": {"source_sha256": "valid", "records": 4},
        "test": {"source_sha256": "test", "records": 4},
    }
    summary = {
        "status": "pass",
        "run_type": "formal_full_relation_transfer",
        "formal_experiment": True,
        "partial_selector_run": True,
        "baseline": {
            "model_dir": str(baseline_model_dir),
            "valid": {"metrics": metrics(baseline_valid)},
            "test": {"metrics": baseline_test},
        },
        "results": [row],
        "provenance": provenance,
    }
    write_json(stage_dir / "run_summary.json", summary)
    write_json(stage_dir / "run_config.json", {"seed": 13, "selector": selector})
    write_json(stage_dir / "selectors" / selector / "metrics.json", row)


def build_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "selector_parallel_manifest.json",
        {
            "seed": 13,
            "git_commit": "abc123",
            "config_sha256": "config-sha",
            "gpus": [{"id": gpu_id, "name": "test-gpu"} for gpu_id in (0, 1, 2)],
        },
    )
    baseline_stage = run_dir / "stages" / "baseline"
    baseline_model_dir = baseline_stage / "baseline"
    write_json(baseline_model_dir / "metrics.json", {"macro_f1": 0.2})
    (baseline_stage / "baseline_train.log").write_text("baseline\n", encoding="utf-8")
    make_stage(
        baseline_stage,
        selector="disabled",
        baseline_model_dir=baseline_model_dir,
        selector_valid=0.2,
    )
    selector_valid = {
        "q_causal_transport": 0.3,
        "classical_causal_transport": 0.25,
        "q_causal_key_only": 0.22,
    }
    selector_dirs: dict[str, Path] = {}
    for selector in TRAINABLE_SELECTORS:
        selector_dir = run_dir / "stages" / selector
        selector_dirs[selector] = selector_dir
        make_stage(
            selector_dir,
            selector=selector,
            baseline_model_dir=baseline_model_dir,
            selector_valid=selector_valid[selector],
        )
    return run_dir, baseline_stage, selector_dirs


def test_aggregate_selector_parallel_run(tmp_path: Path) -> None:
    run_dir, baseline_stage, selector_dirs = build_fixture(tmp_path)
    assignments = {
        "baseline": 0,
        "q_causal_transport": 0,
        "classical_causal_transport": 1,
        "q_causal_key_only": 2,
    }
    summary = aggregate_selector_parallel_run(
        run_dir,
        baseline_stage,
        selector_dirs,
        assignments,
    )
    assert summary["status"] == "pass"
    assert summary["pilot_validation_gate"]["status"] == "pass"
    assert [row["selector"] for row in summary["results"]] == [
        "disabled",
        *TRAINABLE_SELECTORS,
    ]
    assert (run_dir / "run_summary.json").is_file()
    assert (run_dir / "run_summary.md").is_file()
    assert (run_dir / "baseline" / "metrics.json").is_file()
    for selector in ("disabled", *TRAINABLE_SELECTORS):
        assert (run_dir / "selectors" / selector / "metrics.json").is_file()


def test_aggregate_rejects_different_baseline_metrics(tmp_path: Path) -> None:
    run_dir, baseline_stage, selector_dirs = build_fixture(tmp_path)
    worker_summary_path = selector_dirs["q_causal_transport"] / "run_summary.json"
    worker_summary = json.loads(worker_summary_path.read_text(encoding="utf-8"))
    worker_summary["baseline"]["valid"]["metrics"]["macro_f1"] = 0.9
    write_json(worker_summary_path, worker_summary)
    with pytest.raises(ValueError, match="same frozen baseline"):
        aggregate_selector_parallel_run(
            run_dir,
            baseline_stage,
            selector_dirs,
            {
                "baseline": 0,
                "q_causal_transport": 0,
                "classical_causal_transport": 1,
                "q_causal_key_only": 2,
            },
        )


def test_aggregate_marks_validation_ties_for_review(tmp_path: Path) -> None:
    run_dir, baseline_stage, selector_dirs = build_fixture(tmp_path)
    for selector in TRAINABLE_SELECTORS:
        summary_path = selector_dirs[selector] / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["results"][0]["valid"]["metrics"] = metrics(0.2)
        write_json(summary_path, summary)
    aggregated = aggregate_selector_parallel_run(
        run_dir,
        baseline_stage,
        selector_dirs,
        {
            "baseline": 0,
            "q_causal_transport": 0,
            "classical_causal_transport": 1,
            "q_causal_key_only": 2,
        },
    )
    assert aggregated["pilot_validation_gate"]["status"] == "review"


def test_aggregate_rejects_different_worker_config(tmp_path: Path) -> None:
    run_dir, baseline_stage, selector_dirs = build_fixture(tmp_path)
    worker_summary_path = selector_dirs["q_causal_transport"] / "run_summary.json"
    worker_summary = json.loads(worker_summary_path.read_text(encoding="utf-8"))
    worker_summary["provenance"]["config_sha256"] = "different"
    write_json(worker_summary_path, worker_summary)
    with pytest.raises(ValueError, match="different code, data, or formal settings"):
        aggregate_selector_parallel_run(
            run_dir,
            baseline_stage,
            selector_dirs,
            {
                "baseline": 0,
                "q_causal_transport": 0,
                "classical_causal_transport": 1,
                "q_causal_key_only": 2,
            },
        )


def test_selector_dashboard_reports_all_stage_states(tmp_path: Path) -> None:
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    (status_dir / "q_causal_transport.env").write_text(
        "STATUS=running\nGPU_ID=1\n", encoding="utf-8"
    )
    (status_dir / "q_causal_transport.heartbeat").write_text(
        json.dumps(
            {
                "phase": "train",
                "epoch": 2,
                "epochs": 5,
                "batch": 64,
                "batches": 128,
                "eta_seconds": 2120,
                "batches_per_second": 0.5,
            }
        ),
        encoding="utf-8",
    )
    stage_state = {
        "baseline": {"stage": "baseline", "status": "complete", "gpu_id": 0},
        "q_causal_transport": {
            "stage": "q_causal_transport",
            "status": "running",
            "gpu_id": 1,
        },
        "classical_causal_transport": {
            "stage": "classical_causal_transport",
            "status": "queued",
            "gpu_id": None,
        },
        "q_causal_key_only": {
            "stage": "q_causal_key_only",
            "status": "failed",
            "gpu_id": 2,
        },
    }
    snapshot = selector_dashboard_snapshot(tmp_path, stage_state, threading.Lock())
    rendered = render_selector_dashboard(snapshot)
    assert snapshot["counts"] == {
        "complete": 1,
        "running": 1,
        "pending": 1,
        "failed": 1,
        "skipped": 0,
    }
    assert "GPU 1 | q_causal_transport | train | epoch 2/5 | batch 64/128" in rendered
    assert "ETA 35:20" in rendered
    assert "Pending: classical_causal_transport" in rendered


def test_resolve_reused_seed_requires_same_commit_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(multi_seed_runner, "ROOT", tmp_path)
    run_dir = tmp_path / "runs" / "pilot"
    run_dir.mkdir(parents=True)
    (run_dir / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    write_json(
        run_dir / "run_summary.json",
        {
            "status": "pass",
            "run_type": "formal_full_relation_transfer",
            "formal_experiment": True,
            "partial_selector_run": False,
            "selectors": [
                "disabled",
                "q_causal_transport",
                "classical_causal_transport",
                "q_causal_key_only",
            ],
            "results": [{"seed": 13}],
            "provenance": {
                "git_commit": "abc123",
                "git_dirty": False,
                "config_sha256": "config-sha",
            },
        },
    )
    reused = multi_seed_runner.resolve_reused_seeds(
        [f"13={run_dir}"],
        seeds=[7, 13],
        expected_commit="abc123",
        expected_config_sha256="config-sha",
    )
    assert reused == {13: run_dir.resolve()}
    with pytest.raises(ValueError, match="different formal config"):
        multi_seed_runner.resolve_reused_seeds(
            [f"13={run_dir}"],
            seeds=[7, 13],
            expected_commit="abc123",
            expected_config_sha256="other-sha",
        )
