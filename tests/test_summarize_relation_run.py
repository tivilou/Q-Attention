from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_summary_module():
    path = Path("experiments/summarize_relation_run.py")
    spec = importlib.util.spec_from_file_location("summarize_relation_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_relation_run_collects_rows(tmp_path) -> None:
    module = load_summary_module()
    run_dir = tmp_path / "run"
    write_json(run_dir / "baseline" / "metrics.json", {"best_valid": {"macro_f1": 0.2, "accuracy": 0.3, "loss": 1.0}})
    write_json(
        run_dir / "classical_steering_eval" / "metrics.json",
        {"steered": {"macro_f1": 0.25, "accuracy": 0.35, "loss": 0.9}, "delta_vs_baseline": {"macro_f1": 0.05}},
    )
    write_json(
        run_dir / "spectral_filter_sweep" / "summary.json",
        {
            "best_by_macro_f1": {
                "family": "quantum",
                "filter": {"mode": "hard_topk", "rank": 4},
                "gain": 0.25,
                "metrics": {"macro_f1": 0.3, "accuracy": 0.4, "loss": 0.8},
                "delta_vs_baseline": {"macro_f1": 0.1},
            }
        },
    )

    summary = module.summarize_run(run_dir)
    markdown = module.markdown_table(summary)

    assert [row["variant"] for row in summary["rows"]] == ["baseline", "classical_steering", "spectral_sweep_best"]
    assert summary["rows"][2]["note"] == "quantum, hard_topk, rank=4, gain=0.25"
    assert "| spectral_sweep_best | 0.300000" in markdown