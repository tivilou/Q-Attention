from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "q_epvg_report_export.py"
    spec = importlib.util.spec_from_file_location("q_epvg_report_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    config = root / "config.json"
    selectors = ["disabled", "selector_a", "selector_b"]
    config.write_text(json.dumps({"selectors": selectors}), encoding="utf-8")
    run = root / "run"
    for name in ("RUN_COMPLETE", "run_summary.json", "run_summary.data", "run_summary.md", "gpu_assignments.json"):
        write(run / name)
    write(run / "baseline" / "metrics.json")
    for selector in selectors[1:]:
        for name in ("metrics.json", "case_study.json", "sample_trace.json"):
            write(run / "selectors" / selector / name)
    for split in ("train", "valid", "test"):
        write(run / "data" / f"{split}.jsonl", f"{{\"split\": \"{split}\"}}\n")
    return run, config, root / "reports" / "experiment" / "run_seed13"


@pytest.mark.parametrize("failure", ["copy", "validation"])
def test_failed_export_cleans_staging_and_retry_reuses_completed_run(tmp_path: Path, failure: str) -> None:
    module = load_module()
    run, config, report = build_fixture(tmp_path)
    summary_before = (run / "run_summary.json").read_bytes()
    kwargs = {
        "inject_failure_after": 2 if failure == "copy" else None,
        "inject_validation_failure": failure == "validation",
    }
    with pytest.raises(module.ExportError):
        module.export_report(
            run_dir=run,
            report_dir=report,
            config_path=config,
            reporting_commit="abc123",
            **kwargs,
        )
    assert not report.exists()
    assert not list(report.parent.glob(f".{report.name}.staging-*"))
    assert (run / "run_summary.json").read_bytes() == summary_before

    destination = module.export_report(
        run_dir=run,
        report_dir=report,
        config_path=config,
        reporting_commit="abc123",
    )
    assert destination == report.resolve()
    assert (report / "metrics/selector_a.json").is_file()
    assert (report / "case_study/selector_b.sample-trace.json").is_file()
    assert (report / "data.sha256").is_file()
    manifest = json.loads((report / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_run"] == str(run.resolve())
    assert manifest["retry_count"] == 1
    assert "injected" in manifest["failure_reason"]
    assert not list(report.parent.glob(f".{report.name}.staging-*"))
