from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HEAD = "a" * 40
RUN_EXPORTS = (
    "run_summary.json",
    "baseline/metrics.json",
    "classical_steering_eval/metrics.json",
    "quantum_steering_eval/metrics.json",
    "spectral_filter_sweep/summary.json",
    "relation_routing_eval/metrics.json",
)


def load_export_module():
    path = Path("scripts/export_retacred_report.py")
    spec = importlib.util.spec_from_file_location("export_retacred_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_run(repo_root: Path, run_name: str, config_name: str, *, dirty: bool = False) -> Path:
    config_path = repo_root / "configs" / config_name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"name": run_name}), encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_dir = repo_root / "runs" / run_name / "20260723-120000"
    write_json(
        run_dir / "pipeline_summary.json",
        {
            "commands": [{"dry_run": False, "returncode": 0}],
            "device": "cuda",
            "evaluation_protocol": {"final_split": "test", "test_isolated": True},
            "output_dir": str(run_dir.resolve()),
            "reproducibility": {
                "config_sha256": config_hash,
                "cuda_available": True,
                "git": {"commit": HEAD, "dirty": dirty},
            },
        },
    )
    (run_dir / "run_summary.md").write_text("# Run summary\n", encoding="utf-8")
    for relative_path in RUN_EXPORTS:
        write_json(run_dir / relative_path, {"ok": True})
    return run_dir


def patch_clean_git(monkeypatch, module) -> None:
    def fake_git_output(repo_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return HEAD
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git_output", fake_git_output)


def test_export_report_writes_only_public_whitelist(tmp_path, monkeypatch) -> None:
    module = load_export_module()
    patch_clean_git(monkeypatch, module)
    full_run = build_run(tmp_path, "retacred_full_gpu", "retacred_full_gpu.json")
    low_run = build_run(
        tmp_path, "retacred_low_resource_gpu", "retacred_low_resource_gpu.json"
    )
    log_dir = tmp_path / "runs" / "handoff_logs"
    log_dir.mkdir(parents=True)
    full_log = log_dir / "retacred_full_gpu.log"
    low_log = log_dir / "retacred_low_resource_gpu.log"
    full_log.write_text(
        "".join(f"line-{index}\n" for index in range(1005)), encoding="utf-8"
    )
    low_log.write_text("low log\n", encoding="utf-8")

    destination = module.export_report(
        repo_root=tmp_path,
        full_run=full_run,
        low_resource_run=low_run,
        full_log=full_log,
        low_resource_log=low_log,
        report_tag="20260723-120000",
    )

    exported = sorted(
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert len(exported) == 20
    assert exported[0] == "configs/retacred_full_gpu.json"
    assert "full/baseline_metrics.json" in exported
    assert "low_resource/pipeline_summary.json" in exported
    assert exported[-1] == "low_resource/spectral_filter_summary.json"
    assert (
        destination / "logs" / "retacred_full_gpu.tail.txt"
    ).read_text(encoding="utf-8").startswith("line-5\n")


def test_export_report_rejects_dirty_pipeline(tmp_path, monkeypatch) -> None:
    module = load_export_module()
    patch_clean_git(monkeypatch, module)
    full_run = build_run(
        tmp_path, "retacred_full_gpu", "retacred_full_gpu.json", dirty=True
    )
    low_run = build_run(
        tmp_path, "retacred_low_resource_gpu", "retacred_low_resource_gpu.json"
    )
    log_dir = tmp_path / "runs" / "handoff_logs"
    log_dir.mkdir(parents=True)
    full_log = log_dir / "full.log"
    low_log = log_dir / "low.log"
    full_log.write_text("full\n", encoding="utf-8")
    low_log.write_text("low\n", encoding="utf-8")

    try:
        module.export_report(
            repo_root=tmp_path,
            full_run=full_run,
            low_resource_run=low_run,
            full_log=full_log,
            low_resource_log=low_log,
            report_tag="dirty-run",
        )
    except module.ExportError as exc:
        assert "git.dirty=false" in str(exc)
    else:
        raise AssertionError("dirty pipeline was accepted")

    assert not (tmp_path / "reports" / "retacred" / "dirty-run").exists()


def test_export_report_rejects_dirty_worktree_before_copying(tmp_path, monkeypatch) -> None:
    module = load_export_module()
    monkeypatch.setattr(module, "_git_output", lambda repo_root, *args: "?? copy.sh")
    try:
        module.export_report(
            repo_root=tmp_path,
            full_run=tmp_path / "missing-full",
            low_resource_run=tmp_path / "missing-low",
            full_log=tmp_path / "missing-full.log",
            low_resource_log=tmp_path / "missing-low.log",
            report_tag="dirty-worktree",
        )
    except module.ExportError as exc:
        assert "copy.sh" in str(exc)
        assert "worktree is not clean" in str(exc)
    else:
        raise AssertionError("dirty worktree was accepted")
