from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def load_case_fixture():
    path = ROOT / "tests" / "test_q_epvg_report_export.py"
    spec = importlib.util.spec_from_file_location("q_epvg_case_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exporter():
    path = ROOT / "scripts" / "q_epvg_multi_seed_report_export.py"
    spec = importlib.util.spec_from_file_location("q_epvg_multi_seed_exporter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_group(root: Path) -> tuple[Path, Path]:
    fixture = load_case_fixture()
    exporter = load_exporter()
    group = root / "group"
    selectors = ["disabled", "selector_a"]
    config_base = {
        "schema_version": "q-attention.q-epvg-formal-single-seed.v1",
        "name": "retacred_q_epvg_formal_single_seed",
        "seed": 0,
        "selectors": selectors,
        "candidate": "selector_a",
        "matched_control": "selector_a",
        "structural_control": "selector_a",
        "case_study": {
            "records": {"train": [0], "valid": [0], "test": [0]},
            "checkpoints": [
                "initial_or_pre_training",
                "best_valid_or_declared_selection_checkpoint",
                "final",
            ],
        },
    }
    group.mkdir(parents=True)
    manifest = {
        "schema_version": "q-attention.q-epvg.formal-multiseed-manifest.v1",
        "protocol": "q_epvg",
        "git_commit": "abc123",
        "seeds": [13, 29, 53],
        "selectors": selectors,
    }
    (group / "multi_seed_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, value in (
        ("MULTI_SEED_COMPLETE", "done\n"),
        ("multi_seed_status.json", "{}"),
        ("multi_seed_run_summary.json", "{}"),
        ("multi_seed_summary.json", "{}"),
        ("multi_seed_summary.md", "summary\n"),
        ("reporting_commit.txt", "abc123\n"),
    ):
        _write(group / name, value)
    for seed in (13, 29, 53):
        seed_dir = group / f"seed_{seed}"
        config = dict(config_base)
        config["seed"] = seed
        _write(seed_dir / "RUN_COMPLETE", "done\n")
        (seed_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        rows = []
        for selector, value in (("disabled", 0.10), ("selector_a", 0.11 + seed / 10000)):
            rows.append(
                {
                    "selector": selector,
                    "valid": {"metrics": {"macro_f1": value}},
                    "test": {"metrics": {"macro_f1": value}},
                }
            )
        summary = {
            "formal_experiment": True,
            "stage": "formal_single_seed",
            "seed": seed,
            "selectors": selectors,
            "test_used_for_training_or_selection": False,
            "provenance": {"git_dirty": False, "git_revision": "abc123"},
            "rows": rows,
        }
        (seed_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        _write(seed_dir / "run_summary.data", "data\n")
        _write(seed_dir / "run_summary.md", "summary\n")
        _write(seed_dir / "data_counts.txt", "3 train\n")
        _write(seed_dir / "data.sha256", "a" * 64 + "  train.jsonl\n")
        (seed_dir / "provenance.json").write_text(json.dumps(summary["provenance"]), encoding="utf-8")
        _write(seed_dir / "baseline" / "metrics.json", json.dumps({"selector": "disabled", "test": {"metrics": {"macro_f1": 0.1}}}))
        for selector in selectors[1:]:
            selector_dir = seed_dir / "selectors" / selector
            _write(selector_dir / "metrics.json", json.dumps({"selector": selector, "finite": True, "valid": {"metrics": {"macro_f1": 0.11}}, "test": {"metrics": {"macro_f1": 0.11}}}))
            payload, trace = fixture.case_payload(selector)
            trace["experiment"] = {
                "config_sha256": hashlib.sha256(
                    (seed_dir / "run_config.json").read_bytes()
                ).hexdigest()
            }
            for tensor in payload["tensor_manifest"]:
                raw = f"{seed}:{tensor['manifest_id']}\n".encode()
                tensor["sha256"] = hashlib.sha256(raw).hexdigest()
                tensor["byte_count"] = len(raw)
                _write(selector_dir / tensor["path"], raw.decode())
            (selector_dir / "case_study.json").write_text(json.dumps(payload), encoding="utf-8")
            (selector_dir / "sample_trace.json").write_text(json.dumps(trace), encoding="utf-8")
    report = root / "reports" / "q_epvg"
    return group, report


@pytest.mark.parametrize("failure", ["copy", "validation"])
def test_multiseed_export_failure_cleans_and_retries(tmp_path: Path, failure: str) -> None:
    exporter = load_exporter()
    group, report = build_group(tmp_path)
    with pytest.raises(exporter.ExportError):
        exporter.export_report(
            group_dir=group,
            report_dir=report,
            reporting_commit="abc123",
            inject_failure_after=2 if failure == "copy" else None,
            inject_validation_failure=failure == "validation",
        )
    assert not report.exists()
    assert not list(report.parent.glob(f".{report.name}.staging-*"))

    exporter.export_report(group_dir=group, report_dir=report, reporting_commit="abc123")
    manifest = json.loads((report / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["retry_count"] == 1
    assert "injected" in str(manifest["failure_reason"])
    assert (report / "seeds/seed_13/case_study/selector_a.sample-trace.json").is_file()
