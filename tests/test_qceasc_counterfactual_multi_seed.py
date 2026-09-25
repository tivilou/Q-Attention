from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_retacred_qceasc_formal_multi_seed.py"
SUMMARY_PATH = ROOT / "scripts" / "summarize_retacred_qceasc_formal_multi_seed.py"
COUNTERFACTUAL_CONFIG = ROOT / "configs" / "retacred_qceasc_counterfactual_formal_single_seed.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterfactual_config_selects_distinct_protocol_and_selectors():
    runner = _load("counterfactual_protocol_selection", RUNNER_PATH)
    config = json.loads(COUNTERFACTUAL_CONFIG.read_text(encoding="utf-8"))
    runner.configure_protocol(COUNTERFACTUAL_CONFIG, config)
    assert runner.PROTOCOL == "qceasc_counterfactual"
    assert runner.GROUP_ROOT.name == "retacred_qceasc_counterfactual_formal_multi_seed"
    assert runner.SELECTORS == (
        "disabled",
        "q_ceasc_counterfactual",
        "classical_counterfactual",
    )
    assert runner.SELECTOR_TASKS == (
        "q_ceasc_counterfactual",
        "classical_counterfactual",
    )


def test_non_counterfactual_config_rejects_seed13_import(tmp_path: Path):
    runner = _load("counterfactual_import_gate", RUNNER_PATH)
    ordinary = {"counterfactual": False}
    report = tmp_path / "report"
    report.mkdir()
    with pytest.raises(ValueError, match="only for counterfactual"):
        runner.validate_seed13_report(
            report,
            config_path=tmp_path / "ordinary.json",
            config=ordinary,
            current_commit="new",
        )


def test_missing_seed13_report_is_rejected(tmp_path: Path):
    runner = _load("counterfactual_missing_report", RUNNER_PATH)
    config = json.loads(COUNTERFACTUAL_CONFIG.read_text(encoding="utf-8"))
    runner.configure_protocol(COUNTERFACTUAL_CONFIG, config)
    with pytest.raises(ValueError, match="does not exist"):
        runner.validate_seed13_report(
            tmp_path / "missing",
            config_path=COUNTERFACTUAL_CONFIG,
            config=config,
            current_commit="new",
        )


def _write_validation_fixture(
    tmp_path: Path,
    runner,
    *,
    source_hash_ok: bool = True,
    data_hash_ok: bool = True,
) -> tuple[Path, Path, dict]:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = json.loads(COUNTERFACTUAL_CONFIG.read_text(encoding="utf-8"))
    config["seed"] = 13
    config["expected_records"] = {split: 1 for split in ("train", "valid", "test")}
    for split in ("train", "valid", "test"):
        config[f"{split}_path"] = f"data/relation/retacred/{split}.jsonl"
    config_path = repo / "configs" / "counterfactual.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    for split in ("train", "valid", "test"):
        data_path = repo / config[f"{split}_path"]
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(f'{{"split": "{split}"}}\n', encoding="utf-8")
    source_path = repo / "fixture_source.py"
    source_path.write_text("SOURCE = 1\n", encoding="utf-8")
    runner.ROOT = repo
    runner.COUNTERFACTUAL_SOURCE_FILES = {"fixture": "fixture_source.py"}
    runner.configure_protocol(config_path, config)

    report = tmp_path / "report"
    (report / "metrics").mkdir(parents=True)
    (report / "case_study").mkdir()
    (report / "run_config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    (report / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    summary = {
        "formal_experiment": True,
        "stage": "formal_single_seed",
        "seed": 13,
        "selectors": list(runner.SELECTORS),
        "test_used_for_training_or_selection": False,
    }
    (report / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (report / "run_summary.md").write_text("summary\n", encoding="utf-8")
    (report / "metrics" / "baseline.json").write_text("{}", encoding="utf-8")
    for selector in runner.SELECTOR_TASKS:
        (report / "metrics" / f"{selector}.json").write_text(
            json.dumps(
                {
                    "selector": selector,
                    "finite": True,
                    "test": {"metrics": {"macro_f1": 0.2}},
                }
            ),
            encoding="utf-8",
        )
        (report / "case_study" / f"{selector}.json").write_text(
            json.dumps(
                {
                    "schema_version": "q-attention.q-ceasc-case-study.v2",
                    "required_splits": ["train", "valid", "test"],
                    "cases": [{} for _ in range(27)],
                }
            ),
            encoding="utf-8",
        )
        (report / "case_study" / f"{selector}.sample-trace.json").write_text(
            json.dumps(
                {
                    "schema_version": "sample-trace.v1",
                    "experiment": {
                        "config_sha256": runner.sha256(report / "run_config.json")
                    },
                }
            ),
            encoding="utf-8",
        )
    provenance = {
        "git_dirty": False,
        "config_sha256": runner.sha256(config_path),
        "git_revision": "old-report-commit",
        "source_contract": {
            "files": {
                "fixture": {
                    "path": "fixture_source.py",
                    "sha256": runner.sha256(source_path)
                    if source_hash_ok
                    else "0" * 64,
                }
            }
        },
    }
    (report / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    counts = "\n".join(
        f"1 data/relation/retacred/{split}.jsonl" for split in ("train", "valid", "test")
    )
    (report / "data_counts.txt").write_text(counts + "\n", encoding="utf-8")
    hash_rows = []
    for split in ("train", "valid", "test"):
        data_path = repo / config[f"{split}_path"]
        digest = runner.sha256(data_path) if data_hash_ok or split != "train" else "0" * 64
        hash_rows.append(f"{digest} {config[f'{split}_path']}")
    (report / "data.sha256").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")
    return report, config_path, config


def test_seed13_source_hash_mismatch_is_rejected(tmp_path: Path):
    runner = _load("counterfactual_source_gate", RUNNER_PATH)
    report, config_path, config = _write_validation_fixture(
        tmp_path, runner, source_hash_ok=False
    )
    with pytest.raises(ValueError, match="source hash differs"):
        runner.validate_seed13_report(
            report,
            config_path=config_path,
            config=config,
            current_commit="new",
        )


def test_seed13_data_hash_mismatch_is_rejected(tmp_path: Path):
    runner = _load("counterfactual_data_gate", RUNNER_PATH)
    report, config_path, config = _write_validation_fixture(
        tmp_path, runner, data_hash_ok=False
    )
    with pytest.raises(ValueError, match="data hash differs for train"):
        runner.validate_seed13_report(
            report,
            config_path=config_path,
            config=config,
            current_commit="new",
        )


def test_imported_seed13_report_materializes_only_seed13(tmp_path: Path):
    runner = _load("counterfactual_report_import", RUNNER_PATH)
    source = tmp_path / "source"
    (source / "metrics").mkdir(parents=True)
    (source / "case_study").mkdir()
    files = [
        "RUN_COMPLETE",
        "run_config.json",
        "run_summary.json",
        "run_summary.md",
        "data.sha256",
        "data_counts.txt",
        "provenance.json",
        "metrics/baseline.json",
        "metrics/q_ceasc_counterfactual.json",
        "metrics/classical_counterfactual.json",
        "case_study/q_ceasc_counterfactual.json",
        "case_study/q_ceasc_counterfactual.sample-trace.json",
        "case_study/classical_counterfactual.json",
        "case_study/classical_counterfactual.sample-trace.json",
    ]
    for relative in files:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    group = tmp_path / "group"
    config = {"counterfactual": True}
    metadata = {
        "seed": 13,
        "mode": "audited_report_reuse",
        "validated_against_git_commit": "new",
    }
    runner.SELECTOR_TASKS = ("q_ceasc_counterfactual", "classical_counterfactual")
    runner.import_seed13_report(
        source,
        group,
        config_path=tmp_path / "config.json",
        config=config,
        current_commit="new",
        metadata=metadata,
    )
    assert (group / "seed_13" / "imported_report.json").is_file()
    assert not (group / "seed_29").exists()
    assert not (group / "seed_53").exists()


def _write_counterfactual_group(
    group: Path,
    *,
    imported_seed13: bool = True,
    bad_fresh_commit: bool = False,
) -> None:
    group.mkdir()
    manifest = {
        "schema_version": "q-attention.q-ceasc-counterfactual.formal-task-graph.v1",
        "protocol": "qceasc_counterfactual",
        "git_commit": "scheduler-commit",
        "seeds": [13, 29, 53],
    }
    (group / "MULTI_SEED_COMPLETE").write_text("ok\n", encoding="utf-8")
    (group / "multi_seed_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    selectors = ("disabled", "q_ceasc_counterfactual", "classical_counterfactual")
    for seed in (13, 29, 53):
        seed_dir = group / f"seed_{seed}"
        for selector in selectors[1:]:
            selector_dir = seed_dir / "selectors" / selector
            selector_dir.mkdir(parents=True)
            (selector_dir / "case_study.json").write_text("{}", encoding="utf-8")
            (selector_dir / "sample_trace.json").write_text("{}", encoding="utf-8")
        (seed_dir / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
        config = {
            "schema_version": "q-attention.q-ceasc-counterfactual-formal-single-seed.v1",
            "counterfactual": True,
            "seed": seed,
            "selectors": list(selectors),
        }
        (seed_dir / "run_config.json").write_text(
            json.dumps(config, sort_keys=True), encoding="utf-8"
        )
        rows = [
            {"selector": "disabled", "valid": {"metrics": {"macro_f1": 0.2}}, "test": {"metrics": {"macro_f1": 0.2}}},
            {"selector": "q_ceasc_counterfactual", "valid": {"metrics": {"macro_f1": 0.21}}, "test": {"metrics": {"macro_f1": 0.21}}},
            {"selector": "classical_counterfactual", "valid": {"metrics": {"macro_f1": 0.205}}, "test": {"metrics": {"macro_f1": 0.205}}},
        ]
        commit = "old-report-commit" if seed == 13 else ("wrong" if bad_fresh_commit and seed == 29 else "scheduler-commit")
        summary = {
            "formal_experiment": True,
            "stage": "formal_single_seed",
            "seed": seed,
            "selectors": list(selectors),
            "rows": rows,
            "test_used_for_training_or_selection": False,
            "provenance": {"git_revision": commit, "git_dirty": False},
        }
        (seed_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        if seed == 13 and imported_seed13:
            (seed_dir / "imported_report.json").write_text(
                json.dumps(
                    {
                        "seed": 13,
                        "source_git_revision": "old-report-commit",
                        "validated_against_git_commit": "scheduler-commit",
                    }
                ),
                encoding="utf-8",
            )


def test_summary_accepts_imported_seed_and_dynamic_selector_names(tmp_path: Path, monkeypatch):
    summary = _load("counterfactual_dynamic_summary", SUMMARY_PATH)
    monkeypatch.setattr(summary, "_trace_errors", lambda path, expected: [])
    group = tmp_path / "group"
    _write_counterfactual_group(group)
    payload = summary.collect(group)
    assert payload["protocol"] == "qceasc_counterfactual"
    assert payload["candidate"] == "q_ceasc_counterfactual"
    assert payload["matched_control"] == "classical_counterfactual"
    assert payload["seed_records"][0]["imported_report"] is True
    assert payload["source_git_revisions"] == ["old-report-commit", "scheduler-commit"]


def test_summary_rejects_fresh_seed_commit_mismatch(tmp_path: Path, monkeypatch):
    summary = _load("counterfactual_fresh_commit_gate", SUMMARY_PATH)
    monkeypatch.setattr(summary, "_trace_errors", lambda path, expected: [])
    group = tmp_path / "group"
    _write_counterfactual_group(group, bad_fresh_commit=True)
    with pytest.raises(ValueError, match="seed 29 commit differs"):
        summary.collect(group)


def test_counterfactual_wrappers_have_valid_shell_syntax():
    for path in (
        ROOT / "scripts" / "run_retacred_qceasc_counterfactual_formal_multi_seed.sh",
        ROOT / "scripts" / "export_retacred_qceasc_counterfactual_formal_multi_seed_report.sh",
    ):
        result = subprocess.run(["bash", "-n", str(path)], check=False)
        assert result.returncode == 0, path
