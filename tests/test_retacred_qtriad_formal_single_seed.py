from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qtriad_formal_config_is_fixed_and_matched() -> None:
    config = json.loads(
        (ROOT / "configs/retacred_qtriad_formal_single_seed.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["formal_experiment"] is True
    assert config["seed"] == 13
    assert config["expected_records"] == {
        "train": 58465,
        "valid": 19584,
        "test": 13418,
    }
    assert config["candidate"] == "q_triad"
    assert config["matched_control"] == "classical_density_tensor"
    assert config["gates"]["test_used_for_training_or_selection"] is False


def test_qtriad_runner_is_portable_serial_and_auto_exports() -> None:
    text = (ROOT / "scripts/run_retacred_qtriad_formal_single_seed.sh").read_text(
        encoding="utf-8"
    )
    assert "resolve_python_bin" in text
    assert "--started-at-utc" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "export_retacred_qtriad_formal_single_seed_report.sh" in text
    assert not any(line.rstrip().endswith("&") for line in text.splitlines())
    assert "/usr/local/miniconda" not in text


def test_qtriad_exporter_uses_run_basename_by_default_and_has_gates() -> None:
    text = (
        ROOT / "scripts/export_retacred_qtriad_formal_single_seed_report.sh"
    ).read_text(encoding="utf-8")
    assert 'reports/retacred_qtriad_formal_single_seed/${RUN_NAME}' in text
    assert "origin/1.1" in text
    assert "origin/main" in text
    assert "test_used_for_training_or_selection" in text
    assert "provenance" in text
    assert "Forbidden private artifact" in text


def test_qtriad_doc_is_indexed_below_method_overview() -> None:
    readme = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    method = readme.index("current/method_overview_zh.md")
    qtriad = readme.index("current/retacred_qtriad_formal_single_seed_zh.md")
    qvres = readme.index("current/qvres_relation_transfer_full_run_zh.md")
    assert method < qtriad < qvres
