from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_query_conditioned_formal_config_is_fixed_and_matched() -> None:
    config = json.loads(
        (ROOT / "configs/retacred_qsrpa_query_conditioned_formal_single_seed.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["formal_experiment"] is True
    assert config["seed"] == 13
    assert config["expected_records"] == {"train": 58465, "valid": 19584, "test": 13418}
    assert config["candidate"] == "quantum_query_conditioned_soft_role_pair"
    assert config["matched_control"] == "classical_query_conditioned_soft_role_pair"
    assert config["kernel"]["query_scope"] == "all"


def test_query_conditioned_formal_runner_is_serial_and_has_all_controls() -> None:
    text = (ROOT / "scripts/run_retacred_qsrpa_query_conditioned_formal_single_seed.sh").read_text(
        encoding="utf-8"
    )
    for name in (
        "quantum_global_context",
        "classical_global_context",
        "quantum_soft_role_pair",
        "classical_soft_role_pair",
        "quantum_query_conditioned_soft_role_pair",
        "classical_query_conditioned_soft_role_pair",
    ):
        assert name in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh" in text
    assert "--report-dir" in text
    assert "RUN_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)" in text
    assert 'runs/retacred_qsrpa_query_conditioned_formal_single_seed/${RUN_TIMESTAMP}_seed13' in text
    assert 'reports/retacred_qsrpa_query_conditioned_formal_single_seed/${RUN_TIMESTAMP}_seed13' in text
    assert 'REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}' in text
    assert 'bash scripts/export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh --run-dir "${RUN_DIR}" --report-dir "${REPORT_DIR}"' in text
    assert "No Python interpreter found" in text
    assert "/usr/local/miniconda3" not in text
    assert not any(line.rstrip().endswith("&") for line in text.splitlines())


def test_query_conditioned_handoff_scripts_use_portable_python_resolution() -> None:
    for name in (
        "run_retacred_qsrpa_query_conditioned_formal_single_seed.sh",
        "check_retacred_qsrpa_query_conditioned_formal_single_seed.sh",
        "export_retacred_qsrpa_query_conditioned_formal_single_seed_report.sh",
    ):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "resolve_python_bin" in text
        assert "for candidate in python python3" in text
        assert "/usr/local/miniconda3" not in text


def test_query_conditioned_preflight_rejects_dirty_worktree() -> None:
    text = (ROOT / "scripts/check_retacred_qsrpa_query_conditioned_formal_single_seed.sh").read_text(
        encoding="utf-8"
    )
    assert "git status --porcelain --untracked-files=all" in text
    assert "Repository is dirty" in text
