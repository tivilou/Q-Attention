from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_report_exporter_only_publishes_reports_and_pushes_1_1() -> None:
    text = (ROOT / "scripts/export_retacred_qlass_formal_single_seed_report.sh").read_text(
        encoding="utf-8"
    )
    assert "reports/retacred_qlass_formal_single_seed" in text
    assert 'git add -- "${REPORT_REL}"' in text
    assert 'git push origin 1.1' in text
    assert "git push origin main" not in text
    assert "git add -- \"${RUN_DIR}" not in text


def test_report_exporter_rejects_failed_or_incomplete_runs() -> None:
    text = (ROOT / "scripts/export_retacred_qlass_formal_single_seed_report.sh").read_text(
        encoding="utf-8"
    )
    assert "RUN_FAILED" in text
    assert "RUN_COMPLETE" in text
    assert "test_used_for_training_or_selection" in text
