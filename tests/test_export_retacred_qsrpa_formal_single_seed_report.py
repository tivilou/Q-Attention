from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_qsrpa_exporter_is_report_only_and_targets_1_1() -> None:
    text = (ROOT / "scripts/export_retacred_qsrpa_formal_single_seed_report.sh").read_text(encoding="utf-8")
    assert "reports/retacred_qsrpa_formal_single_seed" in text
    assert 'git add -- "${REPORT_REL}"' in text
    assert 'git push origin 1.1' in text
    assert "git push origin main" not in text
    assert "RUN_COMPLETE" in text and "test_used_for_training_or_selection" in text
