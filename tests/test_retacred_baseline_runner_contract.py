from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_retacred_baseline_complete.sh"
EXPORTER = ROOT / "scripts" / "export_retacred_baseline_report.sh"


def test_runner_contains_automatic_export_and_publish_contract():
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        "--skip-export",
        "--no-push",
        "bash scripts/export_retacred_baseline_report.sh",
        "bash scripts/publish_retacred_baseline_report.sh",
        "EXPORT_COMPLETE",
    ):
        assert token in text


def test_exporter_remains_standalone_audit_component():
    text = EXPORTER.read_text(encoding="utf-8")
    assert "Working tree must be clean before export" in text
    assert "git merge-base --is-ancestor origin/main HEAD" in text
    assert "Forbidden private artifact detected" in text
    assert 'mkdir -p "${ROOT}/reports"' in text
    assert "Standalone export complete" in text


def test_dry_run_prints_publication_steps_without_running_them():
    result = subprocess.run(
        ["bash", str(RUNNER), "--seed", "13", "--gpu", "0", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "scripts/export_retacred_baseline_report.sh" in result.stdout
    assert "publish_retacred_baseline_report.sh" in result.stdout


def test_dry_run_can_skip_publication():
    result = subprocess.run(
        ["bash", str(RUNNER), "--seed", "13", "--gpu", "0", "--dry-run", "--skip-export"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "export and publish skipped (--skip-export)" in result.stdout
    assert "git push origin 1.1" not in result.stdout
