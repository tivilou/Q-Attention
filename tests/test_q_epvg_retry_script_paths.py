from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SOURCE = REPO_ROOT / "scripts" / "retry_retacred_q_epvg_formal_single_seed_report.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="the retry runner is Bash-only")
def test_retry_runner_uses_repository_root_when_invoked_from_another_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / SCRIPT_SOURCE.name
    shutil.copy2(SCRIPT_SOURCE, script)

    run_dir = repo / "runs/retacred_q_epvg_formal_single_seed/20260926T011732Z_seed13"
    run_dir.mkdir(parents=True)
    (run_dir / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    (run_dir / "run_summary.data").write_text("summary\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/usr/bin/env bash
set -eu
case "$*" in
  "branch --show-current") echo 1.1 ;;
  "status --short --untracked-files=all") ;;
  "status --short --branch") echo '## 1.1' ;;
  "reset --quiet -- "*) ;;
  "fetch origin --prune") ;;
  "pull --ff-only origin 1.1") ;;
  "merge --no-edit origin/main") ;;
  "merge-base --is-ancestor origin/1.1 HEAD") ;;
  "merge-base --is-ancestor origin/main HEAD") ;;
  "add -- "*) ;;
  "diff --cached --check") ;;
  "diff --cached --name-only")
    echo 'reports/retacred_q_epvg_formal_single_seed/20260926T011732Z_seed13_retry/report.txt'
    echo 'reports/retacred_q_epvg_formal_single_seed/20260926T011732Z_seed13_retry/export_manifest.json'
    ;;
  "diff --cached --quiet") exit 1 ;;
  "commit -m "*) ;;
  "push origin 1.1") ;;
  "log -1 --oneline") echo 'test000 retry report' ;;
  *) echo "unexpected git invocation: $*" >&2; exit 90 ;;
esac
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    exporter = scripts / "export_retacred_q_epvg_formal_single_seed_report.sh"
    exporter.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if env | grep -q '^Q_EPVG_EXPORT_INJECT_FAILURE_AFTER='; then
  echo 'injected copy failure' >&2
  exit 73
fi
RUN_DIR=''
REPORT_DIR=''
while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --report-dir) REPORT_DIR=$2; shift 2 ;;
    --no-commit) shift ;;
    *) echo "unexpected exporter argument: $1" >&2; exit 91 ;;
  esac
done
"$PYTHON_BIN" - "$RUN_DIR" "$REPORT_DIR" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1]).resolve()
report = Path(sys.argv[2]).resolve()
report.mkdir(parents=True, exist_ok=True)
(report / "report.txt").write_text("safe report\\n", encoding="utf-8")
manifest = {
    "status": "complete",
    "attempt_count": 2,
    "retry_count": 1,
    "source_run": str(run),
    "report_identity": str(report),
    "failure_reason": "injected copy failure",
}
(report / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
PY
""",
        encoding="utf-8",
    )
    exporter.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(fake_bin), env.get("PATH", "")))
    env["PYTHON_BIN"] = sys.executable
    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    expected_report = repo / "reports/retacred_q_epvg_formal_single_seed/20260926T011732Z_seed13_retry"
    assert result.returncode == 0, result.stdout + result.stderr
    assert "completed run marker is missing" not in result.stderr
    assert "injected copy failure" in result.stdout + result.stderr
    assert f"REPORT_DIR=reports/retacred_q_epvg_formal_single_seed/20260926T011732Z_seed13_retry" in result.stdout
    assert (expected_report / "export_manifest.json").is_file()
