from __future__ import annotations

import json
from pathlib import Path
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "retacred_q_epvg_formal_single_seed.json"
EXPORTER = ROOT / "scripts" / "export_q_epvg_report.sh"


def test_exporter_selector_mapfile_emits_one_selector_per_line() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise the exporter shell path")

    line = next(
        line for line in EXPORTER.read_text(encoding="utf-8").splitlines()
        if line.startswith("mapfile -t SELECTORS < <(")
    )
    expected = json.loads(CONFIG.read_text(encoding="utf-8"))["selectors"][1:]
    command = "set -euo pipefail\n" + line + '\nprintf \'%s\\n\' "${SELECTORS[@]}"\n'
    env = os.environ.copy()
    env["PYTHON_BIN"] = shutil.which("python3") or shutil.which("python") or "python3"
    result = subprocess.run(
        [bash, "-c", command],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    actual = result.stdout.splitlines()
    assert actual == expected
    assert all("\\n" not in selector for selector in actual)
