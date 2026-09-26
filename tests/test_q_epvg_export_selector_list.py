from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "retacred_q_epvg_formal_single_seed.json"
EXPORTER = ROOT / "scripts" / "q_epvg_report_export.py"


def load_export_module():
    spec = importlib.util.spec_from_file_location("q_epvg_report_export", EXPORTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_names_are_loaded_as_structured_lines() -> None:
    module = load_export_module()
    expected = json.loads(CONFIG.read_text(encoding="utf-8"))["selectors"][1:]
    assert module._selector_names(CONFIG) == expected
    shell = (ROOT / "scripts/export_q_epvg_report.sh").read_text(encoding="utf-8")
    assert "q_epvg_report_export.py" in shell
    assert "mapfile -t SELECTORS" not in shell
