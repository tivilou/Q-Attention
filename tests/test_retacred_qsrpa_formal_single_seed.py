from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_qsrpa_formal_config_is_label_free_and_fixed_seed() -> None:
    config = json.loads((ROOT / "configs/retacred_qsrpa_formal_single_seed.json").read_text(encoding="utf-8"))
    assert config["formal_experiment"] is True
    assert config["seed"] == 13
    assert config["expected_records"] == {"train": 58465, "valid": 19584, "test": 13418}
    assert config["kernel"]["query_scope"] == "all"
    assert config["kernel"]["relation_anchor_mode"] == "soft_role_pair"

def test_qsrpa_runner_is_serial_and_contains_matched_controls() -> None:
    text = (ROOT / "scripts/run_retacred_qsrpa_formal_single_seed.sh").read_text(encoding="utf-8")
    for name in ("quantum_global_context", "classical_global_context", "quantum_srpa", "classical_srpa"):
        assert name in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert not any(line.rstrip().endswith("&") for line in text.splitlines())
