from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_config_is_full_data_single_seed_and_label_free() -> None:
    config = json.loads((ROOT / "configs/retacred_qlass_formal_single_seed.json").read_text(encoding="utf-8"))
    assert config["formal_experiment"] is True
    assert config["seed"] == 13
    assert config["selection_metric"] == "macro_f1_then_loss"
    assert config["expected_records"] == {"train": 58465, "valid": 19584, "test": 13418}
    assert config["kernel"]["input_encoding"] == "joint"
    assert config["kernel"]["query_scope"] == "all"
    assert config["kernel"]["relation_anchor_mode"] == "global_context"


def test_runner_is_explicitly_single_gpu_and_serial() -> None:
    text = (ROOT / "scripts/run_retacred_qlass_formal_single_seed.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "--device cuda" in text
    assert "run_stage baseline" in text
    assert "run_stage quantum_global_context" in text
    assert "run_stage classical_global_context" in text
    assert not any(line.rstrip().endswith("&") for line in text.splitlines())
    assert "--gpus" not in text


def test_summary_does_not_offer_test_selection() -> None:
    text = (ROOT / "scripts/summarize_retacred_qlass_formal_single_seed.py").read_text(encoding="utf-8")
    assert "test_used_for_training_or_selection" in text
    assert "valid" in text
