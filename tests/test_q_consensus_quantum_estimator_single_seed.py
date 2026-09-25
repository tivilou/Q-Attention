from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_consensus_quantum_estimator_single_seed as runner


def test_public_cli_fixes_seed_training_selectors_and_parallelism() -> None:
    options = {
        option
        for action in runner.build_parser()._actions
        for option in action.option_strings
    }
    assert "--seed" not in options
    assert "--steps" not in options
    assert "--selectors" not in options
    assert "--gpus" not in options
    assert "--gpu" in options
    assert "--dry-run" in options


def test_seed_config_is_the_first_frozen_seed_and_uses_cuda() -> None:
    master = runner.frozen.load_config(
        ROOT / "configs/q_consensus_quantum_estimator_frozen_multiseed.json"
    )
    output_dir = ROOT / "runs" / "unit-single-seed"
    config = runner.seed_config(master, output_dir)
    assert config["seed"] == runner.PREFLIGHT_SEED == runner.frozen.FROZEN_SEEDS[0]
    assert config["device"] == "cuda"
    assert tuple(config["selectors"]) == runner.frozen.FROZEN_SELECTORS
    assert config["training"] == master["training"]
    assert config["output_root"] == "runs/unit-single-seed"


def test_promoted_payload_records_complete_seed_runtime_and_one_gpu() -> None:
    payload = {
        "schema_version": "q-attention.q-consensus-quantum-estimator-canary.v1",
        "seed": 7,
        "gate": {"status": "pass", "next_multi_seed_authorized": False},
    }
    result = runner.promote_canary_payload(
        payload,
        elapsed_seconds=23.5,
        physical_gpu_id=2,
        config_path=ROOT / "configs/q_consensus_quantum_estimator_frozen_multiseed.json",
        config_sha256="config-hash",
        provenance={"git_commit": "commit", "git_dirty": False},
    )
    assert result["schema_version"] == runner.SCHEMA_VERSION
    assert result["runtime"]["elapsed_seconds"] == 23.5
    assert result["parallelism"] == {
        "type": "single_seed_single_gpu",
        "ddp": False,
        "physical_gpu_id": 2,
        "workers_on_gpu": 1,
    }
    assert result["gate"]["next_multi_seed_authorized"] is True


def test_write_json_replaces_atomically(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    runner.write_json(path, {"value": 1})
    runner.write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
