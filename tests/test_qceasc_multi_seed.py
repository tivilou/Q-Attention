from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_retacred_qceasc_formal_multi_seed.py"
SUMMARY_PATH = ROOT / "scripts" / "summarize_retacred_qceasc_formal_multi_seed.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_parser_rejects_duplicates_and_non_predeclared_set():
    runner = _load("qceasc_multiseed_runner_parser", RUNNER_PATH)
    assert runner.parse_int_list("13,29,53", label="seeds") == [13, 29, 53]
    with pytest.raises(ValueError):
        runner.parse_int_list("13,13", label="seeds")


def test_child_command_records_exact_target_and_physical_gpu():
    runner = _load("qceasc_multiseed_runner_command", RUNNER_PATH)
    args = type("Args", (), {
        "python_bin": "/env/bin/python",
        "hardware_profile": "adaptive",
        "log_every_batches": 50,
        "checkpoint_every_batches": 50,
    })()
    command = runner.build_child_command(
        seed=29,
        gpu_id=2,
        config_path=ROOT / "runs/group/configs/seed_29.json",
        run_dir=ROOT / "runs/group/seed_29",
        args=args,
    )
    assert command[0] == "/env/bin/python"
    assert command[1].endswith("experiments/run_retacred_qceasc_formal_single_seed.py")
    assert command[command.index("--seed") + 1] == "29"
    assert command[command.index("--gpus") + 1] == "2"
    assert "--replication-child" in command


def test_selector_resume_command_enables_adaptive_elastic_resume():
    runner = _load("qceasc_multiseed_selector_command", RUNNER_PATH)
    args = type("Args", (), {
        "python_bin": "/env/bin/python",
        "log_every_batches": 50,
        "checkpoint_every_batches": 50,
    })()
    command = runner.build_selector_command(
        seed=13,
        selector="q_ceasc",
        gpu_id=1,
        config_path=ROOT / "runs/group/configs/seed_13.json",
        seed_dir=ROOT / "runs/group/seed_13",
        selector_dir=ROOT / "runs/group/seed_13/selectors/q_ceasc",
        args=args,
        profile={
            "pair_chunk_size": None,
            "pair_chunk_divisor": 1,
            "micro_batch_size": 256,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": False,
        },
        adaptive=True,
        resume=True,
    )
    assert "--adaptive-memory" in command
    assert "--resume" in command
    assert "--elastic-resume" in command


def _write_trace(path: Path, config_sha: str, selector: str) -> None:
    observed = ("data", "preprocess", "training", "scoring", "selection", "context", "evaluation", "diagnosis")
    stages = [{"stage": name, "status": "observed", "observed_fields": {"selector": selector}} for name in observed]
    stages.extend([
        {"stage": "retrieval", "status": "not_applicable"},
        {"stage": "generation", "status": "not_applicable"},
    ])
    payload = {
        "schema_version": "sample-trace.v1",
        "trace_id": f"trace-{selector}",
        "experiment": {
            "run_id": "fixture",
            "dataset": "valid",
            "code_revision": "abc",
            "config_sha256": config_sha,
            "model_identity": "fixture-model",
        },
        "sample_selection": {
            "rule": "fixed indices [0]",
            "population_scope": "valid",
            "selected_count": 1,
            "selected_sample_ids": ["s0"],
        },
        "coverage": {name: ("not_applicable" if name in {"retrieval", "generation"} else "observed") for name in ("data", "preprocess", "training", "retrieval", "scoring", "selection", "context", "generation", "evaluation", "diagnosis")},
        "samples": [{"sample_id": "s0", "stages": stages}],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_group(group: Path, *, q_values=(0.210, 0.211, 0.212)) -> None:
    group.mkdir()
    manifest = {
        "schema_version": "q-attention.q-ceasc.formal-task-graph.v2",
        "git_commit": "abc",
        "seeds": [13, 29, 53],
        "protocol_fingerprint": "fixture",
    }
    (group / "MULTI_SEED_COMPLETE").write_text("ok\n", encoding="utf-8")
    (group / "multi_seed_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for seed, q_value in zip((13, 29, 53), q_values, strict=True):
        seed_dir = group / f"seed_{seed}"
        for selector in ("q_ceasc", "classical_ceasc"):
            (seed_dir / "selectors" / selector).mkdir(parents=True)
        (seed_dir / "baseline").mkdir()
        (seed_dir / "RUN_COMPLETE").write_text("ok\n", encoding="utf-8")
        config = {"schema_version": "q-attention.q-ceasc-formal-single-seed.v1", "name": "fixture", "seed": seed, "selectors": ["disabled", "q_ceasc", "classical_ceasc"], "candidate": "q_ceasc", "matched_control": "classical_ceasc"}
        (seed_dir / "run_config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        config_sha = hashlib.sha256((seed_dir / "run_config.json").read_bytes()).hexdigest()
        rows = []
        for selector, value in (("disabled", 0.2), ("q_ceasc", q_value), ("classical_ceasc", 0.205)):
            rows.append({"selector": selector, "valid": {"metrics": {"macro_f1": value}}, "test": {"metrics": {"macro_f1": value}}})
            if selector != "disabled":
                (seed_dir / "selectors" / selector / "metrics.json").write_text("{}", encoding="utf-8")
                (seed_dir / "selectors" / selector / "case_study.json").write_text("{}", encoding="utf-8")
                _write_trace(seed_dir / "selectors" / selector / "sample_trace.json", config_sha, selector)
        (seed_dir / "baseline" / "metrics.json").write_text("{}", encoding="utf-8")
        summary = {"formal_experiment": True, "stage": "formal_single_seed", "seed": seed, "selectors": ["disabled", "q_ceasc", "classical_ceasc"], "rows": rows, "test_used_for_training_or_selection": False, "provenance": {"git_revision": "abc", "git_dirty": False}}
        (seed_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_summary_computes_l2_gate_and_paired_ci(tmp_path: Path):
    summary = _load("qceasc_multiseed_summary", SUMMARY_PATH)
    group = tmp_path / "group"
    _write_group(group)
    payload = summary.collect(group)
    assert payload["gates"]["l2_reproducible_utility_gate"] is True
    assert payload["aggregate"]["q_ceasc"]["delta_test_macro_f1_vs_disabled"]["n"] == 3
    assert payload["aggregate"]["q_ceasc"]["delta_test_macro_f1_vs_disabled"]["ci95"][0] > 0


def test_summary_rejects_missing_sample_trace(tmp_path: Path):
    summary = _load("qceasc_multiseed_summary_missing_trace", SUMMARY_PATH)
    group = tmp_path / "group"
    _write_group(group)
    (group / "seed_29" / "selectors" / "q_ceasc" / "sample_trace.json").unlink()
    with pytest.raises(ValueError, match="sample trace"):
        summary.collect(group)


def test_summary_allows_scheduler_pre_marker_pass(tmp_path: Path):
    summary = _load("qceasc_multiseed_summary_pre_marker", SUMMARY_PATH)
    group = tmp_path / "group"
    _write_group(group)
    (group / "MULTI_SEED_COMPLETE").unlink()
    (group / "multi_seed_run_summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "tasks": [
                    {"task_id": f"task-{index}", "status": "complete"}
                    for index in range(9)
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = summary.collect(group)
    assert payload["claim_ceiling"] in {"L1_utility_candidate", "L2_reproducible_utility"}
