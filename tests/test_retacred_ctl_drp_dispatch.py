"""Actual suite entry/child diagnostic isolation; no formal experiment or transport."""
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import ctl_drp_preflight as gate, orchestrate
from baselines.retacred.common import ROOT, digest, select_models


def options(tmp_path, **overrides):
    args = gate.parser().parse_args([])
    args.cache = Path(os.environ["CTL_DRP_BPE_CACHE"])
    args.paper_source = Path(os.environ["CTL_DRP_NATIVE"])
    args.run_dir = tmp_path / "run"
    args.device = "cpu"
    for key, value in overrides.items():
        setattr(args, key, value)
    gate.validate_options(args)
    return args


def invoke(arguments):
    return subprocess.run(["bash", str(ROOT / "scripts/run_retacred_reference_suite.sh"),
                           "ctl-preflight", *map(str, arguments)], cwd=ROOT,
                          env={**os.environ, "PYTHON_BIN": sys.executable}, capture_output=True, text=True, timeout=100)


@pytest.mark.parametrize("mode", ("tiny", "pretrained", "data"))
def test_actual_unified_descriptor_backend_free_no_effects(tmp_path, mode):
    done = invoke(["--describe", "--mode", mode])
    assert done.returncode == 0, done.stderr
    value = json.loads(done.stdout)
    assert value["formal_training"] is value["publishing"] is value["downloads"] is False
    assert value["diagnostic_updates"] == {"tiny": 2, "pretrained": 1, "data": 0}[mode]
    # Import hook demonstrates that description has no heavy backend import.
    script = """import builtins,sys
old=builtins.__import__
def block(name,*a,**k):
 if name.split('.')[0] in ('torch','transformers','numpy','tokenizers'):raise AssertionError(name)
 return old(name,*a,**k)
builtins.__import__=block
from baselines.retacred import ctl_drp_preflight as gate
assert gate.main(['--describe','--mode',sys.argv[1]])==0
"""
    checked = subprocess.run([sys.executable, "-c", script, mode], cwd=ROOT, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert not list(tmp_path.iterdir())


def test_formal_admission_and_marker_command_remain_unchanged(tmp_path):
    assert len(select_models("ready")) == 5
    for name in ("all", "ctl_drp_supervised", "ctl_drp_unsupervised"):
        with pytest.raises(ValueError, match="not runnable"):
            select_models(name)
    from argparse import Namespace
    args = Namespace(micro_batch=4, logical_batch=64, checkpointing=False, canary=False)
    job = {"model": "typed_marker", "seed": 13, "directory": tmp_path / "marker"}
    command = orchestrate.build_command(job, args, Path("asset"), Path("data"))
    assert command[:3] == [sys.executable, "-m", "baselines.retacred.train"]
    assert command[command.index("--logical-batch") + 1] == "64"
    assert "ctl_drp" not in " ".join(command)


@pytest.mark.parametrize("case", ("weight_in_tiny", "cpu_pretrained", "real_data_in_tiny", "public_output", "missing_receipt", "invalid_micro"))
def test_invalid_mixed_scopes_fail_before_files_or_child(tmp_path, case, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("child launched"))
    bad = {
        "weight_in_tiny": {"weight": tmp_path / "weight"},
        "cpu_pretrained": {"mode": "pretrained", "weight": tmp_path / "weight"},
        "real_data_in_tiny": {"qualification": tmp_path / "qualification"},
        "public_output": {"run_dir": ROOT / "reports/forbidden"},
        "missing_receipt": {"mode": "data"},
        "invalid_micro": {"micro_batch": 513},
    }[case]
    with pytest.raises(ValueError):
        options(tmp_path, **bad)
    assert not list(tmp_path.iterdir())


def test_wrong_weight_rejected_before_backend_or_control_directory(tmp_path, monkeypatch):
    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"not the official model")
    args = options(tmp_path, mode="pretrained", device="cuda", weight=weight)
    old = builtins.__import__
    def block(name, *a, **k):
        if name.split('.')[0] == "torch":
            pytest.fail("backend imported before bad asset rejection")
        return old(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ValueError, match="size/hash"):
        gate.dispatch(args, [])
    assert not args.run_dir.exists() and not args.run_dir.with_name("run.ctl-control").exists()


def test_conservative_disk_gate_before_heavy_child(tmp_path, monkeypatch):
    from collections import namedtuple
    args = options(tmp_path)
    disk = namedtuple("disk", "total used free")
    monkeypatch.setattr(gate.shutil, "disk_usage", lambda _: disk(1, 1, 0))
    with pytest.raises(ValueError, match="disk headroom"):
        gate.require_disk(args, {"size_bytes": 100})


def test_actual_unified_cpu_pause_resume_export_and_parent_integrity(tmp_path):
    base = Path(os.environ.get("CTL_DRP_DISPATCH_EVIDENCE", str(tmp_path))) / "unified-cpu"
    args = ["--mode", "tiny", "--device", "cpu", "--cache", os.environ["CTL_DRP_BPE_CACHE"],
            "--paper-source", os.environ["CTL_DRP_NATIVE"], "--run-dir", base,
            "--local-export-dir", base.with_name("unified-cpu-report")]
    paused = invoke([*args, "--pause-after", "1"])
    assert paused.returncode == 75, paused.stderr + paused.stdout
    complete = invoke([*args, "--resume"])
    assert complete.returncode == 0, complete.stderr + complete.stdout
    assert "2/2" in complete.stdout and "1/2" not in complete.stdout
    assert (base / "DIAGNOSTIC_COMPLETE").is_file() and not (base / "RUN_COMPLETE").exists()
    parent = json.loads((base.with_name("unified-cpu.ctl-control") / "command.json").read_text())
    assert parent["exit_status"] == parent["child_exit_status"] == 0 and parent["selected_gpu"] == "-1"
    assert parent["argv"][:4] == [sys.executable, "-u", "-m", "baselines.retacred.ctl_drp_preflight"]
    assert not (base / "COMPUTE_COMPLETE").exists()
    from baselines.retacred.ctl_drp_export import validate_report
    assert validate_report(base.with_name("unified-cpu-report"))["files"] == 6
    wrong = invoke([*args, "--resume", "--seed", "29"])
    assert wrong.returncode == 1
    assert digest(base / "run_summary.json") == parent_result_hash(base)


def parent_result_hash(base):
    # Export run_identity is the original immutable computation summary hash.
    return json.loads((base.with_name("unified-cpu-report") / "export_manifest.json").read_text())["run_identity"]


def test_actual_one_selected_gpu_child_and_complete_numeric_trace(tmp_path):
    base = tmp_path / "gpu"
    done = invoke(["--mode", "tiny", "--gpu", "auto", "--updates", "1",
                   "--cache", os.environ["CTL_DRP_BPE_CACHE"], "--paper-source", os.environ["CTL_DRP_NATIVE"],
                   "--run-dir", base])
    assert done.returncode == 0, done.stdout + done.stderr
    parent = json.loads((base.with_name("gpu.ctl-control") / "command.json").read_text())
    contract = json.loads((base / "contract.json").read_text())
    assert contract["runtime"]["visible_devices"] == parent["selected_gpu"]
    assert contract["runtime"]["device_uuid"] and contract["budget_updates"] == 1
    gate.validate_child(options(tmp_path, run_dir=base, device="cuda", updates=1))


def test_actual_data_only_physical_audit_and_no_qualification_repair(tmp_path):
    from baselines.retacred.ctl_drp_runner import synthetic_receipt
    receipt = synthetic_receipt(tmp_path / "physical")
    before = digest(receipt)
    done = invoke(["--mode", "data", "--qualification", receipt, "--cache", os.environ["CTL_DRP_BPE_CACHE"],
                   "--paper-source", os.environ["CTL_DRP_NATIVE"], "--run-dir", tmp_path / "audit"])
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((tmp_path / "audit/data-preflight.json").read_text())
    assert result["status"] == "diagnostic_data_verified" and result["data_identity"]["synthetic"]
    assert result["training_executed"] is result["backend_loaded"] is False
    assert result["encoding_audit"]["train"]["sentences"] == 3
    assert not (tmp_path / "audit/checkpoints").exists() and digest(receipt) == before


def test_actual_timeout_reaps_only_own_child_and_records_literal_exit(tmp_path, monkeypatch):
    args = options(tmp_path, timeout_seconds=1)
    monkeypatch.setattr(gate, "build_worker_command", lambda _: [sys.executable, "-c", "import time;time.sleep(20)"])
    assert gate.dispatch(args, []) == 124
    record = json.loads((tmp_path / "run.ctl-control/command.json").read_text())
    assert record["status"] == "failed" and record["exit_status"] == 124 and record["child_exit_status"] != 0
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)
    assert not args.run_dir.exists()


def test_success_exit_without_real_numeric_evidence_is_rejected(tmp_path, monkeypatch):
    args = options(tmp_path)
    monkeypatch.setattr(gate, "build_worker_command", lambda _: [sys.executable, "-c", "print('fake completion')"])
    with pytest.raises(FileNotFoundError):
        gate.dispatch(args, [])
    record = json.loads((tmp_path / "run.ctl-control/command.json").read_text())
    assert record["status"] == "failed" and record["exit_status"] == 1 and record["child_exit_status"] == 0
