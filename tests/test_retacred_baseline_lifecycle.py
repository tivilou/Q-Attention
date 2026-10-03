from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ("check_retacred_baseline_publish_preflight.sh", "publish_retacred_baseline_report.sh",
         "run_retacred_baseline_complete.sh", "validate_retacred_baseline_checkpoint.py")

def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()

def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in FILES:
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    (root / ".gitignore").write_text("runs/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git(root, "init", "-b", "1.1")
    git(root, "config", "user.name", "Test Owner")
    git(root, "config", "user.email", "owner@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-m", "fixture")
    head = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", head)
    git(root, "update-ref", "refs/remotes/origin/1.1", head)
    return root

def call_preflight(root: Path):
    return subprocess.run(
        ["bash", str(root / "scripts/check_retacred_baseline_publish_preflight.sh"),
         "--run-dir", "runs/new-seed13", "--report-dir", "reports/retacred_baseline_complete/new-seed13"],
        cwd=root, text=True, capture_output=True, check=False,
    )

def test_publish_preflight_checks_branch_refs_tree_and_path_collisions(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    assert call_preflight(root).returncode == 0
    git(root, "switch", "-c", "feature")
    assert "branch 1.1" in call_preflight(root).stderr
    git(root, "switch", "1.1")
    (root / "dirty.txt").write_text("dirty", encoding="utf-8")
    assert "clean worktree" in call_preflight(root).stderr
    (root / "dirty.txt").unlink()
    (root / "reports/retacred_baseline_complete/new-seed13").mkdir(parents=True)
    assert "already exists" in call_preflight(root).stderr

@pytest.mark.parametrize("ref", ["main", "1.1"])
def test_publish_preflight_rejects_non_ancestor_refs(tmp_path: Path, ref: str) -> None:
    root = make_repo(tmp_path)
    git(root, "checkout", "--orphan", "unrelated")
    (root / "unrelated.txt").write_text("other history", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-m", "unrelated")
    unrelated = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "1.1")
    git(root, "update-ref", f"refs/remotes/origin/{ref}", unrelated)
    result = call_preflight(root)
    assert result.returncode != 0 and f"origin/{ref}" in result.stderr

def write_checkpoint(root: Path, seed: int = 13, dim: int = 128) -> Path:
    model = root / "runs/existing/baseline"
    model.mkdir(parents=True)
    args = {
        "train_path": "data/relation/retacred/train.jsonl", "valid_path": "data/relation/retacred/valid.jsonl",
        "epochs": 12, "batch_size": 128, "lr": 0.0005, "dim": dim, "num_layers": 4,
        "num_heads": 8, "ff_dim": 256, "dropout": 0.1, "max_length": 128,
        "selection_metric": "macro_f1_then_loss", "device": "cuda", "seed": seed,
    }
    (model / "model.pt").write_bytes(b"checkpoint")
    (model / "metrics.json").write_text(json.dumps({"args": args, "selection_metric": "macro_f1_then_loss",
        "best_epoch": 2, "best_valid": {"macro_f1": 0.2}}), encoding="utf-8")
    (model / "vocab.json").write_text('{"token": 1}', encoding="utf-8")
    (model / "labels.json").write_text('{"relation": 0}', encoding="utf-8")
    return model

def validate(root: Path, model: Path, seed: int = 13):
    return subprocess.run([sys.executable, str(ROOT / "scripts/validate_retacred_baseline_checkpoint.py"),
        "--root", str(root), "--model-dir", str(model), "--expected-seed", str(seed)],
        cwd=root, text=True, capture_output=True, check=False)

def test_checkpoint_validator_binds_seed_and_frozen_training_config(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    model = write_checkpoint(root)
    assert validate(root, model).returncode == 0
    assert "seed mismatch" in validate(root, model, 29).stderr
    metrics = json.loads((model / "metrics.json").read_text(encoding="utf-8"))
    metrics["args"]["dim"] = 64
    (model / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert "dim mismatch" in validate(root, model).stderr

def test_invalid_model_dir_fails_before_creating_run(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = subprocess.run(["bash", str(root / "scripts/run_retacred_baseline_complete.sh"),
        "--skip-preflight", "--skip-export", "--model-dir", str(root / "runs/missing"),
        "--output-dir", "runs/should-not-exist"], cwd=root, text=True, capture_output=True, check=False)
    assert result.returncode != 0 and "does not exist" in result.stderr
    assert not (root / "runs/should-not-exist").exists()

def fake_runtime(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text('#!/usr/bin/env bash\nif [[ "$1" == "experiments/train_relation_baseline.py" ]]; then exit 37; fi\nexit 0\n', encoding="utf-8")
    python.chmod(0o755)
    nvidia = bindir / "nvidia-smi"
    nvidia.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    nvidia.chmod(0o755)
    return python, os.environ | {"PYTHON_BIN": str(python), "PATH": f"{bindir}:{os.environ['PATH']}"}

def test_train_failure_is_marked_with_stage_and_history(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    python, env = fake_runtime(tmp_path)
    result = subprocess.run(["bash", str(root / "scripts/run_retacred_baseline_complete.sh"),
        "--skip-preflight", "--skip-export", "--output-dir", "runs/failed"], cwd=root,
        env=env, text=True, capture_output=True, check=False)
    run = root / "runs/failed"
    assert result.returncode == 37
    assert (run / "RUN_FAILED").is_file()
    status = (run / "status/run.env").read_text(encoding="utf-8")
    assert "STARTED_AT=" in status and "FAILED_STAGE=training" in status
    assert "status=failed stage=training" in (run / "status/history.log").read_text(encoding="utf-8")

def test_publish_retry_rechecks_staging_after_commit_failure(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run = root / "runs/run-seed13"
    (run / "status").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "RUN_COMPLETE").write_text("complete", encoding="utf-8")
    (run / "EXPORT_COMPLETE").write_text("exported", encoding="utf-8")
    report = root / "reports/retacred_baseline_complete/run-seed13"
    report.mkdir(parents=True)
    (report / "run_summary.md").write_text("report\n", encoding="utf-8")
    hook = root / ".git/hooks/pre-commit"
    hook.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    cmd = ["bash", str(root / "scripts/publish_retacred_baseline_report.sh"), "--run-dir", str(run), "--no-push"]
    first = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    assert first.returncode != 0 and not (run / "COMMIT_COMPLETE").exists()
    (root / "intruder.txt").write_text("not allowed\n", encoding="utf-8")
    git(root, "add", "intruder.txt")
    blocked = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    assert blocked.returncode != 0 and "outside the report" in blocked.stderr
    assert not (run / "COMMIT_COMPLETE").exists()
    git(root, "reset", "-q", "--", "intruder.txt")
    (root / "intruder.txt").unlink()
    hook.unlink()
    recovered = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    assert recovered.returncode == 0, recovered.stderr
    assert (run / "COMMIT_COMPLETE").is_file()
    history = (run / "status/history.log").read_text(encoding="utf-8")
    assert "status=failed stage=commit" in history and "status=push-skipped stage=push" in history
    assert git(root, "show", "--pretty=format:", "--name-only", "HEAD").strip() == "reports/retacred_baseline_complete/run-seed13/run_summary.md"

def test_skip_preflight_does_not_skip_publish_guards_but_raw_mode_is_branch_independent(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    git(root, "switch", "-c", "feature")
    python, env = fake_runtime(tmp_path)
    publishing = subprocess.run(["bash", str(root / "scripts/run_retacred_baseline_complete.sh"),
        "--skip-preflight", "--output-dir", "runs/blocked"], cwd=root, env=env, text=True, capture_output=True, check=False)
    assert publishing.returncode != 0 and "branch 1.1" in publishing.stderr
    assert not (root / "runs/blocked").exists()
    raw = subprocess.run(["bash", str(root / "scripts/run_retacred_baseline_complete.sh"),
        "--skip-preflight", "--skip-export", "--output-dir", "runs/raw"], cwd=root, env=env, text=True, capture_output=True, check=False)
    assert raw.returncode == 37
    assert "FAILED_STAGE=training" in (root / "runs/raw/status/run.env").read_text(encoding="utf-8")
