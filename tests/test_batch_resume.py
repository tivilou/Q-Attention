from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

from q_attention.experiments.batch_resume import (
    BatchCheckpointManager,
    BatchCursor,
    RemainingBatchSampler,
    ResumeCompatibilityError,
    atomic_torch_save,
    epoch_permutation,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import train_relation_baseline as baseline_trainer


def test_epoch_permutation_and_remaining_sampler_are_deterministic() -> None:
    permutation = epoch_permutation(11, seed=13, epoch=2)
    assert permutation == epoch_permutation(11, seed=13, epoch=2)
    sampler = RemainingBatchSampler(permutation, batch_size=4, next_batch_index=1)
    assert list(sampler) == [permutation[4:8], permutation[8:]]
    cursor = BatchCursor.fresh(dataset_size=11, seed=13)
    assert cursor.permutation == epoch_permutation(11, seed=13, epoch=1)


def test_checkpoint_is_atomic_and_contract_mismatch_is_rejected(tmp_path) -> None:
    contract = {"stage": "toy", "seed": 13, "batch_size": 2}
    manager = BatchCheckpointManager(tmp_path, contract=contract, resume=False)
    manager.save({"model_state": {"weight": torch.ones(1)}, "cursor": {"global_step": 1}})
    payload = manager.load()
    assert payload["schema_version"] == "q-attention.batch-resume.v1"
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ResumeCompatibilityError):
        BatchCheckpointManager(tmp_path, contract={**contract, "batch_size": 4}, resume=True)


def test_pause_marker_records_post_update_cursor(tmp_path) -> None:
    manager = BatchCheckpointManager(tmp_path, contract={"stage": "toy"}, resume=False)
    cursor = BatchCursor.fresh(dataset_size=5, seed=13)
    cursor.next_batch_index = 2
    cursor.global_step = 2
    manager.save({"cursor": cursor.payload()})
    manager.write_paused_marker(stage="toy", cursor=cursor, reason="SIGTERM")
    marker = json.loads((tmp_path / "RUN_PAUSED").read_text(encoding="utf-8"))
    assert marker["stage"] == "toy"
    assert marker["next_batch_index"] == 2
    assert marker["global_step"] == 2


def test_atomic_torch_save_leaves_readable_checkpoint(tmp_path) -> None:
    path = tmp_path / "checkpoints" / "latest.pt"
    atomic_torch_save(path, {"value": torch.arange(3)})
    assert torch.equal(torch.load(path, map_location="cpu", weights_only=True)["value"], torch.arange(3))


def _baseline_argv(output_dir: Path, *, resume: bool = False) -> list[str]:
    arguments = [
        "train_relation_baseline.py",
        "--train_path", str(ROOT / "examples" / "relation_toy_train.jsonl"),
        "--valid_path", str(ROOT / "examples" / "relation_toy_valid.jsonl"),
        "--output_dir", str(output_dir),
        "--device", "cpu",
        "--epochs", "2",
        "--batch_size", "2",
        "--dim", "16",
        "--num_layers", "1",
        "--num_heads", "2",
        "--ff_dim", "32",
        "--dropout", "0.2",
        "--seed", "17",
        "--log_every_batches", "100",
        "--checkpoint-every-batches", "1",
    ]
    if resume:
        arguments.append("--resume")
    return arguments


def _checkpoint_payload(output_dir: Path) -> dict[str, object]:
    return torch.load(
        output_dir / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )


def test_baseline_resume_matches_uninterrupted_training_after_post_update_pause(
    tmp_path, monkeypatch
) -> None:
    """A paused run must be bitwise-identical to uninterrupted CPU training."""

    class NeverPause:
        requested = False
        reason = None

        def install(self) -> None:
            return None

        def close(self) -> None:
            return None

    uninterrupted_dir = tmp_path / "uninterrupted"
    monkeypatch.setattr(baseline_trainer, "PauseController", NeverPause)
    monkeypatch.setattr(sys, "argv", _baseline_argv(uninterrupted_dir))
    assert baseline_trainer.main() == 0
    uninterrupted = _checkpoint_payload(uninterrupted_dir)

    updates = {"count": 0}
    original_require_finite_parameters = baseline_trainer.require_finite_parameters

    def count_completed_updates(*args, **kwargs) -> None:
        original_require_finite_parameters(*args, **kwargs)
        updates["count"] += 1

    class PauseAfterSecondUpdate:
        reason = "test_pause"

        @property
        def requested(self) -> bool:
            return updates["count"] >= 2

        def install(self) -> None:
            return None

        def close(self) -> None:
            return None

    resumed_dir = tmp_path / "resumed"
    monkeypatch.setattr(
        baseline_trainer, "require_finite_parameters", count_completed_updates
    )
    monkeypatch.setattr(baseline_trainer, "PauseController", PauseAfterSecondUpdate)
    monkeypatch.setattr(sys, "argv", _baseline_argv(resumed_dir))
    assert baseline_trainer.main() == 75
    paused = _checkpoint_payload(resumed_dir)
    assert paused["cursor"]["global_step"] == 2
    assert paused["cursor"]["next_batch_index"] == 2
    assert (resumed_dir / "RUN_PAUSED").is_file()
    assert not (resumed_dir / "metrics.json").exists()

    monkeypatch.setattr(baseline_trainer, "PauseController", NeverPause)
    monkeypatch.setattr(sys, "argv", _baseline_argv(resumed_dir, resume=True))
    assert baseline_trainer.main() == 0
    resumed = _checkpoint_payload(resumed_dir)

    assert resumed["cursor"] == uninterrupted["cursor"]
    assert resumed["history"] == uninterrupted["history"]
    for name, tensor in uninterrupted["model_state"].items():
        assert torch.equal(tensor, resumed["model_state"][name]), name
    assert not (resumed_dir / "RUN_PAUSED").exists()


def test_baseline_elastic_resume_is_execution_only_contract_change(monkeypatch) -> None:
    base_argv = _baseline_argv(Path("unused"))
    monkeypatch.setattr(sys, "argv", base_argv)
    regular = baseline_trainer.parse_args()
    monkeypatch.setattr(sys, "argv", [*base_argv, "--elastic-resume"])
    elastic = baseline_trainer.parse_args()

    assert baseline_trainer._resume_contract(regular) == baseline_trainer._resume_contract(elastic)

    migrated = baseline_trainer._resume_contract(elastic)
    migrated["source"]["git_revision"] = "different-revision"
    migrated["source"]["files"]["trainer"] = "different-trainer-source"
    assert baseline_trainer._resume_contract_compatible(
        baseline_trainer._resume_contract(regular), migrated
    )
