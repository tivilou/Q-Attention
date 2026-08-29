from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BASELINE_ELASTIC_SOURCE_FILES = ("trainer", "batch_resume")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.metrics import classification_metrics, correct_label_margin
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.experiments import RELATION_SELECTION_CHOICES, relation_selection_score
from q_attention.experiments.health import (
    EpochHealthMonitor,
    require_finite_gradients,
    require_finite_parameters,
    require_finite_tensor,
    require_finite_values,
)
from q_attention.experiments.progress import tracked_batches
from q_attention.experiments.batch_resume import (
    PAUSED_EXIT_CODE,
    BatchCheckpointManager,
    BatchCursor,
    PauseController,
    RemainingBatchSampler,
    TrainingPaused,
    capture_rng_state,
    execution_contract_compatible,
    file_contract,
    restore_rng_state,
    sha256_file,
)
from q_attention.tasks.relation import (
    PAD_TOKEN,
    RelationDataset,
    build_label_map,
    build_vocab,
    collate_relation_batch,
    load_relation_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small relation extraction baseline.")
    parser.add_argument("--train_path", default="examples/relation_toy_train.jsonl")
    parser.add_argument("--valid_path", default="examples/relation_toy_valid.jsonl")
    parser.add_argument("--output_dir", default="runs/relation_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--log_every_batches", type=int, default=50)
    parser.add_argument("--health_warning_patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Fixed position-embedding capacity; defaults to observed train/valid length plus four.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume this exact output directory from its compatible batch checkpoint",
    )
    parser.add_argument(
        "--elastic-resume",
        action="store_true",
        help="allow an explicit execution-only contract migration during resume",
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=50,
        help="write an atomic post-update checkpoint at this batch interval",
    )
    parser.add_argument(
        "--selection_metric",
        choices=RELATION_SELECTION_CHOICES,
        default="macro_f1_then_loss",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--model-parallel-gpus",
        default=None,
        help="comma-separated physical GPU IDs for layer-sharded model parallelism",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_model_parallel_devices(spec: str) -> tuple[torch.device, ...]:
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) < 2 or any(not field.isdigit() for field in fields):
        raise ValueError("--model-parallel-gpus must contain at least two GPU IDs")
    physical_ids = [int(field) for field in fields]
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("--model-parallel-gpus must not contain duplicate GPU IDs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_fields = [field.strip() for field in visible.split(",")]
        if all(field.isdigit() for field in visible_fields):
            mapping = {int(field): index for index, field in enumerate(visible_fields)}
            missing = [gpu_id for gpu_id in physical_ids if gpu_id not in mapping]
            if missing:
                raise ValueError(
                    f"model-parallel GPUs {missing} are not visible in CUDA_VISIBLE_DEVICES={visible}"
                )
            return tuple(torch.device(f"cuda:{mapping[gpu_id]}") for gpu_id in physical_ids)
    available = torch.cuda.device_count()
    if any(gpu_id >= available for gpu_id in physical_ids):
        raise ValueError(
            f"model-parallel GPUs {physical_ids} exceed visible device count {available}"
        )
    return tuple(torch.device(f"cuda:{gpu_id}") for gpu_id in physical_ids)


def evaluate(
    model: RelationExtractionModel,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_labels: int,
) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_margin = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
            loss = F.cross_entropy(logits, batch["labels"].to(logits.device))
            total_loss += float(loss.item()) * batch["labels"].shape[0]
            total_margin += float(
                correct_label_margin(logits, batch["labels"].to(logits.device)).sum().item()
            )
            total_items += batch["labels"].shape[0]
            predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["correct_label_margin"] = total_margin / max(total_items, 1)
    return metrics


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _restore_health_monitor(
    summary: dict[str, Any], *, patience: int
) -> EpochHealthMonitor:
    monitor = EpochHealthMonitor(
        "baseline",
        patience=int(summary.get("patience", patience)),
        min_delta=float(summary.get("min_delta", 1e-6)),
    )
    monitor.best_loss = summary.get("best_loss")
    monitor.no_improvement_epochs = int(summary.get("no_improvement_epochs", 0))
    monitor.consecutive_mechanism_failures = int(
        summary.get("consecutive_mechanism_failures", 0)
    )
    monitor.warnings = list(summary.get("warnings", []))
    return monitor


def _resume_contract(args: argparse.Namespace) -> dict[str, Any]:
    semantic_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {
            "output_dir",
            "resume",
            "elastic_resume",
            "checkpoint_every_batches",
            "log_every_batches",
        }
    }
    return {
        "stage": "relation_baseline",
        "training_semantics": semantic_args,
        "data": {
            "train": file_contract(args.train_path),
            "valid": file_contract(args.valid_path),
        },
        "source": {
            "git_revision": _git_revision(),
            "files": {
                name: sha256_file(path)
                for name, path in {
                    "trainer": Path(__file__),
                    "batch_resume": SRC
                    / "q_attention"
                    / "experiments"
                    / "batch_resume.py",
                    "progress": SRC
                    / "q_attention"
                    / "experiments"
                    / "progress.py",
                    "health": SRC
                    / "q_attention"
                    / "experiments"
                    / "health.py",
                    "relation_model": SRC
                    / "q_attention"
                    / "models"
                    / "relation_transformer.py",
                    "relation_task": SRC
                    / "q_attention"
                    / "tasks"
                    / "relation.py",
                }.items()
            },
        },
    }


def _resume_contract_compatible(
    persisted: Any, current: dict[str, Any]
) -> bool:
    return execution_contract_compatible(
        persisted,
        current,
        ignored_source_files=BASELINE_ELASTIC_SOURCE_FILES,
    )


def main() -> int:
    args = parse_args()
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("max_length must be positive when provided")
    if args.log_every_batches <= 0:
        raise ValueError("log_every_batches must be positive")
    if args.health_warning_patience <= 0:
        raise ValueError("health_warning_patience must be positive")
    if args.checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be positive")
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_relation_jsonl(args.train_path)
    valid_records = load_relation_jsonl(args.valid_path)
    vocab = build_vocab(train_records)
    label_to_id = build_label_map(train_records)

    train_data = RelationDataset(train_records, vocab, label_to_id)
    valid_data = RelationDataset(valid_records, vocab, label_to_id)
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_relation_batch(batch, pad_id=vocab[PAD_TOKEN]),
    )

    observed_max_length = max(
        max(len(record.tokens) for record in train_records),
        max(len(record.tokens) for record in valid_records),
    )
    max_length = args.max_length or max(8, observed_max_length + 4)
    if max_length < observed_max_length:
        raise ValueError("max_length is smaller than an observed train/valid sequence")
    config = RelationTransformerConfig(
        vocab_size=len(vocab),
        num_labels=len(label_to_id),
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_length=max_length,
    )
    model = RelationExtractionModel(config)
    model_parallel_devices: tuple[torch.device, ...] = ()
    if args.model_parallel_gpus is not None:
        if args.device == "cpu":
            raise ValueError("model parallelism requires CUDA")
        model_parallel_devices = resolve_model_parallel_devices(args.model_parallel_gpus)
        model.configure_model_parallel(model_parallel_devices)
        device = model_parallel_devices[0]
    else:
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    health_monitor = EpochHealthMonitor("baseline", patience=args.health_warning_patience)
    best_metrics: dict[str, float] | None = None
    best_epoch: int | None = None
    best_score = (float("-inf"), float("-inf"))
    manager = BatchCheckpointManager(
        output_dir,
        contract=_resume_contract(args),
        resume=args.resume,
        resume_contract_compatible=(
            _resume_contract_compatible if args.elastic_resume else None
        ),
    )
    if args.resume:
        checkpoint = manager.load()
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng_state(checkpoint["rng_state"])
        cursor = BatchCursor.from_payload(checkpoint["cursor"])
        history = list(checkpoint.get("history", []))
        best_metrics = checkpoint.get("best_metrics")
        best_epoch = checkpoint.get("best_epoch")
        best_score = tuple(checkpoint.get("best_score", best_score))
        health_monitor = _restore_health_monitor(
            dict(checkpoint.get("health", {})), patience=args.health_warning_patience
        )
        manager.clear_pause_marker()
        print(
            json.dumps(
                {
                    "event": "resume_loaded",
                    "stage": "baseline",
                    "epoch": cursor.epoch,
                    "next_batch_index": cursor.next_batch_index,
                    "global_step": cursor.global_step,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        cursor = BatchCursor.fresh(dataset_size=len(train_data), seed=args.seed)

    pause = PauseController()
    pause.install()

    def save_checkpoint() -> None:
        manager.save(
            {
                "stage": "baseline",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cursor": cursor.payload(),
                "rng_state": capture_rng_state(),
                "history": history,
                "best_metrics": best_metrics,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "health": health_monitor.summary(),
            }
        )

    if not args.resume:
        save_checkpoint()
    if pause.requested:
        manager.write_paused_marker(
            stage="baseline", cursor=cursor, reason=pause.reason
        )
        print(
            json.dumps(
                {"event": "run_paused", "stage": "baseline", "global_step": cursor.global_step},
                sort_keys=True,
            ),
            flush=True,
        )
        pause.close()
        return PAUSED_EXIT_CODE

    try:
        while cursor.epoch <= args.epochs:
            epoch = cursor.epoch
            model.train()
            total_batches = (len(train_data) + args.batch_size - 1) // args.batch_size
            train_loader = DataLoader(
                train_data,
                batch_sampler=RemainingBatchSampler(
                    cursor.permutation, args.batch_size, cursor.next_batch_index
                ),
                generator=torch.Generator(device="cpu").manual_seed(
                    args.seed * 1_000_003 + epoch
                ),
                collate_fn=lambda batch: collate_relation_batch(
                    batch, pad_id=vocab[PAD_TOKEN]
                ),
            )
            for batch_index, batch in enumerate(
                tracked_batches(
                    train_loader,
                    total_batches=total_batches,
                    stage="baseline",
                    phase="train",
                    log_every_batches=args.log_every_batches,
                    epoch=epoch,
                    epochs=args.epochs,
                    completed_batches=cursor.next_batch_index,
                ),
                start=cursor.next_batch_index,
            ):
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["subject_mask"],
                    batch["object_mask"],
                )
                loss = F.cross_entropy(logits, batch["labels"].to(logits.device))
                require_finite_tensor(
                    loss,
                    "objective",
                    stage="baseline",
                    epoch=epoch,
                    batch_index=batch_index,
                )
                loss.backward()
                require_finite_gradients(
                    model,
                    stage="baseline",
                    epoch=epoch,
                    batch_index=batch_index,
                )
                optimizer.step()
                require_finite_parameters(
                    model,
                    stage="baseline",
                    epoch=epoch,
                    batch_index=batch_index,
                )
                cursor.total_loss += float(loss.item()) * batch["labels"].shape[0]
                cursor.total_items += batch["labels"].shape[0]
                cursor.next_batch_index += 1
                cursor.global_step += 1
                if (
                    cursor.next_batch_index % args.checkpoint_every_batches == 0
                    or cursor.next_batch_index == total_batches
                    or pause.requested
                ):
                    save_checkpoint()
                if pause.requested:
                    manager.write_paused_marker(
                        stage="baseline", cursor=cursor, reason=pause.reason
                    )
                    raise TrainingPaused(
                        "baseline pause requested after optimizer update"
                    )

            valid_metrics = evaluate(
                model,
                tracked_batches(
                    valid_loader,
                    total_batches=len(valid_loader),
                    stage="baseline",
                    phase="validation",
                    log_every_batches=args.log_every_batches,
                    epoch=epoch,
                    epochs=args.epochs,
                ),
                device,
                len(label_to_id),
            )
            require_finite_values(valid_metrics, f"baseline.valid.epoch_{epoch}")
            health = health_monitor.observe(
                epoch=epoch, valid_loss=valid_metrics["loss"]
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": cursor.total_loss / max(cursor.total_items, 1),
                "valid": valid_metrics,
                "health": health,
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record, sort_keys=True), flush=True)
            score = relation_selection_score(valid_metrics, args.selection_metric)
            if score > best_score:
                best_score = score
                best_metrics = valid_metrics
                best_epoch = epoch
                torch.save(model.state_dict(), output_dir / "model.pt")
            cursor.advance_epoch(dataset_size=len(train_data), seed=args.seed)
            save_checkpoint()
            if pause.requested:
                manager.write_paused_marker(
                    stage="baseline", cursor=cursor, reason=pause.reason
                )
                raise TrainingPaused(
                    "baseline pause requested after epoch checkpoint"
                )

    except TrainingPaused:
        print(
            json.dumps(
                {"event": "run_paused", "stage": "baseline", "global_step": cursor.global_step},
                sort_keys=True,
            ),
            flush=True,
        )
        return PAUSED_EXIT_CODE
    finally:
        pause.close()

    payload = {
        "args": vars(args),
        "vocab": vocab,
        "label_to_id": label_to_id,
        "best_valid": best_metrics,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "history": history,
        "health": health_monitor.summary(),
        "key_module_paths": model.key_module_paths,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps(label_to_id, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": str(output_dir), "best_valid": best_metrics},
            sort_keys=True,
        ),
        flush=True,
    )
    (output_dir / "RUN_PAUSED").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
