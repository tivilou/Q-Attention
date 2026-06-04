from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.metrics import classification_metrics
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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


def evaluate(model: RelationExtractionModel, loader: DataLoader, device: torch.device, num_labels: int) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
            loss = F.cross_entropy(logits, batch["labels"])
            total_loss += float(loss.item()) * batch["labels"].shape[0]
            total_items += batch["labels"].shape[0]
            predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def main() -> None:
    args = parse_args()
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
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_relation_batch(batch, pad_id=vocab[PAD_TOKEN]),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_relation_batch(batch, pad_id=vocab[PAD_TOKEN]),
    )

    max_length = max(max(len(record.tokens) for record in train_records), max(len(record.tokens) for record in valid_records))
    config = RelationTransformerConfig(
        vocab_size=len(vocab),
        num_labels=len(label_to_id),
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_length=max(8, max_length + 4),
    )
    model = RelationExtractionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] | None = None
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch["labels"].shape[0]
            total_items += batch["labels"].shape[0]

        valid_metrics = evaluate(model, valid_loader, device, len(label_to_id))
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            "valid": valid_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))
        if valid_metrics["macro_f1"] > best_score:
            best_score = valid_metrics["macro_f1"]
            best_metrics = valid_metrics
            torch.save(model.state_dict(), output_dir / "model.pt")

    payload = {
        "args": vars(args),
        "vocab": vocab,
        "label_to_id": label_to_id,
        "best_valid": best_metrics,
        "history": history,
        "key_module_paths": model.key_module_paths,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps(label_to_id, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best_valid": best_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
