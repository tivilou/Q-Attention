from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.evaluate_relation_baseline import main
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.tasks.relation import build_label_map, build_vocab, load_relation_jsonl


def test_evaluator_writes_complete_toy_report(tmp_path: Path, monkeypatch) -> None:
    train_path = Path("examples/relation_toy_train.jsonl")
    valid_path = Path("examples/relation_toy_valid.jsonl")
    test_path = Path("examples/relation_toy_test.jsonl")
    records = load_relation_jsonl(train_path)
    vocab = build_vocab(records)
    label_to_id = build_label_map(records)
    model_dir = tmp_path / "baseline"
    model_dir.mkdir()
    config = RelationTransformerConfig(
        vocab_size=len(vocab),
        num_labels=len(label_to_id),
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        max_length=16,
    )
    torch.manual_seed(13)
    model = RelationExtractionModel(config)
    torch.save(model.state_dict(), model_dir / "model.pt")
    (model_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (model_dir / "labels.json").write_text(json.dumps(label_to_id), encoding="utf-8")
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "args": {
                    "seed": 13,
                    "dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ff_dim": 16,
                    "dropout": 0.0,
                },
                "selection_metric": "macro_f1_then_loss",
                "best_epoch": 1,
                "key_module_paths": list(model.key_module_paths),
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_relation_baseline.py",
            "--model-dir",
            str(model_dir),
            "--train-path",
            str(train_path),
            "--valid-path",
            str(valid_path),
            "--test-path",
            str(test_path),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
    )
    main()

    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    required = {
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "loss",
    }
    assert summary["status"] == "complete"
    assert set(summary["splits"]) == {"train", "valid", "test"}
    for split in summary["splits"].values():
        assert required.issubset(split["metrics"])
        assert split["per_class"]
        assert split["confusion_matrix"]
    assert (output_dir / "run_summary.data").stat().st_size > 0
    assert (output_dir / "run_summary.md").is_file()
