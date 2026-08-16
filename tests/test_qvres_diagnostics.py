from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from diagnose_qvres_relation_transfer import (  # noqa: E402
    relation_metrics,
    resolve_run_layout,
    run_diagnostics,
    spearman,
    topk_overlap,
)
from run_q_causal_value_evidence_relation_transfer import build_kernel  # noqa: E402
from q_attention.models import RelationExtractionModel, RelationTransformerConfig  # noqa: E402
from q_attention.tasks.relation import RelationRecord, write_relation_jsonl  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_serial_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    baseline_dir = run_dir / "baseline"
    baseline_dir.mkdir(parents=True)
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "alice": 2,
        "works": 3,
        "at": 4,
        "acme": 5,
        "bob": 6,
        "visits": 7,
    }
    labels = {"employee_of": 0, "no_relation": 1}
    config = RelationTransformerConfig(
        vocab_size=len(vocab),
        num_labels=len(labels),
        dim=4,
        num_layers=1,
        num_heads=1,
        ff_dim=8,
        dropout=0.0,
        max_length=12,
    )
    torch.manual_seed(7)
    model = RelationExtractionModel(config)
    torch.save(model.state_dict(), baseline_dir / "model.pt")
    _write_json(baseline_dir / "vocab.json", vocab)
    _write_json(baseline_dir / "labels.json", labels)
    _write_json(
        baseline_dir / "metrics.json",
        {
            "args": {
                "dim": 4,
                "num_layers": 1,
                "num_heads": 1,
                "ff_dim": 8,
                "dropout": 0.0,
            },
            "key_module_paths": list(model.key_module_paths),
        },
    )
    records = [
        RelationRecord(("Alice", "works", "at", "Acme"), (0, 1), (3, 4), "employee_of"),
        RelationRecord(("Bob", "visits", "Acme"), (0, 1), (2, 3), "no_relation"),
        RelationRecord(("Bob", "works", "at", "Acme"), (0, 1), (3, 4), "employee_of"),
        RelationRecord(("Alice", "visits", "Acme"), (0, 1), (2, 3), "no_relation"),
    ]
    write_relation_jsonl(records, run_dir / "screen_data" / "valid.jsonl")
    run_config = {
        "seed": 13,
        "register_qubits": 2,
        "depth": 1,
        "angle_scale": 1.0,
        "max_transport": 0.25,
        "initial_transport": 0.05,
        "evidence_floor": 1e-6,
    }
    _write_json(run_dir / "run_config.json", run_config)
    kernel_args = argparse.Namespace(**run_config)
    for selector in ("q_causal_transport", "q_causal_key_only"):
        selector_dir = run_dir / "selectors" / selector
        selector_dir.mkdir(parents=True)
        kernel = build_kernel(selector, model, 13, kernel_args)
        assert kernel is not None
        torch.save(kernel.state_dict(), selector_dir / "best_kernel.pt")
    return run_dir


def test_rank_metrics_handle_ties_and_topk() -> None:
    assert spearman([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0], [2.0, 3.0]) is None
    assert topk_overlap([3.0, 2.0, 1.0], [5.0, 4.0, 0.0]) == 1.0


def test_relation_metrics_report_false_positive_and_false_negative() -> None:
    rows = relation_metrics([0, 0, 1, 1], [0, 1, 1, 1], {0: "a", 1: "b"})
    assert rows["a"]["support"] == 1
    assert rows["a"]["false_positive"] == 1
    assert rows["b"]["false_negative"] == 1
    assert rows["b"]["recall"] == pytest.approx(2 / 3)


def test_resolve_selector_parallel_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "parallel"
    baseline = run_dir / "stages" / "baseline"
    (baseline / "baseline").mkdir(parents=True)
    for name in ("model.pt", "metrics.json"):
        (baseline / "baseline" / name).write_text("fixture", encoding="utf-8")
    (baseline / "screen_data").mkdir()
    (baseline / "screen_data" / "valid.jsonl").write_text("{}\n", encoding="utf-8")
    layout = resolve_run_layout(run_dir, "valid")
    assert layout.parallel_mode == "selectors"
    assert layout.baseline_dir == (baseline / "baseline").resolve()
    assert layout.selector_dirs["q_causal_transport"] == (
        run_dir / "stages" / "q_causal_transport" / "selectors" / "q_causal_transport"
    )


def test_diagnostic_smoke_writes_only_aggregate_outputs(tmp_path: Path) -> None:
    run_dir = _make_serial_run(tmp_path)
    output_dir = tmp_path / "diagnostic"
    args = argparse.Namespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        split="valid",
        allow_test_report=False,
        device="cpu",
        batch_size=2,
        max_records=0,
        sample_seed=13,
        log_every_batches=100,
        selectors="q_causal_transport,q_causal_key_only",
        attribution_selectors="q_causal_transport,q_causal_key_only",
    )
    summary = run_diagnostics(args)
    assert summary["records"] == 4
    assert summary["source_run_name"] == "run"
    assert str(run_dir) not in json.dumps(summary)
    assert summary["privacy"] == {
        "contains_raw_text": False,
        "contains_per_example_predictions": False,
        "contains_tokens": False,
        "contains_checkpoints": False,
        "contains_gradient_tensors": False,
    }
    assert set(summary["selectors"]) == {
        "disabled",
        "q_causal_transport",
        "q_causal_key_only",
    }
    assert summary["selectors"]["q_causal_transport"]["mechanism"]["layers"]
    assert (output_dir / "diagnostic_summary.json").is_file()
    assert (output_dir / "diagnostic_summary.md").is_file()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "diagnostic_summary.json",
        "diagnostic_summary.md",
    ]


def test_test_split_requires_explicit_reporting_acknowledgement(tmp_path: Path) -> None:
    args = argparse.Namespace(
        run_dir=str(tmp_path),
        output_dir=None,
        split="test",
        allow_test_report=False,
        device="cpu",
        batch_size=2,
        max_records=0,
        sample_seed=13,
        log_every_batches=100,
        selectors="q_causal_transport",
        attribution_selectors="q_causal_transport",
    )
    with pytest.raises(ValueError, match="must not drive method selection"):
        run_diagnostics(args)
