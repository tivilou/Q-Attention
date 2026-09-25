from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.tasks.relation import RelationRecord


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterfactual_formal_config_builds_both_selector_kernels():
    runner = _load(
        "qceasc_grouped_counterfactual_formal_runner",
        ROOT / "experiments" / "run_retacred_qceasc_grouped_counterfactual_formal_single_seed.py",
    )
    config = json.loads(
        (ROOT / "configs" / "retacred_qceasc_grouped_counterfactual_formal_single_seed.json").read_text(
            encoding="utf-8"
        )
    )
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=32,
            num_labels=3,
            dim=8,
            num_layers=2,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
            max_length=16,
        )
    )
    for selector in (
        "q_ceasc_grouped_counterfactual",
        "classical_grouped_counterfactual",
        "random_grouped_counterfactual",
    ):
        kernel = runner.build_kernel(selector, model, 13, config, pair_chunk_size=4)
        query = torch.randn(2, 2, 5, 4)
        key = torch.randn(2, 2, 5, 4)
        attention = torch.ones(2, 5, dtype=torch.bool)
        subject = torch.zeros_like(attention)
        object_ = torch.zeros_like(attention)
        subject[:, 0] = True
        object_[:, 1] = True
        residual = kernel(
            query,
            key,
            layer_index=0,
            attention_mask=attention,
            subject_mask=subject,
            object_mask=object_,
        )
        assert residual.shape == (2, 2, 5, 5)
        assert torch.isfinite(residual).all()


def test_counterfactual_case_study_emits_semantic_trace_and_tensor_manifest(tmp_path):
    worker = _load(
        "qceasc_grouped_counterfactual_worker_contract",
        ROOT / "experiments" / "run_qceasc_grouped_counterfactual_selector_worker.py",
    )
    runner = _load(
        "qceasc_grouped_counterfactual_case_runner",
        ROOT / "experiments" / "run_retacred_qceasc_grouped_counterfactual_formal_single_seed.py",
    )
    config_path = ROOT / "configs" / "retacred_qceasc_grouped_counterfactual_formal_single_seed.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["case_study"]["records"] = {
        "train": [0, 1, 2],
        "valid": [0, 1, 2],
        "test": [0, 1, 2],
    }
    model = RelationExtractionModel(
        RelationTransformerConfig(
            vocab_size=32,
            num_labels=3,
            dim=8,
            num_layers=2,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
            max_length=16,
        )
    )
    kernel = runner.build_kernel("q_ceasc_grouped_counterfactual", model, 13, config)
    records = [
        RelationRecord(
            tokens=("alice", "met", "bob", "today"),
            subject=(0, 1),
            object=(2, 3),
            label="per:friends",
            metadata={"subject_type": "PER", "object_type": "PER"},
        ),
        RelationRecord(
            tokens=("carol", "visited", "acme"),
            subject=(0, 1),
            object=(2, 3),
            label="org:visited",
        ),
        RelationRecord(
            tokens=("dave", "joined", "lab"),
            subject=(0, 1),
            object=(2, 3),
            label="org:member",
        ),
    ]
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({token: index + 2 for index, token in enumerate({t for r in records for t in r.tokens})})
    artifacts = SimpleNamespace(
        vocab=vocab,
        label_to_id={"org:member": 0, "org:visited": 1, "per:friends": 2},
        id_to_label={0: "org:member", 1: "org:visited", 2: "per:friends"},
    )
    output_dir = tmp_path / "selector"
    worker.write_case_study(
        model=model,
        kernel=kernel,
        records={"train": records, "valid": records, "test": records},
        artifacts=artifacts,
        device=torch.device("cpu"),
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        selector="q_ceasc_grouped_counterfactual",
        initial_state=kernel.state_dict(),
        final_state=kernel.state_dict(),
    )
    case = json.loads((output_dir / "case_study.json").read_text(encoding="utf-8"))
    trace_path = output_dir / "sample_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    validator = _load(
        "qceasc_grouped_counterfactual_semantic_validator",
        ROOT / "scripts" / "validate_qceasc_grouped_counterfactual_case_study.py",
    )
    assert set(case["required_splits"]) == {"train", "valid", "test"}
    assert {item["split"] for item in case["cases"]} == {"train", "valid", "test"}
    assert set(case["checkpoint_policy"]) == {
        "initial_or_pre_training",
        "best_valid_or_declared_selection_checkpoint",
        "final",
    }
    assert "subject" in case["cases"][0] and "object" in case["cases"][0]
    assert "gold_relation" in case["cases"][0]
    assert trace["schema_version"] == "sample-trace.v1"
    assert trace["semantic_contract"]["required_splits"] == ["train", "valid", "test"]
    required_grouped = {
        "q_ceasc_group_manifest",
        "q_ceasc_group_masked_supports",
        "q_ceasc_group_influence_vectors",
        "q_ceasc_group_scores",
        "q_ceasc_member_scores",
    }
    assert required_grouped.issubset(set(case["representation_inventory"]))
    assert len(case["tensor_manifest"]) >= 15
    assert validator.validate(output_dir) == []
    for entry in case["tensor_manifest"]:
        assert (output_dir / entry["path"]).is_file()


def test_formal_report_contract_requires_data_provenance_and_random_control_summary():
    exporter = (
        ROOT
        / "scripts"
        / "export_retacred_qceasc_grouped_counterfactual_formal_single_seed_report.sh"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT
        / "experiments"
        / "run_retacred_qceasc_grouped_counterfactual_formal_single_seed.py"
    ).read_text(encoding="utf-8")

    assert 'set(data) == {"train", "valid", "test"}' in exporter
    assert 'info.get("records", 0)' in exporter
    assert 'info.get("source_sha256")' in exporter
    assert "structural control:" in runner
    assert "candidate minus random-group test macro-F1:" in runner
    assert "random-group control completed:" in runner
