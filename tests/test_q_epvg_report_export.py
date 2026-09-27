from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "q_epvg_report_export.py"
    spec = importlib.util.spec_from_file_location("q_epvg_report_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def case_payload(selector: str) -> tuple[dict, dict]:
    """Build a compact, complete source-first trace fixture."""
    stage_names = [
        "data", "preprocess", "embedding", "encoder", "training",
        "retrieval", "attention_baseline", "scoring", "selection",
        "attention_intervention", "context", "classifier", "generation",
        "evaluation", "diagnosis",
    ]
    cases = []
    manifests = []
    for split in ("train", "valid", "test"):
        for checkpoint in ("initial_or_pre_training", "best_valid_or_declared_selection_checkpoint", "final"):
            slug = checkpoint.split("_")[0]
            case_id = f"{selector}:{split}:0:{slug}"
            rep_names = {
                "token_embeddings": "embedding",
                "encoder_hidden_states": "encoder",
                "attention_qkv": "attention_baseline",
                "baseline_attention_scores": "attention_baseline",
                "q_epvg_gate": "scoring",
                "q_epvg_score_adjustment": "scoring",
                "steered_attention_scores": "selection",
                "q_epvg_output": "context",
                "classifier_logits_probabilities": "classifier",
                "q_epvg_kernel_parameters": "training",
            }
            representations = {}
            for rep_id, producer in rep_names.items():
                manifest_id = f"{split}_0_{slug}__{rep_id}"
                manifest = {
                    "id": rep_id, "manifest_id": manifest_id,
                    "producer_stage": producer,
                    "path": f"case_study_tensors/{manifest_id}.pt",
                    "shape": [2, 3], "dtype": "torch.float32",
                    "axis_semantics": ["rows", "features"],
                    "sha256": "a" * 64, "byte_count": 24,
                    "preview": {"min": 0.0, "max": 1.0, "mean": 0.5, "l2_norm": 1.0},
                }
                representations[rep_id] = manifest
                manifests.append(manifest)

            def refs(*names: str) -> list[str]:
                return [representations[name]["manifest_id"] for name in names if name in representations]

            def stage(name: str, status: str = "observed", inputs: list[str] | None = None, outputs: list[str] | None = None, semantic_output: str | None = None) -> dict:
                inputs = inputs or []
                outputs = outputs or []
                return {
                    "stage": name, "status": status,
                    "input_refs": ["source_sample"] if name == "preprocess" and not inputs else refs(*inputs),
                    "output_refs": [semantic_output] if semantic_output else (["source_sample"] if name == "data" and not outputs else refs(*outputs)),
                    "inputs": {"representations": [representations[n] for n in inputs if n in representations]},
                    "outputs": {"representations": [representations[n] for n in outputs if n in representations]},
                }

            trace_stages = [
                stage("data", outputs=[]),
                stage("preprocess", inputs=["token_embeddings"], outputs=["token_embeddings"]),
                stage("embedding", inputs=["token_embeddings"], outputs=["token_embeddings"]),
                stage("encoder", inputs=["token_embeddings"], outputs=["encoder_hidden_states"]),
                stage("training", inputs=["encoder_hidden_states"], outputs=["q_epvg_kernel_parameters"]),
                stage("retrieval", "not_applicable"),
                stage("attention_baseline", inputs=["attention_qkv"], outputs=["baseline_attention_scores"]),
                stage("scoring", inputs=["attention_qkv", "baseline_attention_scores"], outputs=["q_epvg_gate", "q_epvg_score_adjustment"]),
                stage("selection", inputs=["baseline_attention_scores", "q_epvg_score_adjustment"], outputs=["steered_attention_scores"]),
                stage("attention_intervention", inputs=["q_epvg_gate"], outputs=["q_epvg_output"]),
                stage("context", inputs=["q_epvg_output"], outputs=["q_epvg_output"]),
                stage("classifier", inputs=["encoder_hidden_states", "q_epvg_output"], outputs=["classifier_logits_probabilities"]),
                stage("generation", "not_applicable"),
                stage("evaluation", inputs=["classifier_logits_probabilities"], outputs=[], semantic_output="gold_relation"),
                stage("diagnosis", inputs=list(representations), outputs=[], semantic_output="diagnosis"),
            ]
            cases.append({
                "case_id": case_id, "split": split, "checkpoint": checkpoint,
                "sentence": "Alice works for Acme .",
                "tokens": ["Alice", "works", "for", "Acme", "."],
                "token_ids": [1, 2, 3, 4, 5],
                "subject": {"text": "Alice", "span": [0, 1], "token_positions": [0]},
                "object": {"text": "Acme", "span": [3, 4], "token_positions": [3]},
                "gold_relation": "per:employee_of", "representations": representations,
                "stages": trace_stages,
            })
    trace = {
        "schema_version": "sample-trace.v1",
        "semantic_contract": {"version": "q-attention.case-study-trace-contract.v3", "lineage": "producer_owned_stage_input_output"},
        "coverage": {name: "not_applicable" if name in {"retrieval", "generation"} else "observed" for name in stage_names},
        "samples": [{"sample_id": case["case_id"], "stages": case["stages"]} for case in cases],
    }
    payload = {
        "schema_version": "q-attention.Q-EPVG-case-study.v2",
        "lineage_schema_version": "q-attention.case-study-lineage.v1",
        "selector": selector, "status": "observed",
        "manifest_inventory": [manifest["manifest_id"] for manifest in manifests],
        "tensor_manifest": manifests, "cases": cases,
    }
    return payload, trace


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.json"
    selectors = ["disabled", "selector_a", "selector_b"]
    config.write_text(json.dumps({"selectors": selectors}), encoding="utf-8")
    run = root / "run"
    for name in ("RUN_COMPLETE", "run_summary.json", "run_summary.data", "run_summary.md", "gpu_assignments.json"):
        write(run / name)
    write(run / "baseline" / "metrics.json")
    for selector in selectors[1:]:
        write(run / "selectors" / selector / "metrics.json")
        payload, trace = case_payload(selector)
        for manifest in payload["tensor_manifest"]:
            raw = f"tensor:{manifest['manifest_id']}\n".encode("utf-8")
            manifest["sha256"] = hashlib.sha256(raw).hexdigest()
            manifest["byte_count"] = len(raw)
            write(run / "selectors" / selector / manifest["path"], raw.decode("utf-8"))
        (run / "selectors" / selector / "case_study.json").write_text(json.dumps(payload), encoding="utf-8")
        (run / "selectors" / selector / "sample_trace.json").write_text(json.dumps(trace), encoding="utf-8")
    for split in ("train", "valid", "test"):
        write(run / "data" / f"{split}.jsonl", f"{{\"split\": \"{split}\"}}\n")
    return run, config, root / "reports" / "experiment" / "run_seed13"


@pytest.mark.parametrize("failure", ["copy", "validation"])
def test_failed_export_cleans_staging_and_retry_reuses_completed_run(tmp_path: Path, failure: str) -> None:
    module = load_module()
    run, config, report = build_fixture(tmp_path)
    summary_before = (run / "run_summary.json").read_bytes()
    kwargs = {
        "inject_failure_after": 2 if failure == "copy" else None,
        "inject_validation_failure": failure == "validation",
    }
    with pytest.raises(module.ExportError):
        module.export_report(
            run_dir=run,
            report_dir=report,
            config_path=config,
            reporting_commit="abc123",
            **kwargs,
        )
    assert not report.exists()
    assert not list(report.parent.glob(f".{report.name}.staging-*"))
    assert (run / "run_summary.json").read_bytes() == summary_before

    destination = module.export_report(
        run_dir=run,
        report_dir=report,
        config_path=config,
        reporting_commit="abc123",
    )
    assert destination == report.resolve()
    assert (report / "metrics/selector_a.json").is_file()
    assert (report / "case_study/selector_b.sample-trace.json").is_file()
    assert (report / "data.sha256").is_file()
    manifest = json.loads((report / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_run"] == str(run.resolve())
    assert manifest["config_sha256"]
    assert manifest["retry_count"] == 1
    assert "injected" in manifest["failure_reason"]
    assert not list(report.parent.glob(f".{report.name}.staging-*"))


def test_stale_staging_is_cleaned_and_attempt_identity_is_bound(tmp_path: Path) -> None:
    module = load_module()
    run, config, report = build_fixture(tmp_path)
    stale = report.parent / f".{report.name}.staging-stale"
    write(stale / "leftover.txt")

    module.export_report(
        run_dir=run,
        report_dir=report,
        config_path=config,
        reporting_commit="abc123",
    )
    manifest = json.loads((report / "export_manifest.json").read_text(encoding="utf-8"))
    assert not stale.exists()
    assert str(stale) in manifest["stale_staging_cleaned"]

    second_run, second_config, second_report = build_fixture(tmp_path / "second")
    with pytest.raises(module.ExportError):
        module.export_report(
            run_dir=second_run,
            report_dir=second_report,
            config_path=second_config,
            reporting_commit="abc123",
            inject_failure_after=1,
        )
    state_path = module._attempt_state_path(second_report.resolve())
    attempts = json.loads(state_path.read_text(encoding="utf-8"))
    attempts[0]["source_run"] = str(run.resolve())
    state_path.write_text(json.dumps(attempts), encoding="utf-8")
    with pytest.raises(module.ExportError, match="identity mismatch"):
        module.export_report(
            run_dir=second_run,
            report_dir=second_report,
            config_path=second_config,
            reporting_commit="abc123",
        )
    state_path.unlink(missing_ok=True)


def test_case_study_validator_rejects_dangling_representation_reference() -> None:
    module = load_module()
    payload, trace = case_payload("selector_a")
    scoring = next(item for item in payload["cases"][0]["stages"] if item["stage"] == "scoring")
    scoring["output_refs"].append("q_epvg_alignment_real")
    with pytest.raises(module.ExportError, match="dangling stage reference"):
        module._validate_case_study_payload(payload, trace, selector="selector_a")


def test_case_study_validator_rejects_incomplete_required_stage() -> None:
    module = load_module()
    payload, trace = case_payload("selector_a")
    scoring = next(item for item in payload["cases"][0]["stages"] if item["stage"] == "scoring")
    scoring["status"] = "failed"
    with pytest.raises(module.ExportError, match="required stage is not fully observed"):
        module._validate_case_study_payload(payload, trace, selector="selector_a")


def test_source_tensor_checksum_failure_is_recorded_and_retryable(tmp_path: Path) -> None:
    module = load_module()
    run, config, report = build_fixture(tmp_path)
    case = json.loads((run / "selectors/selector_a/case_study.json").read_text(encoding="utf-8"))
    tensor = run / "selectors/selector_a" / case["tensor_manifest"][0]["path"]
    original = tensor.read_bytes()
    tensor.write_bytes(b"tampered\n")
    with pytest.raises(module.ExportError, match="checksum mismatch"):
        module.export_report(run_dir=run, report_dir=report, config_path=config, reporting_commit="abc123")
    assert not report.exists()
    assert not list(report.parent.glob(f".{report.name}.staging-*"))
    attempts = json.loads(module._attempt_state_path(report.resolve()).read_text(encoding="utf-8"))
    assert attempts[-1]["status"] == "failed"
    tensor.write_bytes(original)
    module.export_report(run_dir=run, report_dir=report, config_path=config, reporting_commit="abc123")
    manifest = json.loads((report / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["retry_count"] == 1
    assert "checksum mismatch" in manifest["failure_reason"]
