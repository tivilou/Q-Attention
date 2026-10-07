from __future__ import annotations

import ast
import base64
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import sequence_audit, sequence_resources
from baselines.retacred.common import ROOT, cohort, digest, git, metrics, pin_source, select_models


@pytest.fixture(scope="module")
def source():
    # Required source gate: missing pinned checkout is a failure, never a skip.
    return pin_source("sequence_routing", fetch=False)


def test_active_h3_mix_overrides_dormant_41_embedding_size(source):
    facts = sequence_audit.source_facts(source)
    assert facts["settings_class_dimensions"] == {"n_class": 41, "h3_embed_size": 41}
    assert facts["active_h3_mix_recipe"] == {"heads": ["head_3"], "h3_embed_size": 512,
                                            "abstract": "mix", "batch_size": 24, "epcoh_n": 5}
    assert facts["learning_rate"] == 1e-5 and facts["scheduler_step"] == 1200
    assert facts["missing_abstraction_import"] is True


def test_recovered_file_does_not_admit_or_mutate_source(source):
    before = {name: digest(source / name) for name in sequence_resources.resource_lock()["active_file_sha256"]}
    report = sequence_audit.audit(source)
    assert report["historical_abstraction"]["status"] == "historical_source_verified"
    assert report["historical_abstraction"]["strict_comparison_eligible"] is False
    assert report["status"] == "not_verifiable" and report["model_jobs_launched"] == 0
    assert report["strict_comparison_eligible"] is False and report["source_modified"] is False
    assert len(report["unresolved_metadata"]) == 2 and len(report["blockers"]) == 5
    assert before == {name: digest(source / name) for name in before}
    assert not (source / "Abstraction.py").exists()
    assert git("status", "--porcelain", "--untracked-files=no", cwd=source) == ""


def test_actual_audit_cli_returns_expected_not_verifiable_exit(source):
    result = subprocess.run([sys.executable, "-m", "baselines.retacred.sequence_audit"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_verifiable" and payload["model_jobs_launched"] == 0


def test_verify_only_cli_remains_offline(source):
    result = subprocess.run([sys.executable, "-m", "baselines.retacred.sequence_resources", "--verify-only"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "historical_source_verified"


def test_source_and_registry_reject_metric_rank_and_model_admission():
    row = next(row for row in cohort()["models"] if row["id"] == "sequence_routing_h3")
    assert row["reported_f1"] == .922  # Preserve frozen historical evidence, do not fabricate a new score.
    assert row["reported_score_rank_eligible"] is False and row["benchmark_comparable"] is False
    with pytest.raises(ValueError, match="no jobs were launched"):
        select_models("sequence_routing_h3")
    assert len(select_models("ready")) == 5


def test_actual_source_micro_assignment_is_accuracy_not_non_null_f1(source):
    from sklearn.metrics import f1_score
    tree = ast.parse((source / "Train_Eval.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rc_eval")
    assignment = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "f1_micor" for target in node.targets))
    namespace = {"Y": [0] * 9 + [1], "PREDICTIONS": [0] * 10, "f1_score": f1_score}
    # Execute only the exact source metric assignment, no model/data imports.
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "upstream-micro-assignment", "exec"), namespace)
    benchmark = metrics(namespace["Y"], namespace["PREDICTIONS"], ["no_relation", "fixture:positive"], .1)
    assert namespace["f1_micor"] == benchmark["accuracy"] == .9
    assert benchmark["all_label_micro_f1"] == .9 and benchmark["micro_f1"] == 0


def test_counter_bootstrap_precedes_generation_in_real_preprocessor(source):
    tree = ast.parse((source / "rc_data.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in ("_get_pretrained_transformer3", "process_file")]
    factory_line = min(node.lineno for node in calls if node.func.id == "_get_pretrained_transformer3")
    generation_line = min(node.lineno for node in calls if node.func.id == "process_file")
    assert factory_line < generation_line
    util = ast.parse((source / "rc_util.py").read_text())
    set_tokenizer = next(node for node in util.body if isinstance(node, ast.FunctionDef) and node.name == "set_tokenizer")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"
               for node in ast.walk(set_tokenizer))


@pytest.mark.parametrize("tokens,spans", [(["Alice", "met", "Bob"], (0, 0, 2, 2)),
                                           (["Alice", "Smith", "met", "Bob", "Jones"], (0, 1, 3, 4))])
def test_real_mix_preprocessing_has_types_and_no_gold_label_in_input(source, tmp_path, monkeypatch, tokens, spans):
    directory = tmp_path / "essential_files"
    directory.mkdir()
    (directory / "properties-with-labels.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    tree = ast.parse((source / "rc_data.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_data")
    namespace = {"json": json, "tqdm": lambda value: value, "copy": copy, "rel_dic": {},
                 "id2kbid": {str(i): f"p{i}" for i in range(40)}, "SRE_Analogy": {"retacred": []}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream-prepare-data", "exec"), namespace)
    record = {"token": tokens, "subj_start": spans[0], "subj_end": spans[1],
              "obj_start": spans[2], "obj_end": spans[3], "subj_type": "PERSON", "obj_type": "PERSON",
              "docid": "fixture", "id": "fixture", "relation": "no_relation"}
    outputs = []
    for n in range(40):
        record["relation"] = "no_relation" if n == 0 else f"fixture:relation{n}"
        rows = namespace["prepare_data"]("retacred", None, {}, [record.copy()], "train", "abstracted")
        outputs.append(rows[0]["sentecne_entabs_flagged_tokens"])
    expected = (["[e11]", *tokens[spans[0]:spans[1]+1], "*", "PERSON", "@", "[e12]",
                 *tokens[spans[1]+1:spans[2]], "[e21]", *tokens[spans[2]:spans[3]+1],
                 "#", "PERSON", "^", "[e22]"])
    assert all(output == expected for output in outputs)


def test_pinned_source_drift_stops_before_audit(tmp_path, monkeypatch):
    first = next(iter(sequence_resources.resource_lock()["active_file_sha256"]))
    (tmp_path / first).write_text("# not the pinned source\n")
    with pytest.raises(ValueError, match="pinned file drift"):
        sequence_audit.audit(tmp_path)


def test_corrupt_cache_rejected_without_network_or_overwrite(tmp_path, monkeypatch):
    target = tmp_path / "Abstraction.py"
    target.write_text("# corrupt historical source\n")
    monkeypatch.setattr(sequence_resources, "resource_directory", lambda: tmp_path)
    monkeypatch.setattr(sequence_resources.urllib.request, "build_opener",
                        lambda *_: pytest.fail("cache validation must not fetch"))
    with pytest.raises(ValueError, match="source identity mismatch"):
        sequence_resources.fetch_resources()
    assert target.read_text() == "# corrupt historical source\n"


def test_recovery_rejects_redirect_without_following():
    with pytest.raises(ValueError, match="redirect rejected"):
        sequence_resources._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.org/")


def test_recovery_rejects_symlink_cache(tmp_path, monkeypatch):
    destination, target = tmp_path / "cache", tmp_path / "outside"
    target.mkdir()
    destination.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(sequence_resources, "resource_directory", lambda: destination)
    with pytest.raises(ValueError, match="symlink"):
        sequence_resources.fetch_resources()


def test_corrupt_api_payload_rejected_before_file_creation(tmp_path, monkeypatch):
    lock = sequence_resources.resource_lock()["artifact"]
    payload = {"sha": lock["git_blob"], "size": lock["size_bytes"], "encoding": "base64",
               "content": base64.b64encode(b"unlocked source").decode()}
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, _): return json.dumps(payload).encode()
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == lock["url"] and timeout == 30
            return Response()
    monkeypatch.setattr(sequence_resources, "resource_directory", lambda: tmp_path / "cache")
    monkeypatch.setattr(sequence_resources.urllib.request, "build_opener", lambda *_: Opener())
    with pytest.raises(ValueError, match="source identity mismatch"):
        sequence_resources.fetch_resources()
    assert not (tmp_path / "cache").exists()
