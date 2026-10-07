"""Behavior gates for source-backed references; no formal training/data needed."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from baselines.retacred import assets, common, orchestrate, ptr_resources, report, train
from baselines.retacred.fidelity import compare
from baselines.retacred.improved import Improved


def test_manifests_do_not_promote_tiny_fixtures_to_formal_certificates():
    for name in common.cohort()["default_models"]:
        path = common.PACKAGE/"manifests"/name/"baseline-manifest.json"
        manifest = json.loads(path.read_text())
        assert manifest["baseline_id"] == name
        assert manifest["equivalence"]["status"] == "not_verifiable"
        assert manifest["adaptation"]["postprocessing_rules"] == []
        assert not manifest["validation"]["gates"]["cuda_amp_checked"]
        source = common.pin_source(common.select_models(name)[0]["source"],fetch=False)
        for item in manifest["provenance"]["artifacts"]:
            assert common.digest(source/item["path"]) == item["sha256"]


def test_recovered_ptr_metadata_identity_and_class_interface():
    from baselines.retacred.improved import load_module
    from types import SimpleNamespace
    directory = common.ROOT/".external/retacred/ptr_metadata"/ptr_resources.REVISION
    result = ptr_resources.validate_resources(directory)
    processor = load_module(common.pin_source("improved",fetch=False),"prepro").RETACREDProcessor(
        SimpleNamespace(input_format="typed_entity_marker_punct"), SimpleNamespace(add_tokens=lambda _:None))
    assert set(result["labels"]) == set(processor.LABEL_TO_ID)
    # Source-specific numeric orders differ; compare canonical names, not IDs.
    assert result["labels"]["no_relation"] == processor.LABEL_TO_ID["no_relation"] == 0
    assert len(result["verbalizers"]) == 40
    assert all(len(words) == 5 for words in result["verbalizers"].values())
    assert result["status"] == "metadata_verified"


def test_cohort_preserves_blockers_and_deduplicates():
    data = common.cohort()
    assert len([row for row in data["models"] if row["rank"] is not None]) == 15
    assert len(common.select_models("ready")) == 5
    with pytest.raises(ValueError, match="not runnable"):
        common.select_models("all")
    with pytest.raises(ValueError, match="duplicates"):
        common.select_models("typed_marker,typed_marker")


def test_metric_null_is_not_accuracy():
    value = common.metrics([0, 0, 1, 2], [0, 1, 1, 0], ["no_relation", "a", "b"], .3)
    assert value["accuracy"] == .5
    assert value["micro_f1"] == .5
    assert value["micro_precision"] == .5
    assert value["micro_recall"] == .5
    assert sum(sum(row) for row in value["confusion_matrix"]) == 4
    other = common.metrics([0,0,0,1], [0,0,0,0], ["no_relation", "a"], .4)
    assert other["accuracy"] == .75 and other["micro_f1"] == 0


def test_shards_and_assembly_are_hash_checked(tmp_path):
    source = tmp_path/"weights.bin"
    source.write_bytes(b"1234567890")
    parts = assets.shard_file(source,tmp_path/"parts",shard_bytes=4)
    assert [part["size_bytes"] for part in parts] == [4,4,2]
    target = tmp_path/"output.bin"
    assets.assemble([tmp_path/"parts"/part["name"] for part in parts],target,common.digest(source),10)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError,match="identity mismatch"):
        assets.assemble([tmp_path/"parts"/part["name"] for part in parts],target,"0"*64,10)
    assert target.read_bytes() == source.read_bytes()


def test_asset_download_rejects_symlink_and_cross_origin_redirect(tmp_path):
    from urllib.request import Request
    real = tmp_path/"real"; real.mkdir()
    link = tmp_path/"linked"; link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        assets.stream_download("https://example.invalid/config.json", link/"config.json", expected="0"*64)
    guard = assets.AuthenticatedRedirectGuard()
    request = Request("https://exchange.example.invalid/files/model", headers={"Authorization":"Bearer dummy"})
    for url in ("http://exchange.example.invalid/model", "https://other.example.invalid/model"):
        with pytest.raises(ValueError, match="outside exchange"):
            guard.redirect_request(request, None, 302, "Found", {}, url)
    redirected = guard.redirect_request(request, None, 302, "Found", {}, "https://exchange.example.invalid:443/files/mirror")
    assert redirected.get_header("Authorization") == "Bearer dummy"


def test_gpu_parsing_respects_visibility_and_aliases(monkeypatch):
    monkeypatch.setattr(subprocess,"check_output",lambda *a,**k:"0, GPU-a\n1, GPU-b\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    assert orchestrate.gpu_ids("auto") == ["1"]
    with pytest.raises(ValueError,match="outside"):
        orchestrate.gpu_ids("0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError,match="duplicate physical"):
        orchestrate.gpu_ids("0,GPU-a")
    with pytest.raises(ValueError):
        orchestrate.gpu_ids("0,0")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    import torch
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
    root = tmp_path_factory.mktemp("tiny-roberta")
    pinned = common.ROOT/".external/assets"/assets.REVISION/"model"
    assert (pinned/"vocab.json").exists(), "run asset preparation before runtime source tests"
    torch.manual_seed(7)
    config = RobertaConfig(vocab_size=50265,hidden_size=16,num_hidden_layers=1,num_attention_heads=4,
                           intermediate_size=32,max_position_embeddings=514,hidden_dropout_prob=.1,
                           attention_probs_dropout_prob=.1)
    RobertaModel(config).save_pretrained(root)
    RobertaTokenizer(vocab_file=str(pinned/"vocab.json"),merges_file=str(pinned/"merges.txt")).save_pretrained(root)
    return root


RECORD = {"token":["Alice","works","for","Acme","."],"subj_start":0,"subj_end":0,
          "obj_start":3,"obj_end":3,"subj_type":"PERSON","obj_type":"ORGANIZATION","relation":"per:employee_of"}


@pytest.mark.parametrize("model",[row["id"] for row in common.select_models("ready") if row["source"] == "improved"])
def test_actual_upstream_forward_gradient_update_and_capture(model,tiny_checkpoint):
    import torch
    torch.set_num_threads(2)
    row = common.select_models(model)[0]
    adapter = Improved(common.pin_source("improved",fetch=False),tiny_checkpoint,row["input_format"],device="cpu")
    feature = adapter.encode([copy.deepcopy(RECORD)])[0]
    assert feature["ss"] != feature["os"] and 0 <= feature["ss"] < len(feature["input_ids"])
    result = compare(adapter,adapter.batch([feature]))
    assert result["trace_transparency"] == result["optimizer_update"] == "passed"
    assert result["training_dropout_trace_transparency"] == "passed"


def make_jobs(tmp_path):
    return [{"id":f"{model}_seed13","model":model,"seed":13,"directory":tmp_path/"tasks"/model}
            for model in ("typed_marker","entity_marker","entity_marker_punct")]


def test_fidelity_reference_does_not_copy_adapter_mutations(tiny_checkpoint):
    adapter = Improved(common.pin_source("improved",fetch=False),tiny_checkpoint,
                       "typed_entity_marker_punct",device="cpu")
    batch = adapter.batch(adapter.encode([copy.deepcopy(RECORD)]))
    handle = adapter.model.classifier.register_forward_hook(lambda module, inputs, output:output+.5)
    try:
        with pytest.raises(AssertionError):
            compare(adapter,batch)
    finally:
        handle.remove()


def test_real_parent_child_launch_and_failure(tmp_path,monkeypatch):
    args = argparse.Namespace(run_dir=tmp_path,micro_batch=2,logical_batch=4,checkpointing=False,canary=True)
    jobs = make_jobs(tmp_path)
    saved = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    try:
        assert len(orchestrate.run_queue(jobs,["synthetic-a","synthetic-b"],args,Path("fake"),Path("fake"))) == 3
        assert all(signal.getsignal(sig) == handler for sig,handler in saved.items())
        for job in jobs:
            command = json.loads((tmp_path/"logs"/(job["id"]+".command.json")).read_text())
            assert command["argv"][:3] == [sys.executable,"-m","baselines.retacred.train"]
            assert command["cwd"] == str(common.ROOT)
        second = tmp_path/"failure"
        second.mkdir()
        args.run_dir = second
        monkeypatch.setenv("RETACRED_CANARY_FAIL","typed_marker")
        with pytest.raises(RuntimeError,match="exited"):
            orchestrate.run_queue(make_jobs(second),["synthetic-a"],args,Path("fake"),Path("fake"))
        assert not (second/"RUN_COMPLETE").exists()
        assert (second/"RUN_FAILED.json").exists()
    finally:
        for sig,handler in saved.items():
            signal.signal(sig,handler)


def test_tiny_actual_train_test_tensor_and_recovery(tmp_path,tiny_checkpoint,monkeypatch):
    import torch
    torch.set_num_threads(2)
    data = tmp_path/"data"; data.mkdir()
    raw = [copy.deepcopy(RECORD),copy.deepcopy(RECORD),copy.deepcopy(RECORD)]
    manifest = {"splits":{}}
    for split,name in (("train","train"),("valid","dev"),("test","test")):
        path = data/(name+".json")
        path.write_text(json.dumps(raw))
        manifest["splits"][split] = {"records":len(raw),"source_sha256":common.digest(path),"output_sha256":common.digest(path)}
    (data/"data_manifest.json").write_text(json.dumps(manifest))
    args = argparse.Namespace(model="typed_marker_punct",seed=13,data_dir=data,checkpoint=tiny_checkpoint,
                              run_dir=tmp_path/"run",resume=False,device="cpu",epochs=1,logical_batch=2,
                              micro_batch=1,checkpointing=False,checkpoint_every=1,canary=False)
    saved = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    monkeypatch.setattr(train,"parse_args",lambda:args)
    try:
        train.main()
        orchestrate.validate_job(args.run_dir)
        summary = json.loads((args.run_dir/"run_summary.json").read_text())
        assert summary["splits"]["test"]["items"] == 3
        trace_paths = [args.run_dir/"case_study"/row["path"] for row in summary["case_study"]["traces"]]
        assert len(trace_paths) == 2
        for path in trace_paths:
            trace = json.loads(path.read_text())
            values = torch.load(path.parent/trace["tensor_artifact"],weights_only=True)
            assert values["task_output.logits"].shape == (1,40)
            assert values["functional.entity_readout.subject"].shape == (1,16)
            assert values["functional.entity_readout.object"].shape == (1,16)
            assert values["functional.entity_readout.concatenated"].shape == (1,32)
            torch.testing.assert_close(values["functional.entity_readout.concatenated"],
                                       torch.cat([values["functional.entity_readout.subject"],
                                                  values["functional.entity_readout.object"]],-1),rtol=0,atol=0)
            assert any("embeddings" in key for key in values)
            assert any("attention_scores_to_probabilities" in key and key.endswith(".input") for key in values)
            assert any("attention_scores_to_probabilities" in key and key.endswith(".output") for key in values)
            if trace["phase"] == "train":
                assert any(key.endswith("gradient") for key in values)
                assert trace["capture_origin"] == "actual_training_forward_backward"
        before = torch.load(args.run_dir/"checkpoints/final.pt",weights_only=False)["model"]
        args.resume = True
        train.main()
        after = torch.load(args.run_dir/"checkpoints/final.pt",weights_only=False)["model"]
        for key in before:
            torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
        args.micro_batch = 2
        with pytest.raises(ValueError,match="contract mismatch"):
            train.main()
    finally:
        for sig,handler in saved.items():
            signal.signal(sig,handler)


def test_atomic_export_failure_then_same_run_retry(tmp_path,monkeypatch):
    run = tmp_path/"run"; run.mkdir()
    common.atomic_json(run/"run_manifest.json",{"scope":"selected"})
    job = {"id":"typed_seed13","model":"typed","seed":13,"directory":run/"tasks/typed"}
    job["directory"].mkdir(parents=True)
    split = common.metrics([0,1],[0,1],["no_relation","relation"],.1)
    summary = {"schema":"result","status":"complete","model":"typed","seed":13,"updates":1,
               "adaptations":[],"fidelity":"separate","environment":{},"contract":{},"case_study":{"traces":[]},
               "splits":{"test":split},"data_manifest":{"splits":{"test":{"records":2,"source_sha256":"a","output_sha256":"b"}}}}
    common.atomic_json(job["directory"]/"run_summary.json",summary)
    exchange = {"schema_version":1,"status":"complete","exchange_url":"https://example.invalid","target_dir":"p/e/r",
                "files":[],"artifact_identity":{}}
    destination = tmp_path/"reports/result"
    validator = report.validate_export
    monkeypatch.setattr(report,"validate_export",lambda *_: (_ for _ in ()).throw(ValueError("injected export failure")))
    with pytest.raises(ValueError,match="injected"):
        report.export_report(run,[job],exchange,destination)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.stage-*"))
    monkeypatch.setattr(report,"validate_export",validator)
    report.export_report(run,[job],exchange,destination)
    validator(destination)
    report.export_report(run,[job],exchange,destination)
    (destination/"private.pt").write_bytes(b"unexpected")
    with pytest.raises(ValueError,match="extra"):
        validator(destination)


def test_required_tensor_delivery_failure_keeps_a_recoverable_pending_manifest(tmp_path,monkeypatch):
    from types import SimpleNamespace
    run = tmp_path/"run"; run.mkdir()
    case = run/"tasks/model_seed13/case_study"; case.mkdir(parents=True)
    (case/"sample.pt").write_bytes(b"test tensor artifact")
    (case/"sample.json").write_text("{}")
    job = {"id":"model_seed13", "directory":case.parent}
    monkeypatch.setenv("PROJECT_EXCHANGE_URL","https://example.invalid")
    monkeypatch.setenv("PROJECT_EXCHANGE_TOKEN","test-not-a-real-token")
    def failed(**kwargs):
        raise ConnectionError("injected transport outage")
    monkeypatch.setattr(report,"uploader",lambda:SimpleNamespace(upload_declared_files=failed))
    with pytest.raises(ConnectionError):
        report.upload_evidence(run,[job])
    pending = json.loads((run/"exchange_pending.json").read_text())
    assert pending["status"] == "failed"
    assert len(pending["files"]) == 2
    assert "test-not-a-real-token" not in json.dumps(pending)
    assert not (run/"exchange_manifest.json").exists()


def test_mid_batch_pause_resume_is_exact(tmp_path,tiny_checkpoint,monkeypatch):
    import torch
    torch.set_num_threads(2)
    data = tmp_path/"data"; data.mkdir()
    raw = [copy.deepcopy(RECORD) for _ in range(5)]
    manifest = {"splits":{}}
    for split,name in (("train","train"),("valid","dev"),("test","test")):
        path = data/(name+".json")
        path.write_text(json.dumps(raw))
        manifest["splits"][split] = {"records":len(raw),"source_sha256":common.digest(path),"output_sha256":common.digest(path)}
    (data/"data_manifest.json").write_text(json.dumps(manifest))
    args = argparse.Namespace(model="typed_marker_punct",seed=29,data_dir=data,checkpoint=tiny_checkpoint,
                              run_dir=tmp_path/"uninterrupted",resume=False,device="cpu",epochs=1,logical_batch=2,
                              micro_batch=1,checkpointing=False,checkpoint_every=1,canary=False)
    monkeypatch.setattr(train,"parse_args",lambda:args)
    saved_handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    try:
        train.main()
        expected = torch.load(args.run_dir/"checkpoints/final.pt",weights_only=False)["model"]
        args.run_dir = tmp_path/"interrupted"
        original_atomic = train.atomic_json
        def pause_after_first_update(path,value):
            original_atomic(path,value)
            if path.name == "heartbeat.json" and value.get("update") == 1:
                signal.raise_signal(signal.SIGTERM)
        monkeypatch.setattr(train,"atomic_json",pause_after_first_update)
        with pytest.raises(SystemExit) as stopped:
            train.main()
        assert stopped.value.code == 75
        assert not (args.run_dir/"RUN_COMPLETE").exists()
        value = torch.load(args.run_dir/"checkpoints/latest.pt",weights_only=False)
        assert value["cursor"]["cursor"] == 1 and value["cursor"]["permutation"] is not None
        monkeypatch.setattr(train,"atomic_json",original_atomic)
        args.resume = True
        train.main()
        actual = torch.load(args.run_dir/"checkpoints/final.pt",weights_only=False)["model"]
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    finally:
        for sig,handler in saved_handlers.items():
            signal.signal(sig,handler)


def test_publication_index_guard_is_exact(tmp_path,monkeypatch):
    repository = tmp_path/"repo"; repository.mkdir()
    def command(*args):
        return subprocess.run(["git",*args],cwd=repository,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    command("init","-b","1.1")
    command("config","user.name","Test User"); command("config","user.email","test@example.invalid")
    (repository/"code.txt").write_text("source")
    command("add","code.txt"); command("commit","-m","initial")
    command("update-ref","refs/remotes/origin/1.1","HEAD")
    command("update-ref","refs/remotes/origin/main","HEAD")
    monkeypatch.setattr(report,"ROOT",repository)
    monkeypatch.setattr(report,"git",lambda *args:command(*args).stdout.strip())
    destination = repository/"reports/retacred_reference_suite/run"
    destination.mkdir(parents=True)
    (destination/"metrics.json").write_text("{}")
    command("add","reports")
    report.publication_preflight(no_push=True,allowed_report=destination)
    (repository/"code.txt").write_text("unrelated user modification")
    command("add","code.txt")
    with pytest.raises(ValueError,match="unrelated"):
        report.publication_preflight(no_push=True,allowed_report=destination)
