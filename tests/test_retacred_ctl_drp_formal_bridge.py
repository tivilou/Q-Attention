"""Bounded interface/recovery fixtures only; not formal data or HTTP evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import common, ctl_drp_export as exporter, ctl_drp_formal as formal
from baselines.retacred import ctl_drp_qualification as qualification, ctl_drp_runner as runner
from baselines.retacred import report as publishing
from baselines.retacred.ctl_drp_simcse import load_lock
from baselines.retacred.ctl_drp_task import LABELS, PhysicalSplits, RECIPE, UpdatePlan


def options(tmp_path, *extra):
    args = formal.parser().parse_args(["--run-dir", str(tmp_path / "run"), "--review-receipt", str(tmp_path / "review.json"),
        "--qualification", str(tmp_path / "qualification.json"), "--cache", str(tmp_path / "cache"),
        "--paper-source", str(tmp_path / "native.tex"), "--weight", str(tmp_path / "weights.bin"), *extra])
    formal.validate_options(args)
    return args


def review_file(path):
    common.atomic_json(path, {"schema": "ctl-drp-independent-review.v1", "status": "accepted",
        "reviewer_role": "independent", "reviewer": "unit-fixture-not-an-actual-review",
        "models": list(runner.VARIANT_MODELS.values()), "fidelity": "paper_interpretive",
        "source_sha256": formal.source_identity()})


def suite_fixture(tmp_path):
    native = runner.synthetic_receipt(tmp_path / "native")
    spec = json.loads(native.read_text()); prepared = tmp_path / "prepared"; prepared.mkdir()
    manifest = {"schema": "q-attention.retacred-reference-improved-baseline-data.v1", "dataset": "Re-TACRED",
                "format": "upstream_tacred_json", "upstream_label_to_id": {n:i for i,n in enumerate(LABELS)}, "splits": {}}
    for split, source in (("train", "train"), ("dev", "valid"), ("test", "test")):
        rows = json.loads((native.parent / (split + ".json")).read_text())
        for row in rows: row.pop("id")
        path = prepared / (split + ".json"); common.atomic_json(path, rows)
        manifest["splits"][source] = {"output_sha256": common.digest(path), "source_sha256": "1"*64,
            "records": len(rows), "source_path": "/another-machine/source.jsonl", "output_path": "/another-machine/valid.json"}
    common.atomic_json(prepared / "data_manifest.json", manifest)
    receipt = tmp_path / "qualified.json"
    common.atomic_json(receipt, {"status":"qualified", "synthetic": True,
        "task":"supervised_sentence_entity_pair_relation_classification", "labels":40, "null_relation":"no_relation",
        "verification":"converter_validates_labels_entity_spans_types_and_full_split_counts", "splits":manifest["splits"]})
    return receipt, prepared


def test_descriptor_is_backend_free_and_frozen_recipe():
    result = subprocess.run([sys.executable, "-c", "from baselines.retacred import ctl_drp_formal as f; "
        "import sys,json; a=f.parser().parse_args(['--describe']); print(json.dumps(f.describe(a))); "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["model"] == "ctl_drp_supervised" and value["budget_updates"] == 1150
    assert value["recipe"] == RECIPE and not value["formal_execution_admitted"]
    assert not value["default_cohort_changed"] and len(common.select_models("ready")) == 5


@pytest.mark.parametrize("variant", ["sup", "unsup"])
def test_exact_checkpoint_to_registry_id_not_abbreviation(variant):
    spec = load_lock()["facts"]["models"][variant]
    receipt = {"id":spec["id"], "revision":spec["revision"], "weight_sha256":spec["weight"]["sha256"],
               "strict_parameter_equality":True}
    assert runner.formal_model_id(receipt) == runner.VARIANT_MODELS[variant]
    for key, value in (("id", variant), ("weight_sha256", "0"*64), ("revision", "changed"), ("strict_parameter_equality", False)):
        with pytest.raises(ValueError): runner.formal_model_id({**receipt, key:value})


def test_admission_blocks_before_model_data_git_upload_or_directory(tmp_path, monkeypatch):
    args = options(tmp_path)
    monkeypatch.setattr(formal, "require_review", lambda *_:pytest.fail("review should follow registry gate"))
    monkeypatch.setattr(publishing, "publication_preflight", lambda **_:pytest.fail("Git should follow registry gate"))
    with pytest.raises(ValueError, match="not runnable"): formal.dispatch(args)
    assert not args.run_dir.exists() and not formal.control_path(args).exists()


def test_generic_marker_queue_never_accepts_even_a_future_admitted_CTL(monkeypatch):
    from baselines.retacred import orchestrate
    from types import SimpleNamespace
    monkeypatch.setattr(sys,"argv",["orchestrate.py"])
    monkeypatch.setattr(orchestrate,"parse_args",lambda:SimpleNamespace(models="ctl_drp_supervised"))
    monkeypatch.setattr(orchestrate,"select_models",lambda _:[{"id":"ctl_drp_supervised"}])
    with pytest.raises(ValueError,match="dedicated ctl-run"): orchestrate.main()


def test_independent_review_binds_entire_source_and_grade(tmp_path):
    path = tmp_path / "review.json"; review_file(path)
    assert formal.require_review(path, "ctl_drp_supervised")["fidelity"] == "paper_interpretive"
    initial = json.loads(path.read_text())
    for key, value in (("status", "pending"), ("reviewer_role", "self"), ("fidelity", "paper_faithful"),
                       ("models", ["ctl_drp_sup"]), ("source_sha256", {})):
        common.atomic_json(path, {**initial, key:value})
        with pytest.raises(ValueError, match="independent"): formal.require_review(path, "ctl_drp_supervised")


def test_native_qualification_explicit_diagnostic_only(tmp_path):
    receipt = runner.synthetic_receipt(tmp_path / "native")
    with pytest.raises(ValueError, match="non-synthetic"): qualification.materialize(receipt)
    assert qualification.materialize(receipt, allow_synthetic=True) == receipt
    with pytest.raises(ValueError, match="does not take"): qualification.materialize(receipt, prepared=tmp_path)


def test_suite_bridge_preserves_rows_hashes_H_and_ids_with_portable_paths(tmp_path):
    receipt, prepared = suite_fixture(tmp_path)
    output = tmp_path / "bound"
    with pytest.raises(ValueError, match="boundary mismatch"): qualification.materialize(receipt, prepared=prepared, output=output)
    native = qualification.materialize(receipt, prepared=prepared, output=output, allow_synthetic=True)
    data = PhysicalSplits(native)
    assert set(data.records) == {"train", "dev", "test"} and all(len(r) == 3 if s == "train" else len(r) == 2 for s,r in data.records.items())
    assert data.binding["source"] == "physical_train_only" and all(not c["unknown_pair_indices"] for c in data.coverage.values())
    assert data.identity["interface_qualification"]["parent"]["qualification_sha256"] == common.digest(receipt)
    assert data.identity["splits"]["dev"]["qualified_prepared_sha256"] == common.digest(prepared / "dev.json")
    assert all(r["id"].startswith(f"ctl:{s}:") for s,rows in data.records.items() for r in rows)
    assert qualification.materialize(receipt, prepared=prepared, output=output, allow_synthetic=True) == native
    with (prepared / "train.json").open("a") as f: f.write(" ")
    with pytest.raises(ValueError, match="hash drift"): qualification.materialize(receipt, prepared=prepared, output=output, allow_synthetic=True)
    assert common.digest(native) == data.identity["qualification_sha256"]


@pytest.mark.parametrize("fault", ["count", "qualified", "synthetic_to_real", "duplicate_id"])
def test_suite_boundary_rejects_false_qualification_and_never_publishes(tmp_path, fault):
    receipt, prepared = suite_fixture(tmp_path); spec = json.loads(receipt.read_text())
    if fault == "count": spec["splits"]["train"]["records"] += 1
    elif fault == "qualified": spec["status"] = "unqualified"
    elif fault == "synthetic_to_real": spec["synthetic"] = False
    else:
        path = prepared / "train.json"; rows = json.loads(path.read_text())
        for row in rows: row["id"] = "duplicate"
        common.atomic_json(path, rows); spec["splits"]["train"]["output_sha256"] = common.digest(path)
        manifest = json.loads((prepared / "data_manifest.json").read_text()); manifest["splits"] = spec["splits"]
        common.atomic_json(prepared / "data_manifest.json", manifest)
    common.atomic_json(receipt, spec)
    with pytest.raises(ValueError): qualification.materialize(receipt, prepared=prepared, output=tmp_path / "bad", allow_synthetic=True)
    assert not (tmp_path / "bad").exists() and not list(tmp_path.glob("bad.stage-*"))


def transport_fixture(tmp_path):
    run = tmp_path / "run"; (run / "case_study").mkdir(parents=True)
    file = run / "case_study/unit.pt"; file.write_bytes(b"fixture-only-not-a-real-tensor")
    summary = {"schema":"unit-plumbing-only", "status":"compute_complete", "model":"ctl_drp_supervised", "seed":13,
        "updates":1150, "formal_training":True, "formal_comparison_admitted":False, "contract_sha256":"0"*64,
        "splits":{"dev":{"micro_f1":.5}, "test":{"micro_f1":.5}},
        "data":{"synthetic":False, "splits":{s:{"sha256":"1"*64,"count":n} for s,n in formal.COMPLETE_COUNTS.items()}},
        "case_study":{"traces":[{"phase":"train","name":"unit.json","sha256":"2"*64},{"phase":"test","name":"unit-test.json","sha256":"3"*64}]}}
    common.atomic_json(run / "run_summary.json", summary)
    files = {file.name:common.digest(file)}
    delivery = {"status":"complete", "target_dir":f"q-attention/retacred-reference-suite/{run.name}",
        "artifact_identity":{f"ctl_drp_seed13/{file.name}":common.digest(file)},
        "files":[{"path":f"ctl_drp_seed13/{file.name}","sha256":common.digest(file), "size_bytes":file.stat().st_size,"uploaded":True}]}
    return run, summary, files, delivery


@pytest.mark.parametrize("fault", ["missing", "duplicate", "size", "hash", "namespace"])
def test_delivery_binds_exact_file_receipts_not_success_flags(tmp_path, fault):
    run, summary, files, delivery = transport_fixture(tmp_path)
    if fault == "missing": delivery["files"] = []
    elif fault == "duplicate": delivery["files"] *= 2
    elif fault == "namespace": delivery["target_dir"] += "-wrong"
    else: delivery["files"][0]["size_bytes" if fault == "size" else "sha256"] = 0 if fault == "size" else "0"*64
    with pytest.raises(ValueError, match="delivery failed"): exporter.deliver_required(run, summary, files, lambda:delivery)
    assert not (run / "ctl_delivery_receipt.json").exists() and not (run / "RUN_COMPLETE").exists()


def test_shards_join_and_receipt_coverage(tmp_path, monkeypatch):
    from baselines.retacred.assets import shard_file
    run, summary, files, delivery = transport_fixture(tmp_path); file = run / "case_study/unit.pt"
    parts_dir = run / "exchange_parts/ctl_drp_seed13"; parts = shard_file(file, parts_dir, shard_bytes=9)
    manifest = parts_dir / "unit.pt.parts.json"
    common.atomic_json(manifest, {"name":file.name,"sha256":common.digest(file),"size_bytes":file.stat().st_size,"parts":parts})
    monkeypatch.setattr(exporter, "SHARD_AT", 1)
    inventory = exporter.delivery_inventory(run, summary, files)
    delivery["files"] = [{"path":name, **row, "uploaded":True} for name,row in inventory.items()]
    assert len(inventory) == len(parts) + 1
    assert exporter.deliver_required(run, summary, files, lambda:delivery)["status"] == "complete"
    (parts_dir / parts[0]["name"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard bytes drift"): exporter.delivery_inventory(run, summary, files)


def test_export_crash_after_rename_restores_marker_without_delivery_or_training(tmp_path, monkeypatch):
    run, summary, files, delivery = transport_fixture(tmp_path)
    monkeypatch.setattr(exporter, "validate_run", lambda _:(summary, {"diagnostic":False}, files))
    report = tmp_path / "private-unit-report"
    exporter.export(run, report, deliver=lambda:delivery)
    (run / "RUN_COMPLETE").unlink()
    exporter.export(run, report, deliver=lambda:pytest.fail("cached complete receipt must avoid transport"))
    assert (run / "RUN_COMPLETE").is_file()
    public = json.loads((report / "run_summary.json").read_text())
    assert public["fidelity"] == "paper_interpretive" and public["model"] == "ctl_drp_supervised"
    assert public["case_study"]["traces"] == summary["case_study"]["traces"]
    assert "model=ctl_drp_supervised" in (report / "run_summary.data").read_text()
    assert not list(report.glob("*.pt"))
    stored = json.loads((run / "ctl_delivery_receipt.json").read_text()); stored["files"][0]["sha256"] = "0"*64
    common.atomic_json(run / "ctl_delivery_receipt.json", stored); (run / "RUN_COMPLETE").unlink()
    with pytest.raises(ValueError, match="delivery failed"): exporter.export(run, report)
    assert not (run / "RUN_COMPLETE").exists()


def test_export_failure_retains_compute_and_retry_no_optimizer(tmp_path, monkeypatch):
    run, summary, files, delivery = transport_fixture(tmp_path); (run / "COMPUTE_COMPLETE").touch()
    monkeypatch.setattr(exporter, "validate_run", lambda _:(summary, {"diagnostic":False}, files))
    def fail(): raise ConnectionError("injected transport failure")
    with pytest.raises(ConnectionError): exporter.export(run, tmp_path / "report", deliver=fail)
    assert (run / "COMPUTE_COMPLETE").exists() and not (run / "RUN_COMPLETE").exists()
    assert not (tmp_path / "report").exists()
    exporter.export(run, tmp_path / "report", deliver=lambda:delivery)
    assert (run / "RUN_COMPLETE").exists()


def test_build_worker_no_marker_settings_and_initialization_retry(tmp_path):
    args = options(tmp_path, "--resume", "--gpu", "auto")
    command = formal.build_command(args)
    assert "baselines.retacred.ctl_drp_formal" in command and "--batch-size" not in command
    assert "--resume" not in command  # parent exists, initial worker never wrote a generation
    args.run_dir.mkdir(); common.atomic_json(args.run_dir / "contract.json", {})
    assert "--resume" in formal.build_command(args)
    assert UpdatePlan(58465,13).total == 1150


def test_export_only_has_no_worker_weight_gpu_or_qualification_path(tmp_path, monkeypatch):
    args = options(tmp_path, "--export-only", "--no-push"); review_file(args.review_receipt)
    monkeypatch.setattr(formal, "select_models", lambda _:[{}])
    model, review = formal.gates(args)
    control = formal.control_path(args); control.mkdir()
    common.atomic_json(control / "contract.json", {"model":model,"review":review,"seed":13})
    monkeypatch.setattr(publishing, "publication_preflight", lambda **_:None)
    monkeypatch.setattr(formal, "validate_compute", lambda *_:{"fixture":True})
    seen = []
    monkeypatch.setattr(formal, "publish", lambda *a,**k:seen.append((a,k)))
    monkeypatch.setattr(formal, "launch", lambda *_:pytest.fail("export-only launched a worker"))
    monkeypatch.setattr(formal, "require_disk", lambda *_:pytest.fail("export-only needs no weight/disk preflight"))
    assert formal.dispatch(args) == 0 and len(seen) == 1


def git_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir()
    def cmd(*args): return subprocess.check_output(["git",*args],cwd=root,text=True).strip()
    cmd("init"); cmd("symbolic-ref","HEAD","refs/heads/1.1")
    cmd("config","user.name","Fixture Owner"); cmd("config","user.email","fixture@example.invalid")
    (root / "anchor").write_text("fixture"); cmd("add","anchor"); cmd("commit","-m","initial")
    for ref in ("origin/1.1","origin/main"): cmd("update-ref", "refs/remotes/"+ref, "HEAD")
    monkeypatch.setattr(formal,"ROOT",root); monkeypatch.setattr(publishing,"ROOT",root)
    monkeypatch.setattr(formal,"git",cmd); monkeypatch.setattr(publishing,"git",cmd)
    return root, cmd


def test_report_only_real_git_commit_and_retry_without_changes(tmp_path, monkeypatch):
    root, git = git_repo(tmp_path, monkeypatch)
    run, summary, files, delivery = transport_fixture(tmp_path / "inputs")
    monkeypatch.setattr(exporter, "validate_run", lambda _:(summary,{"diagnostic":False},files))
    monkeypatch.setattr(publishing,"upload_evidence",lambda *_:delivery)
    report = formal.report_path(run); original = git("rev-parse","HEAD")
    formal.publish(run, report, no_push=True)
    head = git("rev-parse","HEAD"); assert head != original and git("rev-parse","HEAD^") == original
    names = git("diff-tree","--no-commit-id","--name-only","-r","HEAD").splitlines()
    assert len(names) == 7 and all(n.startswith(report.relative_to(root).as_posix()+"/") for n in names)
    assert not git("status","--porcelain") and git("show","-s","--format=%an") == "Fixture Owner"
    formal.publish(run, report, no_push=True); assert git("rev-parse","HEAD") == head
    (root / "foreign").write_text("do not stage"); git("add","foreign")
    with pytest.raises(ValueError, match="unrelated"): formal.publish(run, report, no_push=True)
    assert git("diff","--cached","--name-only") == "foreign"


def test_disk_gate_is_conservative_for_checkpoint_event_pins(tmp_path):
    args = options(tmp_path)
    with pytest.raises(ValueError, match="disk headroom"): formal.require_disk(args,{"size_bytes":10**15})


def test_actual_parent_child_command_environment_exit_and_handler_cleanup(tmp_path, monkeypatch):
    import signal
    args = options(tmp_path); control = formal.control_path(args); control.mkdir()
    command = [sys.executable,"-c","import os; print('VISIBLE='+os.environ['CUDA_VISIBLE_DEVICES']); print('FIXTURE_CHILD_EXIT0')"]
    monkeypatch.setattr(formal,"build_command",lambda _:command)
    handlers = {s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    assert formal.launch(args,control,"0") == 0
    record = json.loads((control / "command.json").read_text())
    assert record["argv"] == command and record["selected_gpu"] == "0" and record["child_exit_status"] == 0
    assert "VISIBLE=0" in (control / "worker.log").read_text()
    assert all(signal.getsignal(s) == h for s,h in handlers.items())
    with pytest.raises(ChildProcessError): __import__('os').waitpid(record["pid"],__import__('os').WNOHANG)


@pytest.mark.parametrize("fault", [None,"recipe","budget","model","physical_identity"])
def test_compute_bridge_checks_method_specific_contract_not_marker_shape(tmp_path, monkeypatch, fault):
    args = options(tmp_path); init = {"id":"locked/unit-fixture","weight_sha256":"1"*64}
    bound = {"model":"ctl_drp_supervised","seed":13,"micro_batch":4,
             "weight":{"id":init["id"],"sha256":init["weight_sha256"]},"physical_qualification_sha256":"2"*64}
    data = {"qualification_sha256":"2"*64,"synthetic":False}
    task = {"recipe":copy.deepcopy(RECIPE),"physical_micro_batch_sentences":4,"initialization":init,"data_H":{"data":data}}
    summary = {"model":bound["model"],"seed":13,"updates":1150,"data":data}
    if fault == "recipe": task["recipe"]["effective_batch_sentences"] = 64
    elif fault == "budget": summary["updates"] = 100
    elif fault == "model": summary["model"] = "ctl_drp_sup"
    elif fault == "physical_identity": data["qualification_sha256"] = "3"*64
    monkeypatch.setattr(exporter,"validate_run",lambda _:(summary,{"diagnostic":False,"task":task},{}))
    if fault is None: assert formal.validate_compute(args,bound) == summary
    else:
        with pytest.raises(ValueError,match="result mismatch"): formal.validate_compute(args,bound)


def test_push_failure_retries_report_only_without_duplicate_commit_or_upload(tmp_path, monkeypatch):
    root, git = git_repo(tmp_path, monkeypatch)
    run, summary, files, delivery = transport_fixture(tmp_path / "inputs")
    monkeypatch.setattr(exporter,"validate_run",lambda _:(summary,{"diagnostic":False},files))
    uploads = []
    monkeypatch.setattr(publishing,"upload_evidence",lambda *_:uploads.append(True) or delivery)
    actual = subprocess.run; pushes = []
    def transport(argv, *a, **k):
        if argv[:2] == ["git","ls-remote"]: return subprocess.CompletedProcess(argv,0)
        if "push" in argv:
            pushes.append(argv)
            if len(pushes) == 1: raise subprocess.CalledProcessError(1,argv)
            return subprocess.CompletedProcess(argv,0)
        return actual(argv,*a,**k)
    monkeypatch.setattr(subprocess,"run",transport)
    report = formal.report_path(run)
    with pytest.raises(subprocess.CalledProcessError): formal.publish(run,report,no_push=False)
    head = git("rev-parse","HEAD")
    assert json.loads((run / "publication.json").read_text())["status"] == "committed"
    formal.publish(run,report,no_push=False)
    assert len(pushes) == 2 and len(uploads) == 1 and git("rev-parse","HEAD") == head
    assert json.loads((run / "publication.json").read_text())["status"] == "pushed"
