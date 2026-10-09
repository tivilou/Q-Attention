"""Bounded actual runner failures/recovery/traces; no full task or HTTP/Git action."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from baselines.retacred import ctl_drp_runner as runner
from baselines.retacred import ctl_drp_trace as trace
from baselines.retacred import ctl_drp_export as export
from baselines.retacred.common import atomic_json, digest, fingerprint
from baselines.retacred.ctl_drp_cuda_smoke import state_hash
from baselines.retacred.ctl_drp_neural import tiny_model, tokenizer_from_cache, descriptions_from_native
from baselines.retacred.ctl_drp_task import CTLDRPTask, PhysicalSplits


@pytest.fixture(scope="module")
def assets():
    import torch
    torch.set_num_threads(1)
    return (tokenizer_from_cache(Path(os.environ["CTL_DRP_BPE_CACHE"])),
            descriptions_from_native(Path(os.environ["CTL_DRP_NATIVE"])))


def make(path,assets,*,device="cpu",micro=2):
    data=PhysicalSplits(runner.synthetic_receipt(path))
    return CTLDRPTask(data,*assets,tiny_model(seed=13,dropout=.1),device=device,micro_batch=micro)


def load_latest(path):
    import torch
    pointer=json.loads((path/"latest.json").read_text())
    return torch.load(path/"checkpoints"/pointer["generation"]/"state.pt",map_location="cpu",weights_only=True,mmap=True)


@pytest.fixture(scope="module")
def completed(tmp_path_factory,assets):
    base=Path(os.environ.get("CTL_DRP_RUNNER_EVIDENCE",str(tmp_path_factory.mktemp("ctl-runner"))))
    base.mkdir(parents=True,exist_ok=True)
    adapter=make(base/"physical-fixture",assets)
    summary=runner.run(adapter,base/"complete-cpu")
    return base,adapter,summary


def test_actual_completion_markers_budget_and_full_split_metrics(completed):
    base,adapter,summary=completed;path=base/"complete-cpu"
    assert summary["status"]=="diagnostic_complete" and summary["updates"]==2 and not summary["formal_training"]
    assert (path/"DIAGNOSTIC_COMPLETE").is_file() and not (path/"RUN_COMPLETE").exists() and not (path/"COMPUTE_COMPLETE").exists()
    assert summary["splits"]["test"]["items"]==2 and len(summary["splits"]["test"]["per_class"])==40
    assert runner.validate_complete(path,adapter)==summary
    assert json.loads((path/"heartbeat.json").read_text())["batch"]==2


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_pause_resume_exact_next_update_and_final_evaluation(tmp_path,assets,device):
    import torch
    if device.startswith("cuda"):assert torch.cuda.is_available()
    full=make(tmp_path/"a",assets,device=device);runner.run(full,tmp_path/"full")
    expected=load_latest(tmp_path/"full")
    partial=make(tmp_path/"b",assets,device=device)
    paused=runner.run(partial,tmp_path/"partial",pause_after=1)
    assert paused=={"status":"paused","updates":1} and not (tmp_path/"partial/DIAGNOSTIC_COMPLETE").exists()
    resume=make(tmp_path/"b",assets,device=device);result=runner.run(resume,tmp_path/"partial",resume=True)
    actual=load_latest(tmp_path/"partial")
    for key in ("model","optimizer","torch_rng","python_rng"):
        assert state_hash(expected[key])==state_hash(actual[key]),key
    if device.startswith("cuda"):
        for key in ("cuda_rng","numpy_rng"):assert state_hash(expected[key])==state_hash(actual[key]),key
    assert result["updates"]==2 and not (tmp_path/"partial/RUN_PAUSED.json").exists()
    assert result["splits"]==json.loads((tmp_path/"full/run_summary.json").read_text())["splits"]


def test_resume_before_first_update_and_reject_existing_run(tmp_path,assets):
    adapter=make(tmp_path/"data",assets)
    assert runner.run(adapter,tmp_path/"run",pause_after=0)["updates"]==0
    assert load_latest(tmp_path/"run")["cursor"]==0
    with pytest.raises(ValueError,match="fresh run"):runner.run(make(tmp_path/"data",assets),tmp_path/"run")
    assert runner.run(make(tmp_path/"data",assets),tmp_path/"run",resume=True)["updates"]==2


def test_crash_after_uncommitted_update_quarantines_orphan_trace_and_replays(tmp_path,assets):
    adapter=make(tmp_path/"data",assets)
    def crash(cursor):raise RuntimeError("injected after update before durable checkpoint")
    with pytest.raises(RuntimeError,match="injected"):
        runner.run(adapter,tmp_path/"run",inject_after_update=crash)
    assert json.loads((tmp_path/"run/checkpoints"/json.loads((tmp_path/"run/latest.json").read_text())["generation"]/"state.json").read_text())["cursor"]==0
    assert list((tmp_path/"run/case_study").glob("*.pt"))
    result=runner.run(make(tmp_path/"data",assets),tmp_path/"run",resume=True)
    assert result["updates"]==2 and len(list((tmp_path/"run/orphaned_trace_attempts").iterdir()))==2


def test_crash_after_final_evaluation_retries_only_finalization(tmp_path,assets,monkeypatch):
    adapter=make(tmp_path/"data",assets);original=adapter.evaluate
    def fail(split,**kwargs):
        result=original(split,**kwargs)
        if split=="test":raise RuntimeError("injected finalization failure")
        return result
    monkeypatch.setattr(adapter,"evaluate",fail)
    with pytest.raises(RuntimeError,match="finalization"):runner.run(adapter,tmp_path/"run")
    assert load_latest(tmp_path/"run")["cursor"]==2
    fresh=make(tmp_path/"data",assets)
    monkeypatch.setattr(fresh,"train_update",lambda *_a,**_k:pytest.fail("training was repeated"))
    assert runner.run(fresh,tmp_path/"run",resume=True)["updates"]==2


def test_contract_and_corrupt_checkpoint_rejected_before_parameter_rng_mutation(tmp_path,assets):
    import torch
    runner.run(make(tmp_path/"data",assets),tmp_path/"run",pause_after=1)
    other=make(tmp_path/"data",assets,micro=1);before=state_hash(other.model.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="contract drift"):runner.run(other,tmp_path/"run",resume=True)
    assert before==state_hash(other.model.state_dict()) and torch.equal(rng,torch.get_rng_state())
    pointer=json.loads((tmp_path/"run/latest.json").read_text());weight=tmp_path/"run/checkpoints"/pointer["generation"]/"state.pt"
    with weight.open("ab") as f:f.write(b"drift")
    other=make(tmp_path/"data",assets);before=state_hash(other.model.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="numeric integrity drift"):runner.run(other,tmp_path/"run",resume=True)
    assert before==state_hash(other.model.state_dict()) and torch.equal(rng,torch.get_rng_state())


def test_shared_v2_structure_and_all_numeric_representation_values(completed):
    base,adapter,_=completed
    path=Path(os.environ["CTL_DRP_TRACE_VALIDATOR"])
    spec=importlib.util.spec_from_file_location("shared_trace_validator",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for file in (base/"complete-cpu/case_study").glob("*.json"):
        payload=json.loads(file.read_text());errors,warnings=module.validate(payload,require_complete=True)
        assert not errors and not warnings,(errors,warnings)
        readiness=trace.validate(file,layers=2)
        assert readiness["status"]=="complete_local_values_verified" and readiness["representations"]>=29
        reader=trace.ValueReader(file)
        if payload["samples"][0]["split"]=="train":
            assert reader.tensor("head_before").shape==reader.tensor("head_after").shape
            assert not __import__('torch').equal(reader.tensor("head_before"),reader.tensor("head_after"))
            assert reader.tensor("selected_loss").item()>=0


def test_arbitrary_prefix_matrix_vector_reader_not_a_fixed_preview(completed):
    import torch
    file=next((completed[0]/"complete-cpu/case_study").glob("test*.json"));reader=trace.ValueReader(file)
    tensor=reader.tensor("layer0.attention_probabilities")
    prefix=(tensor.shape[0]-1,tensor.shape[1]-1)
    result=reader.read("layer0.attention_probabilities",prefix=prefix,window=[(1,4),(5,9)])
    assert result["values"]==tensor[prefix+(slice(1,4),slice(5,9))].tolist() and result["shape"]==[3,4]
    vector=reader.tensor("first_token")
    assert reader.read("first_token",prefix=(vector.shape[0]-1,))["values"]==vector[-1].tolist()
    for prefix in ((-1,), (True,), (0,0,0,0,0)):
        with pytest.raises(ValueError):reader.read("layer0.attention_probabilities",prefix=prefix)
    with pytest.raises(ValueError,match="prefix axes"):reader.read("layer0.attention_probabilities")


def test_recorder_does_not_change_training_or_eval_mechanism(tmp_path,assets):
    recorded=make(tmp_path/"a",assets);runner.run(recorded,tmp_path/"recorded")
    plain=make(tmp_path/"b",assets);runner.seed_runtime(13);optimizer=plain.make_optimizer()
    for cursor in range(2):plain.train_update(plain.plan.at(cursor)["indices"],optimizer)
    dev,_=plain.evaluate("dev");test,_=plain.evaluate("test")
    assert state_hash(recorded.model.state_dict())==state_hash(plain.model.state_dict())
    summary=json.loads((tmp_path/"recorded/run_summary.json").read_text())
    assert summary["splits"]=={"dev":dev,"test":test}


def test_private_export_is_idempotent_complete_and_no_raw_values(completed,tmp_path):
    base,_,_=completed;report=tmp_path/"private-report"
    with pytest.raises(ValueError,match="synthetic evidence"):export.export(base/"complete-cpu",report)
    assert export.export(base/"complete-cpu",report,allow_diagnostic=True)==report
    assert export.validate_report(report)=={"status":"whitelist_verified","files":6}
    assert export.export(base/"complete-cpu",report,allow_diagnostic=True)==report
    assert all((report/n).read_text().strip() for n in ("data.sha256","data_counts.txt","run_summary.data"))
    for path in report.iterdir():
        text=path.read_text()
        assert "Alice" not in text and "Bob" not in text and "Acme" not in text and "input_ids" not in text
    assert not (base/"complete-cpu/RUN_COMPLETE").exists()
    (report/"unexpected.pt").touch()
    with pytest.raises(ValueError,match="whitelist"):export.validate_report(report)


def test_missing_values_and_incomplete_manifest_prevent_export(completed,tmp_path):
    import shutil
    source=completed[0]/"complete-cpu";copy_path=tmp_path/"broken"
    shutil.copytree(source,copy_path)
    file=next((copy_path/"case_study").glob("test*.json"));payload=json.loads(file.read_text())
    payload["samples"][0]["representations"][0]["value_capture"]["stored_element_count"]=1
    atomic_json(file,payload)
    summary=json.loads((copy_path/"run_summary.json").read_text())
    for t in summary["case_study"]["traces"]:
        if t["name"]==file.name:t["sha256"]=digest(file)
    atomic_json(copy_path/"run_summary.json",summary)
    with pytest.raises(ValueError,match="coordinate coverage"):export.export(copy_path,tmp_path/"report",allow_diagnostic=True)
    assert not (tmp_path/"report").exists() and not (copy_path/"RUN_COMPLETE").exists()


def test_diagnostic_identity_cannot_be_promoted_by_changing_completion_flags(completed,tmp_path):
    import shutil
    path=tmp_path/"forged";shutil.copytree(completed[0]/"complete-cpu",path)
    contract=json.loads((path/"contract.json").read_text());contract["diagnostic"]=False;atomic_json(path/"contract.json",contract)
    summary=json.loads((path/"run_summary.json").read_text());summary["status"]="compute_complete";summary["contract_sha256"]=fingerprint(contract);atomic_json(path/"run_summary.json",summary)
    (path/"COMPUTE_COMPLETE").touch()
    with pytest.raises(ValueError,match="cannot be relabeled"):export.export(path,tmp_path/"public-report",deliver=lambda:pytest.fail("transport should not be called"))


def test_required_transport_receipt_and_retry_without_training(tmp_path):
    # Transport-unit fixture only: neither a formal run nor an actual service call.
    (tmp_path/"case_study").mkdir();file=tmp_path/"case_study/fixture.pt";file.write_bytes(b"transport-unit-fixture")
    files={"fixture.pt":digest(file)};expected={"ctl_drp_seed13/fixture.pt":digest(file)}
    item={"path":"ctl_drp_seed13/fixture.pt","sha256":digest(file),"size_bytes":file.stat().st_size,"uploaded":False}
    bad={"status":"complete","target_dir":f"q-attention/retacred-reference-suite/{tmp_path.name}","artifact_identity":expected,"files":[item]}
    with pytest.raises(ValueError,match="delivery failed"):
        export.deliver_required(tmp_path,{"seed":13},files,lambda:bad)
    assert json.loads((tmp_path/"ctl_delivery_pending.json").read_text())["status"]=="pending"
    good={**bad,"files":[{**item,"uploaded":True}]}
    assert export.deliver_required(tmp_path,{"seed":13},files,lambda:good)==good
    assert json.loads((tmp_path/"ctl_delivery_pending.json").read_text())["status"]=="complete"
    assert not (tmp_path/"RUN_COMPLETE").exists()


def test_lock_prevents_duplicate_owner_before_checkpoint_load(tmp_path,assets):
    import fcntl
    runner.run(make(tmp_path/"data",assets),tmp_path/"run",pause_after=0)
    with (tmp_path/"run/runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match="active owner"):
            runner.run(make(tmp_path/"data",assets),tmp_path/"run",resume=True)
