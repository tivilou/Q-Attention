"""Mechanism parity and actual parent/child recovery and export contracts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from q_attention.adapters.ecpa_formal import ECPAExecution,ECPAFormalModel
from q_attention.experiments import ecpa_protocol as p
from q_attention.experiments.ecpa_runner import child_spec,gpu_ids,completed
from q_attention.analysis.ecpa_formal_trace import validate
from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig
from q_attention.plugins.q_ecpa import EntityCoherentPath,ECPAConfig


@pytest.mark.parametrize("mode",["quantum","dephased","classical","random_phase"])
def test_fast_branch_equals_full_equations_and_gradient(mode):
    torch.set_num_threads(2)
    torch.manual_seed(77)
    full=EntityCoherentPath(ECPAConfig(heads=2,head_dim=4),mode).double()
    fast=ECPAExecution(full.config,mode).double()
    fast.load_state_dict(full.state_dict())
    q=torch.randn(2,2,7,4,dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
    s=torch.zeros(2,7,dtype=torch.bool);s[:,0]=True
    o=torch.zeros(2,7,dtype=torch.bool);o[:,3]=True
    scores=q@q.transpose(-1,-2)
    a=full(q,scores,mask,s,o)[0]; b=fast(q,scores,mask,s,o)[0]
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    parameters=[q,*[v for v in full.parameters() if v.requires_grad]]
    ga=torch.autograd.grad(a.square().sum(),parameters,retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),[q,*[v for v in fast.parameters() if v.requires_grad]])
    for left,right in zip(ga,gb):
        torch.testing.assert_close(left,right,atol=2e-10,rtol=2e-10)


@pytest.mark.parametrize("mode",["disabled","quantum","original_qk"])
def test_capture_recomputation_and_entity_boundary(mode):
    torch.manual_seed(17)
    config=p.load_config(True)
    base=RelationExtractionModel(RelationTransformerConfig(vocab_size=20,num_labels=3,max_length=9,**config["model"]))
    untouched=copy.deepcopy(base)
    model=ECPAFormalModel(base,mode,config)
    ids=torch.tensor([[2,3,4,5,6,0]])
    mask=ids!=0;s=ids==2;o=ids==5
    a=model(ids,mask,s,o)
    b=model(ids,mask,s,o,capture=True)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    assert model.observed["embedding"].shape==(6,32)
    assert "layer1_hidden" in model.observed
    if mode=="disabled":
        torch.testing.assert_close(a,untouched(ids,mask,s,o),atol=0,rtol=0)
    else:
        a.sum().backward()
        before=[x.grad.clone() for x in model.parameters() if x.requires_grad]
        model.zero_grad(set_to_none=True);model.recompute=True
        model(ids,mask,s,o).sum().backward()
        for x,y in zip(before,[x.grad for x in model.parameters() if x.requires_grad]):
            torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
        assert model.scoped is None and not model.capture


def test_gpu_and_linear_tiers():
    assert gpu_ids("auto",inventory=[dict(id="0",free=9000),dict(id="1",free=4000)])==["0"]
    with pytest.raises(ValueError):gpu_ids("0,0",inventory=[dict(id="0",free=9000)])
    tiers=p.memory_tiers(256)
    assert [t["micro"] for t in tiers[:4]]==[256,248,240,232]
    assert tiers[-2:]==[dict(micro=1,recompute=False),dict(micro=1,recompute=True)]


def command(run, *extra):
    return [sys.executable,str(p.ROOT/"experiments/run_ecpa_formal.py"),"--canary","--gpu","cpu:0,cpu:1","--resume" if (run/"run_manifest.json").exists() else "--output-dir",str(run),*extra]


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    if existing:=os.environ.get("ECPA_FORMAL_TEST_RUN"):
        run=Path(existing)
        assert (run/"RUN_COMPLETE.json").is_file()
        return run
    run=tmp_path_factory.mktemp("ecpa-lifecycle")/"run with spaces_seed13"
    first=subprocess.run(command(run,"--inject-oom-once","--stop-after","2"),cwd=p.ROOT,capture_output=True,text=True,timeout=240)
    assert first.returncode==p.PAUSED_EXIT,first.stdout+first.stderr
    assert p.read(run/"jobs/quantum/memory_state.json")["tier"]==1
    state=torch.load(run/"jobs/quantum/latest.pt",weights_only=False)
    assert state["next_update"]==2
    second=subprocess.run(command(run),cwd=p.ROOT,capture_output=True,text=True,timeout=240)
    assert second.returncode==0,second.stdout+second.stderr+"\n"+"\n".join(f"{x.name}: {x.read_text()[-1500:]}" for x in (run/"logs").glob("*.log"))
    return run


def test_actual_queue_complete_full_values_and_topology_resume(finished):
    run=finished;manifest=p.read(run/"run_manifest.json");config=p.read(run/"run_config.json")
    for mode in ("baseline",*p.MODES):
        assert completed(run,mode,manifest)
        report=validate(run/"jobs"/mode/"sample_trace.json",mode,config)
        assert report["samples"]==(27 if mode in ("disabled","random_phase") else 28)
    assert p.read(run/"jobs/quantum/memory_state.json")["tier"]==1
    latest=run/"jobs/quantum/latest.pt";digest=p.sha(latest)
    # Three simulated CPU slots -> one: exercises topology-independent cursor.
    result=subprocess.run([*command(run)[:4],"cpu",*command(run)[5:]],cwd=p.ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert p.sha(latest)==digest
    assert not (run/"RUN_PAUSED.json").exists()


def test_interrupted_validation_does_not_skip_selection_or_overwrite_latest(finished,monkeypatch):
    from q_attention.experiments.ecpa_worker import Worker,main
    worker=Worker(finished,"quantum",device="cpu")
    original=torch.load(worker.latest,weights_only=False)
    # Simulate a durable optimizer boundary whose validation was interrupted.
    state=copy.deepcopy(original);state["selection_update"]=0
    p.save_state(worker.latest,state)
    worker=Worker(finished,"quantum",device="cpu")
    calls=[];evaluate=worker.evaluate
    def observe(split):
        calls.append((split,worker.update));return evaluate(split)
    monkeypatch.setattr(worker,"evaluate",observe)
    assert worker.run()==0
    assert calls[0]==("valid",worker.total)
    committed=p.sha(worker.latest)
    # An interruption after BEST is loaded must leave final LATEST untouched.
    monkeypatch.setattr(Worker,"run",lambda self: (_ for _ in ()).throw(InterruptedError()))
    assert main(finished,"quantum",device="cpu")==p.PAUSED_EXIT
    assert p.sha(worker.latest)==committed


def fake_exchange(run,manifest):
    return dict(status="uploaded",fingerprint=manifest["fingerprint"],target_dir="q-attention/ecpa/canary",
        parts=[dict(name="test.part00000",sha256="a"*64,size_bytes=1,verified=True)])


def test_export_failure_retry_immutable_and_whitelist(finished,tmp_path):
    from q_attention.experiments.ecpa_export import export_report
    destination=tmp_path/"reports/run"
    hashes={str(f):p.sha(f) for f in finished.glob("jobs/*/latest.pt")}
    with pytest.raises(RuntimeError,match="injected"):
        export_report(finished,exchange_fn=fake_exchange,inject_failure=True,destination=destination)
    assert not destination.exists() and list(destination.parent.iterdir())==[]
    report=export_report(finished,exchange_fn=fake_exchange,destination=destination)
    assert export_report(finished,exchange_fn=fake_exchange,destination=destination)==report
    assert all(p.sha(Path(f))==digest for f,digest in hashes.items())
    for path in report.rglob("*"):
        assert path.suffix not in (".pt",".npy",".log",".jsonl")
        if path.is_file():
            text=path.read_text()
            assert "sentence\"" not in text and "data_dir\"" not in text and "Bearer " not in text
    assert p.read(report/"REPORT_COMPLETE.json")["fingerprint"]==p.read(finished/"run_manifest.json")["fingerprint"]


def test_full_value_tamper_blocks_export(finished,tmp_path):
    from q_attention.experiments.ecpa_export import export_report
    file=next((finished/"jobs/quantum/values").rglob("*.npy"))
    original=file.read_bytes()
    try:
        file.write_bytes(original+b"tamper")
        with pytest.raises(ValueError,match="corrupted"):
            export_report(finished,exchange_fn=fake_exchange,destination=tmp_path/"report")
    finally:
        file.write_bytes(original)


def test_report_only_git_push_and_unrelated_changes(tmp_path,monkeypatch):
    from q_attention.experiments.ecpa_export import publish
    root=tmp_path/"repo";root.mkdir()
    remote=tmp_path/"remote.git"
    def git(*args):
        return subprocess.run(["git",*args],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    git("init","-b","1.1")
    git("config","user.name","Fixture Owner");git("config","user.email","fixture@example.invalid")
    (root/"README").write_text("fixture")
    git("add","README");git("commit","-m","fixture")
    git("init","--bare",str(remote));git("remote","add","origin",str(remote))
    git("push","origin","HEAD:1.1","HEAD:main");git("fetch","origin")
    report=root/"reports"/p.EXPERIMENT/"test_seed13";report.mkdir(parents=True)
    (report/"metrics.json").write_text('{}')
    monkeypatch.setattr(p,"ROOT",root)
    (root/"unrelated.txt").write_text("preserve")
    with pytest.raises(ValueError,match="unrelated"):publish(report)
    assert git("diff","--cached","--name-only")==""
    (root/"unrelated.txt").unlink()
    publish(report)
    assert git("status","--porcelain")==""
    assert git("rev-parse","HEAD")==git("rev-parse","origin/1.1")
    # A clean unpublished merge of already-public main is allowed. Use the
    # same file with disjoint hunks, which a combined-diff heuristic mislabels.
    shared="\n".join(str(i) for i in range(50))+"\n"
    (root/"shared.txt").write_text(shared);git("add","shared.txt");git("commit","-m","shared base")
    git("push","origin","HEAD:1.1","HEAD:main");git("fetch","origin")
    git("switch","-c","main")
    (root/"shared.txt").write_text(shared.replace("1\n","main\n",1))
    git("add","shared.txt");git("commit","-m","public main edit");git("push","origin","HEAD:main")
    git("switch","1.1")
    (root/"shared.txt").write_text(shared.replace("48\n","report branch\n"))
    git("add","shared.txt");git("commit","-m","public result-branch edit");git("push","origin","HEAD:1.1")
    git("fetch","origin");git("merge","--no-edit","origin/main")
    publish(report)
    assert git("rev-parse","HEAD")==git("rev-parse","origin/1.1")
    remote_before=git("rev-parse","origin/1.1")
    (root/"unrelated.txt").write_text("committed private work")
    git("add","unrelated.txt");git("commit","-m","unrelated local commit")
    with pytest.raises(ValueError,match="unpublished"):publish(report)
    assert git("rev-parse","origin/1.1")==remote_before
