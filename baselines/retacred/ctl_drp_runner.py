"""Portable CTL lifecycle; formal dispatch stays gated, tiny canary is explicit."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import tempfile
import time

from .common import atomic_json, digest, fingerprint, git, select_models
from .ctl_drp_task import CTLDRPTask, LABELS, PhysicalSplits
from .ctl_drp_trace import TraceObserver, observability, validate
from .reference_execution import Cadence,metadata,poll_archives,schedule_archive

VARIANT_MODELS = {"sup": "ctl_drp_supervised", "unsup": "ctl_drp_unsupervised"}


def formal_model_id(initialization):
    """Resolve only exact locked checkpoint IDs, never sup/unsup abbreviations."""
    from .ctl_drp_simcse import load_lock
    models = load_lock()["facts"]["models"]
    for variant, spec in models.items():
        if initialization.get("id") == spec["id"]:
            if (initialization.get("revision") != spec["revision"]
                    or initialization.get("weight_sha256") != spec["weight"]["sha256"]
                    or initialization.get("strict_parameter_equality") is not True):
                raise ValueError("formal initialization identity differs from pinned checkpoint")
            return VARIANT_MODELS[variant]
    raise ValueError("formal initialization is not an exact locked variant ID")


def safe(path):
    path=Path(path).absolute()
    if path==Path(path.anchor) or path.resolve()!=path or any(p.is_symlink() for p in path.parents):
        raise ValueError("regular non-root owned run path required")
    return path


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def runtime(adapter):
    import torch
    import numpy as np
    if getattr(adapter,'distributed',None) is not None:
        result=adapter.distributed.runtime
    elif adapter.device=="cuda:0":
        from .ctl_drp_cuda_checkpoint import runtime as cuda_runtime
        result=cuda_runtime()
    elif adapter.device=="cpu" and adapter.diagnostic:
        result={"device":"cpu_diagnostic_only","torch":str(torch.__version__)}
    else:raise ValueError("scope is one visible CUDA:0 FP32 or explicit tiny CPU diagnostic")
    return {**result,"python":platform.python_version(),"numpy":str(np.__version__)}


def seed_runtime(seed):
    import torch
    import numpy as np
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG",":4096:8")!=":4096:8":raise ValueError("cuBLAS configuration drift")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    configure_determinism()


def configure_determinism():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def initial_snapshot(adapter, optimizer, contract):
    import torch
    from .ctl_drp_cuda_checkpoint import numpy_rng
    from .ctl_drp_cuda_smoke import state_hash
    if optimizer.state:raise ValueError("initial snapshot requires fresh optimizer")
    return {"schema":"ctl-initial-reconstructible.v1","contract":contract,"cursor":0,
            "initial_model_sha256":state_hash(adapter.model.state_dict()),"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state(0) if adapter.device=="cuda:0" else None,
            "python_rng":random.getstate(),"numpy_rng":numpy_rng()}


def checkpoint(adapter, optimizer, run, contract, cursor, traces, *, prior=None):
    from .reference_storage import owner, write_reservation, serialized_bytes, clean_abandoned_ctl_stages
    directory=run/'checkpoints';directory.mkdir(exist_ok=True)
    clean_abandoned_ctl_stages(run,contract)
    size=16*1024**2 if cursor==0 and not getattr(adapter,'distributed',None) else serialized_bytes({'model':adapter.model.state_dict(),'optimizer':optimizer.state_dict()})
    with write_reservation(directory/'state.pt',size):
        result=_checkpoint(adapter,optimizer,run,contract,cursor,traces,prior=prior)
    if owner(run) is not None and traces and (prior is None or len(traces)>len(prior['traces'])):
        frozen=copy.deepcopy(result)
        schedule_archive(lambda:_archive_sample_pins(run,frozen))
    return result


def _archive_sample_pins(run,result):
    from .reference_archive import archive_files,_archive_traces
    # Pin identities are immutable. Never scan/delete rolling generations in
    # the background: the training thread may have published a newer latest.
    for name in result['pinned_generations']:
        p=safe(run/'checkpoints'/name)
        if (p/'state.pt').is_file():
            archive_files(run,[p/'state.pt',p/'state.json'])
            # A sampled post-update pin may still be the live latest.
            latest=json.loads((run/'latest.json').read_text())['generation']
            if name!=latest:archive_files(run,[p/'state.pt'],remove=True)
    _archive_traces(run,result['traces'])


def _checkpoint(adapter, optimizer, run, contract, cursor, traces, *, prior=None):
    import torch
    directory=run/"checkpoints";directory.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=".generation-",dir=directory))
    from .reference_storage import durable_json
    durable_json(stage/'storage-owner.json',{'schema':'ctl-owned-stage.v1','contract_sha256':fingerprint(contract)})
    target=directory/(f"update{cursor:08d}-"+stage.name.removeprefix(".generation-"))
    weights=stage/"state.pt"
    if getattr(adapter,'distributed',None) is not None:
        adapter.distributed.save(weights,optimizer,contract=contract,cursor=cursor)
    elif cursor==0:
        with weights.open("xb") as f:torch.save(initial_snapshot(adapter,optimizer,contract),f)
    elif adapter.device=="cuda:0":
        from .ctl_drp_cuda_checkpoint import save
        save(weights,adapter.model,optimizer,contract=contract,cursor=cursor)
    else:
        from .ctl_drp_neural import save_checkpoint
        save_checkpoint(weights,adapter.model,optimizer,contract=contract,cursor=cursor)
    with weights.open("rb") as f:os.fsync(f.fileno())
    pinned=list(prior.get("pinned_generations",[])) if prior else []
    if prior and traces and not prior["traces"]:
        pinned=sorted(set(pinned+[prior["generation"],target.name]))
    receipt={"schema":"ctl-durable-update.v1","generation":target.name,"cursor":cursor,"contract_sha256":fingerprint(contract),
             "weights_sha256":digest(weights),"traces":copy.deepcopy(traces),"pinned_generations":pinned}
    atomic_json(stage/"state.json",receipt);sync_dir(stage)
    os.rename(stage,target);sync_dir(directory)
    (target/'storage-owner.json').unlink();sync_dir(target)
    pointer={"generation":target.name,"state_sha256":digest(target/"state.json")}
    atomic_json(run/"latest.json",pointer);sync_dir(run)
    # Two rolling generations plus source snapshots pinned for the sampled train
    # event. Never scan/delete another run or follow a symlink.
    owned=[]
    for p in directory.iterdir():
        if re.fullmatch(r"update\d{8}-[A-Za-z0-9_]+",p.name):
            if safe(p).parent!=directory or not p.is_dir():raise ValueError("unsafe checkpoint generation")
            meta=json.loads((p/"state.json").read_text())
            if meta["generation"]!=p.name or meta["contract_sha256"]!=fingerprint(contract):raise ValueError("foreign checkpoint generation")
            owned.append((meta["cursor"],p))
    keep={p.name for _,p in sorted(owned,key=lambda x:(x[0],x[1].name))[-2:]}|set(pinned)|{target.name}
    for _,p in owned:
        if p.name not in keep:shutil.rmtree(p)
    return receipt


def restore(adapter,optimizer,run,contract):
    import torch
    import numpy as np
    from .ctl_drp_cuda_checkpoint import decode_numpy_rng
    from .ctl_drp_cuda_smoke import state_hash
    pointer=json.loads((run/"latest.json").read_text());name=pointer["generation"]
    if not re.fullmatch(r"update\d{8}-[A-Za-z0-9_]+",name):raise ValueError("unsafe latest generation")
    directory=safe(run/"checkpoints"/name)
    if digest(directory/"state.json")!=pointer["state_sha256"]:raise ValueError("checkpoint pointer integrity drift")
    state=json.loads((directory/"state.json").read_text())
    if state["generation"]!=name or state["contract_sha256"]!=fingerprint(contract):raise ValueError("checkpoint contract drift")
    cursor=state["cursor"]
    if type(cursor) is not int or not 0<=cursor<=contract["budget_updates"]:raise ValueError("checkpoint cursor drift")
    for item in state["traces"]:
        trace=safe(run/"case_study"/item["name"])
        if trace.parent!=run/"case_study" or digest(trace)!=item["sha256"]:raise ValueError("durable trace identity drift")
        validate(trace,layers=adapter.model.encoder.config.num_hidden_layers)
    weights=safe(directory/"state.pt")
    if digest(weights)!=state["weights_sha256"]:raise ValueError("checkpoint numeric integrity drift")
    if getattr(adapter,'distributed',None) is not None:
        adapter.distributed.restore(weights,optimizer,contract=contract,cursor=cursor)
    elif cursor==0:
        value=torch.load(weights,map_location="cpu",weights_only=True,mmap=True)
        if value.get("schema")!="ctl-initial-reconstructible.v1" or value.get("contract")!=contract or value.get("cursor")!=0 or optimizer.state or value["initial_model_sha256"]!=state_hash(adapter.model.state_dict()):
            raise ValueError("initial reconstruction drift before mutation")
        rng=value["torch_rng"]
        if rng.dtype!=torch.uint8 or rng.shape!=torch.get_rng_state().shape:raise ValueError("initial CPU RNG drift")
        torch.Generator().set_state(rng);random.Random().setstate(value["python_rng"])
        np_state=decode_numpy_rng(value["numpy_rng"])
        if adapter.device=="cuda:0":
            cuda=value["cuda_rng"]
            if cuda.dtype!=torch.uint8 or cuda.shape!=torch.cuda.get_rng_state(0).shape:raise ValueError("initial CUDA RNG drift")
        torch.set_rng_state(rng);random.setstate(value["python_rng"]);np.random.set_state(np_state)
        if adapter.device=="cuda:0":torch.cuda.set_rng_state(value["cuda_rng"],0)
    elif adapter.device=="cuda:0":
        from .ctl_drp_cuda_checkpoint import load
        if load(weights,adapter.model,optimizer,contract=contract)!=cursor:raise ValueError("numeric cursor mismatch")
    else:
        from .ctl_drp_neural import load_checkpoint
        value=torch.load(weights,map_location="cpu",weights_only=True,mmap=True)
        if value.get("optimizer",{}).get("param_groups")!=optimizer.state_dict()["param_groups"]:raise ValueError("CPU optimizer settings drift")
        del value
        if load_checkpoint(weights,adapter.model,optimizer,contract=contract)!=cursor:raise ValueError("numeric cursor mismatch")
    # Uncommitted event files from a crash before checkpoint publication cannot
    # impersonate an observed durable event. Preserve them in a bounded quarantine.
    case=run/"case_study";declared={r["name"] for r in state["traces"]}
    declared|={Path(n).with_suffix(".pt").name for n in declared}
    for path in list(case.iterdir()) if case.exists() else []:
        if path.name not in declared:
            if not re.fullmatch(r"(?:train|test)-index\d+-update\d+\.(?:json|pt)",path.name) or safe(path).parent!=case:
                raise ValueError("foreign case-study artifact")
            quarantine=run/"orphaned_trace_attempts";quarantine.mkdir(exist_ok=True)
            target=quarantine/(str(time.time_ns())+"-"+path.name)
            path.rename(target)
    return state


def run(adapter, directory, *, resume=False, pause_after=None, diagnostic_budget=2, inject_after_update=None, formal_review_receipt=None):
    import fcntl
    run_dir=safe(directory)
    if run_dir.exists() and not resume:raise ValueError("fresh run required; existing run needs explicit --resume")
    if resume and not (run_dir/"contract.json").is_file():raise ValueError("resume run contract missing")
    if adapter.diagnostic:
        if type(diagnostic_budget) is not int or not 1<=diagnostic_budget<=2:raise ValueError("bounded diagnostic budget is1..2 updates")
        budget=min(diagnostic_budget,adapter.plan.total)
    else:
        # Candidate rows remain blocked until independent review/admission.
        if formal_review_receipt is None:
            select_models(formal_model_id(adapter.initialization))
        else:
            from .reference_complete import require_review
            require_review(formal_review_receipt)
            if formal_model_id(adapter.initialization) not in ('ctl_drp_supervised','ctl_drp_unsupervised'):
                raise ValueError('complete-cohort gate only accepts a pinned CTL variant')
        budget=adapter.plan.total
    configure_determinism()
    from .execution_compat import source_map,review_sha256
    from .reference_execution import reset_update_peak,memory_metadata
    memory_device=next(adapter.model.parameters()).device
    observe=observability(adapter)
    contract={"schema":"ctl-runner-contract.v1","task":adapter.contract,"budget_updates":budget,
              "runtime":runtime(adapter),"observability":observe,"diagnostic":adapter.diagnostic,
              "checkpoint_policy":getattr(adapter,'execution_policy',None) or ("every_completed_logical_update; latest_local; sample_event_pins_verified_archive" if os.environ.get('QATTN_REFERENCE_STORAGE_ROOT') else "every_completed_logical_update; two_rolling_plus_sample_event_pins"),
              "complete_cohort_review_sha256":review_sha256(formal_review_receipt) if formal_review_receipt else None,
              "source":source_map(("ctl_drp_runner.py","ctl_drp_trace.py","ctl_drp_ddp.py"))}
    # Configuration maps may use integer keys/tuples in Python; JSON receipts
    # use string keys/lists. Freeze the same serialized identity on both sides.
    contract=json.loads(json.dumps(contract,sort_keys=True,allow_nan=False))
    if resume and json.loads((run_dir/"contract.json").read_text())!=contract:raise ValueError("immutable runner contract drift before mutation")
    run_dir.mkdir(parents=True,exist_ok=True)
    with (run_dir/"runner.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError("this run already has an active owner") from exc
        if not resume:
            atomic_json(run_dir/"contract.json",contract);atomic_json(run_dir/"observability.json",observe)
        complete="DIAGNOSTIC_COMPLETE" if adapter.diagnostic else "COMPUTE_COMPLETE"
        if (run_dir/complete).exists():
            # A committed result owns its final test trace through the summary,
            # not the last training checkpoint. Validate it before any restore
            # cleanup; normalization retries must not quarantine valid evidence
            # or reload an optimizer that will never perform another update.
            return validate_complete(run_dir,adapter)
        optimizer=adapter.make_optimizer()
        from .reference_storage import initial_save_pending,durable_json,fsync_parent
        initial_marker=run_dir/'INITIALIZATION_PENDING.json'
        if resume and not (run_dir/'latest.json').exists() and initial_save_pending(run_dir,contract):
            seed_runtime(adapter.seed);adapter.model.train()
            checkpoint(adapter,optimizer,run_dir,contract,0,[])
            initial_marker.unlink();fsync_parent(initial_marker)
        if resume:
            state=restore(adapter,optimizer,run_dir,contract)
        else:
            seed_runtime(adapter.seed);adapter.model.train()
            durable_json(initial_marker,{'schema':'reference-initial-save.v1','contract_sha256':fingerprint(contract),'completed_updates':0})
            state=checkpoint(adapter,optimizer,run_dir,contract,0,[])
            initial_marker.unlink();fsync_parent(initial_marker)
        pause=[False]
        handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        for s in handlers:signal.signal(s,lambda *_:pause.__setitem__(0,True))
        experiment={"run_id":run_dir.name,"dataset":"synthetic_retacred_fixture" if adapter.diagnostic else "Re-TACRED",
                    "code_revision":git("rev-parse","HEAD"),"config_sha256":fingerprint(contract),
                    "model_identity":adapter.initialization or "random_tiny_not_pretrained","seed":adapter.seed}
        started=time.monotonic();start_cursor=state["cursor"]
        cadence=Cadence(getattr(adapter,'execution_policy',None),cursor=state['cursor'])
        try:
            while state["cursor"]<budget:
                poll_archives()
                if pause[0] or pause_after is not None and state["cursor"]>=pause_after:
                    if state['cursor']!=cadence.cursor:
                        state=checkpoint(adapter,optimizer,run_dir,contract,state['cursor'],state['traces'],prior=state)
                        cadence.saved(state['cursor'])
                    atomic_json(run_dir/"RUN_PAUSED.json",{"cursor":state["cursor"],"reason":"post_update_pause"})
                    return {"status":"paused","updates":state["cursor"]}
                cursor=state["cursor"];next_batch=adapter.plan.at(cursor)
                if not state['traces'] and observe['selected']['train']['index'] in next_batch['indices'] and cursor!=cadence.cursor:
                    state=checkpoint(adapter,optimizer,run_dir,contract,cursor,state['traces'],prior=state)
                    cadence.saved(cursor)
                observer=None
                if not state["traces"]:
                    observer=TraceObserver(adapter,run_dir/"case_study",phase="train",index=observe["selected"]["train"]["index"],update=cursor+1,
                                           experiment={**experiment,"pre_update_generation":state["generation"],"pre_update_weights_sha256":state["weights_sha256"]})
                reset_update_peak(memory_device)
                update=adapter.train_update(next_batch["indices"],optimizer,observer=observer)
                update_memory=memory_metadata(memory_device)
                if inject_after_update:inject_after_update(cursor+1)
                traces=list(state["traces"])
                if observer and observer.path:traces.append({"phase":"train","name":observer.path.name,"sha256":digest(observer.path)})
                if cadence.due() or len(traces)>len(state['traces']) or (cursor+1)%adapter.plan.per_epoch==0 or cursor+1==budget:
                    state=checkpoint(adapter,optimizer,run_dir,contract,cursor+1,traces,prior=state)
                    cadence.saved(state['cursor'])
                else:
                    state={**state,'cursor':cursor+1,'traces':traces}
                elapsed=time.monotonic()-started;rate=(state["cursor"]-start_cursor)/max(elapsed,1e-9)
                heartbeat={"phase":"training","batch":state["cursor"],"batches":budget,"batches_per_second":rate,
                           "eta_seconds":(budget-state["cursor"])/rate,"loss":update["loss"],
                           **metadata(adapter,state['cursor'],cadence.cursor)}
                heartbeat.update(update_memory)
                if getattr(adapter,'distributed',None):
                    heartbeat.update(world_size=adapter.distributed.world_size,rank_memory=adapter.distributed.last_memory)
                atomic_json(run_dir/"heartbeat.json",heartbeat)
                print(f"[CTL-DRP] {state['cursor']}/{budget} | {rate:.2f} batch/s | ETA {heartbeat['eta_seconds']:.1f}s | loss {update['loss']:.6f}",flush=True)
            if not state["traces"]:raise ValueError("required selected training event never captured")
            from .reference_execution import drain_archives
            drain_archives()
            dev,_=adapter.evaluate("dev")
            observer=TraceObserver(adapter,run_dir/"case_study",phase="test",index=observe["selected"]["test"]["index"],update=budget,experiment=experiment)
            test,predictions=adapter.evaluate("test",observer=observer)
            if observer.path is None:raise ValueError("final test event absent")
            traces=state["traces"]+[{"phase":"test","name":observer.path.name,"sha256":digest(observer.path)}]
            summary={"schema":"ctl-drp-run-summary.v1","status":"diagnostic_complete" if adapter.diagnostic else "compute_complete",
                     "formal_training":not adapter.diagnostic,"formal_comparison_admitted":False,"updates":budget,
                     "model":"ctl_drp" if adapter.diagnostic else formal_model_id(adapter.initialization),
                     "seed":adapter.seed,"splits":{"dev":dev,"test":test},"contract_sha256":fingerprint(contract),
                     "data":adapter.data.identity,"case_study":{"status":"complete_local_values_verified","schema":"sample-trace.v2","traces":traces}}
            atomic_json(run_dir/"predictions.private.json",predictions);atomic_json(run_dir/"run_summary.json",summary)
            validate_complete(run_dir,adapter,require_marker=False)
            (run_dir/complete).touch();sync_dir(run_dir)
            (run_dir/"RUN_PAUSED.json").unlink(missing_ok=True)
            return summary
        finally:
            for s,handler in handlers.items():signal.signal(s,handler)


def validate_complete(run_dir,adapter,*,require_marker=True):
    marker="DIAGNOSTIC_COMPLETE" if adapter.diagnostic else "COMPUTE_COMPLETE"
    if require_marker and not (run_dir/marker).is_file():raise ValueError("compute completion marker absent")
    summary=json.loads((run_dir/"run_summary.json").read_text())
    contract=json.loads((run_dir/"contract.json").read_text())
    if summary["contract_sha256"]!=fingerprint(contract) or summary["updates"]!=contract["budget_updates"] or {t["phase"] for t in summary["case_study"]["traces"]}!={"train","test"}:
        raise ValueError("incomplete budget or split trace coverage")
    for item in summary["case_study"]["traces"]:
        path=safe(run_dir/"case_study"/item["name"])
        if path.parent!=run_dir/"case_study" or digest(path)!=item["sha256"]:raise ValueError("completed trace integrity drift")
        validate(path,layers=adapter.model.encoder.config.num_hidden_layers)
    return summary


def synthetic_receipt(directory):
    from .ctl_drp_smoke import examples
    directory=safe(directory);directory.mkdir(parents=True,exist_ok=True)
    rows={s:examples() for s in ("train","dev","test")}
    extra=copy.deepcopy(rows["train"][0]);extra.update(id="synthetic-c",relation="per:schools_attended");rows["train"].append(extra)
    spec={"schema":"ctl-drp-physical-splits.v1","status":"qualified","task":"retacred_sentence_relation_40","canonical_labels":LABELS,"synthetic":True,"splits":{}}
    for split,records in rows.items():
        for i,r in enumerate(records):r["id"]=f"synthetic-{split}-{i}"
        path=directory/(split+".json")
        if path.exists() and json.loads(path.read_text())!=records:raise ValueError("synthetic fixture drift")
        if not path.exists():atomic_json(path,records)
        spec["splits"][split]={"file":path.name,"format":"json_array","count":len(records),"sha256":digest(path)}
    receipt=directory/"qualification.json"
    if receipt.exists() and json.loads(receipt.read_text())!=spec:raise ValueError("synthetic qualification drift")
    if not receipt.exists():atomic_json(receipt,spec)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--describe",action="store_true");p.add_argument("--tiny",action="store_true")
    p.add_argument("--cache",type=Path);p.add_argument("--paper-source",type=Path);p.add_argument("--run-dir",type=Path)
    p.add_argument("--resume",action="store_true");p.add_argument("--pause-after",type=int);p.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    p.add_argument("--local-export-dir",type=Path,help="explicit private diagnostic whitelist export, no upload or Git action")
    a=p.parse_args()
    if a.describe:
        print(json.dumps({"schema":"ctl-drp-runner.v1","formal_dispatch":False,"diagnostic_updates":2,"checkpoint":"every_completed_logical_update",
                          "trace":"sample-trace.v2_complete_private_values_and_local_reader","publishing":False}));return
    if not a.tiny or not all((a.cache,a.paper_source,a.run_dir)):p.error("explicit --tiny, --cache, --paper-source, --run-dir required while formal admission is pending")
    from .ctl_drp_neural import tiny_model,tokenizer_from_cache,descriptions_from_native
    seed_runtime(13)
    data=PhysicalSplits(synthetic_receipt(a.run_dir.absolute().with_name(a.run_dir.name+".synthetic-data")))
    adapter=CTLDRPTask(data,tokenizer_from_cache(a.cache),descriptions_from_native(a.paper_source),tiny_model(seed=13,dropout=.1),micro_batch=2,device=a.device)
    result=run(adapter,a.run_dir,resume=a.resume,pause_after=a.pause_after)
    if result["status"]!="paused" and a.local_export_dir:
        from .ctl_drp_export import export
        export(a.run_dir,a.local_export_dir,allow_diagnostic=True)
    print(json.dumps({"status":result["status"],"updates":result["updates"],"formal_training":False}))
    if result["status"]=="paused":raise SystemExit(75)


if __name__=="__main__":main()
