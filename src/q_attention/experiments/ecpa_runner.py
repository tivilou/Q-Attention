"""Dependency-aware single-seed queue: shared baseline, then independent controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from q_attention.experiments import ecpa_protocol as p

WORKER_TARGET = p.ROOT/"experiments/run_ecpa_formal.py"


def gpu_ids(spec, *, canary=False, inventory=None):
    if canary and spec.startswith("cpu"):
        values = spec.split(",")
        if any(value not in ("cpu","cpu:0","cpu:1","cpu:2") for value in values) or len(values)!=len(set(values)):
            raise ValueError("invalid canary CPU slots")
        return values
    if inventory is None:
        raw = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,uuid","--format=csv,noheader,nounits"],text=True)
        inventory = [dict(id=parts[0].strip(),free=int(parts[1]),uuid=parts[2].strip()) for line in raw.splitlines() if (parts:=line.split(","))]
    allowed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allowed is not None:
        visible = set(allowed.split(","))
        inventory = [g for g in inventory if g["id"] in visible or g.get("uuid") in visible]
    values = [g["id"] for g in inventory if g["free"]>=8192] if spec=="auto" else spec.split(",")
    available = {g["id"] for g in inventory if g["free"]>=8192}
    if not values or len(values)!=len(set(values)) or any(not g.isdecimal() or g not in available for g in values):
        raise ValueError("GPUs must be unique available indexes with >=8GiB free")
    return values


def child_spec(run, mode, device, *, inject_oom=False, stop_after=None):
    if not WORKER_TARGET.is_file():
        raise FileNotFoundError(WORKER_TARGET)
    argv = [sys.executable,str(WORKER_TARGET),"--worker",mode,"--run-dir",str(Path(run).resolve()),"--device",device]
    if inject_oom:
        argv.append("--inject-oom-once")
    if stop_after is not None:
        argv += ["--stop-after",str(stop_after)]
    return dict(target=str(WORKER_TARGET.relative_to(p.ROOT)),cwd=str(p.ROOT),argv=argv)


def stop(process, timeout=60):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        process.wait(timeout=10)


def completed(run, job, contract):
    directory = Path(run)/"jobs"/job
    path = directory/"RUN_COMPLETE.json"
    if not path.exists():
        return False
    marker = p.read(path)
    if marker["fingerprint"] != contract["fingerprint"] or marker["metrics_sha256"] != p.sha(directory/"metrics.json") or marker["trace_sha256"] != p.sha(directory/"sample_trace.json"):
        raise ValueError(f"damaged complete marker for {job}")
    return True


def duration(seconds):
    seconds = max(0,int(seconds))
    return f"{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}"


def dashboard(run, slots, jobs, active, started, tty):
    complete = [j for j,s in jobs.items() if s["status"]=="complete"]
    pending = [j for j,s in jobs.items() if s["status"]=="pending"]
    failed = [j for j,s in jobs.items() if s["status"]=="failed"]
    lines = [f"Q-ECPA seed 13 | {len(complete)}/{len(jobs)} complete | {len(active)} running | {duration(time.monotonic()-started)}"]
    for gpu in slots:
        job = next((j for j,item in active.items() if item["gpu"]==gpu),None)
        if job is None:
            lines.append(f"GPU {gpu} | waiting")
            continue
        path = run/"jobs"/job/"heartbeat.json"
        event = p.read(path) if path.exists() else {}
        done,total,rate = event.get("done",0),event.get("total",0),event.get("rate",0)
        percent = done/max(1,total)
        bar = "#"*int(16*percent)+"-"*(16-int(16*percent))
        lines.append(f"GPU {gpu} | {job:16} | {event.get('phase','initializing')} | [{bar}] {done}/{total} | {rate:.3f}/s | ETA {duration((total-done)/rate) if rate else '--:--:--'} | VRAM {event.get('allocated_gib',0):.2f}G | T{jobs[job]['tier']} m{event.get('micro','-')} save@{event.get('checkpoint_update','-')}")
    lines += ["PENDING | "+(", ".join(pending) or "empty"),"COMPLETED | "+(", ".join(complete) or "empty"),"FAILED | "+(", ".join(failed) or "none")]
    width,height = shutil.get_terminal_size((120,30))
    if tty:
        print("\x1b[H\x1b[2J"+"\n".join(line[:max(20,width-1)] for line in lines[:height-1]),end="",flush=True)
    else:
        print("\n".join(lines),flush=True)
    p.atomic_json(run/"scheduler_heartbeat.json",dict(updated_at=datetime.now(timezone.utc).isoformat(),complete=complete,pending=pending,failed=failed))


def publication_preflight():
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=p.ROOT,text=True).strip()
    if git("branch","--show-current") != "1.1":
        raise ValueError("formal collaborator execution requires branch 1.1")
    if git("status","--porcelain"):
        raise ValueError("formal collaborator execution requires a clean worktree")
    git("var","GIT_AUTHOR_IDENT")
    git("var","GIT_COMMITTER_IDENT")
    for branch in ("origin/1.1","origin/main"):
        subprocess.run(["git","merge-base","--is-ancestor",branch,"HEAD"],cwd=p.ROOT,check=True)


def run(args):
    config = p.load_config(args.canary)
    if not args.canary:
        publication_preflight()
        if not os.environ.get("PROJECT_EXCHANGE_URL") or not os.environ.get("PROJECT_EXCHANGE_TOKEN"):
            raise ValueError("configure project artifact-exchange URL/token before formal training")
    slots = gpu_ids(args.gpu,canary=args.canary)
    # Execute the exact declared module target through harmless help before any
    # formal output directory is made, including paths with spaces.
    spec = child_spec(Path(args.output_dir or args.resume or "."),"baseline","cpu")
    subprocess.run(spec["argv"]+["--help"],cwd=spec["cwd"],check=True,stdout=subprocess.DEVNULL)
    if args.dry_run:
        print(spec)
        print(dict(config=config,gpus=slots))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.resume or args.output_dir or p.ROOT/"runs"/p.EXPERIMENT/f"{stamp}_seed13").resolve()
    args.resolved_run = run
    data_dir = Path(args.data_dir or p.ROOT/"data/relation/retacred").resolve()
    if args.canary:
        data_dir = run/"prepared_data"
    if args.resume:
        previous = p.read(run/"run_manifest.json")
        data_dir = Path(previous["data_dir"])
    elif run.exists():
        raise ValueError("fresh output directory must not already exist")
    # Canary fixture creation is explicitly synthetic; full-data checks happen
    # before making a fresh formal run directory.
    if args.canary and not args.resume:
        p.synthetic_data(data_dir)
    _,_,_,_,_,data = p.load_data(data_dir,config)
    source = p.source_contract()
    fingerprint = p.identity(dict(config=config,data=data,source_sha256=p.identity(source)))
    if args.check:
        print(dict(status="preflight_passed",config=config["profile"],data=data,gpus=slots,source_sha256=p.identity(source)))
        return 0
    run.mkdir(parents=True,exist_ok=True)
    with p.exclusive(run/".scheduler.lock"):
        manifest = dict(schema_version="ecpa-run.v1",run_id=run.name,git_revision=p.revision(),fingerprint=fingerprint,
            source_sha256=p.identity(source),source_files=source,data=data,data_dir=str(data_dir),
            environment=dict(python=sys.version.split()[0],torch=__import__("torch").__version__),profile=config["profile"])
        if args.resume:
            previous = p.read(run/"run_manifest.json")
            if previous["fingerprint"] != fingerprint or p.read(run/"run_config.json") != config:
                raise ValueError("source/config/data changed; resume refused")
            manifest = previous  # topology is an execution detail, not identity
        else:
            p.atomic_json(run/"run_manifest.json",manifest)
            p.atomic_json(run/"run_config.json",config)
            (run/"data.sha256").write_text("".join(f"{v['sha256']}  {k}.jsonl\n" for k,v in data.items()))
            (run/"data_counts.txt").write_text("".join(f"{k}: {v['records']}\n" for k,v in data.items()))
        jobs = {j:dict(status="complete" if completed(run,j,manifest) else "pending",tier=0) for j in ("baseline",*p.MODES)}
        for job in jobs:
            state = run/"jobs"/job/"memory_state.json"
            if state.exists():
                jobs[job]["tier"] = p.read(state)["tier"]
            else:
                p.atomic_json(state,dict(tier=0,retries=0,events=[]))
        active,paused = {},False
        def pause(signum,frame):
            nonlocal paused
            paused = True
        signals = (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)
        old = {sig:signal.signal(sig,pause) for sig in signals}
        tty = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        if tty:
            print("\x1b[?1049h\x1b[?25l",end="",flush=True)
        started,last_frame,pause_started = time.monotonic(),0,None
        try:
            while True:
                if paused and pause_started is None:
                    pause_started = time.monotonic()
                    for item in active.values():
                        if item["process"].poll() is None:
                            os.killpg(item["process"].pid,signal.SIGTERM)
                if paused and time.monotonic()-pause_started>600:
                    for item in active.values():
                        stop(item["process"],timeout=1)
                occupied = {item["gpu"] for item in active.values()}
                for gpu in slots:
                    if gpu in occupied or paused:
                        continue
                    ready = [j for j,s in jobs.items() if s["status"]=="pending" and (j=="baseline" or jobs["baseline"]["status"]=="complete")]
                    if not ready:
                        break
                    job = ready[0]
                    device = "cpu" if gpu.startswith("cpu") else "cuda"
                    spec = child_spec(run,job,device,inject_oom=args.inject_oom_once,stop_after=args.stop_after if job=="quantum" else None)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES="" if device=="cpu" else gpu,CUDA_DEVICE_ORDER="PCI_BUS_ID")
                    logs = run/"logs"
                    logs.mkdir(exist_ok=True)
                    stream = (logs/f"{job}.log").open("a")
                    p.atomic_json(logs/f"{job}.command.json",dict(**spec,gpu=gpu,source_sha256=manifest["source_sha256"]))
                    process = subprocess.Popen(spec["argv"],cwd=spec["cwd"],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[job] = dict(process=process,stream=stream,gpu=gpu,started_at=time.time())
                    jobs[job].update(status="running",gpu=gpu,pid=process.pid)
                for job,item in list(active.items()):
                    code = item["process"].poll()
                    if code is None:
                        continue
                    item["stream"].close()
                    del active[job]
                    jobs[job].update(exit_code=code,finished_at=time.time())
                    if code==p.OOM_EXIT:
                        path = run/"jobs"/job/"memory_state.json"
                        state = p.read(path)
                        tiers = p.memory_tiers(config["baseline" if job=="baseline" else "kernel"]["batch_size"])
                        if state["tier"]+1 < len(tiers):
                            event = p.read(run/"jobs"/job/"OOM.json")
                            state["events"].append(dict(from_tier=state["tier"],to_tier=state["tier"]+1,**event))
                            state.update(tier=state["tier"]+1,retries=state["retries"]+1)
                            p.atomic_json(path,state)
                            jobs[job].update(status="pending",tier=state["tier"])
                        else:
                            jobs[job].update(status="failed",error="all finite memory tiers exhausted")
                    elif code==p.PAUSED_EXIT:
                        jobs[job]["status"] = "paused"
                        paused = True
                    elif code==0 and completed(run,job,manifest):
                        jobs[job]["status"] = "complete"
                    else:
                        jobs[job].update(status="failed",error=f"worker exit {code}; see logs/{job}.log")
                p.atomic_json(run/"gpu_assignments.json",dict(requested=args.gpu,resolved=slots,jobs=jobs,scheduler="first-free independent selectors after shared baseline"))
                if time.monotonic()-last_frame> (2 if tty else 30):
                    dashboard(run,slots,jobs,active,started,tty)
                    last_frame = time.monotonic()
                if not active and (paused or all(j["status"] in ("complete","failed") for j in jobs.values()) or jobs["baseline"]["status"]=="failed"):
                    break
                time.sleep(.2)
        finally:
            for item in active.values():
                stop(item["process"])
                item["stream"].close()
            for sig,handler in old.items():
                signal.signal(sig,handler)
            if tty:
                print("\x1b[?25h\x1b[?1049l",end="",flush=True)
        if paused:
            p.atomic_json(run/"RUN_PAUSED.json",dict(jobs=jobs))
            print(f"Paused safely. Resume: --resume {run}")
            return p.PAUSED_EXIT
        if not all(job["status"]=="complete" for job in jobs.values()):
            p.atomic_json(run/"RUN_FAILED.json",dict(jobs=jobs))
            raise RuntimeError(f"one or more jobs failed; inspect {run}/logs")
        rows = {mode:p.read(run/"jobs"/mode/"metrics.json") for mode in p.MODES}
        baseline = rows["disabled"]["reports"]["test"]["metrics"][config["primary"]]
        differences = {mode:row["reports"]["test"]["metrics"][config["primary"]]-baseline for mode,row in rows.items()}
        result = dict(schema_version="ecpa-summary.v1",experiment=config["name"],profile=config["profile"],fingerprint=fingerprint,
            primary=config["primary"],rows=rows,delta_vs_disabled=differences,
            gates=dict(single_seed_L1_utility=config["profile"]=="formal_seed13" and differences["quantum"]>0,
                quantum_inspired_candidate=config["profile"]=="formal_seed13" and differences["classical"]/max(abs(baseline),1e-12)>.01,
                quantum_specific_advantage_claimed=False),test_used_for_training_or_selection=False)
        p.atomic_json(run/"run_summary.json",result)
        p.atomic_json(run/"run_summary.data",result)
        (run/"run_summary.md").write_text("# Q-ECPA seed 13\n\n"+"\n".join(f"- {m}: test Macro-F1 {r['reports']['test']['metrics']['macro_f1']:.8f}" for m,r in rows.items())+"\n")
        p.atomic_json(run/"RUN_COMPLETE.json",dict(fingerprint=fingerprint,summary_sha256=p.sha(run/"run_summary.json")))
        for stale in ("RUN_PAUSED.json","RUN_FAILED.json"):
            (run/stale).unlink(missing_ok=True)
        print(f"Complete: {run}")
    return 0


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group()
    group.add_argument("--output-dir",type=Path)
    group.add_argument("--resume",type=Path)
    cli.add_argument("--gpu",default="auto")
    cli.add_argument("--hardware-profile",choices=["adaptive"],default="adaptive")
    cli.add_argument("--data-dir",type=Path)
    cli.add_argument("--canary",action="store_true")
    cli.add_argument("--check",action="store_true")
    cli.add_argument("--dry-run",action="store_true")
    cli.add_argument("--inject-oom-once",action="store_true")
    cli.add_argument("--stop-after",type=int)
    cli.add_argument("--export-only",action="store_true",help="retry export/publication from completed --resume; never train")
    cli.add_argument("--no-push",action="store_true",help="export without report push")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.inject_oom_once or args.stop_after is not None) and not args.canary:
        raise ValueError("fault injection is canary-only")
    from q_attention.experiments.ecpa_export import finish
    if args.export_only:
        if args.resume is None or args.check or args.dry_run or args.canary:
            raise ValueError("export-only requires the completed formal --resume directory")
        return finish(args.resume,push=not args.no_push)
    code = run(args)
    if code==0 and not (args.canary or args.check or args.dry_run):
        return finish(args.resolved_run,push=not args.no_push)
    return code
