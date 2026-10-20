"""Portable single-seed DAG: one shared warm-start, then seven independent jobs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import importlib.metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from q_attention.experiments import cfra_protocol as p

WORKER_TARGET=p.ROOT/'experiments/run_cfra_formal.py'
HARDWARE_TARGET=p.ROOT/'experiments/validate_cfra_handoff.py'

def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,uuid','--format=csv,noheader,nounits'],text=True)
    inventory=[dict(id=part[0].strip(),free=int(part[1]),uuid=part[2].strip()) for line in raw.splitlines() if (part:=line.split(','))]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        allowed=set(visible.split(','));inventory=[g for g in inventory if g['id'] in allowed or g['uuid'] in allowed]
    return inventory

def gpu_ids(spec,*,canary=False,inventory=None):
    if canary and spec=='cpu':return ['cpu']
    inventory=gpu_inventory() if inventory is None else inventory
    available={g['id'] for g in inventory if g['free']>=8192}
    result=[g['id'] for g in inventory if g['id'] in available] if spec=='auto' else spec.split(',')
    if not result or len(result)!=len(set(result)) or any(not s.isdecimal() or s not in available for s in result):
        raise ValueError('select unique visible physical GPU indexes with at least8GiB free')
    return result

def execution_profile(slots,inventory):
    if slots==['cpu']:return dict(profile='synthetic_cpu',micro_batch=2,weakest_free_mib=None)
    weakest=min(g['free'] for g in inventory if g['id'] in slots)
    profile,micro=('low_memory',1) if weakest<12288 else ('balanced',2) if weakest<24576 else ('high_memory',4)
    return dict(profile=profile,micro_batch=micro,weakest_free_mib=weakest)

def child_spec(run,mode,device,*,stop_after=None,recover_previous=False):
    if not WORKER_TARGET.is_file():raise FileNotFoundError(WORKER_TARGET)
    argv=[sys.executable,str(WORKER_TARGET),'--worker',mode,'--run-dir',str(Path(run).resolve()),'--device',device]
    if stop_after is not None:argv.extend(['--stop-after',str(stop_after)])
    if recover_previous:argv.append('--recover-previous')
    return dict(target=str(WORKER_TARGET.relative_to(p.ROOT)),cwd=str(p.ROOT),argv=argv)

def publication_preflight():
    def git(*args):return subprocess.check_output(['git',*args],cwd=p.ROOT,text=True).strip()
    if git('branch','--show-current')!='1.1' or git('status','--porcelain'):
        raise ValueError('formal execution requires clean collaborator branch1.1')
    for upstream in ('origin/1.1','origin/main'):
        subprocess.run(['git','merge-base','--is-ancestor',upstream,'HEAD'],cwd=p.ROOT,check=True)
    git('var','GIT_AUTHOR_IDENT');git('var','GIT_COMMITTER_IDENT')

def completed(run,job,manifest):
    directory=Path(run)/'jobs'/job;path=directory/'RUN_COMPLETE.json'
    if not path.exists():return False
    marker=p.read(path)
    pointer=p.read(directory/'latest.json')
    if pointer['schema_version']!='cfra-checkpoint-pointer.v1':raise ValueError('checkpoint schema differs')
    p.checkpoint_file(directory,pointer['latest'])
    expected=dict(fingerprint=manifest['fingerprint'],metrics_sha256=p.sha(directory/'metrics.json'),
        trace_sha256=p.sha(directory/'sample_trace.json'),best_sha256=p.sha(directory/'best.pt'),checkpoint_sha256=pointer['latest']['sha256'])
    if marker!=expected:raise ValueError('damaged completed job: '+job)
    return True


def prepare_completed_recovery(run,job,manifest,config):
    """Invalidate only an explicitly requested damaged completed checkpoint.

    A valid predecessor must already contain the complete optimizer budget.
    The worker verifies its full payload; finalization may replay diagnostics,
    but cannot train or change the completed summary/report identity.
    """
    import math
    directory=Path(run)/'jobs'/job;marker_path=directory/'RUN_COMPLETE.json'
    if not marker_path.exists():return
    marker=p.read(marker_path);pointer=p.read(directory/'latest.json')
    expected=dict(fingerprint=manifest['fingerprint'],metrics_sha256=p.sha(directory/'metrics.json'),
        trace_sha256=p.sha(directory/'sample_trace.json'),best_sha256=p.sha(directory/'best.pt'),checkpoint_sha256=pointer['latest']['sha256'])
    if marker!=expected:raise ValueError('completed recovery requires intact prior result identities')
    try:p.checkpoint_file(directory,pointer['latest'])
    except (OSError,ValueError):pass
    else:raise ValueError('previous-generation recovery only applies to damaged latest')
    previous=pointer['previous']
    if not previous:raise ValueError('completed job has no retained predecessor')
    path=p.checkpoint_file(directory,previous)
    budget=config['baseline' if job=='baseline' else 'kernel']
    total=math.ceil(manifest['data']['train']['records']/budget['batch_size'])*budget['epochs']
    if int(path.name.split('-')[0])!=total:raise ValueError('completed repair predecessor lacks full optimizer budget; stop for owner audit')
    receipt=directory/'completed_checkpoint_recovery.json'
    if receipt.exists():
        pending=p.read(receipt)
        if pending['fingerprint']!=manifest['fingerprint'] or pending['old_completion']!=marker or pending['pointer']!=pointer:
            raise ValueError('pending completed-repair identity differs')
    else:
        summary=Path(run)/'run_summary.json'
        p.atomic_json(receipt,dict(reason='explicit_damaged_completed_latest',fingerprint=manifest['fingerprint'],
            old_completion=marker,old_summary=p.read(summary) if summary.exists() else None,pointer=pointer,
            policy='finalization only; no optimizer updates; unchanged summary and public report identity'))
    marker_path.unlink();p.syncdir(directory)
    (Path(run)/'RUN_COMPLETE.json').unlink(missing_ok=True)
    if (Path(run)/'EXPORT_COMPLETE.json').exists():
        p.atomic_json(Path(run)/'EXPORT_INVALIDATED.json',dict(reason='explicit completed checkpoint repair',prior=p.read(Path(run)/'EXPORT_COMPLETE.json')))
        (Path(run)/'EXPORT_COMPLETE.json').unlink()

def stop(process,timeout=600):
    if process.poll() is not None:return
    try:
        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    except ProcessLookupError:process.wait(timeout=10)

def dashboard(run,slots,jobs,active,started):
    groups={s:[j for j,v in jobs.items() if v['status']==s] for s in ('complete','pending','failed','paused')}
    print(f"Q-CFRA seed13 | {len(groups['complete'])}/8 complete | {len(active)} running | elapsed{time.monotonic()-started:.0f}s",flush=True)
    for gpu in slots:
        job=next((j for j,v in active.items() if v['gpu']==gpu),None)
        if not job:print(f'GPU {gpu}: waiting',flush=True);continue
        path=run/'jobs'/job/'heartbeat.json';event=p.read(path) if path.exists() else {}
        done,total,rate=event.get('done',0),event.get('total',0),event.get('rate',0)
        eta=(total-done)/rate if rate else None
        print(f"GPU{gpu} {job}: {event.get('phase','startup')} {done}/{total} rate={rate:.3f}/s ETA={eta} micro={event.get('micro')} save@{event.get('checkpoint_update')}",flush=True)
    print(' | '.join(f'{s}: '+(','.join(v) or '-') for s,v in groups.items()),flush=True)
    p.atomic_json(run/'scheduler_heartbeat.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),groups=groups))

def hardware_check(slots,destination):
    if not HARDWARE_TARGET.is_file():raise FileNotFoundError(HARDWARE_TARGET)
    spec=dict(target=str(HARDWARE_TARGET.relative_to(p.ROOT)),cwd=str(p.ROOT),
        argv=[sys.executable,str(HARDWARE_TARGET),'--gpu',','.join(slots),'--output',str(destination)])
    subprocess.run(spec['argv']+['--help'],cwd=spec['cwd'],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(spec['argv'],cwd=spec['cwd'],check=True)
    receipt=p.read(Path(destination)/'readiness.json')
    if receipt['status']!='passed' or receipt['source_sha256']!=p.identity(p.source_contract()) or receipt['gpus']!=slots:
        raise ValueError('hardware canary receipt differs')
    return receipt

def run(args):
    config=p.load_config(args.canary)
    if not args.canary:
        publication_preflight();p.validate_release()
        if not os.environ.get('PROJECT_EXCHANGE_URL','').startswith('https://') or not os.environ.get('PROJECT_EXCHANGE_TOKEN'):
            raise ValueError('configure HTTPS project artifact exchange before formal training')
    inventory=[] if args.canary and args.gpu=='cpu' else gpu_inventory()
    slots=gpu_ids(args.gpu,canary=args.canary,inventory=inventory)
    execution=execution_profile(slots,inventory)
    spec=child_spec(args.output_dir or args.resume or '.', 'baseline','cpu')
    subprocess.run(spec['argv']+['--help'],cwd=spec['cwd'],check=True,stdout=subprocess.DEVNULL)
    if args.dry_run:
        print(dict(launch=spec,gpus=slots,execution=execution,profile=config['profile']));return 0
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=Path(args.resume or args.output_dir or p.ROOT/'runs'/p.EXPERIMENT/f'{stamp}_seed13').resolve()
    args.resolved_run=run
    data_dir=Path(args.data_dir or p.ROOT/'data/relation/retacred').resolve()
    if args.resume:data_dir=Path(p.read(run/'run_manifest.json')['data_dir'])
    elif run.exists():raise ValueError('fresh output directory already exists')
    if args.canary and not args.resume:
        data_dir=run/'prepared_data';p.synthetic_data(data_dir)
    data=p.inspect_data(data_dir,config);source=p.source_contract()
    fingerprint=p.identity(dict(config=config,data=data,source_sha256=p.identity(source)))
    if args.resume and (p.read(run/'run_manifest.json')['fingerprint']!=fingerprint or p.read(run/'run_config.json')!=config):
        raise ValueError('resume source/config/data changed')
    if args.check:
        print(dict(status='preflight_passed',profile=config['profile'],data=data,gpus=slots,execution=execution));return 0
    # A complete GPU launch/recovery/export canary is automatically required on
    # this executor, before making any full-data run directory.
    hardware=None
    if not args.canary:
        hardware=hardware_check(slots,p.ROOT/'runs/q_cfra_executor_canary'/f'{stamp}-{time.time_ns()}')
    run.mkdir(parents=True,exist_ok=True)
    with p.exclusive(run/'.scheduler.lock'):
        if not args.resume and (run/'run_manifest.json').exists():raise ValueError('fresh run was initialized concurrently; refuse overwrite')
        manifest=dict(schema_version='cfra-run.v1',run_id=run.name,git_revision=p.revision(),fingerprint=fingerprint,
            source_sha256=p.identity(source),source_files=source,data=data,data_dir=str(data_dir),execution=execution,
            environment=dict(python=sys.version.split()[0],torch=importlib.metadata.version('torch')),profile=config['profile'])
        if args.resume:
            old=p.read(run/'run_manifest.json')
            if old['fingerprint']!=fingerprint or p.read(run/'run_config.json')!=config:raise ValueError('resume source/config/data changed')
            manifest=old
        else:
            p.atomic_json(run/'run_manifest.json',manifest);p.atomic_json(run/'run_config.json',config)
            (run/'data.sha256').write_text(''.join(f"{v['sha256']}  {s}.jsonl\n" for s,v in data.items()))
            (run/'data_counts.txt').write_text(''.join(f"{s}: {v['records']}\n" for s,v in data.items()))
        if hardware:p.atomic_json(run/'executor_hardware_readiness.json',hardware)
        if args.recover_previous:prepare_completed_recovery(run,args.recover_previous,manifest,config)
        jobs={j:dict(status='complete' if completed(run,j,manifest) else 'pending') for j in ('baseline',*p.MODES)}
        active={};paused=False;failed=False;started=time.monotonic();last_frame=0;pause_started=None
        def pause(signum,frame):
            nonlocal paused
            paused=True
        previous={sig:signal.signal(sig,pause) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            while True:
                if (paused or failed) and pause_started is None:
                    pause_started=time.monotonic()
                    for item in active.values():
                        if item['process'].poll() is None:os.killpg(item['process'].pid,signal.SIGTERM)
                if pause_started is not None and time.monotonic()-pause_started>600:
                    for item in active.values():stop(item['process'],timeout=1)
                occupied={item['gpu'] for item in active.values()}
                for gpu in slots:
                    if gpu in occupied or paused or failed:continue
                    ready=[j for j,s in jobs.items() if s['status']=='pending' and (j=='baseline' or jobs['baseline']['status']=='complete')]
                    if not ready:break
                    job=ready[0];device='cpu' if gpu=='cpu' else 'cuda'
                    spec=child_spec(run,job,device,stop_after=args.stop_after if job==args.stop_job else None,
                        recover_previous=args.recover_previous==job)
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if device=='cpu' else gpu,
                        CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                    logs=run/'logs';logs.mkdir(exist_ok=True);stream=(logs/f'{job}.log').open('a')
                    p.atomic_json(logs/f'{job}.command.json',dict(**spec,gpu=gpu,source_sha256=manifest['source_sha256']))
                    try:process=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:stream.close();raise
                    active[job]=dict(process=process,stream=stream,gpu=gpu)
                    jobs[job].update(status='running',gpu=gpu,pid=process.pid,started_at=time.time())
                for job,item in list(active.items()):
                    code=item['process'].poll()
                    if code is None:continue
                    item['stream'].close();del active[job]
                    jobs[job].update(exit_code=code,finished_at=time.time())
                    if code==p.PAUSED_EXIT:jobs[job]['status']='paused';paused=True
                    elif code==0 and completed(run,job,manifest):jobs[job]['status']='complete'
                    else:jobs[job].update(status='failed',error=f'worker exit{code}; see logs/{job}.log');failed=True
                p.atomic_json(run/'gpu_assignments.json',dict(requested=args.gpu,resolved=slots,jobs=jobs,
                    scheduler='first-free independent conditions after one immutable shared warm-start'))
                if time.monotonic()-last_frame>=30:
                    dashboard(run,slots,jobs,active,started);last_frame=time.monotonic()
                if not active and (paused or failed or all(v['status']=='complete' for v in jobs.values())):break
                time.sleep(.1)
        except BaseException as exc:
            p.atomic_json(run/'RUN_FAILED.json',dict(error_type=type(exc).__name__,jobs=jobs));raise
        finally:
            for item in active.values():stop(item['process']);item['stream'].close()
            for sig,handler in previous.items():signal.signal(sig,handler)
        if failed:
            p.atomic_json(run/'RUN_FAILED.json',dict(jobs=jobs));raise RuntimeError('worker failed; retain same run and checkpoints')
        if paused:
            p.atomic_json(run/'RUN_PAUSED.json',dict(jobs=jobs));return p.PAUSED_EXIT
        from q_attention.experiments.cfra_export import summarize
        summarize(run)
        for stale in ('RUN_FAILED.json','RUN_PAUSED.json'):(run/stale).unlink(missing_ok=True)
    return 0

def parser():
    cli=argparse.ArgumentParser(description=__doc__)
    group=cli.add_mutually_exclusive_group();group.add_argument('--output-dir',type=Path);group.add_argument('--resume',type=Path)
    cli.add_argument('--gpu',default='auto');cli.add_argument('--data-dir',type=Path)
    cli.add_argument('--canary',action='store_true');cli.add_argument('--check',action='store_true');cli.add_argument('--dry-run',action='store_true')
    cli.add_argument('--stop-after',type=int);cli.add_argument('--stop-job',choices=('baseline',*p.MODES),default='quantum')
    cli.add_argument('--recover-previous',choices=('baseline',*p.MODES))
    cli.add_argument('--export-only',action='store_true');cli.add_argument('--no-push',action='store_true')
    return cli

def main(argv=None):
    args=parser().parse_args(argv)
    if args.stop_after is not None and not args.canary:raise ValueError('fault injection is canary-only')
    if args.recover_previous and not args.resume:raise ValueError('predecessor recovery requires explicit --resume')
    if args.export_only:
        if args.resume is None or args.canary or args.check or args.dry_run:raise ValueError('export-only requires completed formal --resume')
        from q_attention.experiments.cfra_export import finish
        return finish(args.resume,push=not args.no_push)
    code=run(args)
    if code==0 and not (args.canary or args.check or args.dry_run):
        from q_attention.experiments.cfra_export import finish
        return finish(args.resolved_run,push=not args.no_push)
    return code
