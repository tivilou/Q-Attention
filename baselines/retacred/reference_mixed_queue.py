"""Atomic GPU-group reservations for CTL DDP mixed with independent singles."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

PRIORITY={'ctl_drp_supervised':38,'ctl_drp_unsupervised':37.5,'ptr':21.6,
          'sequence_routing_h3':21.5,'metaentail':11.6,'luke_core':6.6,'curriculum':6.2,
          'luke_cfie':5.8,'luke_resample':5.2,'luke_focal':5.1}


def group_size(requested, devices):
    if not devices or len(set(devices))!=len(devices):
        raise ValueError('nonempty unique GPU devices required')
    size=min(2,len(devices)) if requested=='auto' else int(requested)
    if not 1<=size<=min(6,len(devices)):
        raise ValueError('CTL GPU group must fit selected devices (maximum6)')
    return size


def allocation(job, free, width):
    pinned=job.get('retry_gpu_ids')
    count=width if job['model'].startswith('ctl_drp_') else 1
    if pinned:
        if len(pinned)!=count:
            raise ValueError('retry GPU group/world size drift')
        return list(pinned) if all(g in free for g in pinned) else None
    return list(free[:count]) if len(free)>=count else None


def queue(jobs,devices,args,assets,prepared):
    from .common import atomic_json
    from .reference_complete import command,validate_task
    from .reference_terminal import Dashboard
    from .reference_execution import ledger_path
    width=group_size(getattr(args,'ctl_gpus_per_task','auto'),devices)
    pending=sorted(jobs,key=lambda j:-PRIORITY.get(j['model'],3))
    if getattr(args,'resume',None):
        for job in pending:
            path=args.run_dir/'logs'/f"{job['id']}.command.json"
            if path.exists():
                saved=json.loads(path.read_text())
                group=saved.get('gpu_ids')
                expected=width if job['model'].startswith('ctl_drp_') else 1
                if (not isinstance(group,list) or len(group)!=expected or len(set(group))!=expected
                        or any(g not in devices for g in group)):
                    raise ValueError('resume requires original selected GPU group and world size')
                job['retry_gpu_ids']=group
    active={};done=[];failures=[];paused=[False]
    dashboard=Dashboard(devices,len(jobs))
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for sig in handlers:signal.signal(sig,lambda *_:paused.__setitem__(0,True))
    try:
        while pending or active:
            if paused[0]:break
            occupied={g for _,j,_ in active.values() for g in j['gpu_ids']}
            free=[g for g in devices if g not in occupied]
            for job in list(pending):
                group=allocation(job,free,width)
                if group is None:continue
                if os.environ.get('QATTN_REFERENCE_STORAGE_ROOT'):
                    from .reference_storage import task_peak,owned_bytes,FLOOR,durable_json
                    committed=sum(max(0,task_peak(j['model'])-owned_bytes(j['directory'])) for _,j,_ in active.values())
                    additional=max(0,task_peak(job['model'])-owned_bytes(job['directory']))
                    if shutil.disk_usage(args.run_dir).free<additional+committed+FLOOR:
                        if active:continue
                        durable_json(args.run_dir/'STORAGE_PAUSED.json',{'status':'paused','task':job['id'],
                            'recovery':'free disk space, resume same suite'})
                        paused[0]=True;break
                job['gpu_ids']=group
                job['launch_tier']=json.loads(ledger_path(job['directory']).read_text())['index'] if ledger_path(job['directory']).exists() else 0
                pending.remove(job)
                free=[g for g in free if g not in group]
                log=args.run_dir/'logs'/f"{job['id']}.log";log.parent.mkdir(parents=True,exist_ok=True)
                argv=command(job,args,assets,prepared)
                atomic_json(log.with_suffix('.command.json'),dict(argv=argv,cwd=str(Path(__file__).resolve().parents[2]),
                    gpu_ids=group,world_size=len(group),logical_batch=512 if job['model'].startswith('ctl_drp_') else None))
                handle=log.open('a')
                env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(group),'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
                    'PYTHONDONTWRITEBYTECODE':'1','USE_TF':'0','USE_FLAX':'0','OMP_NUM_THREADS':'1',
                    'MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                try:
                    process=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[2],env=env,
                        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                except Exception:
                    handle.close();raise
                active[job['id']]=(process,job,handle)
            for key,(process,job,handle) in list(active.items()):
                rc=process.poll()
                if rc is None:continue
                handle.close();del active[key]
                if rc==0:
                    try:
                        validate_task(job['directory'])
                        from .reference_archive import archive_completed
                        archive_completed(job['directory']);done.append(job['id'])
                    except Exception as error:
                        failures.append(dict(task=key,gpu_ids=job['gpu_ids'],error=str(error)))
                elif rc==76:
                    state=json.loads(ledger_path(job['directory']).read_text())
                    if state['index']!=job['launch_tier']+1 or not state['events'] or state['events'][-1]['to_tier']!=state['index']:
                        raise ValueError('OOM retry needs exactly one committed tier transition')
                    job['retry_gpu_ids']=list(job['gpu_ids']);pending.insert(0,job)
                    dashboard.notice(f'[OOM] {key}: tier{state["index"]}; retry same GPU group {job["gpu_ids"]}')
                elif rc==75:
                    paused[0]=True
                else:
                    failures.append(dict(task=key,gpu_ids=job['gpu_ids'],exit_code=rc))
            if failures:break
            rows=[]
            for _,job,_ in active.values():
                hb=job['directory']/'heartbeat.json'
                try:value=json.loads(hb.read_text()) if hb.exists() else {}
                except json.JSONDecodeError:value={}
                rank_memory=value.pop('rank_memory',[])
                for rank,gpu in enumerate(job['gpu_ids']):
                    memory=rank_memory[rank] if rank<len(rank_memory) else {}
                    rows.append({'gpu':gpu,'task':job['id'],**value,'rank':rank,'world_size':len(job['gpu_ids']),
                        **({'memory_allocated_bytes':memory['allocated'],'memory_reserved_bytes':memory['reserved'],
                            'memory_peak_allocated_bytes':memory.get('peak_allocated'),
                            'memory_peak_reserved_bytes':memory.get('peak_reserved'),
                            'memory_total_bytes':memory.get('total')} if memory else {})})
            atomic_json(args.run_dir/'queue_status.json',dict(completed=done,active=rows,pending=[j['id'] for j in pending]))
            dashboard.render(done,rows,[j['id'] for j in pending])
            time.sleep(2 if active and dashboard.live else 10 if active else .1)
    finally:
        for process,_,_ in active.values():
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+300
        while any(p.poll() is None for p,_,_ in active.values()) and time.monotonic()<deadline:
            time.sleep(.2)
        for process,_,handle in active.values():
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            process.wait();handle.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if failures or paused[0]:
        atomic_json(args.run_dir/'RUN_FAILED.json',dict(failures=failures,paused=paused[0],pending=[j['id'] for j in pending]))
        for failure in failures:
            log=args.run_dir/'logs'/f"{failure['task']}.log"
            dashboard.notice(f'[FAILED] {failure} | log: {log}')
            with log.open('rb') as reader:
                reader.seek(max(0,log.stat().st_size-8192))
                for line in reader.read().decode(errors='replace').splitlines()[-10:]:dashboard.notice(line)
        return 1 if failures else 75
    dashboard.notice(f'[COMPUTE COMPLETE] {len(done)}/{len(jobs)}; export follows')
    return 0
