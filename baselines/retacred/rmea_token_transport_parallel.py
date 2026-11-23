"""Independent condition workers inside one scheduler GPU allocation.

The scientific train/evaluate path is shared with serial execution. No DDP,
batch-size changes, fleet allocation, cloud credentials or automatic retries.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import rmea_token_transport_protocol as p


from .rmea_training_options import per_epoch, exact_tree, validate_retained_checkpoints, expected_training_identities, validate_training_envelope


def equal_state(a,b):
    return exact_tree(a,b)


def commit_rank(row):
    cursor=row['cursor']
    return row['done'],cursor['mode'],cursor['epoch'],cursor['next_start'],len(cursor['history'])


def validate_outer_phase(output,cfg,protocol,*,allow_pending=False):
    from .rmea_token_transport_recovery import regular_paths
    output=Path(output);p=protocol;root=output.parent;folder=root.parent
    phase=folder/'restore-install.json';identity=folder/'run-identity.json';pin=root/'restore-pin.json'
    regular_paths(files=(phase,identity,pin),directories=(folder,root,output))
    intent=dict(run_id=root.name,profile=cfg['profile'],source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg))
    if identity.exists() and p.read(identity)!=intent:raise ValueError('Outer run identity differs')
    if phase.exists():
        value=p.read(phase)
        if (value.get('status') not in (('pending','complete') if allow_pending else ('complete',))
                or any(value.get(k)!=v for k,v in intent.items()) or value.get('pin_sha256')!=p.digest(pin)):
            raise ValueError('Finish the matching outer pinned installation before training')


def pin_receipts(pin, *, run_id, identity, total, sha256, latest_bytes, best_sha256, best_bytes):
    """Validate the complete acknowledged prefix before installing any receipt."""
    import hashlib
    from .rmea_token_transport_recovery import GENERATIONS,validate_cloud_receipt
    if (pin.get('run_id')!=run_id or pin.get('identity')!=identity or pin.get('total')!=total
            or pin.get('sha256')!=sha256 or not pin.get('version_id')):
        raise ValueError('Restored aggregate identity differs')
    selected=pin.get('generation','')
    names=list(GENERATIONS)
    if selected not in names[1:]:raise ValueError('A progress generation is required')
    rows=pin.get('acknowledged_generations',[])
    if [r['request'].get('generation') for r in rows]!=names[:names.index(selected)+1]:
        raise ValueError('Incomplete acknowledged generation history')
    previous=0
    for row in rows:
        request,ack=row['request'],row['ack'];name=request['generation']
        raw=(json.dumps(request,ensure_ascii=False,indent=2)+'\n').encode()
        validate_cloud_receipt(request,ack,request_sha256=hashlib.sha256(raw).hexdigest(),run_id=run_id,identity=identity,total=total,previous=previous)
        if request['done']>pin['done']:raise ValueError('Acknowledged cursor exceeds pin')
        previous=request['done']
        if name==selected:
            key='cos-channel/'+name+'/latest.pt';obj=ack['objects'].get(key,{})
            if (request['done']!=pin['done'] or obj.get('sha256')!=sha256
                    or obj.get('version_id')!=pin['version_id'] or obj.get('bytes')!=latest_bytes
                    or ack['objects']['cos-channel/'+name+'/best.pt'].get('sha256')!=best_sha256
                    or ack['objects']['cos-channel/'+name+'/best.pt'].get('bytes')!=best_bytes):
                raise ValueError('Selected aggregate VersionId/hash/cursor differs')
    return rows


def completed_worker(output,cfg,kind,protocol,identity,cache,proposed,*,runtime=None):
    """Read-only validation of an existing receipt before recovery may write."""
    import torch
    from .rmea_token_transport_recovery import Recovery
    p=protocol;output=Path(output);work=output/'.workers';per_head=cfg['epochs']*per_epoch(cfg)
    folder=work/kind;marker=folder/'WORKER_COMPLETE';result=folder/'result.json'
    if not marker.exists():return None
    rec=p.read(marker)
    if (rec.get('condition')!=kind or rec.get('source')!=identity['source']
            or rec.get('result_sha256')!=p.digest(result)
            or rec.get('checkpoint_sha256')!=p.digest(output/'checkpoints'/(kind+'-final.pt'))):
        raise ValueError('Completed condition receipt differs')
    row=p.read(result)
    if set(row['results'])!={kind} or row['results'][kind]['updates']!=per_head:
        raise ValueError('Condition completion budget differs')
    state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    if rec.get('terminal_sha256')!=p.digest(folder/'latest.pt'):raise ValueError('Receipt terminal commit differs')
    validator=object.__new__(Recovery);validator.cfg=cfg;validator.modes=(kind,);validator.total=per_head
    if state['identity']!=dict(identity,condition=kind) or state['cache']!=cache:
        raise ValueError('Completed worker identity/cache differs')
    validator._validate_cursor(state)
    if not equal_state(state,proposed):raise ValueError('Receipt differs from proposed worker state')
    if state['done']!=per_head or state['cursor']['mode']!=1 or set(state['completed'])!={kind}:
        raise ValueError('Receipt lacks the same terminal worker state')
    from .rmea_token_transport_study import state_identity
    cp=output/'checkpoints'/(kind+'-final.pt');metric=row['results'][kind]
    if (metric['checkpoint_sha256']!=p.digest(cp) or metric['final_state_sha256']!=state_identity(state['completed'][kind]['state'])
            or state_identity(torch.load(cp,map_location='cpu',weights_only=True))!=metric['final_state_sha256']):
        raise ValueError('Result checkpoint differs from terminal model')
    if cfg.get('reported_checkpoint','final')=='validation_best':
        validate_retained_checkpoints(output,state,(kind,),cfg)
        chosen=state['best'][kind];selected=output/'checkpoints'/(kind+'-selected.pt')
        expected=dict(path='checkpoints/'+kind+'-selected.pt',sha256=p.digest(selected),state_sha256=state_identity(chosen['state']),epoch=chosen['epoch'])
        selected_sha=state_identity(torch.load(selected,map_location='cpu',weights_only=True))
        if selected_sha!=expected['state_sha256'] or metric.get('evaluation_checkpoint')!=expected or metric['selected_epoch']!=chosen['epoch']:
            raise ValueError('Reported evaluation checkpoint differs from terminal best')

    if (metric['history']!=state['completed'][kind]['history'] or metric['updates']!=state['completed'][kind]['steps']
            or row['initial_sha']!=state['initial_sha'] or row['order_sha']!=state['order_sha']
            or metric['initial_state_sha256']!=state['initial_sha']):raise ValueError('Result history/order differs')
    if runtime is not None:
        runtime.validate_completed(output,kind,row,rec)
    else:
        if kind=='cubic_coherent' and rec.get('geometry_sha256')!=p.digest(output/'private-geometry.json'):
            raise ValueError('Cubic geometry changed')
        if {s['sample_id'] for s in row['samples']}!={f'{kind}:final:{split}:{i}' for split in cfg['counts'] for i in range(2)}:
            raise ValueError('Condition sample ownership differs')
        for sample in row['samples']:
            provenance=sample['roles']['provenance']
            if provenance['checkpoint_file']!='checkpoints/'+kind+'-final.pt' or provenance['checkpoint_sha256']!=p.digest(cp):
                raise ValueError('Sample checkpoint ownership differs')
            for rep in sample['representations']:
                for ref in rep['artifact_refs']:
                    path=output/ref['path']
                    if not path.resolve().is_relative_to(output.resolve()) or p.digest(path)!=ref['sha256']:
                        raise ValueError('Completed trace changed')
    return row


def preflight_retained_workers(output,cfg,protocol,states,identity,cache,*,runtime=None):
    """Check every proposed worker and receipt before the first repair write."""
    import torch
    from .rmea_token_transport_recovery import Recovery,regular_paths,validate_retained_best
    output=Path(output);work=output/'.workers'
    regular_paths(directories=(output,output/'checkpoints',work,output/'training-started'))
    if set(states)!=set(protocol.MODES):raise ValueError('Proposed worker inventory differs')
    expected_initial,expected_order=expected_training_identities(cfg) if any(row is not None for row in states.values()) else (None,None)
    for kind,row in states.items():
        folder=work/kind
        regular_paths(directories=(folder,),files=tuple(folder/n for n in ('latest.pt','best.pt','progress.json','RUN_PAUSED','WORKER_COMPLETE','result.json')))
        orphan=Path(output)/'.workers'/kind/'best.pt'
        if row is None and orphan.exists() and torch.load(orphan,map_location='cpu',weights_only=True):
            raise ValueError('Worker best lacks candidate latest commit')
        if row is not None:
            if row.get('schema')!='q-rmea.resume.v1' or row.get('initial_sha')!=expected_initial or row.get('order_sha')!=expected_order:
                raise ValueError('Proposed worker schema/initial/order identity differs')
            validator=object.__new__(Recovery);validator.cfg=cfg;validator.modes=(kind,);validator.total=cfg['epochs']*per_epoch(cfg)
            if row['identity']!=dict(identity,condition=kind) or row['cache']!=cache:
                raise ValueError('Proposed worker identity/cache differs')
            validator._validate_cursor(row)
            validate_retained_best(folder/'best.pt',row['best'],cfg)
        validate_retained_checkpoints(output,row,(kind,),cfg)
        completed_worker(output,cfg,kind,protocol,identity,cache,row,runtime=runtime)


def validate_resume_evidence(output,cfg,protocol,states,recovery,aggregate,*,runtime=None,authenticated_receipts=()):
    """Read-only input/progress bounds for all in-memory repair candidates."""
    from .rmea_token_transport_recovery import cache_inventory,validate_retained_best,condition_progress_bound,run_progress_bound
    p=protocol;output=Path(output);work=output/'.workers'
    if cache_inventory(output/'cache')!=recovery.cache:raise ValueError('Shared resume cache differs')
    cloud_bound=recovery.verify_cloud_history(authenticated_receipts=authenticated_receipts)
    validate_retained_best(recovery.best,{k:row['best'][k] for k,row in states.items() if row and k in row['best']},cfg)
    for kind,row in states.items():
        prior=(aggregate or {}).get('workers',{}).get(kind)
        minimum,started=condition_progress_bound(output,kind,p,recovery.identity,cfg['epochs']*per_epoch(cfg))
        minimum=max(minimum,prior['done'] if prior else 0)
        for payload in recovery._cloud_evidence['payloads']:
            cloud_worker=payload.get('workers',{}).get(kind)
            if cloud_worker is not None:
                if row is None or commit_rank(row)<commit_rank(cloud_worker) or commit_rank(row)==commit_rank(cloud_worker) and not equal_state(row,cloud_worker):
                    raise ValueError('Repair candidate precedes or conflicts with durable cloud worker')
        if minimum and (row is None or row['done']<minimum):raise ValueError('Repair candidate precedes durable committed progress')
        if started and row is None:raise ValueError('Started condition has no recoverable commit')
    available=sum(row['done'] for row in states.values() if row is not None)
    if max(cloud_bound,run_progress_bound(output,p,recovery.identity,recovery.total))>available:
        raise ValueError('Available candidates precede durable run cursor')
    observed=Path(os.environ.get('QATTN_PROGRESS_PATH',str(output.parent/'progress.json')))
    if observed.is_file():
        row=p.read(observed)
        if row.get('run_id')==output.parent.name and row.get('identity')==recovery.identity and row.get('completed',0)>available:
            raise ValueError('Run-bound progress exceeds repair candidates')
        if row.get('run_id')==output.parent.name and row.get('identity')==recovery.identity:
            for kind,job in row.get('jobs',{}).items():
                if kind not in states or job.get('completed',0)>(states[kind]['done'] if states[kind] else 0):
                    raise ValueError('Run-bound condition progress exceeds repair candidate')


def install_pinned_workers(output,cfg,protocol,*,runtime=None,install=True):
    """Validate the selected aggregate and durably install all workers before admission."""
    import torch
    from .rmea_token_transport_recovery import Recovery,atomic_torch,cache_inventory,sync_directory,regular_paths
    p=protocol;output=Path(output);work=output/'.workers'
    validate_outer_phase(output,cfg,p,allow_pending=True)
    latest=output/'checkpoints/latest.pt';best=output/'checkpoints/best.pt';pin=p.read(output.parent/'restore-pin.json')
    channel=output.parent/'cos-channel'
    directories=[output,output/'checkpoints',work,channel]+[work/k for k in p.MODES]
    directories += [channel/r['request']['generation'] for r in pin['acknowledged_generations']]
    for directory in directories:
        if directory.is_symlink() or directory.exists() and not directory.is_dir():
            raise ValueError('Pinned destination directory is nonregular')
    entries=[latest,best,work/'restore-pin.json',work/'restore-complete.json']
    entries += [work/k/n for k in p.MODES for n in ('latest.pt','best.pt')]
    for entry in entries:
        if entry.is_symlink() or entry.exists() and not entry.is_file():raise ValueError('Pinned checkpoint or marker is nonregular')
    recovery=Recovery(output,cfg,protocol=p)
    receipts=pin_receipts(pin,run_id=output.parent.name,identity=recovery.identity,total=recovery.total,
        sha256=p.digest(latest),latest_bytes=latest.stat().st_size,best_sha256=p.digest(best),best_bytes=best.stat().st_size)
    state=torch.load(latest,map_location='cpu',weights_only=True)
    if (state.get('schema')!='q-rmea.parallel-resume.v1' or state['identity']!=recovery.identity
            or state['cache']!=cache_inventory(output/'cache') or state['total']!=recovery.total
            or state['done']!=pin['done'] or set(state['workers'])!=set(p.MODES)):
        raise ValueError('Pinned aggregate scope/cursor differs')
    preflight_retained_workers(output,cfg,p,state['workers'],recovery.identity,state['cache'],runtime=runtime)
    validate_resume_evidence(output,cfg,p,state['workers'],recovery,state,runtime=runtime,authenticated_receipts=receipts)
    done=sum(row['done'] for row in state['workers'].values() if row is not None)
    if done!=pin['done']:raise ValueError('Pinned worker cursor sum differs')
    for kind,row in state['workers'].items():
        target=work/kind/'latest.pt'
        if row is None and (work/kind).exists() and any((work/kind).iterdir()):
            raise ValueError('Nonempty worker outside selected pin')
        if target.exists() and (row is None or not equal_state(torch.load(target,map_location='cpu',weights_only=True),row)):
            raise ValueError('Existing worker differs from selected pin')
    expected_best={k:row['best'][k] for k,row in state['workers'].items() if row and k in row['best']}
    if not equal_state(torch.load(best,map_location='cpu',weights_only=True),expected_best):
        raise ValueError('Pinned aggregate best disagrees with pinned workers')
    for receipt in receipts:
        directory=output.parent/'cos-channel'/receipt['request']['generation']
        for name,key in (('request.json','request'),('ack.json','ack')):
            target=directory/name
            if target.is_symlink() or (target.exists() and p.read(target)!=receipt[key]):
                raise ValueError('Existing pinned receipt prefix differs')
            if name=='request.json' and target.exists() and p.digest(target)!=receipt['ack']['request_sha256']:
                raise ValueError('Existing pinned request bytes differ')
    allowed={r['request']['generation'] for r in receipts}
    channel=output.parent/'cos-channel'
    for entry in channel.iterdir() if channel.exists() else ():
        if entry.name=='cloud-latest.json':continue
        if entry.name not in allowed and (not entry.is_dir() or any(entry.iterdir())):
            raise ValueError('COS evidence outside selected pinned prefix')
    last=receipts[-1]
    pointer=dict(generation=last['request']['generation'],done=last['request']['done'],index_commit=last['ack']['index_commit'],request_sha256=last['ack']['request_sha256'])
    target=output.parent/'cos-channel/cloud-latest.json'
    if target.is_symlink() or (target.exists() and p.read(target)!=pointer):
        raise ValueError('Existing cloud pointer differs from selected pin')
    installed=work/'restore-pin.json'
    if installed.exists() and p.read(installed)!=pin:raise ValueError('Different interrupted worker installation')
    if not installed.exists() and any(work.glob('*/latest.pt')):raise ValueError('Fresh pinned worker destination required')
    completion=work/'restore-complete.json';complete=dict(pin_sha256=p.digest(output.parent/'restore-pin.json'))
    if completion.exists() and p.read(completion)!=complete:raise ValueError('Different pinned installation completion')
    if not install:return dict(state=state,receipts=receipts,pointer=pointer)
    work.mkdir(exist_ok=True)
    if not installed.exists():p.atomic_json(installed,pin);sync_directory(work)
    for kind,row in state['workers'].items():
        target=work/kind/'latest.pt'
        if row is None:
            if target.exists():raise ValueError('Worker outside selected pin')
            continue
        if not target.exists():atomic_torch(target,row)
        sync_directory(target.parent)
    if done!=pin['done']:raise ValueError('Pinned worker cursor sum differs')
    sync_directory(work);sync_directory(output)


def complete_pinned_install(output,protocol):
    """Fence ordinary resume until the authenticated prefix and pointer are durable."""
    from .rmea_token_transport_recovery import sync_directory
    output=Path(output);p=protocol
    pin=p.read(output.parent/'restore-pin.json')
    for receipt in pin['acknowledged_generations']:
        directory=output.parent/'cos-channel'/receipt['request']['generation']
        if (p.read(directory/'request.json')!=receipt['request'] or p.read(directory/'ack.json')!=receipt['ack']
                or p.digest(directory/'request.json')!=receipt['ack']['request_sha256']):
            raise ValueError('Pinned receipt installation incomplete')
        sync_directory(directory)
    last=pin['acknowledged_generations'][-1]
    pointer=dict(generation=last['request']['generation'],done=last['request']['done'],index_commit=last['ack']['index_commit'],request_sha256=last['ack']['request_sha256'])
    if p.read(output.parent/'cos-channel/cloud-latest.json')!=pointer:raise ValueError('Pinned pointer installation incomplete')
    sync_directory(output.parent/'cos-channel')
    p.atomic_json(output/'.workers/restore-complete.json',dict(pin_sha256=p.digest(output.parent/'restore-pin.json')))
    sync_directory(output/'.workers')


def verify_completed_pin(output,cfg,protocol,recovery):
    """Authenticate the installed prefix independently of whether COS writes are enabled."""
    output=Path(output);p=protocol;work=output/'.workers';outer=output.parent/'restore-pin.json'
    from .rmea_token_transport_recovery import regular_paths,preflight_recovery_paths
    preflight_recovery_paths(output,p.MODES)
    validate_outer_phase(output,cfg,p)
    regular_paths(directories=(work,output.parent/'cos-channel'),files=(work/'restore-pin.json',work/'restore-complete.json'))
    pin=p.read(outer)
    expected=dict(pin_sha256=p.digest(outer))
    phase=output.parent.parent/'restore-install.json'
    if phase.exists() and p.read(phase).get('status')!='complete':
        raise ValueError('Outer pinned installation is incomplete')
    if (not (work/'restore-pin.json').is_file() or p.read(work/'restore-pin.json')!=pin
            or not (work/'restore-complete.json').is_file() or p.read(work/'restore-complete.json')!=expected):
        raise ValueError('Interrupted pinned installation requires explicit --restore-pin')
    last=pin['acknowledged_generations'][-1];files=last['request']['files'];prefix='cos-channel/'+pin['generation']+'/'
    receipts=pin_receipts(pin,run_id=output.parent.name,identity=recovery.identity,total=recovery.total,
        sha256=pin['sha256'],latest_bytes=files[prefix+'latest.pt']['bytes'],
        best_sha256=files[prefix+'best.pt']['sha256'],best_bytes=files[prefix+'best.pt']['bytes'])
    for receipt in receipts:
        directory=output.parent/'cos-channel'/receipt['request']['generation']
        if (directory.is_symlink() or not directory.is_dir()
                or any((directory/n).is_symlink() for n in ('request.json','ack.json'))):
            raise ValueError('Installed prefix path is nonregular')
        if (p.read(directory/'request.json')!=receipt['request'] or p.read(directory/'ack.json')!=receipt['ack']
                or p.digest(directory/'request.json')!=receipt['ack']['request_sha256']):
            raise ValueError('Installed pinned prefix differs')
    recovery.verify_cloud_history()
    pointer=recovery._cloud_evidence['pointer']
    if pointer is None:raise ValueError('Completed install lacks acknowledged prefix')
    if pointer['done']<pin['done'] or pointer['generation']<pin['generation']:
        raise ValueError('Installed pointer precedes selected pin')


def devices(requested, visible, free):
    """Return only eligible devices inside the caller's CUDA allocation."""
    if not visible or len(set(visible)) != len(visible):
        raise ValueError('Empty or duplicate allocation')
    if requested == 'auto':
        selected = list(range(len(visible)))
    else:
        fields = requested.split(',')
        if any(not x.isdecimal() for x in fields): raise ValueError('GPU indexes must be nonnegative integers')
        selected = [int(x) for x in fields]
        if len(set(selected)) != len(selected) or any(i >= len(visible) for i in selected):
            raise ValueError('Duplicate or out-of-allocation GPU')
    eligible = [visible[i] for i in selected if free[i] >= 8*1024**3]
    if not eligible or (requested != 'auto' and len(eligible) != len(selected)):
        raise ValueError('Insufficient GPU memory; keep the scientific budget fixed')
    return eligible


def launch_spec(output, kind, device, *, resume):
    if kind not in p.MODES or not str(device).strip(): raise ValueError('Undeclared worker assignment')
    argv = [sys.executable, '-B', '-m', __name__, 'worker', '--output', str(Path(output).resolve()), '--condition', kind,
            '--parent-pid',str(os.getpid())]
    if resume: argv.append('--resume')
    return dict(argv=argv, cwd=str(p.ROOT), physical_gpu=str(device), condition=kind,
                revision=p.git('rev-parse', 'HEAD'))


def condition_recovery(output, cfg, condition, *, protocol=None, resume=False):
    import torch
    from .rmea_token_transport_recovery import Recovery, atomic_torch,sync_directory,regular_paths,condition_progress_bound,validate_retained_best
    p = protocol if protocol is not None else globals()['p']
    output=Path(output);folder=output/'.workers'/condition
    regular_paths(directories=(output,output/'.workers',folder,output/'training-started'),files=tuple(folder/n for n in ('latest.pt','best.pt','progress.json','RUN_PAUSED','WORKER_COMPLETE')))
    if not resume and (folder/'latest.pt').exists():raise ValueError('Fresh worker must not overwrite committed state')
    class ConditionRecovery(Recovery):
        def __init__(self):
            super().__init__(output, cfg, modes=(condition,), protocol=p)
            self.latest, self.best = folder/'latest.pt', folder/'best.pt'
            for path in (self.latest,self.best):
                if path.is_symlink() or path.exists() and not path.is_file():raise ValueError('Nonregular condition checkpoint')
            self.identity['condition']=condition
            self.stop=False
            self.start_marker=output/'training-started'/(condition+'.json')
            global_identity={k:v for k,v in self.identity.items() if k!='condition'}
            minimum,self.started=condition_progress_bound(output,condition,p,global_identity,self.total)
            prior=None
            aggregate_file=output/'checkpoints/latest.pt'
            if aggregate_file.exists():
                aggregate=torch.load(aggregate_file,map_location='cpu',weights_only=True)
                if aggregate.get('schema')!='q-rmea.parallel-resume.v1' or aggregate.get('identity')!=global_identity:
                    raise ValueError('Worker aggregate identity differs')
                prior=aggregate['workers'][condition]
                if prior is not None:
                    if prior.get('identity')!=self.identity or prior.get('cache')!=self.cache:raise ValueError('Prior condition identity/cache differs')
                    validate_training_envelope(prior,cfg);self._validate_cursor(prior)
                    minimum=max(minimum,prior['done'])
            if not self.latest.exists():
                if minimum or self.started or (folder/'WORKER_COMPLETE').exists() or prior is not None:
                    raise ValueError('Missing condition latest contradicts retained progress')
                validate_retained_best(self.best,{},cfg)
                validate_retained_checkpoints(output,None,(condition,),cfg)
            elif resume:
                self.state=torch.load(self.latest,map_location='cpu',weights_only=True)
                if self.state['identity']!=self.identity or self.state['cache']!=self.cache:raise ValueError('Condition resume identity differs')
                validate_training_envelope(self.state,cfg);self._validate_cursor(self.state)
                if self.state['done']<minimum:raise ValueError('Condition checkpoint precedes durable committed progress')
                if prior is not None and (commit_rank(self.state)<commit_rank(prior) or commit_rank(self.state)==commit_rank(prior) and not equal_state(self.state,prior)):
                    raise ValueError('Condition checkpoint regressed or conflicts with aggregate')
                validate_retained_checkpoints(output,self.state,(condition,),cfg)
                validate_retained_best(self.best,self.state['best'],cfg)
                atomic_torch(self.best,self.state['best'])
        def save(self, **kwargs):
            result = super().save(**kwargs)
            import fcntl
            with (output/'.workers/commit.lock').open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                control=p.read(output/'.workers/control.json')
                control['committed'][condition]=result['done']
                if sum(control['committed'].values())>=control['threshold']:
                    (output/'.workers/pause').touch()
                p.atomic_json(output/'.workers/control.json',control)
            if self.stop or (output/'.workers/pause').exists():
                p.atomic_json(folder/'RUN_PAUSED', dict(done=result['done']))
                raise SystemExit(75)
            return result
        def before_update(self):
            if self.stop or (output/'.workers/pause').exists():
                p.atomic_json(folder/'RUN_PAUSED',dict(done=self.state['done']))
                raise SystemExit(75)
            if not self.started:
                self.start_marker.parent.mkdir(exist_ok=True)
                p.atomic_json(self.start_marker,dict(run_id=output.parent.name,identity=self.identity))
                sync_directory(self.start_marker.parent);sync_directory(output)
                self.started=True
        def progress(self, stage):
            p.atomic_json(folder/'progress.json',dict(condition=condition,stage=stage,
                completed=self.state['done'] if self.state else 0,total=self.total,
                observed_at=time.time(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    result=ConditionRecovery();folder.mkdir(parents=True,exist_ok=True)
    return result


def worker(args):
    # Set before importing Torch; a killed parent cannot strand a GPU worker.
    import ctypes
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Parent-death fencing failed')
    if os.getppid()!=args.parent_pid:os.kill(os.getpid(),signal.SIGKILL)
    import torch
    from .rmea_token_transport_recovery import Recovery, atomic_torch, regular_paths
    from .rmea_token_transport_study import train
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
        raise ValueError('Worker needs exactly one allocated GPU')
    output = args.output.resolve(); cfg = p.read(output/'manifest.json')['config']
    folder = output/'.workers'/args.condition
    regular_paths(directories=(output,folder.parent,folder),files=(folder/'.lock',))
    folder.mkdir(parents=True, exist_ok=True)
    from .rmea_token_transport_recovery import run_lock
    with run_lock(folder/'.lock'):
        recovery = condition_recovery(output,cfg,args.condition,resume=args.resume)
        def pause(*_): recovery.stop = True
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, pause)
        labels = p.read(output.parent/'recovery-inputs/labels.json')
        result = train(cfg, output, labels, recovery=recovery, modes=(args.condition,), finalize=False)
        p.atomic_json(folder/'result.json', result)
        (folder/'RUN_PAUSED').unlink(missing_ok=True)
        p.atomic_json(folder/'WORKER_COMPLETE',dict(result_sha256=p.digest(folder/'result.json'),
            terminal_sha256=p.digest(folder/'latest.pt'),
            checkpoint_sha256=p.digest(output/'checkpoints'/(args.condition+'-final.pt')),
            source=p.identity(p.sources()),condition=args.condition,
            geometry_sha256=p.digest(output/'private-geometry.json') if args.condition=='cubic_coherent' else None))


def parallel_train(cfg, output, labels, allocated, *, data_dir=None, assets=None, cos=False, resume=False, pause_after=0, restore_pin=False, runtime=None):
    import torch
    from .rmea_token_transport_recovery import Recovery, atomic_torch
    from .rmea_token_transport_study import finish
    p = runtime.protocol if runtime is not None else globals()['p']
    finish = runtime.finish if runtime is not None else finish
    output=Path(output);work=output/'.workers'
    from .rmea_token_transport_recovery import sync_directory,regular_paths
    regular_paths(directories=(output,output/'checkpoints',work),files=(work/'pause',work/'control.json',output/'parallel.json'))
    validate_outer_phase(output,cfg,p)
    for kind in p.MODES:
        regular_paths(directories=(work/kind,),files=(work/kind/'latest.pt',work/kind/'best.pt'))
    per_head = cfg['epochs']*per_epoch(cfg)
    if not allocated or len(set(allocated)) != len(allocated): raise ValueError('Invalid resolved allocation')
    pause_file = work/'pause'
    class BatchRecovery(Recovery):
        def _validate_cursor(self, state):
            if (state.get('schema') != 'q-rmea.parallel-resume.v1' or state['identity'] != self.identity
                    or state['cache'] != self.cache or state['total'] != self.total
                    or set(state['workers']) != set(p.MODES)):
                raise ValueError('Parallel recovery identity/scope differs')
            done = 0
            for kind, row in state['workers'].items():
                if row is None: continue
                if row.get('schema') != 'q-rmea.resume.v1' or row['identity'] != dict(self.identity,condition=kind) or row['cache'] != self.cache:
                    raise ValueError('Cross-condition checkpoint differs')
                validator = object.__new__(Recovery);validator.cfg=cfg;validator.modes=(kind,);validator.total=per_head
                validator._validate_cursor(row);done += row['done']
            if done != state['done']: raise ValueError('Parallel cursor sum differs')
    recovery = BatchRecovery(output, cfg, enabled=cos, protocol=p)
    if not resume and (recovery.latest.exists() or recovery.best.exists() or (work.exists() and any(work.iterdir())) or (output.parent/'restore-pin.json').exists() or (output/'training-started').exists()
            or list((output/'checkpoints').glob('*-final.pt')) or list((output/'checkpoints').glob('*-selected.pt'))):
        raise ValueError('Existing training state requires explicit resume')
    if resume and not restore_pin and ((output.parent/'restore-pin.json').exists() or (work/'restore-pin.json').exists()):
        verify_completed_pin(output,cfg,p,recovery)
    aggregate=None;pin_plan=None
    if restore_pin and (not resume or not recovery.latest.is_file()):raise ValueError('Verified aggregate pin required')
    if resume and recovery.latest.exists():
        aggregate=torch.load(recovery.latest,map_location='cpu',weights_only=True)
        recovery._validate_cursor(aggregate)
    if restore_pin:pin_plan=install_pinned_workers(output,cfg,p,runtime=runtime,install=False)
    proposed={}
    for kind in p.MODES:
        target=work/kind/'latest.pt'
        local=torch.load(target,map_location='cpu',weights_only=True) if target.exists() else None
        base=aggregate['workers'][kind] if aggregate else None
        if resume and not restore_pin and (output/'training-started'/(kind+'.json')).exists() and local is None:
            raise ValueError('Started original worker checkpoint missing; explicit verified restore required')
        if restore_pin:
            if local is not None and not equal_state(local,base):raise ValueError('Local worker differs from selected pin')
            proposed[kind]=base
        elif local is not None:
            if base is not None:
                if commit_rank(local)<commit_rank(base) or commit_rank(local)==commit_rank(base) and not equal_state(local,base):
                    raise ValueError('Local worker regressed or conflicts with aggregate')
            proposed[kind]=local
        else:proposed[kind]=base
    if resume:
        preflight_retained_workers(output,cfg,p,proposed,recovery.identity,recovery.cache,runtime=runtime)
        validate_resume_evidence(output,cfg,p,proposed,recovery,aggregate,runtime=runtime,
            authenticated_receipts=pin_plan['receipts'] if restore_pin else ())
    # Every identity, retained receipt, cache and durable progress bound passed.
    work.mkdir(exist_ok=True);sync_directory(work);sync_directory(output)
    if resume:recovery.preserve_pending_payloads()
    if resume and not restore_pin:recovery.repair_cloud_pointer()
    if restore_pin:
        install_pinned_workers(output,cfg,p,runtime=runtime)
        for receipt in pin_plan['receipts']:
            directory=recovery.channel/receipt['request']['generation'];directory.mkdir(parents=True,exist_ok=True)
            for name,key in (('request.json','request'),('ack.json','ack')):
                if not (directory/name).exists():p.atomic_json(directory/name,receipt[key])
        if not (recovery.channel/'cloud-latest.json').exists():p.atomic_json(recovery.channel/'cloud-latest.json',pin_plan['pointer'])
        complete_pinned_install(output,p)
    for kind,row in proposed.items():
        if row is None:continue
        target=work/kind/'latest.pt'
        if not target.exists():atomic_torch(target,row)
        atomic_torch(work/kind/'best.pt',row['best'])
    recovery.state=aggregate
    pause_file.unlink(missing_ok=True)
    def snapshot():
        states = {k:torch.load(work/k/'latest.pt',map_location='cpu',weights_only=True)
                  if (work/k/'latest.pt').exists() else None for k in p.MODES}
        state=dict(schema='q-rmea.parallel-resume.v1',identity=recovery.identity,cache=recovery.cache,
                   workers=states,total=recovery.total,done=sum(r['done'] for r in states.values() if r is not None))
        recovery._validate_cursor(state);atomic_torch(recovery.latest,state)
        atomic_torch(recovery.best,{k:r['best'][k] for k,r in states.items() if r and k in r['best']})
        recovery.state=state
    # Reconcile sealed payloads while the original aggregate is still available.
    recovery.drain_pending()
    snapshot()
    if cfg['profile']=='discovery' or runtime is not None and cos:
        if runtime is None: recovery.prepare(labels,data_dir)
        else: runtime.prepare(recovery,labels,data_dir)
    recovery.drain_pending()
    recovery.verify_cloud_history()
    if recovery.cloud_cursor()>recovery.state['done']:raise ValueError('Available local states precede verified COS cursor')
    active={}; pending=[]; completed={}; stopping=False
    commands=p.read(output/'parallel.json')['commands'] if resume and (output/'parallel.json').exists() else []
    def valid_complete(kind):
        file=work/kind/'latest.pt'
        proposed=torch.load(file,map_location='cpu',weights_only=True) if file.exists() else None
        return completed_worker(output,cfg,kind,p,recovery.identity,recovery.cache,proposed,runtime=runtime)
    for kind in p.MODES:
        result=valid_complete(kind) if resume else None
        if result is None:pending.append(kind)
        else:completed[kind]=result
    def request_pause(*_):
        nonlocal stopping
        stopping=True;pause_file.touch()
    previous={sig:signal.signal(sig,request_pause) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    threshold=next((math.ceil(recovery.total*x/100) for x in range(10,101,10)
                    if recovery.cloud_cursor()<math.ceil(recovery.total*x/100)),recovery.total+1)
    last_progress=0.;barrier=False;last_done=recovery.state['done'];jobs={}
    def control():
        committed={k:r['done'] if r else 0 for k,r in recovery.state['workers'].items()}
        target=min(threshold if cos else recovery.total+1,pause_after or recovery.total+1)
        if sum(committed.values())==recovery.total and not pause_after:target=recovery.total+1
        p.atomic_json(work/'control.json',dict(threshold=target,committed=committed))
        if sum(p.read(work/'control.json')['committed'].values())>=p.read(work/'control.json')['threshold']:pause_file.touch()
    control()
    def progress(stage):
        nonlocal last_done,jobs
        jobs={}
        for k in p.MODES:
            file=work/k/'progress.json';row=p.read(file) if file.exists() else {}
            jobs[k]=dict(row,status='complete' if k in completed else 'running' if k in active else 'pending')
        committed=p.read(work/'control.json')['committed']
        for k in jobs:jobs[k]['completed']=committed[k]
        last_done=sum(committed.values())
        attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name if os.environ.get('RESEARCH_OUTPUT_DIR') else 'standalone'
        p.atomic_json(Path(os.environ.get('QATTN_PROGRESS_PATH',str(output.parent/'progress.json'))),dict(
            schema='research.progress.v1',attempt=attempt,stage=stage,observed_at=time.time(),run_id=output.parent.name,identity=recovery.identity,
            completed=last_done,total=recovery.total,jobs=jobs,cloud_verified_updates=recovery.cloud_cursor()))
    try:
        while pending or active:
            for kind,(proc,device,log) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del active[kind]
                if code==0:completed[kind]=valid_complete(kind)
                elif code==75 and (barrier or stopping or pause_file.exists()):pending.insert(0,kind)
                else:raise RuntimeError('Condition failed without automatic retry: '+kind+' exit='+str(code))
            if time.monotonic()-last_progress>=2:
                progress('pausing' if stopping else 'training');last_progress=time.monotonic()
            if pause_file.exists() and not stopping and not barrier:
                barrier=True;pause_file.touch()
            if (barrier or stopping) and not active:
                snapshot()
                if stopping or pause_after and recovery.state['done']>=pause_after:
                    p.atomic_json(output/'RUN_PAUSED',dict(done=recovery.state['done'],schema='q-rmea.parallel-pause.v1'))
                    raise SystemExit(75)
                recovery.milestone(final=not pending)
                threshold=next((math.ceil(recovery.total*x/100) for x in range(10,101,10)
                    if recovery.cloud_cursor()<math.ceil(recovery.total*x/100)),recovery.total+1)
                control();pause_file.unlink(missing_ok=True);barrier=False
            if not barrier and not stopping:
                busy={v[1] for v in active.values()}
                for device in allocated:
                    if device in busy or not pending:continue
                    kind=pending.pop(0);folder=work/kind;folder.mkdir(exist_ok=True)
                    sync_directory(folder);sync_directory(work)
                    launcher = runtime.launch_spec if runtime is not None else launch_spec
                    spec=launcher(output,kind,device,resume=(folder/'latest.pt').exists())
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
                    log=(folder/'worker.log').open('ab')
                    proc=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[kind]=(proc,device,log);commands.append(dict(spec,pid=proc.pid,started_at=time.time()))
                    p.atomic_json(output/'parallel.json',dict(schema='q-rmea.parallel.v1',allocated=allocated,commands=commands))
            time.sleep(.2)
        snapshot()
        if recovery.state['done']!=recovery.total or any(r is None or r['cursor']['mode']!=1 for r in recovery.state['workers'].values()):
            raise ValueError('Full aggregate terminal budget required before sealing')
        recovery.milestone(final=True)
        finish(cfg,output,[completed[k] for k in p.MODES],data_dir=data_dir,assets=assets)
        (output/'RUN_PAUSED').unlink(missing_ok=True)
        failure=output/'RUN_FAILED'
        if failure.exists():failure.rename(output/('failure-history-'+str(time.time_ns())+'.json'))
        progress('audit')
    except BaseException as exc:
        if not isinstance(exc,SystemExit) or exc.code!=75:
            p.atomic_json(output/'RUN_FAILED',dict(error=type(exc).__name__,pending=pending,running=list(active)))
        raise
    finally:
        pause_file.touch()
        for proc,_,_ in active.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        for proc,_,log in active.values():
            try:proc.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['worker'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--condition',choices=p.MODES,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--parent-pid',type=int,required=True)
    worker(parser.parse_args())


if __name__=='__main__':main()
