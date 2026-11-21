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


def equal_state(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def pin_receipts(pin, *, run_id, identity, total, sha256, latest_bytes, best_sha256, best_bytes):
    """Validate the complete acknowledged prefix before installing any receipt."""
    import hashlib
    if (pin.get('run_id')!=run_id or pin.get('identity')!=identity or pin.get('total')!=total
            or pin.get('sha256')!=sha256 or not pin.get('version_id')):
        raise ValueError('Restored aggregate identity differs')
    selected=pin.get('generation','')
    names=['000-preparation']+['%03d-progress'%x for x in range(10,101,10)]
    if selected not in names[1:]:raise ValueError('A progress generation is required')
    rows=pin.get('acknowledged_generations',[])
    if [r['request'].get('generation') for r in rows]!=names[:names.index(selected)+1]:
        raise ValueError('Incomplete acknowledged generation history')
    previous=0
    for row in rows:
        request,ack=row['request'],row['ack'];name=request['generation']
        raw=(json.dumps(request,ensure_ascii=False,indent=2)+'\n').encode()
        if (request.get('schema')!='q-rmea.cos-request.v1' or request.get('run_id')!=run_id
                or request.get('identity')!=identity or request.get('total')!=total
                or type(request.get('done')) is not int or not previous<=request['done']<=pin['done']
                or ack.get('request_sha256')!=hashlib.sha256(raw).hexdigest()
                or ack.get('done')!=request['done'] or not ack.get('index_commit')
                or not request.get('files') or set(ack.get('objects',{}))!=set(request['files'])):
            raise ValueError('Acknowledged generation lineage differs')
        previous=request['done']
        if name=='000-preparation':
            if previous!=0 or any(not k.startswith('cos-channel/000-preparation/preparation.') or not k.endswith('.tarpart') for k in request['files']):
                raise ValueError('Preparation cursor/inventory differs')
        else:
            if set(request['files'])!={'cos-channel/'+name+'/'+f for f in ('latest.pt','best.pt')}:
                raise ValueError('Progress generation requires latest and best')
            if previous<math.ceil(total*int(name[:3])/100):raise ValueError('Generation precedes its threshold')
        for key,item in request['files'].items():
            obj=ack['objects'][key]
            if (Path(key).is_absolute() or '..' in Path(key).parts or not obj.get('version_id')
                    or any(obj.get(k)!=item.get(k) for k in ('bytes','sha256'))):
                raise ValueError('Pinned object differs')
        if name==selected:
            key='cos-channel/'+name+'/latest.pt';obj=ack['objects'].get(key,{})
            if (request['done']!=pin['done'] or obj.get('sha256')!=sha256
                    or obj.get('version_id')!=pin['version_id'] or obj.get('bytes')!=latest_bytes
                    or ack['objects']['cos-channel/'+name+'/best.pt'].get('sha256')!=best_sha256
                    or ack['objects']['cos-channel/'+name+'/best.pt'].get('bytes')!=best_bytes):
                raise ValueError('Selected aggregate VersionId/hash/cursor differs')
    return rows


def install_pinned_workers(output,cfg,protocol):
    """Validate the selected aggregate and durably install all workers before admission."""
    import torch
    from .rmea_token_transport_recovery import Recovery,atomic_torch,cache_inventory,sync_directory
    p=protocol;output=Path(output);work=output/'.workers';work.mkdir(exist_ok=True)
    latest=output/'checkpoints/latest.pt';best=output/'checkpoints/best.pt';pin=p.read(output.parent/'restore-pin.json')
    recovery=Recovery(output,cfg,protocol=p)
    pin_receipts(pin,run_id=output.parent.name,identity=recovery.identity,total=recovery.total,
        sha256=p.digest(latest),latest_bytes=latest.stat().st_size,best_sha256=p.digest(best),best_bytes=best.stat().st_size)
    state=torch.load(latest,map_location='cpu',weights_only=True)
    if (state.get('schema')!='q-rmea.parallel-resume.v1' or state['identity']!=recovery.identity
            or state['cache']!=cache_inventory(output/'cache') or state['total']!=recovery.total
            or state['done']!=pin['done'] or set(state['workers'])!=set(p.MODES)):
        raise ValueError('Pinned aggregate scope/cursor differs')
    done=0
    installed=work/'restore-pin.json'
    if installed.exists() and p.read(installed)!=pin:raise ValueError('Different interrupted worker installation')
    if not installed.exists():
        if any(work.glob('*/latest.pt')):raise ValueError('Fresh pinned worker destination required')
        p.atomic_json(installed,pin);sync_directory(work)
    for kind,row in state['workers'].items():
        target=work/kind/'latest.pt'
        if row is None:
            if target.exists():raise ValueError('Worker outside selected pin')
            continue
        validator=object.__new__(Recovery);validator.cfg=cfg;validator.modes=(kind,);validator.total=cfg['epochs']*math.ceil(cfg['counts']['train']/32)
        if row['identity']!=dict(recovery.identity,condition=kind) or row['cache']!=state['cache']:
            raise ValueError('Pinned condition identity differs')
        validator._validate_cursor(row);done+=row['done']
        if target.exists() and not equal_state(torch.load(target,map_location='cpu',weights_only=True),row):
            raise ValueError('Existing worker differs from selected pin')
        if not target.exists():atomic_torch(target,row)
        sync_directory(target.parent)
    if done!=pin['done']:raise ValueError('Pinned worker cursor sum differs')
    sync_directory(work);sync_directory(output)


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
    from .rmea_token_transport_recovery import Recovery, atomic_torch,sync_directory
    p = protocol if protocol is not None else globals()['p']
    folder = output/'.workers'/condition; folder.mkdir(parents=True,exist_ok=True)
    if not resume and (folder/'latest.pt').exists():raise ValueError('Fresh worker must not overwrite committed state')
    class ConditionRecovery(Recovery):
        def __init__(self):
            super().__init__(output, cfg, modes=(condition,), protocol=p)
            self.latest, self.best = folder/'latest.pt', folder/'best.pt'
            self.stop = False
            self.start_marker=output/'training-started'/(condition+'.json')
            self.started=self.start_marker.exists()
            if resume and self.latest.exists():
                self.state = torch.load(self.latest, map_location='cpu', weights_only=True)
                if self.state['identity'] != self.identity or self.state['cache'] != self.cache:
                    raise ValueError('Condition resume identity differs')
                self._validate_cursor(self.state)
                atomic_torch(self.best, self.state['best'])
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
            if protocol is not None and not self.started:
                self.start_marker.parent.mkdir(exist_ok=True)
                p.atomic_json(self.start_marker,dict(run_id=output.parent.name,identity=self.identity))
                sync_directory(self.start_marker.parent);sync_directory(output)
                self.started=True
        def progress(self, stage):
            p.atomic_json(folder/'progress.json',dict(condition=condition,stage=stage,
                completed=self.state['done'] if self.state else 0,total=self.total,
                observed_at=time.time(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    return ConditionRecovery()


def worker(args):
    # Set before importing Torch; a killed parent cannot strand a GPU worker.
    import ctypes
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Parent-death fencing failed')
    if os.getppid()!=args.parent_pid:os.kill(os.getpid(),signal.SIGKILL)
    import torch
    from .rmea_token_transport_recovery import Recovery, atomic_torch
    from .rmea_token_transport_study import train
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
        raise ValueError('Worker needs exactly one allocated GPU')
    output = args.output.resolve(); cfg = p.read(output/'manifest.json')['config']
    folder = output/'.workers'/args.condition; folder.mkdir(parents=True, exist_ok=True)
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
    output = Path(output); work = output/'.workers'; work.mkdir(exist_ok=True)
    from .rmea_token_transport_recovery import sync_directory
    sync_directory(work);sync_directory(output)
    per_head = cfg['epochs']*math.ceil(cfg['counts']['train']/32)
    if not allocated or len(set(allocated)) != len(allocated): raise ValueError('Invalid resolved allocation')
    pause_file = work/'pause'; pause_file.unlink(missing_ok=True)
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
    # Inspect original worker presence before any aggregate-to-worker repair.
    if resume and runtime is not None and not restore_pin:
        for kind in p.MODES:
            if (output/'training-started'/(kind+'.json')).exists() and not (work/kind/'latest.pt').is_file():
                raise ValueError('Started original worker checkpoint missing; explicit verified restore required')
    if restore_pin and (not resume or not recovery.latest.is_file()):
        raise ValueError('Verified aggregate pin required')
    if restore_pin:
        pin=p.read(output.parent/'restore-pin.json')
        staged=torch.load(recovery.latest,map_location='cpu',weights_only=True)
        recovery._validate_cursor(staged)
        if pin.get('done')!=staged['done']:raise ValueError('Pin cursor differs')
        receipts=pin_receipts(pin,run_id=output.parent.name,identity=recovery.identity,
            total=recovery.total,sha256=p.digest(recovery.latest),latest_bytes=recovery.latest.stat().st_size,
            best_sha256=p.digest(recovery.best),best_bytes=recovery.best.stat().st_size)
        installed=work/'restore-pin.json'
        if installed.exists() and p.read(installed)!=pin:raise ValueError('Different interrupted restore pin')
        if not installed.exists() and any(work.glob('*/latest.pt')):raise ValueError('Fresh pin destination required')
        p.atomic_json(installed,pin)
        for receipt in receipts:
            name=receipt['request']['generation']
            directory=recovery.channel/name;directory.mkdir(parents=True,exist_ok=True)
            for filename,key in [('request.json','request'),('ack.json','ack')]:
                target=directory/filename
                if target.exists() and p.read(target)!=receipt[key]:raise ValueError('Interrupted receipt differs')
                p.atomic_json(target,receipt[key])
    if resume and recovery.latest.exists():
        recovery.state = torch.load(recovery.latest,map_location='cpu',weights_only=True)
        recovery._validate_cursor(recovery.state)
        if restore_pin and (pin.get('done')!=recovery.state['done'] or pin.get('total')!=recovery.total):
            raise ValueError('Pinned update count differs')
        for kind,row in recovery.state['workers'].items():
            if row is None:
                if restore_pin and (work/kind).exists() and any((work/kind).iterdir()):
                    raise ValueError('Local worker exists outside selected pin')
                continue
            target=work/kind/'latest.pt'
            if restore_pin and target.exists():
                if not equal_state(torch.load(target,map_location='cpu',weights_only=True),row):
                    raise ValueError('Interrupted worker restore differs from pin')
            elif not target.exists():
                atomic_torch(target,row)
            else:
                local=torch.load(target,map_location='cpu',weights_only=True)
                validator=object.__new__(Recovery);validator.cfg=cfg;validator.modes=(kind,);validator.total=per_head
                validator._validate_cursor(local)
                rank=lambda s:(s['done'],s['cursor']['mode'],s['cursor']['epoch'],s['cursor']['next_start'],len(s['cursor']['history']))
                if local['identity']!=row['identity'] or local['cache']!=row['cache'] or rank(local)<rank(row):
                    raise ValueError('Local commit regressed relative to aggregate')
                if rank(local)==rank(row) and not equal_state(local,row):
                    raise ValueError('Equal-cursor local and aggregate commits disagree')
        if restore_pin:
            installed_states={k:torch.load(work/k/'latest.pt',map_location='cpu',weights_only=True)
                if (work/k/'latest.pt').exists() else None for k in p.MODES}
            if not equal_state(installed_states,recovery.state['workers']):raise ValueError('Installed aggregate differs from pin')
    def snapshot():
        states = {k:torch.load(work/k/'latest.pt',map_location='cpu',weights_only=True)
                  if (work/k/'latest.pt').exists() else None for k in p.MODES}
        state=dict(schema='q-rmea.parallel-resume.v1',identity=recovery.identity,cache=recovery.cache,
                   workers=states,total=recovery.total,done=sum(r['done'] for r in states.values() if r is not None))
        recovery._validate_cursor(state);atomic_torch(recovery.latest,state)
        atomic_torch(recovery.best,{k:r['best'][k] for k,r in states.items() if r and k in r['best']})
        recovery.state=state
    # Parent validates the shared cache once; children use that immutable manifest.
    if resume:
        from .rmea_token_transport_recovery import cache_inventory
        if cache_inventory(output/'cache') != recovery.cache: raise ValueError('Shared resume cache differs')
        control_file=work/'control.json'
        previous_committed=p.read(control_file)['committed'] if control_file.exists() else {}
        for kind in p.MODES:
            file=work/kind/'latest.pt';available=torch.load(file,map_location='cpu',weights_only=True) if file.exists() else None
            observed=p.read(work/kind/'progress.json') if (work/kind/'progress.json').exists() else {}
            minimum=max(previous_committed.get(kind,0),observed.get('completed',0),(recovery.state or {}).get('workers',{}).get(kind,{}).get('done',0) if (recovery.state or {}).get('workers',{}).get(kind) else 0)
            if minimum and (available is None or available['done']<minimum):raise ValueError('Missing state behind durable commit evidence')
            if (output/'checkpoints'/(kind+'-final.pt')).exists() and (available is None or available['done']!=per_head):
                raise ValueError('Missing state behind final checkpoint evidence')
            if (work/kind/'WORKER_COMPLETE').exists() and (available is None or available['cursor']['mode']!=1):
                raise ValueError('Missing terminal state behind completion receipt')
    if resume and recovery.cloud_cursor()>sum(torch.load(work/k/'latest.pt',map_location='cpu',weights_only=True)['done'] if (work/k/'latest.pt').exists() else 0 for k in p.MODES):
        raise ValueError('Available local states precede verified COS cursor')
    if resume and runtime is not None:
        available=sum(torch.load(work/k/'latest.pt',map_location='cpu',weights_only=True)['done'] if (work/k/'latest.pt').exists() else 0 for k in p.MODES)
        for kind in p.MODES:
            marker=output/'training-started'/(kind+'.json')
            if marker.exists():
                if p.read(marker)!=dict(run_id=output.parent.name,identity=dict(recovery.identity,condition=kind)):
                    raise ValueError('Training-start evidence identity differs')
                if not (work/kind/'latest.pt').is_file():raise ValueError('Started worker state missing; verified restore required')
        observed=Path(os.environ.get('QATTN_PROGRESS_PATH',str(output.parent/'progress.json')))
        if observed.is_file():
            row=p.read(observed)
            if row.get('run_id')==output.parent.name and row.get('identity')==recovery.identity and row.get('completed',0)>available:
                raise ValueError('Run-bound scheduler progress exceeds available states; verified restore required')
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
        folder=work/kind;marker=folder/'WORKER_COMPLETE';result=folder/'result.json'
        if not marker.exists():return None
        rec=p.read(marker)
        if (rec.get('condition')!=kind or rec.get('source')!=recovery.identity['source']
                or rec.get('result_sha256')!=p.digest(result)
                or rec.get('checkpoint_sha256')!=p.digest(output/'checkpoints'/(kind+'-final.pt'))):
            raise ValueError('Completed condition receipt differs')
        row=p.read(result)
        if set(row['results'])!={kind} or row['results'][kind]['updates']!=per_head:
            raise ValueError('Condition completion budget differs')
        state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if rec.get('terminal_sha256')!=p.digest(folder/'latest.pt'):raise ValueError('Receipt terminal commit differs')
        recovery._validate_cursor(dict(recovery.state,workers={k:state if k==kind else None for k in p.MODES},done=per_head))
        if state['done']!=per_head or state['cursor']['mode']!=1 or set(state['completed'])!={kind}:
            raise ValueError('Receipt lacks the same terminal worker state')
        atomic_torch(folder/'best.pt',state['best'])
        from .rmea_token_transport_study import state_identity
        cp=output/'checkpoints'/(kind+'-final.pt');metric=row['results'][kind]
        if (metric['checkpoint_sha256']!=p.digest(cp) or metric['final_state_sha256']!=state_identity(state['completed'][kind]['state'])
                or state_identity(torch.load(cp,map_location='cpu',weights_only=True))!=metric['final_state_sha256']):
            raise ValueError('Result checkpoint differs from terminal model')
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
