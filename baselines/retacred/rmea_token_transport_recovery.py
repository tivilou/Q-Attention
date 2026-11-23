"""Atomic local training state and blocking, sealed ten-percent COS handoffs.

No cloud credentials live here. A separately supervised owner publisher verifies
version-pinned objects and the private index before acknowledging a generation.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tarfile
import tempfile
import time

from . import rmea_token_transport_protocol as p


@contextmanager
def run_lock(path):
    """Kernel ownership survives neither SIGKILL nor a host reboot."""
    path=Path(path)
    if path.is_symlink():raise ValueError('Symlinked run lock')
    with path.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    regular_paths(files=(path,),directories=(path.parent,))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.pending',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as handle:
            torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def regular_paths(*,files=(),directories=()):
    """Inspect destinations before repair; missing entries are allowed."""
    for path in map(Path,directories):
        if path.is_symlink() or path.exists() and not path.is_dir():
            raise ValueError('Nonregular recovery directory: '+str(path))
    for path in map(Path,files):
        if path.is_symlink() or path.exists() and not path.is_file():
            raise ValueError('Nonregular recovery file: '+str(path))


def preflight_recovery_paths(output,modes):
    output=Path(output);root=output.parent;work=output/'.workers';channel=root/'cos-channel'
    regular_paths(directories=(root.parent,root,output,output/'checkpoints',work,channel))
    regular_paths(files=(root.parent/'restore-install.json',root.parent/'run-identity.json',root/'restore-pin.json',
        output/'manifest.json',output/'checkpoints/latest.pt',output/'checkpoints/best.pt',
        work/'restore-pin.json',work/'restore-complete.json',channel/'cloud-latest.json'))
    for kind in modes:
        regular_paths(directories=(work/kind,),files=tuple(work/kind/n for n in ('latest.pt','best.pt','WORKER_COMPLETE','result.json')))
    for name in GENERATIONS:
        regular_paths(directories=(channel/name,),files=tuple(channel/name/n for n in ('request.json','ack.json','latest.pt','best.pt')))


GENERATIONS=('000-preparation',)+tuple('%03d-progress'%x for x in range(10,101,10))


def validate_cloud_request(request,*,run_id,identity,total,previous=0):
    name=request.get('generation')
    if (name not in GENERATIONS or request.get('schema')!='q-rmea.cos-request.v1'
            or request.get('run_id')!=run_id or request.get('identity')!=identity
            or type(request.get('total')) is not int or request['total']!=total
            or type(request.get('done')) is not int or not previous<=request['done']<=total):
        raise ValueError('Cloud request identity/cursor differs')
    files=request.get('files')
    if not isinstance(files,dict) or not files:raise ValueError('Missing cloud file inventory')
    if name=='000-preparation':
        expected={'cos-channel/'+name+'/preparation.%03d.tarpart'%i for i in range(len(files))}
        if request['done']!=0 or set(files)!=expected:raise ValueError('Preparation inventory differs')
    elif (set(files)!={'cos-channel/'+name+'/'+n for n in ('latest.pt','best.pt')}
            or request['done']<math.ceil(total*int(name[:3])/100)):
        raise ValueError('Progress inventory/threshold differs')
    for key,item in files.items():
        if (not isinstance(item,dict) or Path(key).is_absolute() or '..' in Path(key).parts or '\\' in key
                or type(item.get('bytes')) is not int or item['bytes']<=0
                or not isinstance(item.get('sha256'),str) or re.fullmatch('[0-9a-f]{64}',item['sha256']) is None):
            raise ValueError('Invalid cloud object descriptor')
    return request['done']


def validate_cloud_receipt(request,ack,*,request_sha256,run_id,identity,total,previous=0):
    done=validate_cloud_request(request,run_id=run_id,identity=identity,total=total,previous=previous)
    if (ack.get('request_sha256')!=request_sha256 or type(ack.get('done')) is not int or ack['done']!=done
            or not isinstance(ack.get('index_commit'),str) or not ack['index_commit']
            or not isinstance(ack.get('objects'),dict) or set(ack['objects'])!=set(request['files'])):
        raise ValueError('Invalid cloud acknowledgement')
    for key,item in request['files'].items():
        obj=ack['objects'][key]
        if (not isinstance(obj,dict) or not isinstance(obj.get('version_id'),str) or not obj['version_id']
                or type(obj.get('bytes')) is not int or obj['bytes']!=item['bytes'] or obj.get('sha256')!=item['sha256']):
            raise ValueError('Cloud version/size/digest differs')
    return dict(generation=request['generation'],done=done,index_commit=ack['index_commit'],request_sha256=request_sha256)


def inspect_cloud_history(channel,*,protocol,run_id,identity,total,authenticated_receipts=()):
    """Read all durable requests/acks, including acknowledgements ahead of the pointer."""
    import torch
    channel=Path(channel);p=protocol;regular_paths(directories=(channel,),files=(channel/'cloud-latest.json',))
    receipts=[];pointers=[];unsealed=[];pending=[];high=0;previous=0;gap=False
    authenticated={r['request']['generation']:r for r in authenticated_receipts}
    if channel.exists():
        for entry in channel.iterdir():
            if entry.name not in GENERATIONS and entry.is_dir() and any((entry/n).exists() for n in ('request.json','ack.json','latest.pt')):
                raise ValueError('Unknown recovery generation')
    for name in GENERATIONS:
        directory=channel/name;regular_paths(directories=(directory,))
        request_file=directory/'request.json';ack_file=directory/'ack.json'
        regular_paths(files=(request_file,ack_file,directory/'latest.pt',directory/'best.pt'))
        restored=authenticated.get(name)
        if restored is not None:
            if request_file.exists() and (p.read(request_file)!=restored['request'] or p.digest(request_file)!=restored['ack']['request_sha256']):
                raise ValueError('Installed request differs from authenticated restore prefix')
            if ack_file.exists() and p.read(ack_file)!=restored['ack']:
                raise ValueError('Installed acknowledgement differs from authenticated restore prefix')
        if not request_file.exists() and restored is None:
            if ack_file.exists():raise ValueError('Acknowledgement lacks immutable request')
            gap=True
            if (directory/'latest.pt').exists():
                row=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
                if (row.get('identity')!=identity or row.get('total')!=total or type(row.get('done')) is not int
                        or not 0<=row['done']<=total):raise ValueError('Unsealed generation identity/cursor differs')
                high=max(high,row['done']);unsealed.append(row)
            elif (directory/'best.pt').exists():raise ValueError('Unsealed best lacks latest')
            continue
        request=p.read(request_file) if request_file.exists() else restored['request']
        if gap:raise ValueError('Request history is not a contiguous prefix')
        if request.get('generation')!=name:raise ValueError('Generation directory differs from request')
        high=max(high,validate_cloud_request(request,run_id=run_id,identity=identity,total=total,previous=previous))
        if not ack_file.exists() and restored is None:pending.append(request);gap=True;continue
        if gap:raise ValueError('Acknowledged history is not a contiguous prefix')
        ack=p.read(ack_file) if ack_file.exists() else restored['ack']
        request_sha=p.digest(request_file) if request_file.exists() else hashlib.sha256((json.dumps(request,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
        pointer=validate_cloud_receipt(request,ack,request_sha256=request_sha,run_id=run_id,identity=identity,total=total,previous=previous)
        previous=pointer['done'];receipts.append(dict(request=request,ack=ack));pointers.append(pointer)
    actual=p.read(channel/'cloud-latest.json') if (channel/'cloud-latest.json').exists() else None
    if (channel/'cloud-latest.json').exists() and (not isinstance(actual,dict) or set(actual)!={'generation','done','index_commit','request_sha256'} or type(actual.get('done')) is not int):
        raise ValueError('Cloud pointer fields/cursor differ')
    if actual is not None and actual not in pointers:raise ValueError('Cloud pointer is not an authenticated prefix record')
    return dict(receipts=receipts,pointer=pointers[-1] if pointers else None,actual_pointer=actual,high_watermark=high,unsealed=unsealed,pending=pending)


def validate_retained_best(path,committed,cfg):
    """Allow a stale derived best, reject any retained improvement absent from latest."""
    import torch
    from .rmea_training_options import exact_tree
    path=Path(path);regular_paths(files=(path,))
    if not path.exists():return
    retained=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(retained,dict) or not set(retained)<=set(committed):raise ValueError('Retained best coverage exceeds candidate')
    score='validation_'+selection_metric(cfg)
    for kind,old in retained.items():
        new=committed[kind]
        if (not isinstance(old,dict) or set(old)!=set(new) or type(old.get('epoch')) is not int
                or not 1<=old['epoch']<=new['epoch'] or old.get('selection')!=new['selection']
                or type(old.get(score)) not in (int,float) or not math.isfinite(old[score]) or not 0<=old[score]<=new[score]
                or not isinstance(old.get('state'),dict) or set(old['state'])!=set(new['state'])):
            raise ValueError('Retained best is ahead of or incompatible with candidate')
        for key,tensor in old['state'].items():
            reference=new['state'][key]
            if (not torch.is_tensor(tensor) or tensor.dtype!=reference.dtype or tensor.shape!=reference.shape or not torch.isfinite(tensor).all()):
                raise ValueError('Retained best tensor schema differs')
        if old['epoch']==new['epoch'] or old[score]==new[score]:
            if not exact_tree(old,new):raise ValueError('Retained best epoch/score/state conflicts')


def run_progress_bound(output,protocol,identity,total):
    output=Path(output);p=protocol;bound=0
    paused=output/'RUN_PAUSED';progress=Path(os.environ.get('QATTN_PROGRESS_PATH',str(output.parent/'progress.json')))
    regular_paths(files=(paused,progress))
    if paused.exists():bound=p.read(paused).get('done',0)
    if progress.exists():
        row=p.read(progress)
        if row.get('run_id')==output.parent.name and row.get('identity')==identity:
            value=row.get('completed',0)
            if type(value) is not int or not 0<=value<=total:raise ValueError('Invalid run progress cursor')
            bound=max(bound,value)
    if type(bound) is not int or not 0<=bound<=total:raise ValueError('Invalid run pause cursor')
    return bound


def condition_progress_bound(output,kind,protocol,identity,total):
    output=Path(output);p=protocol;folder=output/'.workers'/kind;minimum=0
    paths=[(folder/'progress.json','completed'),(folder/'RUN_PAUSED','done')]
    for path,key in paths:
        regular_paths(files=(path,))
        if path.exists():
            value=p.read(path).get(key,0)
            if type(value) is not int or not 0<=value<=total:raise ValueError('Invalid condition progress cursor')
            minimum=max(minimum,value)
    control=output/'.workers/control.json';regular_paths(files=(control,))
    if control.exists():
        value=p.read(control).get('committed',{}).get(kind,0)
        if type(value) is not int or not 0<=value<=total:raise ValueError('Invalid condition committed cursor')
        minimum=max(minimum,value)
    progress=Path(os.environ.get('QATTN_PROGRESS_PATH',str(output.parent/'progress.json')));regular_paths(files=(progress,))
    if progress.exists():
        row=p.read(progress)
        if row.get('run_id')==output.parent.name and row.get('identity')==identity:
            value=row.get('jobs',{}).get(kind,{}).get('completed',0)
            if type(value) is not int or not 0<=value<=total:raise ValueError('Invalid run-bound condition cursor')
            minimum=max(minimum,value)
    marker=output/'training-started'/(kind+'.json');regular_paths(files=(marker,),directories=(marker.parent,folder))
    if marker.exists() and p.read(marker)!=dict(run_id=output.parent.name,identity=dict(identity,condition=kind)):
        raise ValueError('Condition training-start identity differs')
    return minimum,marker.exists()


def sync_directory(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def cpu_tree(value):
    import torch
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def rng_state():
    import numpy as np
    import torch
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python']); n = value['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state_all(value['cuda'])


def cache_inventory(root):
    root = Path(root)
    return {f.relative_to(root).as_posix(): dict(bytes=f.stat().st_size, sha256=p.digest(f))
            for f in sorted(root.rglob('*')) if f.is_file() and not f.is_symlink()}


class _Parts:
    """Write a tar stream once, in individually uploadable <=512MiB objects."""
    def __init__(self, directory):
        self.directory = directory; self.handle = None; self.size = 0; self.number = 0
    def write(self, data):
        view = memoryview(data); count = len(view)
        while view:
            if self.handle is None:
                self.handle = (self.directory / ('preparation.%03d.tarpart' % self.number)).open('xb')
                self.number += 1; self.size = 0
            length = min(len(view), 512*1024**2 - self.size)
            self.handle.write(view[:length]); view = view[length:]; self.size += length
            if self.size == 512*1024**2: self.close()
        return count
    def close(self):
        if self.handle is not None:
            self.handle.flush(); os.fsync(self.handle.fileno()); self.handle.close(); self.handle = None


from .rmea_training_options import batch_size, per_epoch, dtype_name, selection_metric, selection_description, validate_retained_checkpoints, validate_training_envelope


class Recovery:
    def __init__(self, output, cfg, *, enabled=False, resume=False, pause_after=0, modes=None, protocol=None):
        p = protocol if protocol is not None else globals()['p']
        self.protocol=p
        self.output = Path(output); self.cfg = cfg; self.enabled = enabled
        self.modes = tuple(p.MODES if modes is None else modes)
        if cfg.get('reported_checkpoint','final') == 'validation_best' and not cfg.get('parameter_schemas'):
            raise ValueError('Validation-best reporting requires declared per-condition parameter schemas')
        if self.modes != p.MODES and (len(self.modes) != 1 or self.modes[0] not in p.MODES):
            raise ValueError("Recovery scope must be all conditions or one declared condition")
        self.parent = self.output.parent; self.channel = self.parent / 'cos-channel'
        self.total = len(self.modes) * cfg['epochs'] * per_epoch(cfg)
        self.pause_after = pause_after
        self.identity = dict(config=p.identity(cfg), source=p.identity(p.sources()))
        if self.modes != p.MODES: self.identity['condition'] = self.modes[0]
        self.latest = self.output / 'checkpoints/latest.pt'
        self.best = self.output / 'checkpoints/best.pt'
        regular_paths(files=(self.latest,self.best),directories=(self.output,self.latest.parent))
        self._verified_cloud_cursor=0
        self.state = None
        if resume and self.latest.exists():
            import torch
            self.state = torch.load(self.latest, map_location='cpu', weights_only=True)
            if self.state.get('schema') != 'q-rmea.resume.v1' or self.state['identity'] != self.identity:
                raise ValueError('Resume source/config differs')
            if self.state['cache'] != cache_inventory(self.output/'cache'):
                raise ValueError('Resume cache differs')
            self.cache=self.state['cache']
            validate_training_envelope(self.state,self.cfg)
            self._validate_cursor(self.state)
            validate_retained_checkpoints(self.output,self.state,self.modes,self.cfg)
            validate_retained_best(self.best,self.state['best'],cfg)
            if run_progress_bound(self.output,p,self.identity,self.total)>self.state['done']:
                raise ValueError('Durable run progress exceeds serial checkpoint')
            if self.verify_cloud_history()>self.state['done']:
                raise ValueError('Durable cloud progress exceeds serial checkpoint')
            if self.modes!=p.MODES:
                minimum,_=condition_progress_bound(self.output,self.modes[0],p,{k:v for k,v in self.identity.items() if k!='condition'},self.total)
                if minimum>self.state['done']:raise ValueError('Durable condition progress exceeds serial checkpoint')
            self.preserve_pending_payloads()
            atomic_torch(self.best, self.state['best'])
            self.repair_cloud_pointer()
        manifest_path=self.output/'manifest.json'
        manifest=p.read(manifest_path) if manifest_path.exists() else {}
        if self.state is None:
            self.cache=cache_inventory(self.output/'cache') if resume or not manifest else manifest['cache']
            if resume:
                if (manifest.get('cache')!=self.cache or manifest.get('source_sha256')!=self.identity['source']
                        or manifest.get('config')!=self.cfg):raise ValueError('Zero-update preparation identity differs')
                if run_progress_bound(self.output,p,self.identity,self.total) or self.verify_cloud_history():
                    raise ValueError('Missing latest contradicts durable run progress')
                if self.modes!=p.MODES:
                    minimum,started=condition_progress_bound(self.output,self.modes[0],p,{k:v for k,v in self.identity.items() if k!='condition'},self.total)
                    if minimum or started:raise ValueError('Missing latest contradicts durable condition progress')
                import torch
                progress=Path(os.environ.get('QATTN_PROGRESS_PATH',str(self.parent/'progress.json')))
                observed=p.read(progress) if progress.exists() else {}
                if ((list((self.output/'checkpoints').glob('*-final.pt')) + list((self.output/'checkpoints').glob('*-selected.pt'))) or (self.output/'summary.json').exists()
                        or list(self.channel.glob('*-progress/request.json'))
                        or (self.best.exists() and torch.load(self.best,map_location='cpu',weights_only=True))
                        or (observed.get('total')==self.total and observed.get('completed',0)>0)):
                    raise ValueError('Missing latest contradicts committed progress; restore a verified generation')
        else:self.cache=self.state['cache']

    def _validate_cursor(self, state, *, payload=True):
        cursor = state['cursor']; epoch_steps = per_epoch(self.cfg)
        per_mode = self.cfg['epochs'] * epoch_steps
        mode = cursor['mode']; epoch = cursor['epoch']; start = cursor['next_start']
        if (type(mode) is not int or not 0 <= mode <= len(self.modes)
                or type(epoch) is not int or not 0 <= epoch <= self.cfg['epochs']
                or type(start) is not int or not 0 <= start <= self.cfg['counts']['train']
                or (start != self.cfg['counts']['train'] and start % batch_size(self.cfg))
                or (epoch == self.cfg['epochs'] and start != 0)):
            raise ValueError('Invalid committed training cursor')
        expected = mode*per_mode + epoch*epoch_steps + math.ceil(start/batch_size(self.cfg))
        if (state.get('total') != self.total or state['done'] != expected or not 0 <= expected <= self.total
                or set(state['completed']) != set(self.modes[:mode])
                or len(cursor['history']) != epoch
                or (mode == len(self.modes) and (epoch or start or state['active'] is not None))):
            raise ValueError('Resume update/history/head coverage differs')
        if epoch == 0 and start == 0 and (state['active'] is not None or state.get('optimizer') is not None):
            raise ValueError('Fresh-head boundary must not supply model or optimizer state')
        if set(state.get('rng',{}))!={'python','numpy','torch','cuda'}:
            raise ValueError('Missing full RNG state')
        if state['done']>mode*per_mode and (not isinstance(state.get('active'),dict) or not isinstance(state.get('optimizer'),dict)):
            raise ValueError('Active commit lacks model/AdamW state')
        for kind,row in state['completed'].items():
            if not isinstance(row.get('state'),dict) or not isinstance(row.get('optimizer'),dict):
                raise ValueError('Terminal commit lacks model/AdamW state')
        if not payload:return
        import numpy as np
        import torch
        rng=state['rng']
        random.Random().setstate(rng['python'])
        n=rng['numpy'];np.random.RandomState().set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        torch.Generator().set_state(rng['torch'])
        if (not isinstance(rng['cuda'],list) or len(rng['cuda'])!=1
                or any(not torch.is_tensor(v) or v.dtype!=torch.uint8 or v.ndim!=1 or v.numel()<16 for v in rng['cuda'])):
            raise ValueError('Invalid one-worker CUDA RNG payload')
        def parameter_template(kind):
            schemas = self.cfg.get('parameter_schemas')
            if schemas is not None:
                return {k:torch.empty(tuple(v), dtype=getattr(torch,dtype_name(self.cfg))) for k,v in schemas[kind].items()}
            if len(self.modes) == 1:
                from .rmea import RMEAHead
                with torch.random.fork_rng(devices=[]):
                    return RMEAHead('quantum').to(dtype=getattr(torch,dtype_name(self.cfg))).state_dict()
            return None
        def history(rows,length):
            if len(rows)!=length or any(r.get('epoch')!=i+1 or not math.isfinite(r.get('train_ce',float('nan'))) for i,r in enumerate(rows)):
                raise ValueError('Committed epoch history differs')
        def model_optimizer(model,opt,steps,kind):
            template=parameter_template(kind)
            if not model or (template is not None and set(model)!=set(template)):
                raise ValueError('Model parameter inventory differs')
            for key,tensor in model.items():
                if (not torch.is_tensor(tensor) or tensor.dtype!=getattr(torch,dtype_name(self.cfg)) or not torch.isfinite(tensor).all()
                        or template is not None and tensor.shape!=template[key].shape):
                    raise ValueError('Invalid committed model tensor')
            groups=opt.get('param_groups',[])
            if len(groups)!=1:raise ValueError('Missing AdamW parameter group')
            group=groups[0];ids=group['params']
            if len(ids)!=len(model) or len(set(ids))!=len(ids):raise ValueError('AdamW parameter coverage differs')
            for key,expected in [('lr',self.cfg['lr']),('weight_decay',self.cfg['weight_decay']),('eps',self.cfg['optimizer_eps'])]:
                if group.get(key)!=expected:raise ValueError('AdamW configuration differs')
            if tuple(group.get('betas',[]))!=tuple(self.cfg['betas']):raise ValueError('AdamW betas differ')
            if any(group.get(k,False) not in (False,None) for k in ('maximize','amsgrad','capturable','differentiable','fused')):
                raise ValueError('AdamW execution flags differ')
            states=opt.get('state',{})
            if set(states)!=(set(ids) if steps else set()):raise ValueError('AdamW state coverage differs')
            for identifier,tensor in zip(ids,model.values()):
                if not steps:continue
                item=states[identifier]
                if (not torch.is_tensor(item.get('step')) or item['step'].numel()!=1 or item['step'].item()!=steps
                        or any(not torch.is_tensor(item.get(k)) or item[k].shape!=tensor.shape
                               or item[k].dtype!=tensor.dtype or not torch.isfinite(item[k]).all() for k in ('exp_avg','exp_avg_sq'))):
                    raise ValueError('AdamW moments/step differ')
        history(cursor['history'],epoch)
        if not math.isfinite(cursor['loss_sum']) or cursor['loss_sum']<0 or not start and cursor['loss_sum']!=0:
            raise ValueError('Committed loss accumulator differs')
        if state['active'] is not None:model_optimizer(state['active'],state['optimizer'],epoch*epoch_steps+math.ceil(start/batch_size(self.cfg)),self.modes[mode])
        elif state.get('optimizer') is not None:raise ValueError('Boundary has an active optimizer')
        for kind,row in state['completed'].items():
            history(row['history'],self.cfg['epochs'])
            if row['steps']!=per_mode:raise ValueError('Completed optimizer budget differs')
            model_optimizer(row['state'],row['optimizer'],per_mode,kind)
        if len(self.modes)!=1 and self.cfg.get('parameter_schemas') is None:return  # Preserve legacy serial schema; new explicit schemas validate every condition.
        reached=set(state['completed'])|({self.modes[mode]} if mode<len(self.modes) and epoch else set())
        if set(state['best'])!=reached:raise ValueError('Validation-best coverage differs')
        for kind,row in state['best'].items():
            template=parameter_template(kind)
            limit=self.cfg['epochs'] if kind in state['completed'] else epoch
            if (type(row['epoch']) is not int or not 1<=row['epoch']<=limit or not 0<=row['validation_'+selection_metric(self.cfg)]<=1
                    or row.get('selection')!=selection_description(self.cfg)
                    or template is not None and set(row['state'])!=set(template)):
                raise ValueError('Validation-best selection differs')
            if any(not torch.is_tensor(t) or not torch.isfinite(t).all() or
                   template is not None and (t.shape!=template[k].shape or t.dtype!=template[k].dtype)
                   for k,t in row['state'].items()):raise ValueError('Validation-best model differs')


    def save(self, *, mode, epoch, next_start, history, loss_sum, head, optimizer,
             completed, best, initial_sha, order_sha):
        epoch_steps = per_epoch(self.cfg)
        state = dict(schema='q-rmea.resume.v1', identity=self.identity, cache=self.cache,
            total=self.total, done=mode*self.cfg['epochs']*epoch_steps+epoch*epoch_steps+math.ceil(next_start/batch_size(self.cfg)),
            cursor=dict(mode=mode, epoch=epoch, next_start=next_start, history=history, loss_sum=loss_sum),
            active=cpu_tree(head.state_dict()) if head is not None else None,
            optimizer=cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
            completed=completed, best=best, rng=rng_state(), initial_sha=initial_sha, order_sha=order_sha)
        self._validate_cursor(state,payload=False)
        atomic_torch(self.latest, state); self.state = state
        self.progress('training')
        if self.pause_after and ((self.pause_after>0 and state['done']==self.pause_after) or (self.pause_after<0 and state['cursor']['mode']==-self.pause_after and state['active'] is None)):
            p.atomic_json(self.output/'RUN_PAUSED', dict(reason='bounded recovery canary fault', done=state['done']))
            raise SystemExit(75)
        return state

    def progress(self, stage):
        state = self.state
        progress_path=Path(os.environ.get('QATTN_PROGRESS_PATH',str(self.parent/'progress.json')))
        attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name if os.environ.get('RESEARCH_OUTPUT_DIR') else 'standalone'
        p.atomic_json(progress_path, dict(schema='research.progress.v1',attempt=attempt,observed_at=time.time(),
            run_id=self.parent.name,identity=self.identity,
            stage=stage, completed=state['done'] if state else 0, total=self.total,
            percent=100*state['done']/self.total if state else 0,
            local_latest_updates=state['done'] if state else 0,
            cloud_verified_updates=self.cloud_cursor(), updated_at=time.time()))

    def cloud_cursor(self):
        path = self.channel/'cloud-latest.json'
        value=self.protocol.read(path).get('done',0) if path.exists() else 0
        if type(value) is not int or not 0<=value<=self.total:raise ValueError('Invalid cloud cursor')
        return max(value,self._verified_cloud_cursor)

    def _validate_cloud_payload(self,state):
        if state.get('schema')!='q-rmea.parallel-resume.v1':
            if state.get('schema')!='q-rmea.resume.v1':raise ValueError('Cloud payload schema differs')
            self._validate_cursor(state);return
        if (state.get('identity')!=self.identity or state.get('cache')!=self.cache or state.get('total')!=self.total
                or set(state.get('workers',{}))!=set(self.protocol.MODES) or type(state.get('done')) is not int):
            raise ValueError('Parallel cloud payload identity/scope differs')
        done=0
        for kind,row in state['workers'].items():
            if row is None:continue
            if row.get('identity')!=dict(self.identity,condition=kind) or row.get('cache')!=self.cache:
                raise ValueError('Parallel cloud worker identity/cache differs')
            validator=object.__new__(Recovery);validator.cfg=self.cfg;validator.modes=(kind,);validator.total=self.cfg['epochs']*per_epoch(self.cfg)
            validate_training_envelope(row,self.cfg);validator._validate_cursor(row);done+=row['done']
        if done!=state['done']:raise ValueError('Parallel cloud payload cursor sum differs')

    def verify_cloud_history(self,*,authenticated_receipts=()):
        """Validate every durable generation before any resume repair."""
        evidence=inspect_cloud_history(self.channel,protocol=self.protocol,run_id=self.parent.name,identity=self.identity,total=self.total,authenticated_receipts=authenticated_receipts)
        import torch
        from .rmea_training_options import exact_tree
        evidence['fallbacks']=[];evidence['payloads']=[]
        for request in evidence['pending']:
            for rel,item in request['files'].items():
                target=self.parent/rel
                regular_paths(files=(target,),directories=(target.parent,))
                if not target.exists() and not self.enabled:
                    raise ValueError('Pending cloud payload missing; publishing recovery required before local continuation')
                candidate=target if target.exists() else self.output/'checkpoints'/target.name
                regular_paths(files=(candidate,),directories=(candidate.parent,))
                if not candidate.is_file() or candidate.stat().st_size!=item['bytes'] or self.protocol.digest(candidate)!=item['sha256']:
                    raise ValueError('Pending cloud payload lacks identical durable source')
                if candidate!=target:evidence['fallbacks'].append((candidate,target,item))
                if target.name=='latest.pt':
                    row=torch.load(candidate,map_location='cpu',weights_only=True)
                    if row.get('identity')!=self.identity or row.get('cache')!=self.cache or row.get('done')!=request['done']:
                        raise ValueError('Pending cloud payload identity/cursor differs')
                    self._validate_cloud_payload(row)
                    evidence['payloads'].append(row)
                    expected=row.get('best',{k:r['best'][k] for k,r in row.get('workers',{}).items() if r and k in r['best']})
                    best_path=target.parent/'best.pt'
                    best_source=best_path if best_path.exists() else self.output/'checkpoints/best.pt'
                    if not exact_tree(torch.load(best_source,map_location='cpu',weights_only=True),expected):
                        raise ValueError('Pending cloud best differs from embedded latest best')
        for state in evidence['unsealed']:
            if state.get('cache')!=self.cache:raise ValueError('Unsealed generation cache differs')
            self._validate_cloud_payload(state)
            evidence['payloads'].append(state)
        for name in GENERATIONS:
            directory=self.channel/name
            if (directory/'latest.pt').exists() and not (directory/'request.json').exists():
                latest=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
                expected=latest.get('best',{k:r['best'][k] for k,r in latest.get('workers',{}).items() if r and k in r['best']})
                if (directory/'best.pt').exists() and not exact_tree(torch.load(directory/'best.pt',map_location='cpu',weights_only=True),expected):
                    raise ValueError('Unsealed cloud best differs from embedded latest best')
        self._cloud_evidence=evidence
        self._verified_cloud_cursor=evidence['pointer']['done'] if evidence['pointer'] else 0
        return evidence['high_watermark']

    def preserve_pending_payloads(self):
        for candidate,target,item in self._cloud_evidence['fallbacks']:
            if not target.exists():os.link(candidate,target);sync_directory(target.parent)
            if target.stat().st_size!=item['bytes'] or self.protocol.digest(target)!=item['sha256']:
                raise ValueError('Preserved pending cloud payload differs')

    def repair_cloud_pointer(self):
        evidence=self._cloud_evidence
        if evidence['pointer'] is not None and evidence['pointer']!=evidence['actual_pointer']:
            self.protocol.atomic_json(self.channel/'cloud-latest.json',evidence['pointer'])
            sync_directory(self.channel);sync_directory(self.parent)

    def _publish(self, name, files, done):
        p=self.protocol
        if name not in GENERATIONS:raise ValueError('Unknown cloud generation')
        directory = self.channel/name
        regular_paths(directories=(self.channel,directory),files=(directory/'request.json',directory/'ack.json',self.channel/'cloud-latest.json'))
        directory.mkdir(parents=True, exist_ok=True)
        sync_directory(self.channel);sync_directory(self.parent)
        request_path=directory/'request.json'
        ack = directory/'ack.json'
        if request_path.exists() and ack.exists():
            request=p.read(request_path)
            if (request.get('schema')!='q-rmea.cos-request.v1' or request.get('run_id')!=self.parent.name
                    or request.get('generation')!=name or request.get('identity')!=self.identity
                    or request.get('done')!=done or request.get('total')!=self.total
                    or set(request.get('files',{}))!={f.relative_to(self.parent).as_posix() for f in files}):
                raise ValueError('Sealed COS request scope changed')
        else:
            request = dict(schema='q-rmea.cos-request.v1', run_id=self.parent.name, generation=name,
                identity=self.identity, done=done, total=self.total,
                files={str(f.relative_to(self.parent).as_posix()):dict(bytes=f.stat().st_size,sha256=p.digest(f)) for f in files})
            if request_path.exists():
                if p.read(request_path)!=request:raise ValueError('Sealed COS request changed')
            else:
                p.atomic_json(request_path,request)
                sync_directory(directory);sync_directory(directory.parent)
        # The owner may retry these same immutable files; training never retries.
        deadline = time.monotonic()+1800
        while not ack.exists():
            self.progress('awaiting_cos_'+name)
            if time.monotonic() > deadline: raise TimeoutError('COS protection pending; preserved local state, no retraining')
            time.sleep(2)
        receipt = p.read(ack)
        validate_cloud_receipt(request,receipt,request_sha256=p.digest(request_path),run_id=self.parent.name,identity=self.identity,total=self.total)
        with ack.open('rb') as handle:os.fsync(handle.fileno())
        sync_directory(directory);sync_directory(self.channel);sync_directory(self.parent)
        self.verify_cloud_history();self.repair_cloud_pointer()
        self.progress('training')

    def drain_pending(self):
        if not self.enabled:return
        import torch
        for directory in sorted(self.channel.glob('*')):
            if not directory.is_dir():continue
            request=directory/'request.json'
            if request.exists():
                sealed=p.read(request)
                if not (directory/'ack.json').exists():
                    for rel,item in sealed['files'].items():
                        target=self.parent/rel
                        if not target.exists():
                            candidate=self.output/'checkpoints'/target.name
                            if (not candidate.is_file() or candidate.stat().st_size!=item['bytes']
                                    or p.digest(candidate)!=item['sha256']):
                                raise ValueError('Missing sealed payload has no identical durable source')
                            os.link(candidate,target);sync_directory(directory)
                        if target.stat().st_size!=item['bytes'] or p.digest(target)!=item['sha256']:
                            raise ValueError('Unacknowledged sealed payload differs')
                self._publish(directory.name,[self.parent/rel for rel in sealed['files']],sealed['done'])
            elif (directory/'latest.pt').exists():
                # Crash after hardlinking but before the request commit.
                sealed=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
                self._validate_cloud_payload(sealed)
                if not (directory/'best.pt').exists():atomic_torch(directory/'best.pt',sealed.get('best',{k:r['best'][k] for k,r in sealed.get('workers',{}).items() if r and k in r['best']}))
                self._publish(directory.name,[directory/'latest.pt',directory/'best.pt'],sealed['done'])

    def prepare(self, labels, data_dir):
        if not self.enabled: return
        directory = self.channel/'000-preparation'; directory.mkdir(parents=True, exist_ok=True)
        if (directory/'ack.json').exists():
            request=p.read(directory/'request.json');self._publish(directory.name,[self.parent/rel for rel in request['files']],request['done']);return
        if not (directory/'request.json').exists():
            # A process may have died while packaging, before publishing a request.
            for orphan in directory.glob('preparation.*.tarpart'): orphan.unlink()
            files = [self.parent/'handoff.json', self.output/'manifest.json']
            files += [f for f in (self.parent/'canary').rglob('*') if f.is_file() and not f.name.startswith('.')]
            files += [self.output/'cache'/name for name in self.cache]
            raw = self.parent/'recovery-inputs'; raw.mkdir(exist_ok=True)
            for split, (count, sha) in p.EXPECTED.items():
                src = Path(data_dir)/(split+'.jsonl')
                if p.digest(src) != sha: raise ValueError('Raw recovery source differs')
                target = raw/src.name
                if not target.exists() or p.digest(target)!=sha:
                    temporary=target.with_suffix('.pending')
                    shutil.copyfile(src,temporary)
                    if p.digest(temporary)!=sha:raise ValueError('Raw staging digest differs')
                    os.replace(temporary,target)
                if p.digest(target)!=sha:raise ValueError('Raw recovery input differs')
                files.append(target)
            p.atomic_json(raw/'labels.json', labels); files.append(raw/'labels.json')
            if any(f.is_symlink() or not f.resolve().is_relative_to(self.parent) for f in files):
                raise ValueError('Nonregular recovery preparation input')
            size = sum(f.stat().st_size for f in files)
            if size > 20*1024**3: raise ValueError('Preparation exceeds 20GiB recovery ceiling')
            parts = _Parts(directory)
            try:
                with tarfile.open(fileobj=parts, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                    for f in sorted(files): archive.add(f, arcname=f.relative_to(self.parent).as_posix(), recursive=False)
            finally: parts.close()
        parts = sorted(directory.glob('*.tarpart'))
        self._publish('000-preparation', parts, 0)
        # Only these acknowledged, run-owned packaging copies are disposable.
        for f in parts: f.unlink()

    def milestone(self, *, final=False):
        if not self.enabled or self.state is None: return
        done = self.state['done']
        for percent in range(10,101,10):
            threshold = math.ceil(self.total*percent/100)
            if done < threshold or (percent==100 and not final): continue
            name = '%03d-progress' % percent; directory = self.channel/name
            if (directory/'ack.json').exists():
                sealed=p.read(directory/'request.json');self._publish(name,[self.parent/rel for rel in sealed['files']],sealed['done']);continue
            directory.mkdir(parents=True, exist_ok=True)
            for src in (self.latest, self.best):
                dst = directory/src.name
                if not dst.exists(): os.link(src, dst)
            sync_directory(directory);sync_directory(directory.parent)
            self._publish(name, [directory/'latest.pt', directory/'best.pt'], done)


def restore_preparation(parts, destination):
    """Parts must already be version/hash verified by the owner recovery client."""
    class Reader:
        def __init__(self): self.index=0; self.handle=None
        def read(self, size=-1):
            if size < 0: raise ValueError('Bounded reads required')
            data=bytearray()
            while len(data)<size:
                if self.handle is None:
                    if self.index == len(parts): break
                    self.handle=Path(parts[self.index]).open('rb'); self.index+=1
                chunk=self.handle.read(size-len(data))
                if chunk: data.extend(chunk)
                else: self.handle.close(); self.handle=None
            return bytes(data)
        def close(self):
            if self.handle: self.handle.close()
    destination=Path(destination).resolve()
    if destination.exists(): raise ValueError('Fresh restore destination required')
    destination.mkdir(parents=True); reader=Reader(); seen=set(); total=0
    try:
        with tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                rel=Path(member.name); target=destination/rel
                if (not member.isfile() or rel.is_absolute() or '..' in rel.parts or member.name in seen
                        or rel.parts[0] not in ('handoff.json','canary','discovery','recovery-inputs')
                        or not 0<=member.size<=20*1024**3 or not target.resolve().is_relative_to(destination)):
                    raise ValueError('Unsafe recovery member')
                total+=member.size; seen.add(member.name)
                if total>20*1024**3: raise ValueError('Restore byte ceiling exceeded')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src, target.open('xb') as dst: shutil.copyfileobj(src,dst,1024**2)
    finally: reader.close()
