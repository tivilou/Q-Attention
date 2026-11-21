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
import shutil
import tarfile
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
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.pending')
    try:
        with temporary.open('wb') as handle:
            torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


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


class Recovery:
    def __init__(self, output, cfg, *, enabled=False, resume=False, pause_after=0):
        self.output = Path(output); self.cfg = cfg; self.enabled = enabled
        self.parent = self.output.parent; self.channel = self.parent / 'cos-channel'
        self.total = len(p.MODES) * cfg['epochs'] * math.ceil(cfg['counts']['train']/32)
        self.pause_after = pause_after
        self.identity = dict(config=p.identity(cfg), source=p.identity(p.sources()))
        self.latest = self.output / 'checkpoints/latest.pt'
        self.best = self.output / 'checkpoints/best.pt'
        self.state = None
        if resume and self.latest.exists():
            import torch
            self.state = torch.load(self.latest, map_location='cpu', weights_only=True)
            if self.state.get('schema') != 'q-rmea.resume.v1' or self.state['identity'] != self.identity:
                raise ValueError('Resume source/config differs')
            if self.state['cache'] != cache_inventory(self.output/'cache'):
                raise ValueError('Resume cache differs')
            self._validate_cursor(self.state)
            atomic_torch(self.best, self.state['best'])
        manifest_path=self.output/'manifest.json'
        manifest=p.read(manifest_path) if manifest_path.exists() else {}
        if self.state is None:
            self.cache=cache_inventory(self.output/'cache') if resume or not manifest else manifest['cache']
            if resume:
                if (manifest.get('cache')!=self.cache or manifest.get('source_sha256')!=self.identity['source']
                        or manifest.get('config')!=self.cfg):raise ValueError('Zero-update preparation identity differs')
                import torch
                progress=Path(os.environ.get('QATTN_PROGRESS_PATH',str(self.parent/'progress.json')))
                observed=p.read(progress) if progress.exists() else {}
                if (list((self.output/'checkpoints').glob('*-final.pt')) or (self.output/'summary.json').exists()
                        or list(self.channel.glob('*-progress/request.json'))
                        or (self.best.exists() and torch.load(self.best,map_location='cpu',weights_only=True))
                        or (observed.get('total')==self.total and observed.get('completed',0)>0)):
                    raise ValueError('Missing latest contradicts committed progress; restore a verified generation')
        else:self.cache=self.state['cache']

    def _validate_cursor(self, state):
        cursor = state['cursor']; per_epoch = math.ceil(self.cfg['counts']['train']/32)
        per_mode = self.cfg['epochs'] * per_epoch
        mode = cursor['mode']; epoch = cursor['epoch']; start = cursor['next_start']
        if (type(mode) is not int or not 0 <= mode <= len(p.MODES)
                or type(epoch) is not int or not 0 <= epoch <= self.cfg['epochs']
                or type(start) is not int or not 0 <= start <= self.cfg['counts']['train']
                or (start != self.cfg['counts']['train'] and start % 32)):
            raise ValueError('Invalid committed training cursor')
        expected = mode*per_mode + epoch*per_epoch + math.ceil(start/32)
        if (state.get('total') != self.total or state['done'] != expected or not 0 <= expected <= self.total
                or set(state['completed']) != set(p.MODES[:mode])
                or len(cursor['history']) != epoch
                or (mode == len(p.MODES) and (epoch or start or state['active'] is not None))):
            raise ValueError('Resume update/history/head coverage differs')

    def save(self, *, mode, epoch, next_start, history, loss_sum, head, optimizer,
             completed, best, initial_sha, order_sha):
        per_epoch = math.ceil(self.cfg['counts']['train']/32)
        state = dict(schema='q-rmea.resume.v1', identity=self.identity, cache=self.cache,
            total=self.total, done=mode*self.cfg['epochs']*per_epoch+epoch*per_epoch+math.ceil(next_start/32),
            cursor=dict(mode=mode, epoch=epoch, next_start=next_start, history=history, loss_sum=loss_sum),
            active=cpu_tree(head.state_dict()) if head is not None else None,
            optimizer=cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
            completed=completed, best=best, rng=rng_state(), initial_sha=initial_sha, order_sha=order_sha)
        self._validate_cursor(state)
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
            stage=stage, completed=state['done'] if state else 0, total=self.total,
            percent=100*state['done']/self.total if state else 0,
            local_latest_updates=state['done'] if state else 0,
            cloud_verified_updates=self.cloud_cursor(), updated_at=time.time()))

    def cloud_cursor(self):
        path = self.channel/'cloud-latest.json'
        return p.read(path).get('done', 0) if self.enabled and path.exists() else 0

    def _publish(self, name, files, done):
        directory = self.channel/name; directory.mkdir(parents=True, exist_ok=True)
        request = dict(schema='q-rmea.cos-request.v1', run_id=self.parent.name, generation=name,
            identity=self.identity, done=done, total=self.total,
            files={str(f.relative_to(self.parent).as_posix()):dict(bytes=f.stat().st_size,sha256=p.digest(f)) for f in files})
        request_path=directory/'request.json'
        if request_path.exists():
            if p.read(request_path)!=request:raise ValueError('Sealed COS request changed')
        else:p.atomic_json(request_path,request)
        # The owner may retry these same immutable files; training never retries.
        ack = directory/'ack.json'; deadline = time.monotonic()+1800
        while not ack.exists():
            self.progress('awaiting_cos_'+name)
            if time.monotonic() > deadline: raise TimeoutError('COS protection pending; preserved local state, no retraining')
            time.sleep(2)
        receipt = p.read(ack)
        if (receipt.get('request_sha256') != p.digest(directory/'request.json')
                or receipt.get('done') != done or not receipt.get('index_commit')
                or set(receipt.get('objects',{})) != set(request['files'])):
            raise ValueError('COS/index acknowledgement differs')
        for key, row in receipt['objects'].items():
            if (not row.get('version_id') or row.get('sha256') != request['files'][key]['sha256']
                    or row.get('bytes') != request['files'][key]['bytes']):
                raise ValueError('COS pinned object identity differs')
        p.atomic_json(self.channel/'cloud-latest.json', dict(generation=name, done=done,
            index_commit=receipt['index_commit'], request_sha256=receipt['request_sha256']))
        self.progress('training')

    def drain_pending(self):
        if not self.enabled:return
        import torch
        for directory in sorted(self.channel.glob('*')):
            if not directory.is_dir() or (directory/'ack.json').exists():continue
            request=directory/'request.json'
            if request.exists():
                sealed=p.read(request)
                self._publish(directory.name,[self.parent/rel for rel in sealed['files']],sealed['done'])
            elif (directory/'latest.pt').exists():
                # Crash after hardlinking but before the request commit.
                sealed=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
                self._validate_cursor(sealed)
                if not (directory/'best.pt').exists():atomic_torch(directory/'best.pt',sealed['best'])
                self._publish(directory.name,[directory/'latest.pt',directory/'best.pt'],sealed['done'])

    def prepare(self, labels, data_dir):
        if not self.enabled: return
        directory = self.channel/'000-preparation'; directory.mkdir(parents=True, exist_ok=True)
        if (directory/'ack.json').exists(): return
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
                if not target.exists(): shutil.copyfile(src, target)
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
            if (directory/'ack.json').exists(): continue
            directory.mkdir(parents=True, exist_ok=True)
            for src in (self.latest, self.best):
                dst = directory/src.name
                if not dst.exists(): os.link(src, dst)
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
