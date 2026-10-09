"""Suite-scoped disk accounting; never evict another run or a live checkpoint."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager

from .common import ROOT, PACKAGE, atomic_json, digest

GIB = 1024**3
FLOOR = 256 * 1024**2  # Leave enough room to record a clean pause.
ENV = 'QATTN_REFERENCE_STORAGE_ROOT'


class DiskPressure(RuntimeError):
    """Exit75: replay from the last durable batch, never delete it to make room."""


def configure(suite):
    suite = Path(suite).absolute()
    regular(suite)
    if not suite.is_relative_to(ROOT / 'runs'):
        raise ValueError('storage governor requires a suite under runs')
    os.environ[ENV] = str(suite)
    return suite


def regular(path):
    path = Path(path).absolute()
    if path == Path(path.anchor) or any(p.is_symlink() for p in (path, *path.parents)) or path.resolve() != path:
        raise ValueError('regular non-root storage path required')
    return path


def owner(path):
    raw = os.environ.get(ENV)
    if not raw:
        return None
    root = regular(Path(raw))
    path = regular(path)
    if not (path.is_relative_to(root) or path.is_relative_to(ROOT / '.external' / 'assets')):
        raise ValueError('storage operation is not owned by this suite/cache')
    return root


def fsync_parent(path):
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    regular(path)
    atomic_json(path, value)
    fsync_parent(path)


def serialized_bytes(value):
    """Count physical tensor storages once, including views and optimizer moments."""
    import torch
    seen, objects = set(), set()
    total = 0
    def walk(v):
        nonlocal total
        if id(v) in objects:
            return
        objects.add(id(v))
        if torch.is_tensor(v):
            storage = v.untyped_storage()
            key = (str(v.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(v, dict):
            for item in v.values(): walk(item)
        elif isinstance(v, (tuple, list)):
            for item in v: walk(item)
        elif hasattr(v, 'nbytes'):
            total += int(v.nbytes)
        elif isinstance(v, (str, bytes)):
            total += len(v)
    walk(value)
    return int(total * 1.05) + 2 * 1024**2


def process_identity(pid):
    try:
        # Start time prevents PID reuse from retaining an abandoned reservation.
        text = Path(f'/proc/{pid}/stat').read_text()
        return text.rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


@contextmanager
def write_reservation(path, size):
    root = owner(path)
    if root is None:
        yield
        return
    directory = root / 'storage'
    directory.mkdir(parents=True, exist_ok=True)
    regular(directory)
    lock = directory / 'allocation.lock'
    regular(lock)
    claim = directory / ('claim-' + uuid.uuid4().hex + '.json')
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reserved = 0
        for item in directory.glob('claim-*.json'):
            regular(item)
            v = json.loads(item.read_text())
            if v.get('start') == process_identity(v['pid']) and v.get('start') is not None:
                reserved += v['bytes']
        free = shutil.disk_usage(Path(path).parent).free
        if free < int(size) + reserved + FLOOR:
            message = f'disk pause before write: {Path(path).name}; need {int(size)/GIB:.2f}GiB + pending {reserved/GIB:.2f}GiB, free {free/GIB:.2f}GiB; durable latest preserved'
            durable_json(root / 'STORAGE_PAUSED.json', {'status':'paused','reason':message,'recovery':'free space, then resume the same suite','path':str(path)})
            raise DiskPressure(message)
        durable_json(claim, {'pid':os.getpid(),'start':process_identity(os.getpid()),'bytes':int(size),'path':str(path)})
        fcntl.flock(handle, fcntl.LOCK_UN)
    try:
        yield
    finally:
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            claim.unlink(missing_ok=True)
            fsync_parent(claim)


def owned_bytes(path):
    path = regular(path)
    if not path.exists(): return 0
    seen, total = set(), 0
    for p in path.rglob('*'):
        regular(p)
        if p.is_file():
            stat = p.stat();key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key);total += stat.st_size
    return total


def task_peak(model):
    """A scheduling estimate, not a claim about all possible case tensor shapes."""
    if model.startswith('luke_'):
        lock = json.loads((PACKAGE/'luke_assets.lock.json').read_text())
    elif model == 'metaentail':
        lock = json.loads((PACKAGE/'metaentail_assets.lock.json').read_text())
    elif model.startswith('ctl_drp_'):
        lock = json.loads((PACKAGE/'reference_resources.lock.json').read_text())
        weight = lock['ctl_weights']['sup' if model.endswith('supervised') and not model.endswith('unsupervised') else 'unsup']['size_bytes']
        return int(weight*6.3) + 4*GIB + FLOOR
    else:
        lock = json.loads((PACKAGE/'assets.lock.json').read_text())
    weight = max(r['size_bytes'] for r in lock['files'])
    # latest(model+Adam) + best(weights) + atomic next-latest staging.
    # Curriculum reviewers are sealed/offloaded before the next reviewer.
    return int(weight*7.35) + (8 if model=='curriculum' else 4)*GIB + FLOOR


def asset_rows(models):
    resources = json.loads((PACKAGE/'reference_resources.lock.json').read_text())
    base = ROOT/'.external/assets/reference_resources'/resources['revision']
    rows = [(base/resources['source_bundle']['name'], resources['source_bundle'])]
    if any(n in ('curriculum','luke_cfie') for n in models):
        rows.append((base/resources['parser_asset']['name'],resources['parser_asset']))
    for variant, name in [('sup','ctl_drp_supervised'),('unsup','ctl_drp_unsupervised')]:
        if name in models: rows.append((base/variant/resources['ctl_weights'][variant]['name'],resources['ctl_weights'][variant]))
    for filename, needed, folder in [
        ('assets.lock.json',any(not n.startswith(('luke_','ctl_drp_')) and n!='metaentail' for n in models),None),
        ('luke_assets.lock.json',any(n.startswith('luke_') for n in models),'luke'),
        ('metaentail_assets.lock.json','metaentail' in models,'metaentail')]:
        if not needed: continue
        spec = json.loads((PACKAGE/filename).read_text())
        # MNLI uses the directory declared by its fetcher.
        if filename=='metaentail_assets.lock.json':
            from .mnli_assets import REVISION
            cache = ROOT/'.external/assets/mnli'/REVISION
        else:
            cache = ROOT/'.external/assets'/spec['revision'] if folder is None else ROOT/'.external/assets'/folder/spec['revision']
        rows += [(cache/'model'/r['name'],r) for r in spec['files']]
    return rows


def disk_plan(models, seeds, *, slots=1, resume=None):
    missing = []
    for path, row in asset_rows(models):
        regular(path)
        if not path.is_file() or path.stat().st_size != row['size_bytes'] or digest(path)!=row['sha256']:
            missing.append(row['size_bytes'])
    # Shared final files once; one file's shard+assembly overlap; not15 copies.
    asset_extra = sum(missing) + max(missing, default=0)
    tasks = [n for n in models for seed in seeds if not resume or not (Path(resume)/'tasks'/f'{n}_seed{seed}'/'RUN_COMPLETE').exists()]
    peaks = sorted((task_peak(n) for n in tasks),reverse=True)
    runtime = sum(peaks[:max(1,slots)])
    return {'schema':'retacred-disk-plan.v2','shared_asset_additional_bytes':asset_extra,
            'runtime_peak_estimate_bytes':runtime,'required_additional_bytes':asset_extra+runtime+FLOOR,
            'hard_start_bytes':asset_extra+FLOOR,'slots':slots,'remaining_tasks':len(tasks),
            'policy':'shared_assets_then_space_admitted_tasks; every_actual_write_guarded; completed_payloads_offloaded',
            'estimate_not_longest_input_guarantee':True}


def clean_verified_parts(target, row, part_directory):
    """Only declared shards, only after the final locked file is fully verified."""
    target, part_directory = regular(target), regular(part_directory)
    if not target.is_file() or target.stat().st_size!=row['size_bytes'] or digest(target)!=row['sha256']:
        raise ValueError('final asset must be verified before shard cleanup')
    for part in row['parts']:
        name = part['name']
        if Path(name).name != name or name.startswith('.'):raise ValueError('unsafe locked shard')
        path = regular(part_directory/name)
        if path.exists():
            if path.stat().st_size!=part['size_bytes'] or digest(path)!=part['sha256']:
                raise ValueError('foreign or corrupt shard preserved')
            path.unlink();fsync_parent(path)


def pause_cli(operation):
    try:
        return operation()
    except DiskPressure as error:
        print('[storage] '+str(error),flush=True)
        return 75


def initialize_checkpoint(run,path,contract,save):
    """An interrupted first save is reconstructible; this never covers a trained batch."""
    from .common import fingerprint
    marker=Path(run)/'INITIALIZATION_PENDING.json'
    durable_json(marker,{'schema':'reference-initial-save.v1','contract_sha256':fingerprint(contract),'completed_updates':0})
    durable_json(Path(run)/'contract.json',contract)
    save()
    marker.unlink();fsync_parent(marker)


def initial_save_pending(run,contract):
    from .common import fingerprint
    marker=Path(run)/'INITIALIZATION_PENDING.json'
    if not marker.is_file() or marker.is_symlink():return False
    if json.loads(marker.read_text())!={'schema':'reference-initial-save.v1','contract_sha256':fingerprint(contract),'completed_updates':0}:
        raise ValueError('initial reconstruction contract drift')
    return True


def clean_abandoned_ctl_stages(run, contract):
    """Caller holds this run's exclusive writer lock; only marked staging is owned."""
    from .common import fingerprint
    directory=regular(Path(run)/'checkpoints')
    if not directory.exists():return
    for path in directory.glob('.generation-*'):
        regular(path)
        marker=path/'storage-owner.json'
        if not marker.is_file():continue  # unknown historical/user staging is preserved
        regular(marker)
        value=json.loads(marker.read_text())
        if value!={'schema':'ctl-owned-stage.v1','contract_sha256':fingerprint(contract)}:
            raise ValueError('foreign CTL temporary stage preserved')
        files=list(path.iterdir())
        if any(regular(p).name not in ('storage-owner.json','state.pt','state.pt.part','state.json') or not p.is_file() for p in files):
            raise ValueError('unknown CTL temporary contents preserved')
        for p in files:p.unlink()
        path.rmdir();fsync_parent(path)
