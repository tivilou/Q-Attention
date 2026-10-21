"""Two retained atomic checkpoint generations; latest owns cursor and best state."""
from pathlib import Path
import time
import torch
from q_attention.experiments import cfra_protocol as p
from q_attention.experiments.ecpa_protocol import save_state

def _entry(directory,entry):
    path=p.checkpoint_file(directory,entry)
    return torch.load(path,map_location='cpu',weights_only=False)

def load(directory,*,recover_previous=False):
    directory=Path(directory);pointer=p.read(directory/'latest.json')
    if pointer['schema_version']!='cfra-checkpoint-pointer.v1':raise ValueError('checkpoint schema differs')
    if not recover_previous:return _entry(directory,pointer['latest'])
    try:_entry(directory,pointer['latest'])
    except (OSError,ValueError,RuntimeError):pass
    else:raise ValueError('previous-generation recovery only applies to damaged latest')
    if not pointer['previous']:raise ValueError('no retained checkpoint predecessor')
    state=_entry(directory,pointer['previous'])
    p.atomic_json(directory/'checkpoint_recovery.json',dict(reason='damaged_latest',
        failed=pointer['latest'],restored=pointer['previous'],restored_update=state['next_update'],
        guarantee='explicit rollback; work after the retained committed cursor may repeat'))
    p.atomic_json(directory/'latest.json',dict(schema_version='cfra-checkpoint-pointer.v1',
        latest=pointer['previous'],previous=None))
    return state

def save(directory,state,*,inject_failure=False):
    directory=Path(directory);generations=directory/'checkpoints';generations.mkdir(parents=True,exist_ok=True)
    pointer=directory/'latest.json';old=p.read(pointer) if pointer.exists() else None
    path=generations/f"{state['next_update']:09d}-{time.time_ns()}.pt"
    try:
        save_state(path,state)
        entry=dict(file=path.name,sha256=p.sha(path))
        if inject_failure:raise RuntimeError('injected checkpoint publication failure')
        p.atomic_json(pointer,dict(schema_version='cfra-checkpoint-pointer.v1',latest=entry,
            previous=old['latest'] if old else None))
    except BaseException:
        # The pointer rename may already have committed before its directory
        # fsync reports an error. Never remove a possibly published generation.
        try:published=pointer.exists() and p.read(pointer)['latest']['file']==path.name
        except (OSError,ValueError,KeyError):published=True
        if not published:path.unlink(missing_ok=True)
        raise
    keep={entry['file']}
    if old:keep.add(old['latest']['file'])
    for obsolete in generations.glob('*.pt'):
        if obsolete.name not in keep:obsolete.unlink()
    p.syncdir(generations)

def exists(directory):
    return (Path(directory)/'latest.json').exists()
