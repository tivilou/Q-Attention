import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


def guard():
    if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ:
        raise RuntimeError('Non-optimized execution required')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def save(path,value):
    path=Path(path)
    assert not path.is_symlink()
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    temporary=path.with_name(path.name+'.writing')
    assert not temporary.is_symlink()
    temporary.write_bytes(raw)
    os.replace(temporary,path)


def safe(root,name):
    name=PurePosixPath(name)
    assert not name.is_absolute() and '..' not in name.parts and str(name) not in ('','.','/')
    root=Path(root).absolute()
    target=root.joinpath(*name.parts)
    assert target.resolve().is_relative_to(root.resolve())
    assert all(not p.is_symlink() for p in [target,*target.parents])
    return target


def complete(output,phase,report):
    output=Path(output)
    assert not output.is_symlink()
    output.mkdir(parents=True,exist_ok=True)
    assert not list(output.iterdir()),'Fresh scheduler output required'
    save(output/'compact-report.json',report)
    save(output/'RUN_COMPLETE',dict(schema='qattn.original-scores.closed.v1',phase=phase,
        files={'compact-report.json':sha(output/'compact-report.json')},training=False))
