"""One source-pinned, execution-only upgrade; never relax scientific contracts.

Historical contracts and checkpoint bytes stay unchanged. The separately bound
release and upgrade receipt describe the executing code. Only the published
a90af7d suite can opt into this path, with its data/runtime/topology still checked
by the ordinary exact contract comparison and checkpoint restorers.
"""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path

from .common import PACKAGE, atomic_json, fingerprint

ORIGIN_ENV = 'QATTN_REFERENCE_EXECUTION_ORIGIN'


@lru_cache(maxsize=1)
def binding():
    return json.loads((PACKAGE/'execution_compat.lock.json').read_text())


def verify_release():
    from .reference_complete import identity, require_review
    lock=binding(); current=identity(); old=lock['origin_source']
    if set(current)-set(old)!=set(lock['allowed_added']) or set(old)-set(current):
        raise ValueError('execution upgrade source set drift')
    if any(current[name]!=sha for name,sha in old.items() if name not in lock['allowed_changed']):
        raise ValueError('execution upgrade changed a scientific or unapproved source')
    require_review(PACKAGE/'reference_review.json')
    return current


def active():
    origin=os.environ.get(ORIGIN_ENV)
    if origin is None:return False
    if origin!=fingerprint(binding()['origin_source']):
        raise ValueError('unknown historical execution origin')
    return True


def activate_suite(manifest):
    # Always start from the explicit suite receipt, never a caller's leftover env.
    os.environ.pop(ORIGIN_ENV,None)
    saved=json.loads(Path(manifest).read_text())
    from .reference_complete import identity
    if saved.get('source')==identity():return
    if saved.get('source')!=binding()['origin_source']:
        raise ValueError('only the pinned a90af7d suite supports this execution upgrade')
    verify_release()
    os.environ[ORIGIN_ENV]=fingerprint(binding()['origin_source'])


def source_map(names=None):
    from .reference_complete import identity
    current=identity() if names is None else None
    if not active():
        from .common import digest
        return current if names is None else {name:digest(PACKAGE/name) for name in names}
    # Every child revalidates the exact release, including unchanged model code.
    verify_release()
    old=binding()['origin_source']
    if names is not None:
        if any(name not in old for name in names):raise ValueError('unbound historical source')
        return {name:old[name] for name in names}
    return dict(old)


def review_sha256(path):
    from .common import digest
    if active():
        verify_release()
        if Path(path).resolve()!=(PACKAGE/'reference_review.json').resolve():
            raise ValueError('historical upgrade requires the release review binding')
        return binding()['origin_review_sha256']
    return digest(Path(path))


def record_upgrade(run):
    if not active():return
    from .common import digest
    run=Path(run);path=run/'execution_upgrade.json'
    value=dict(schema='reference-execution-upgrade.v1',origin_commit=binding()['origin_commit'],
        origin_source=binding()['origin_source'],executing_source=verify_release(),
        executing_review_sha256=digest(PACKAGE/'reference_review.json'),
        checkpoint_contract='original bytes and hashes retained; scientific/runtime/topology checks unchanged',
        controls='absolute-step coarse/fine OOM descent; once-only +1 micro probe for upgraded auto tasks')
    if path.is_symlink():raise ValueError('symlinked execution upgrade receipt')
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('execution upgrade receipt drift')
    atomic_json(path,value)
