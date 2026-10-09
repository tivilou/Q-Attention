"""Pinned MNLI assets via the project exchange, no model/backend imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import ssl
import time
from urllib.parse import quote

from .assets import assemble, reject_symlinks, safe_name
from .common import ROOT, PACKAGE, atomic_json, digest
from .luke_assets import stream_part, validate_url
from .metaentail_checkpoint import REPO, REVISION, checkpoint_spec, verify_metadata

NAMESPACE = 'q-attention/mnli-reference-assets/'+REVISION
LOCK = PACKAGE/'metaentail_assets.lock.json'
SHARD_BYTES = 512*1024*1024
CA = ROOT/'scripts/collab/certs/qattn-exchange-ca.crt'
HEX = re.compile(r'^[0-9a-f]{64}$')


def originals():
    checkpoint = checkpoint_spec()['checkpoint']
    rows = dict(checkpoint['files'])
    rows[checkpoint['selected_weight_file']] = checkpoint['weights'][checkpoint['selected_weight_file']]
    return rows


def validate_lock(spec):
    if (spec.get('schema')!='mnli-assets.v1' or spec.get('repo')!=REPO
            or spec.get('revision')!=REVISION or spec.get('exchange_namespace')!=NAMESPACE
            or spec.get('original_files_locally_hash_verified') is not True
            or spec.get('model_load_certified') is not False):
        raise ValueError('MNLI asset identity/admission mismatch')
    rows=spec.get('files',[]); names=[row.get('name') for row in rows]; original=originals()
    if len(names)!=len(set(names)) or set(names)!=set(original):
        raise ValueError('MNLI assets require six small files and selected safetensors')
    for row in rows:
        name=safe_name(row['name']); expected=original[name]
        if any(row.get(key)!=expected[key] for key in ('size_bytes','sha256','git_blob')):
            raise ValueError('MNLI original asset identity mismatch: '+name)
        parts=row.get('parts',[])
        if len(parts)!=(row['size_bytes']+SHARD_BYTES-1)//SHARD_BYTES:
            raise ValueError('MNLI shard count mismatch')
        for index,part in enumerate(parts):
            if (safe_name(part['name'])!=f'{name}.part{index:04d}'
                    or type(part.get('size_bytes')) is not int
                    or part['size_bytes']!=min(SHARD_BYTES,row['size_bytes']-index*SHARD_BYTES)
                    or not HEX.fullmatch(part.get('sha256',''))):
                raise ValueError('MNLI shard order/size/hash mismatch')
    return spec


def verify_checkpoint(directory):
    """Stream complete original SHA checks, then small-file config binding."""
    directory=Path(directory).absolute(); reject_symlinks(directory)
    hashes={}
    for name,row in originals().items():
        path=directory/name; reject_symlinks(path)
        if not path.is_file() or path.stat().st_size!=row['size_bytes']:
            raise ValueError('MNLI checkpoint asset size/path mismatch: '+name)
        actual=digest(path)
        if actual!=row['sha256']:
            raise ValueError('MNLI checkpoint asset SHA mismatch: '+name)
        hashes[name]=actual
    metadata=verify_metadata(directory)
    return {'status':'checkpoint_bytes_verified','files':hashes,'metadata':metadata,
            'weights_locally_verified':True,'neural_runtime_verified':False,'formal_admission':False}


def fetch(lock_path=LOCK, *, cache_dir=None):
    lock_path=Path(lock_path); reject_symlinks(lock_path)
    spec=validate_lock(json.loads(lock_path.read_text()))
    cache=Path(cache_dir) if cache_dir else ROOT/'.external/assets/mnli'/REVISION
    reject_symlinks(cache); model=cache/'model'; reject_symlinks(model); model.mkdir(parents=True,exist_ok=True)
    needed=[]
    for row in spec['files']:
        path=model/row['name']; reject_symlinks(path)
        if path.exists():
            if not path.is_file() or path.stat().st_size!=row['size_bytes'] or digest(path)!=row['sha256']:
                raise ValueError('existing MNLI cache is corrupt; preserved for diagnosis')
        else: needed.append(row)
    receipt={'schema':'mnli-assets-fetch.v1','revision':REVISION,'status':'pending',
             'network_downloads':0,'files':{},'model_jobs':0,'neural_runtime_verified':False}
    atomic_json(cache/'fetch-verification.json',receipt)
    if needed:
        url=validate_url(os.environ.get('PROJECT_EXCHANGE_URL',''))
        token=os.environ.get('PROJECT_EXCHANGE_TOKEN','')
        if not token: raise ValueError('PROJECT_EXCHANGE_TOKEN required for missing MNLI assets')
        context=ssl.create_default_context(cafile=CA)
        for row in needed:
            parts=[]
            for part in row['parts']:
                path=cache/'parts'/part['name']
                time.sleep(1)
                result=stream_part(url+'/files/'+NAMESPACE+'/'+quote(part['name']),path,
                    sha256=part['sha256'],size=part['size_bytes'],token=token,context=context)
                receipt['network_downloads']+=result=='downloaded'; parts.append(path)
            assemble(parts,model/row['name'],row['sha256'],row['size_bytes'])
            receipt['files'][row['name']]={'sha256':digest(model/row['name']),'size_bytes':row['size_bytes']}
            atomic_json(cache/'fetch-verification.json',receipt)
            print('[mnli-assets] verified '+row['name'],flush=True)
    evidence=verify_checkpoint(model)
    from .reference_storage import clean_verified_parts
    for row in spec['files']:clean_verified_parts(model/row['name'],row,cache/'parts')
    receipt.update(status='verified',files=evidence['files'],weights_locally_verified=True)
    atomic_json(cache/'fetch-verification.json',receipt)
    return model


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['fetch','check'])
    parser.add_argument('--lock',type=Path,default=LOCK)
    parser.add_argument('--cache-dir',type=Path)
    args=parser.parse_args(argv)
    if args.action=='fetch': print(fetch(args.lock,cache_dir=args.cache_dir))
    else:
        reject_symlinks(args.lock); spec=validate_lock(json.loads(args.lock.read_text()))
        print(json.dumps({'status':'mnli_asset_lock_valid','files':len(spec['files']),
                          'parts':sum(len(row['parts']) for row in spec['files']),'model_jobs':0}))


if __name__=='__main__': main()
