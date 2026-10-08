"""Pinned LUKE assets on the authenticated exchange; no model or torch loads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import tempfile
import time
from urllib.parse import quote, urlsplit
from urllib.request import HTTPSHandler, Request, build_opener

from .assets import AuthenticatedRedirectGuard, assemble, reject_symlinks, safe_name
from .common import ROOT, PACKAGE, atomic_json, digest
from .luke import lock as foundation_lock, verify_checkpoint

REVISION = '0729d044dfe301d9ecabc222d60633f92ac450eb'
REPO = 'studio-ousia/luke-large'
NAMESPACE = 'q-attention/luke-reference-assets/' + REVISION
CA = ROOT / 'scripts/collab/certs/qattn-exchange-ca.crt'
SHARD_BYTES = 512 * 1024 * 1024
LOCK_PATH = PACKAGE / 'luke_assets.lock.json'
HEX = re.compile(r'^[0-9a-f]{64}$')


def validate_lock(spec):
    original = foundation_lock()['checkpoint']
    if (spec.get('schema') != 'luke-assets.v1' or spec.get('repo') != REPO or
        spec.get('revision') != REVISION or original['revision'] != REVISION or
        spec.get('exchange_namespace') != NAMESPACE):
        raise ValueError('LUKE asset revision/schema/namespace mismatch')
    rows = spec.get('files', [])
    names = [row.get('name') for row in rows]
    if len(names) != len(set(names)) or set(names) != set(original['files']):
        raise ValueError('LUKE assets must cover exactly the eight pinned files')
    for row in rows:
        name = safe_name(row['name']); source = original['files'][name]
        if row.get('size_bytes') != source['size'] or row.get('git_blob') != source['git_blob']:
            raise ValueError('LUKE original asset identity mismatch')
        if not HEX.fullmatch(row.get('sha256', '')) or source.get('sha256', row['sha256']) != row['sha256']:
            raise ValueError('LUKE original asset SHA mismatch')
        parts = row.get('parts', [])
        if not parts or len(parts) != (row['size_bytes'] + SHARD_BYTES - 1) // SHARD_BYTES:
            raise ValueError('LUKE shard count mismatch')
        for index, part in enumerate(parts):
            expected_size = min(SHARD_BYTES, row['size_bytes'] - index * SHARD_BYTES)
            if (safe_name(part['name']) != f'{name}.part{index:04d}' or
                type(part.get('size_bytes')) is not int or part['size_bytes'] != expected_size or
                not HEX.fullmatch(part.get('sha256', ''))):
                raise ValueError('LUKE shard order/size/SHA mismatch')
    return spec


def verify_original(path, name):
    spec = foundation_lock()['checkpoint']['files'][name]
    reject_symlinks(path)
    if path.stat().st_size != spec['size']:
        raise ValueError('pinned original size mismatch: ' + name)
    if spec.get('sha256'):
        actual = digest(path)
        if actual != spec['sha256']:
            raise ValueError('pinned original SHA mismatch: ' + name)
    else:
        value = hashlib.sha1(b'blob ' + str(spec['size']).encode() + b'\0')
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                value.update(block)
        if value.hexdigest() != spec['git_blob']:
            raise ValueError('pinned original Git blob mismatch: ' + name)
    return digest(path)


def validate_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or
        parsed.query or parsed.fragment):
        raise ValueError('configure credential-free HTTPS PROJECT_EXCHANGE_URL')
    return url.rstrip('/')


def stream_part(url, target, *, sha256, size, token, context):
    """One authenticated attempt. Atomic shard cache, no partial append/retry."""
    reject_symlinks(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == size and digest(target) == sha256:
            return 'cache_hit'
        raise ValueError('existing LUKE shard is corrupt; preserved for diagnosis')
    fd, temporary = tempfile.mkstemp(prefix=target.name + '.', dir=target.parent)
    try:
        request = Request(url, headers={'Authorization': 'Bearer ' + token})
        opener = build_opener(HTTPSHandler(context=context), AuthenticatedRedirectGuard())
        total, value = 0, hashlib.sha256()
        with os.fdopen(fd, 'wb') as output:
            with opener.open(request, timeout=120) as response:
                if response.status != 200:
                    raise ValueError('exchange shard response must be HTTP200')
                while True:
                    block = response.read(min(1024 * 1024, size - total + 1))
                    if not block: break
                    total += len(block)
                    if total > size: raise ValueError('exchange shard exceeds pinned size')
                    value.update(block); output.write(block)
            output.flush(); os.fsync(output.fileno())
        if total != size or value.hexdigest() != sha256:
            raise ValueError('exchange shard size/SHA mismatch')
        os.replace(temporary, target)
        return 'downloaded'
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch(lock_path=LOCK_PATH, *, cache_dir=None):
    spec = validate_lock(json.loads(Path(lock_path).read_text(encoding='utf-8')))
    cache = Path(cache_dir) if cache_dir else ROOT / '.external/assets/luke' / REVISION
    reject_symlinks(cache)
    model = cache / 'model'; reject_symlinks(model); model.mkdir(parents=True, exist_ok=True)
    needed = []
    for row in spec['files']:
        target = model / row['name']
        if target.exists():
            # A full cache hit is still checked against original immutable identity.
            if verify_original(target, row['name']) != row['sha256']:
                raise ValueError('existing LUKE model cache SHA mismatch')
        else:
            needed.append(row)
    receipt = {'schema':'luke-asset-fetch.v1','revision':REVISION,'status':'pending',
               'files':{},'network_downloads':0,'no_model_load':True}
    atomic_json(cache/'fetch-verification.json', receipt)
    if needed:
        url = validate_url(os.environ.get('PROJECT_EXCHANGE_URL', ''))
        token = os.environ.get('PROJECT_EXCHANGE_TOKEN', '')
        if not token: raise ValueError('PROJECT_EXCHANGE_TOKEN is required for missing assets')
        context = ssl.create_default_context(cafile=CA)
        for row in needed:
            parts = []
            for part in row['parts']:
                path = cache / 'parts' / part['name']
                time.sleep(1)
                result = stream_part(url+'/files/'+NAMESPACE+'/'+quote(part['name']), path,
                                     sha256=part['sha256'], size=part['size_bytes'], token=token, context=context)
                receipt['network_downloads'] += result == 'downloaded'
                parts.append(path)
            assemble(parts, model/row['name'], row['sha256'], row['size_bytes'])
            if verify_original(model/row['name'], row['name']) != row['sha256']:
                raise ValueError('assembled asset differs from pinned original')
            receipt['files'][row['name']] = {'sha256':row['sha256'],'size_bytes':row['size_bytes']}
            atomic_json(cache/'fetch-verification.json',receipt)
            print('[luke-assets] verified '+row['name'],flush=True)
    verify_checkpoint(model)
    receipt.update(status='verified',files={row['name']:{'sha256':row['sha256'],'size_bytes':row['size_bytes']} for row in spec['files']})
    atomic_json(cache/'fetch-verification.json',receipt)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch', 'check'])
    parser.add_argument('--lock', type=Path, default=LOCK_PATH)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'fetch':
        print(fetch(args.lock,cache_dir=args.cache_dir))
    else:
        spec = validate_lock(json.loads(args.lock.read_text(encoding='utf-8')))
        print(json.dumps({'status':'lock_valid','files':len(spec['files']),
                          'parts':sum(len(row['parts']) for row in spec['files']),
                          'revision':REVISION,'model_loaded':False}))


if __name__ == '__main__':
    main()
