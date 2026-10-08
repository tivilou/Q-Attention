"""Prepare only the hash-locked official LUKE source files, never import them."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import tempfile
import time
from urllib.parse import quote
from urllib.request import HTTPSHandler, Request, build_opener

from .assets import AuthenticatedRedirectGuard, reject_symlinks, safe_name
from .common import ROOT, digest
from .luke import lock, verify_source


def source_spec():
    spec = lock()['source']
    if (spec['repository'] != 'https://github.com/studio-ousia/luke'
            or not re.fullmatch(r'[0-9a-f]{40}', spec['revision']) or not spec['files']):
        raise ValueError('LUKE source repository/revision invalid')
    names = []
    for name, row in spec['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
            raise ValueError('unsafe upstream source path')
        names.append(safe_name(row['file']))
        if (type(row['size_bytes']) is not int or not 0 < row['size_bytes'] <= 64*1024**2
                or not re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
            raise ValueError('LUKE source size/hash invalid')
    if len(names) != len(set(names)):
        raise ValueError('duplicate source cache filename')
    return spec


def prepare(directory=None, *, offline=False):
    spec = source_spec()
    directory = Path(directory) if directory else ROOT / spec['cache_dir']
    directory = directory.absolute()
    reject_symlinks(directory)
    missing = []
    for name, row in spec['files'].items():
        path = directory / row['file']
        reject_symlinks(path)
        if path.exists():
            if (not path.is_file() or path.stat().st_size != row['size_bytes']
                    or digest(path) != row['sha256']):
                raise ValueError('corrupt LUKE source cache preserved: ' + row['file'])
        else:
            missing.append((name, row))
    if missing and offline:
        raise FileNotFoundError('offline LUKE source cache missing ' + str(len(missing)) + ' pinned files')
    if missing:
        directory.mkdir(parents=True, exist_ok=True)
        opener = build_opener(HTTPSHandler(context=ssl.create_default_context()), AuthenticatedRedirectGuard())
        for index, (name, row) in enumerate(missing):
            if index:
                time.sleep(1)
            url = 'https://raw.githubusercontent.com/studio-ousia/luke/' + spec['revision'] + '/' + quote(name, safe='/')
            request = Request(url, headers={'User-Agent':'q-attention-source-preflight/1', 'Accept-Encoding':'identity'})
            with opener.open(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError('pinned source response must be HTTP200')
                data = response.read(row['size_bytes'] + 1)
            if len(data) != row['size_bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('pinned source response size/SHA mismatch: ' + name)
            target = directory / row['file']
            reject_symlinks(target)
            fd, temporary = tempfile.mkstemp(prefix=target.name + '.', dir=directory)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if target.stat().st_size != len(data) or digest(target) != row['sha256']:
                        raise ValueError('concurrent source cache changed; preserved')
            finally:
                Path(temporary).unlink(missing_ok=True)
    return directory, {'revision':spec['revision'], 'files':verify_source(directory),
                       'network_downloads':len(missing), 'model_imported':False}
