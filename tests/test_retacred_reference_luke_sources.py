"""Pinned source transport fixtures; no real network or model execution."""
import copy
import hashlib
import io
from pathlib import Path

import pytest

from baselines.retacred import luke_sources as s


def fixture(monkeypatch):
    blobs = {'LICENSE':b'fixture-license', 'src/model.py':b'fixture-source'}
    rows = {name:{'file':name.replace('/','--'), 'size_bytes':len(data),
                  'sha256':hashlib.sha256(data).hexdigest()} for name,data in blobs.items()}
    spec = {'repository':'https://github.com/studio-ousia/luke','revision':'1'*40,
            'cache_dir':'.external/fixture','files':rows}
    monkeypatch.setattr(s,'lock',lambda:{'source':spec})
    def verify(directory):
        result = {name:s.digest(directory/row['file']) for name,row in rows.items()}
        assert all(result[name] == row['sha256'] for name,row in rows.items())
        return result
    monkeypatch.setattr(s,'verify_source',verify)
    monkeypatch.setattr(s.time,'sleep',lambda _:None)
    return blobs,spec


class Response(io.BytesIO):
    status = 200


class Opener:
    def __init__(self, blobs, calls):
        self.blobs,self.calls = blobs,calls

    def open(self,request,timeout):
        self.calls.append((request,timeout))
        name = request.full_url.split('/'+'1'*40+'/')[1]
        return Response(self.blobs[name])


def test_actual_source_identity_has_fifteen_locked_files():
    spec = s.source_spec()
    assert spec['revision'] == '251654548a15b9a8ddc708a53b715bdf3a112102'
    assert len(spec['files']) == 15
    assert spec['files']['LICENSE']['file'] == 'LICENSE'


def test_pinned_fetch_atomic_cache_and_offline_reuse(tmp_path,monkeypatch):
    blobs,spec = fixture(monkeypatch)
    calls = []
    monkeypatch.setattr(s,'build_opener',lambda *_:Opener(blobs,calls))
    monkeypatch.setenv('PROJECT_EXCHANGE_TOKEN','fixture-private-token')
    cache = tmp_path/'cache'
    source,evidence = s.prepare(cache)
    assert source == cache and evidence['network_downloads'] == 2 and evidence['model_imported'] is False
    for name,data in blobs.items():
        assert (source/spec['files'][name]['file']).read_bytes() == data
    for request,timeout in calls:
        assert request.full_url.startswith('https://raw.githubusercontent.com/studio-ousia/luke/'+'1'*40+'/')
        assert not request.has_header('Authorization') and timeout == 30
    assert s.prepare(cache,offline=True)[1]['network_downloads'] == 0 and len(calls) == 2
    assert sorted(p.name for p in cache.iterdir()) == ['LICENSE','src--model.py']


def test_missing_offline_source_no_download_or_directory(tmp_path,monkeypatch):
    fixture(monkeypatch)
    monkeypatch.setattr(s,'build_opener',lambda *_:pytest.fail('offline transport called'))
    with pytest.raises(FileNotFoundError,match='missing'):
        s.prepare(tmp_path/'missing',offline=True)
    assert not (tmp_path/'missing').exists()


@pytest.mark.parametrize('kind',['size','hash','symlink'])
def test_corrupt_existing_source_preserved_before_network(tmp_path,monkeypatch,kind):
    blobs,spec = fixture(monkeypatch)
    cache = tmp_path/'cache'
    cache.mkdir()
    target = cache/'LICENSE'
    if kind == 'symlink':
        original = tmp_path/'original'
        original.write_bytes(blobs['LICENSE'])
        target.symlink_to(original)
    else:
        target.write_bytes(b'x' if kind == 'size' else b'x'*len(blobs['LICENSE']))
    before = target.read_bytes()
    monkeypatch.setattr(s,'build_opener',lambda *_:pytest.fail('corrupt cache fetched'))
    with pytest.raises(ValueError):
        s.prepare(cache)
    assert target.read_bytes() == before


@pytest.mark.parametrize('mode',['short','long','hash'])
def test_bad_response_never_published(tmp_path,monkeypatch,mode):
    blobs,spec = fixture(monkeypatch)
    response = dict(blobs)
    response['LICENSE'] = (b'x' if mode == 'short' else blobs['LICENSE']+b'x' if mode == 'long'
                           else b'x'*len(blobs['LICENSE']))
    monkeypatch.setattr(s,'build_opener',lambda *_:Opener(response,[]))
    with pytest.raises(ValueError,match='size/SHA'):
        s.prepare(tmp_path/'cache')
    assert not list((tmp_path/'cache').iterdir())


@pytest.mark.parametrize('kind',['repository','revision','path','cache_name','duplicate','size','sha'])
def test_invalid_lock_rejected(tmp_path,monkeypatch,kind):
    blobs,spec = fixture(monkeypatch)
    if kind == 'repository':
        spec['repository'] = 'https://other.invalid/repo'
    elif kind == 'revision':
        spec['revision'] = 'main'
    elif kind == 'path':
        spec['files']['../model.py'] = spec['files'].pop('src/model.py')
    elif kind == 'cache_name':
        spec['files']['LICENSE']['file'] = '../LICENSE'
    elif kind == 'duplicate':
        spec['files']['src/model.py']['file'] = 'LICENSE'
    elif kind == 'size':
        spec['files']['LICENSE']['size_bytes'] = 0
    else:
        spec['files']['LICENSE']['sha256'] = 'invalid'
    with pytest.raises(ValueError):
        s.prepare(tmp_path/'cache')
    assert not (tmp_path/'cache').exists()
