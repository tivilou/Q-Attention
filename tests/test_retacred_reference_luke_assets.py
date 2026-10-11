"""Asset integrity/transport gates, not model-runtime certification."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import luke_assets as client
from baselines.retacred import luke

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.external/assets/luke'/client.REVISION


def actual_lock():
    return json.loads(client.LOCK_PATH.read_text())


def test_real_lock_and_immutable_original_identities():
    spec=client.validate_lock(actual_lock())
    assert len(spec['files'])==8
    assert sum(len(row['parts']) for row in spec['files'])==12
    weight=next(row for row in spec['files'] if row['name']=='pytorch_model.bin')
    assert weight['size_bytes']==2246566835 and len(weight['parts'])==5
    assert weight['sha256']=='9ea73dbb7e0625e88c646f26230902c3df000e71320139e99b0703e1654f7bcc'
    assert all(part['size_bytes']<=512*1024**2 for row in spec['files'] for part in row['parts'])
    assert 'exchange_url' not in spec
    assert all(value not in json.dumps(spec) for value in ('PROJECT_EXCHANGE_TOKEN', 'Authorization', 'Bearer '))


def test_real_private_roundtrip_proof_required_not_skipped():
    progress=json.loads((CACHE/'mirror-progress.json').read_text())
    receipt=json.loads((CACHE/'roundtrip/fetch-verification.json').read_text())
    assert progress['status']=='roundtrip_verified' and progress['published_parts']==12
    assert receipt['status']=='verified' and receipt['network_downloads']==12 and receipt['no_model_load']
    assert len(luke.verify_checkpoint(CACHE/'roundtrip/model'))==8


@pytest.mark.parametrize('field,value',[('repo','other'),('revision','0'*40),('schema','other'),('exchange_namespace','another/project/namespace')])
def test_lock_top_identity_negatives(field,value):
    spec=actual_lock();spec[field]=value
    with pytest.raises(ValueError):client.validate_lock(spec)


@pytest.mark.parametrize('mode',['missing','duplicate','size','git_blob','weight_sha','part_order','part_size','part_sha','part_name','part_missing'])
def test_lock_file_and_shard_negatives(mode):
    spec=actual_lock();row=next(r for r in spec['files'] if r['name']=='pytorch_model.bin')
    if mode=='missing':spec['files'].pop()
    elif mode=='duplicate':spec['files'].append(copy.deepcopy(spec['files'][0]))
    elif mode=='size':row['size_bytes']+=1
    elif mode=='git_blob':row['git_blob']='0'*40
    elif mode=='weight_sha':row['sha256']='0'*64
    elif mode=='part_order':row['parts'].reverse()
    elif mode=='part_size':row['parts'][0]['size_bytes']+=1
    elif mode=='part_sha':row['parts'][0]['sha256']='x'*64
    elif mode=='part_name':row['parts'][0]['name']='../bad'
    elif mode=='part_missing':row['parts'].pop()
    with pytest.raises(ValueError):client.validate_lock(spec)


@pytest.mark.parametrize('url',['http://exchange.example','https://u:p@exchange.example','https://exchange.example?token=secret','https://exchange.example#fragment',''])
def test_url_security_contract(url):
    with pytest.raises(ValueError):client.validate_url(url)


class Response(io.BytesIO):
    status=200


class Opener:
    def __init__(self,data,calls):self.data,self.calls=data,calls
    def open(self,request,timeout):
        self.calls.append((request.full_url,timeout))
        return Response(self.data)


def test_stream_shard_atomic_cache_and_failure(tmp_path,monkeypatch):
    data=b'bounded-fixture';calls=[]
    monkeypatch.setattr(client,'build_opener',lambda *_:Opener(data,calls))
    path=tmp_path/'shard';digest=hashlib.sha256(data).hexdigest()
    assert client.stream_part('https://fixture.example/file',path,sha256=digest,size=len(data),token='fixture',context=None)=='downloaded'
    assert path.read_bytes()==data
    assert client.stream_part('https://fixture.example/file',path,sha256=digest,size=len(data),token='fixture',context=None)=='cache_hit'
    assert len(calls)==1
    with pytest.raises(ValueError,match='size/SHA'):
        client.stream_part('https://fixture.example/file',tmp_path/'bad',sha256='0'*64,size=len(data),token='fixture',context=None)
    assert not (tmp_path/'bad').exists() and not list(tmp_path.glob('bad.*'))
    path.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='preserved'):
        client.stream_part('https://fixture.example/file',path,sha256=digest,size=len(data),token='fixture',context=None)
    assert path.read_bytes()==b'wrong'


def test_oversize_response_never_published(tmp_path,monkeypatch):
    monkeypatch.setattr(client,'build_opener',lambda *_:Opener(b'x'*40,[]))
    with pytest.raises(ValueError,match='exceeds'):
        client.stream_part('https://fixture.example/file',tmp_path/'bad',sha256='0'*64,size=8,token='fixture',context=None)
    assert not list(tmp_path.iterdir())


def test_token_redirect_not_forwarded_to_other_origin():
    from urllib.request import Request
    guard=client.AuthenticatedRedirectGuard()
    request=Request('https://fixture.example/file',headers={'Authorization':'Bearer fixture'})
    with pytest.raises(ValueError,match='outside exchange origin'):
        guard.redirect_request(request,None,302,'redirect',{},'https://other.example/file')


def fixture_lock(monkeypatch):
    blobs={'config.json':b'config-data','pytorch_model.bin':b'weight-data'}
    original={};rows=[]
    for name,data in blobs.items():
        sha=hashlib.sha256(data).hexdigest()
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        original[name]={'size':len(data),'sha256':sha,'git_blob':blob}
        rows.append({'name':name,'size_bytes':len(data),'sha256':sha,'git_blob':blob,
                     'parts':[{'name':name+'.part0000','sha256':sha,'size_bytes':len(data)}]})
    value={'checkpoint':{'revision':client.REVISION,'files':original}}
    monkeypatch.setattr(client,'foundation_lock',lambda:value)
    monkeypatch.setattr(client,'verify_checkpoint',lambda p:{name:client.verify_original(p/name,name) for name in original})
    spec={'schema':'luke-assets.v1','repo':client.REPO,'revision':client.REVISION,'exchange_namespace':client.NAMESPACE,'files':rows}
    return blobs,spec


def test_fetch_reassembly_integrity_and_offline_cache_hit(tmp_path,monkeypatch):
    blobs,spec=fixture_lock(monkeypatch)
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(spec));cache=tmp_path/'cache'
    calls=[]
    def fake_download(url,path,**kwargs):
        calls.append(url);path.parent.mkdir(parents=True,exist_ok=True)
        name=path.name.removesuffix('.part0000');path.write_bytes(blobs[name]);return 'downloaded'
    monkeypatch.setattr(client,'stream_part',fake_download)
    monkeypatch.setattr(client.time,'sleep',lambda _:None)
    monkeypatch.setenv('PROJECT_EXCHANGE_URL','https://fixture.example')
    monkeypatch.setenv('PROJECT_EXCHANGE_TOKEN','fixture')
    model=client.fetch(lock,cache_dir=cache)
    assert all((model/name).read_bytes()==data for name,data in blobs.items())
    assert len(calls)==2
    monkeypatch.delenv('PROJECT_EXCHANGE_URL');monkeypatch.delenv('PROJECT_EXCHANGE_TOKEN')
    assert client.fetch(lock,cache_dir=cache)==model and len(calls)==2
    receipt=json.loads((cache/'fetch-verification.json').read_text())
    assert receipt['status']=='verified' and receipt['network_downloads']==0
    (model/'config.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError):client.fetch(lock,cache_dir=cache)
    assert (model/'config.json').read_bytes()==b'corrupt'


def test_source_blob_verification_non_lfs(tmp_path,monkeypatch):
    data=b'git-blob-fixture';blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    monkeypatch.setattr(client,'foundation_lock',lambda:{'checkpoint':{'files':{'vocab.json':{'size':len(data),'git_blob':blob}}}})
    path=tmp_path/'vocab.json';path.write_bytes(data)
    assert client.verify_original(path,'vocab.json')==hashlib.sha256(data).hexdigest()
    path.write_bytes(b'x'*len(data))
    with pytest.raises(ValueError,match='Git blob'):client.verify_original(path,'vocab.json')


def test_symlink_cache_rejected_before_transfer(tmp_path,monkeypatch):
    _,spec=fixture_lock(monkeypatch)
    path=tmp_path/'lock.json';path.write_text(json.dumps(spec))
    real=tmp_path/'real';real.mkdir();link=tmp_path/'link';link.symlink_to(real)
    with pytest.raises(ValueError,match='symlink'):client.fetch(path,cache_dir=link)


def test_actual_lock_cli_imports_no_torch_or_numpy():
    code=('import runpy,sys; sys.argv=["gate","check"]; runpy.run_module("baselines.retacred.luke_assets",run_name="__main__"); '
          'assert "torch" not in sys.modules and "numpy" not in sys.modules')
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['parts']==12


def test_shell_entrypoint_does_not_hardcode_server_or_python():
    path=ROOT/'scripts/fetch_retacred_luke_assets.sh'
    text=path.read_text()
    assert '/home/Q-Attention' not in text and '/root/projects' not in text and 'miniconda' not in text
    assert 'PYTHON_BIN' in text and 'python3' in text and 'baselines.retacred.luke_assets fetch' in text
    assert subprocess.run(['bash','-n',str(path)]).returncode==0
