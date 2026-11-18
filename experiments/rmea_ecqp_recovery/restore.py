"""Pinned prefix recovery; a prefix never certifies the complete inner H hash."""
import hashlib
import io
import json
import os
import re
import ssl
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from experiments.rmea_original_scores.common import safe, save, read, sha


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def read_range(root,stage,offset,length,chunk_size):
    result=bytearray()
    while length:
        index,within=divmod(offset,chunk_size)
        path=safe(root,f'chunks/{stage}/{index}.bin')
        with path.open('rb') as f:
            f.seek(within)
            block=f.read(min(length,chunk_size-within))
        assert block,'Missing sealed chunk range'
        result.extend(block);offset+=len(block);length-=len(block)
    return bytes(result)


def download(root,plan,urls):
    journal_path=safe(root,'network-state.json')
    journal=read(journal_path) if journal_path.exists() else dict(plan_sha256=plan['_sha256'],chunks={})
    assert journal['plan_sha256']==plan['_sha256']
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    for entry in plan['chunks']:
        identifier=entry['id'];stage=entry['stage'];index=entry['index']
        assert re.fullmatch(r'seed(29|53|71|97|113)',stage)
        path=safe(root,f'chunks/{stage}/{index}.bin');pending=path.with_suffix('.pending')
        assert not pending.is_symlink()
        previous=journal['chunks'].get(identifier)
        if path.exists():
            assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
            assert previous and previous.get('headers_verified') and previous['requests']==1
            previous.update(status='COMPLETE',bytes=entry['bytes']);save(journal_path,journal);continue
        if pending.exists() and pending.stat().st_size==entry['bytes'] and sha(pending)==entry['sha256']:
            assert previous and previous.get('headers_verified') and previous['requests']==1
            os.replace(pending,path);previous.update(status='COMPLETE',bytes=entry['bytes']);save(journal_path,journal);continue
        assert previous is None,'Existing incomplete GET: preserve state; no automatic retry'
        url=urls[identifier];parts=urllib.parse.urlsplit(url)
        assert parts.scheme=='https' and parts.hostname==entry['host'] and parts.port in (None,443)
        assert urllib.parse.unquote(parts.path)=='/'+entry['key']
        assert not parts.username and not parts.password and not parts.fragment
        assert urllib.parse.parse_qs(parts.query).get('versionId')==[entry['version']]
        path.parent.mkdir(parents=True,exist_ok=True)
        attempt=dict(status='STARTED',requests=1,bytes=0,headers_verified=False)
        journal['chunks'][identifier]=attempt;save(journal_path,journal)
        assert sum(v['requests'] for v in journal['chunks'].values())<=plan['maximum_get_requests']
        try:
            request=urllib.request.Request(url,method='GET',headers={'Accept-Encoding':'identity'})
            with opener.open(request,timeout=90) as response:
                assert response.status==200 and int(response.headers['Content-Length'])==entry['bytes']
                assert response.headers.get('x-cos-version-id')==entry['version']
                assert response.headers.get('Content-Encoding') in (None,'identity')
                attempt['headers_verified']=True;save(journal_path,journal)
                h=hashlib.sha256();remaining=entry['bytes']
                with pending.open('xb') as output:
                    while remaining:
                        block=response.read(min(1024**2,remaining))
                        assert block and len(block)<=remaining
                        output.write(block);h.update(block);remaining-=len(block);attempt['bytes']+=len(block)
                    output.flush();os.fsync(output.fileno())
                assert h.hexdigest()==entry['sha256'] and pending.stat().st_size==entry['bytes']
            os.replace(pending,path);attempt['status']='COMPLETE';save(journal_path,journal)
            print(json.dumps(dict(stage='restore_chunk',completed=sum(v['status']=='COMPLETE' for v in journal['chunks'].values()),
                planned=len(plan['chunks']))),flush=True)
        except BaseException as exc:
            attempt['status']='FAILED_PRESERVE';attempt['error_class']=type(exc).__name__;save(journal_path,journal)
            raise RuntimeError('Pinned chunk GET failed; state preserved; no automatic retry') from None
    assert sum(x['bytes'] for x in journal['chunks'].values())==plan['maximum_payload_bytes']
    return journal



def validate_plan(plan):
    assert plan['schema']=='qattn.ecqp.train-prefix-recovery.v1'
    assert plan['source_stage']=='seed29' and plan['attempts_per_chunk']==1
    assert plan['chunk_bytes']==64*1024**2
    assert plan['prefix_examples']==4096 and plan['prefix_tokens']==151571 and plan['hidden_size']==1024
    assert plan['expected_dtype']=='<f2' and plan['fortran_order'] is False
    assert plan['maximum_get_requests']==len(plan['chunks'])==6
    assert plan['maximum_payload_bytes']==sum(c['bytes'] for c in plan['chunks'])==402653184
    assert [c['index'] for c in plan['chunks']]==list(range(14,20))
    assert len({c['id'] for c in plan['chunks']})==6
    for c in plan['chunks']:
        assert c['stage']=='seed29' and c['id']==f"seed29:{c['index']}" and c['bytes']==plan['chunk_bytes']
        assert c['version'] and c['version']!='null' and re.fullmatch('[0-9a-f]{64}',c['sha256'])
    assert plan['npy_header_maximum_bytes']==4096 and plan['cloud_writes']==0 and plan['training'] is False


def validate_chunks(root,plan):
    for c in plan['chunks']:
        path=safe(root,f"chunks/{c['stage']}/{c['index']}.bin")
        assert path.is_file() and path.stat().st_size==c['bytes'] and sha(path)==c['sha256']


def parse_header(root,plan):
    e=plan['original_H'];stage=plan['source_stage']
    raw=read_range(root,stage,e['header_offset'],512,plan['chunk_bytes'])
    assert hashlib.sha256(raw).hexdigest()==e['header_sha256']
    member=tarfile.TarInfo.frombuf(raw,encoding='utf8',errors='strict')
    assert member.isfile() and member.name=='cache/train-h.npy' and member.size==e['bytes']
    assert member.mode==0o600 and member.mtime==0 and e['data_offset']==e['header_offset']+512
    raw=read_range(root,stage,e['data_offset'],plan['npy_header_maximum_bytes'],plan['chunk_bytes'])
    f=io.BytesIO(raw);version=np.lib.format.read_magic(f)
    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
    elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
    else:raise ValueError('Only original NPY v1/v2 accepted')
    assert dtype==np.dtype('<f2') and dtype.hasobject is False and not fortran
    assert shape==(plan['expected_shape_first_dimension'],1024)
    assert 0<f.tell()<=plan['npy_header_maximum_bytes']
    assert f.tell()+int(np.prod(shape))*dtype.itemsize==e['bytes']
    assert 0<plan['prefix_tokens']<=shape[0]
    return dict(npy_header_bytes=f.tell(),npy_version=list(version),shape=list(shape),dtype=dtype.str,
        value_offset=e['data_offset']+f.tell(),tar_header_verified=True)


def extract(root,plan,sidecars):
    """Require original full sidecar hashes; publish newly hashed prefix files."""
    metadata={}
    for name,expected in plan['trusted_sidecars'].items():
        path=safe(sidecars,name)
        assert path.is_file() and path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']
    offsets=np.load(safe(sidecars,'train-offsets.npy'),mmap_mode='r',allow_pickle=False)
    roles=np.load(safe(sidecars,'train-roles.npy'),mmap_mode='r',allow_pickle=False)
    targets=np.load(safe(sidecars,'train-targets.npy'),mmap_mode='r',allow_pickle=False)
    count=plan['prefix_examples'];tokens=plan['prefix_tokens']
    assert offsets.dtype==np.dtype('<i8') and offsets.shape==(58466,) and offsets[0]==0
    assert np.all(offsets[1:]>=offsets[:-1]) and offsets[count]==tokens
    assert roles.dtype==np.dtype('<f4') and roles.shape==(int(offsets[-1]),4)
    assert targets.dtype==np.dtype('<i8') and targets.shape==(58465,)
    assert int(offsets[-1])==plan['expected_shape_first_dimension']
    subset=read(safe(sidecars,'metadata.json'))['train'][:count]
    assert len(subset)==count and len({r['id_hash'] for r in subset})==count
    assert all(int(offsets[i+1]-offsets[i])==r['words'] for i,r in enumerate(subset))
    prefix_roles=np.asarray(roles[:tokens]);prefix_offsets=np.asarray(offsets[:count+1])
    assert hashlib.sha256(prefix_roles.tobytes(order='C')).hexdigest()==plan['prefix_role_values_sha256']
    assert hashlib.sha256(prefix_offsets.tobytes(order='C')).hexdigest()==plan['prefix_offsets_values_sha256']
    header=parse_header(root,plan)
    folder=safe(root,'assets');folder.mkdir(exist_ok=True)
    h=safe(folder,'train-h.npy');pending=h.with_name(h.name+'.pending')
    assert not h.exists() and not pending.exists()
    values_hash=hashlib.sha256();remaining=tokens*1024*2;offset=header['value_offset']
    with pending.open('xb') as f:
        np.lib.format.write_array_header_1_0(f,dict(descr='<f2',fortran_order=False,shape=(tokens,1024)))
        while remaining:
            size=min(1024**2,remaining)
            raw=read_range(root,plan['source_stage'],offset,size,plan['chunk_bytes'])
            f.write(raw);values_hash.update(raw);remaining-=size;offset+=size
        f.flush();os.fsync(f.fileno())
    os.replace(pending,h)
    arrays={'train-offsets.npy':prefix_offsets,'train-roles.npy':prefix_roles,'train-targets.npy':np.asarray(targets[:count])}
    for name,array in arrays.items():
        target=safe(folder,name);assert not target.exists()
        with target.open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
    save(safe(folder,'metadata.json'),dict(train=subset))
    for path in folder.iterdir():
        assert path.is_file() and not path.is_symlink()
        metadata[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path));path.chmod(0o444)
    assert set(metadata)=={'train-h.npy','train-offsets.npy','train-roles.npy','train-targets.npy','metadata.json'}
    restored=np.load(h,mmap_mode='r',allow_pickle=False)
    assert restored.shape==(tokens,1024) and restored.dtype==np.dtype('<f2')
    return dict(files=metadata,header=header,prefix_values_sha256=values_hash.hexdigest(),
        full_original_H_sha256=plan['original_H']['sha256'],full_original_H_hash_recomputed=False,
        sidecar_original_hashes_verified=True,prefix_examples=count,prefix_tokens=tokens)


def run(plan_path,plan_sha256,urls_path,sidecars,root):
    assert sha(plan_path)==plan_sha256
    plan=read(plan_path);validate_plan(plan);plan['_sha256']=plan_sha256
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=True)
    assert all(not p.is_symlink() for p in [root,*root.parents])
    sidecars=Path(sidecars).absolute();assert all(not p.is_symlink() for p in [sidecars,*sidecars.parents])
    lock=safe(root,'recovery.lock')
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        accepted=safe(root,'prefix-acceptance.json')
        if accepted.exists():
            result=read(accepted)
            assert result['plan_sha256']==plan_sha256 and result['status'] in ['PREFIX_VERIFIED_CLEANUP_PENDING','PASS_TRAIN_PREFIX_RECOVERY']
            assert set(result['files'])=={'train-h.npy','train-offsets.npy','train-roles.npy','train-targets.npy','metadata.json'}
            for name,entry in result['files'].items():
                path=safe(root,'assets/'+name)
                assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
            for c in plan['chunks']:
                path=safe(root,f"chunks/{c['stage']}/{c['index']}.bin")
                if path.exists():assert path.stat().st_size==c['bytes'] and sha(path)==c['sha256'];path.unlink()
            result.update(status='PASS_TRAIN_PREFIX_RECOVERY',reproducible_chunks_removed=True,new_GET_requests=0)
            save(accepted,result);return result
        for name,expected in plan['trusted_sidecars'].items():
            path=safe(sidecars,name);assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']
        assert not safe(root,'assets').exists(),'Preserve unsealed extraction; inspect, do not overwrite/restart'
        journal=download(root,plan,read(urls_path));validate_chunks(root,plan)
        result=extract(root,plan,sidecars)
        result.update(schema='qattn.ecqp.prefix-receipt.v1',status='PREFIX_VERIFIED_CLEANUP_PENDING',plan_sha256=plan_sha256,
            COS_GET_requests=sum(c['requests'] for c in journal['chunks'].values()),
            payload_bytes=sum(c['bytes'] for c in journal['chunks'].values()),new_GET_requests=6,
            automatic_retries=0,long_lived_credentials_on_worker=False,reproducible_chunks_removed=False,
            model_forwards=0,task_training=False,L5='NOT_ACCEPTED')
        save(accepted,result)
        for c in plan['chunks']:
            path=safe(root,f"chunks/{c['stage']}/{c['index']}.bin");assert sha(path)==c['sha256'];path.unlink()
        result.update(status='PASS_TRAIN_PREFIX_RECOVERY',reproducible_chunks_removed=True)
        save(accepted,result);return result
    finally:lock.unlink()
