"""One-attempt pinned COS chunks -> hash-verified selected tar members.

Long-lived credentials never enter this process. Presigned URLs are private
inputs, never argv or output. No automatic retry or endpoint redirect is allowed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import tarfile
import time
import urllib.parse
import urllib.request
from common import complete,guard,read,safe,save,sha


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


def extract(root,plan):
    verified={}
    for entry in plan['files']:
        destination=safe(root,'assets/'+entry['destination'])
        header=read_range(root,entry['stage'],entry['header_offset'],entry['header_bytes'],plan['chunk_bytes'])
        assert len(header)==512 and hashlib.sha256(header).hexdigest()==entry['header_sha256']
        member=tarfile.TarInfo.frombuf(header,encoding='utf-8',errors='strict')
        assert member.isfile() and member.name==entry['source_name'] and member.size==entry['bytes']
        assert member.mode==0o600 and member.mtime==0
        if destination.exists():
            assert destination.is_file() and destination.stat().st_size==entry['bytes'] and sha(destination)==entry['sha256']
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            pending=destination.with_name(destination.name+'.pending')
            assert not pending.is_symlink()
            h=hashlib.sha256();offset=entry['data_offset'];remaining=entry['bytes']
            with pending.open('wb') as output:
                while remaining:
                    n=min(1024**2,remaining)
                    block=read_range(root,entry['stage'],offset,n,plan['chunk_bytes'])
                    output.write(block);h.update(block);offset+=n;remaining-=n
                output.flush();os.fsync(output.fileno())
            assert pending.stat().st_size==entry['bytes'] and h.hexdigest()==entry['sha256']
            os.replace(pending,destination);destination.chmod(0o444)
        verified[entry['destination']]=dict(bytes=entry['bytes'],sha256=entry['sha256'],tar_header_verified=True)
    return verified


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


def main():
    guard();parser=argparse.ArgumentParser()
    for name in ['plan','urls','root','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    assert sha(args.plan)==args.plan_sha256
    plan=read(args.plan);assert plan['schema']=='qattn.original-score-restore.plan.v1'
    assert plan['maximum_payload_bytes']==sum(x['bytes'] for x in plan['chunks'])
    assert plan['maximum_get_requests']==len(plan['chunks']) and plan['attempts_per_chunk']==1
    assert len({x['id'] for x in plan['chunks']})==len(plan['chunks'])
    assert len({x['destination'] for x in plan['files']})==len(plan['files'])
    assert plan['chunk_bytes']==64*1024**2
    root=args.root.absolute();root.mkdir(parents=True,exist_ok=True)
    assert all(not p.is_symlink() for p in [root,*root.parents])
    lock=safe(root,'restore.lock')
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=time.monotonic()
    try:
        plan['_sha256']=args.plan_sha256
        receipt_path=safe(root,'restore-acceptance.json')
        if receipt_path.exists():
            old=read(receipt_path);assert old['plan_sha256']==args.plan_sha256
            assert old['status'] in ['FILES_VERIFIED_CLEANUP_PENDING','PASS_SELECTED_FILE_RESTORE']
            assert set(old['files'])=={x['destination'] for x in plan['files']}
            for entry in plan['files']:
                assert old['files'][entry['destination']]==dict(bytes=entry['bytes'],sha256=entry['sha256'],tar_header_verified=True)
            for name,row in old['files'].items():assert sha(safe(root,'assets/'+name))==row['sha256']
            for entry in plan['chunks']:
                path=safe(root,f"chunks/{entry['stage']}/{entry['index']}.bin")
                if path.exists():assert sha(path)==entry['sha256'];path.unlink()
            old.update(status='PASS_SELECTED_FILE_RESTORE',reproducible_large_chunks_removed=True)
            save(receipt_path,old)
            complete(args.output,'restore',old)
            print(json.dumps(dict(status='PASS_REUSED_SEALED_RESTORE',new_get_requests=0)));return
        urls=read(args.urls)
        journal=download(root,plan,urls)
        files=extract(root,plan)
        assert len(files)==len(plan['files'])
        receipt=dict(schema='qattn.original-scores.restore.v1',status='FILES_VERIFIED_CLEANUP_PENDING',
            plan_sha256=args.plan_sha256,files=files,cos_get_requests=sum(v['requests'] for v in journal['chunks'].values()),
            payload_bytes=sum(v['bytes'] for v in journal['chunks'].values()),automatic_retries=0,
            long_lived_credentials_on_worker=False,all_tar_headers_verified=True,all_inner_hashes_verified=True,
            reproducible_large_chunks_removed=False,elapsed_seconds=time.monotonic()-started,
            full_archive_restored=False,training=False,model_forwards=0)
        # Seal verified files before deleting reproducible chunks so a crash can
        # finish cleanup without paying for a second GET.
        save(receipt_path,receipt)
        for entry in plan['chunks']:
            path=safe(root,f"chunks/{entry['stage']}/{entry['index']}.bin")
            assert sha(path)==entry['sha256'];path.unlink()
        receipt.update(status='PASS_SELECTED_FILE_RESTORE',reproducible_large_chunks_removed=True)
        save(receipt_path,receipt);complete(args.output,'restore',receipt)
        print(json.dumps({k:receipt[k] for k in ['status','cos_get_requests','payload_bytes','model_forwards']}))
    finally:lock.unlink()


if __name__=='__main__':main()
