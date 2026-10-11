"""Hash-verified streaming offload; preserve complete tensors and resumable states."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import ssl
import tempfile
from contextlib import contextmanager
from urllib.parse import quote
from urllib.request import Request, HTTPSHandler, build_opener

from .common import atomic_json, digest, fingerprint
from .assets import CA, uploader, AuthenticatedRedirectGuard
from .reference_storage import owner, regular, durable_json, write_reservation, fsync_parent

SHARD = 512*1024**2


def source_identity():
    from .reference_complete import identity
    return fingerprint(identity())


def receipt_path(task):return Path(task)/'reference_archive.json'


def load(task):
    task=regular(task);path=receipt_path(task)
    regular(path)
    if not path.exists():return None
    r=json.loads(path.read_text())
    if (r.get('schema')!='retacred-reference-archive.v1' or r.get('source_identity')!=source_identity()
            or r.get('catalog_sha256')!=fingerprint(r['entries'])):
        raise ValueError('archive ledger source/catalog drift')
    uploader().safe_relative(r['target_dir'],label='archive namespace')
    if not r['exchange_url'].startswith('https://'):raise ValueError('HTTPS archive required')
    for key,entry in r['entries'].items():
        uploader().safe_relative(key,label='archived artifact')
        if entry['readback_sha256']!=entry['sha256'] or entry['readback_size_bytes']!=entry['size_bytes'] or sum(p['size_bytes'] for p in entry['parts'])!=entry['size_bytes']:
            raise ValueError('incomplete archive byte coverage')
        for part in entry['parts']:uploader().safe_relative(part['path'],label='archived shard')
    return r


def locate(path):
    path=regular(path)
    for task in path.parents:
        if receipt_path(task).is_file():
            receipt=load(task);key=path.relative_to(task).as_posix()
            entry=receipt['entries'].get(key)
            if entry is not None:
                if entry['readback_sha256']!=entry['sha256'] or entry['readback_size_bytes']!=entry['size_bytes']:
                    raise ValueError('archive lacks whole-file readback proof')
                return task,receipt,key,entry
    raise ValueError('missing local artifact has no verified archive entry')


def verify_available(path, sha, size=None):
    path=regular(path)
    if path.is_file():
        if digest(path)!=sha or size is not None and path.stat().st_size!=size:
            raise ValueError('local artifact identity drift')
        return
    _,_,_,entry=locate(path)
    if entry['sha256']!=sha or size is not None and entry['size_bytes']!=size:
        raise ValueError('archive artifact identity drift')


def transport():
    url=os.environ.get('PROJECT_EXCHANGE_URL','').rstrip('/');token=os.environ.get('PROJECT_EXCHANGE_TOKEN','')
    if not url.startswith('https://') or not token:raise ValueError('required HTTPS project exchange credentials missing')
    helper=uploader();helper.parse_base_url(url)
    context=ssl.create_default_context(cafile=CA)
    opener=build_opener(HTTPSHandler(context=context),AuthenticatedRedirectGuard())
    return url,token,helper,context,opener


def readback(url, token, opener, remote, expected, size, whole=None, sink=None):
    request=Request(url+'/files/'+quote(remote,safe='/._-'),headers={'Authorization':'Bearer '+token})
    h=hashlib.sha256();count=0
    with opener.open(request,timeout=300) as response:
        if response.status!=200 or int(response.headers.get('Content-Length',-1))!=size:
            raise ValueError('archive readback response identity mismatch')
        while True:
            block=response.read(1024**2)
            if not block:break
            h.update(block);count+=len(block)
            if whole is not None:whole.update(block)
            if sink is not None:sink.write(block)
    if count!=size or h.hexdigest()!=expected:raise ValueError('archive readback bytes/hash mismatch')
    return count


def archive_files(task, paths, *, remove=False):
    task=regular(task);paths=list(paths)
    if owner(task) is None:return None
    if not paths:return load(task)
    keys=[]
    for p in paths:
        p=regular(p)
        if not p.is_relative_to(task):raise ValueError('archive path not inside task')
        keys.append(p.relative_to(task).as_posix())
    pending={'schema':'retacred-reference-archive-pending.v1','status':'pending','paths':keys,
        'cleanup_after_full_verification':remove,'recovery':'resume the same suite; completed tasks only retry offload/export'}
    durable_json(task/'reference_archive_pending.json',pending)
    try:result=_archive_files(task,paths,remove=remove)
    except Exception as error:
        durable_json(task/'reference_archive_pending.json',{**pending,'status':'failed','error_type':type(error).__name__})
        raise
    durable_json(task/'reference_archive_pending.json',{**pending,'status':'complete'})
    return result


def _archive_files(task, paths, *, remove=False):
    """Crash-safe: upload -> full streamed GET verification -> fsynced ledger -> unlink."""
    task=regular(task);suite=owner(task)
    if suite is None:return None
    paths=list(paths)
    if not paths:return load(task)
    url,token,helper,context,opener=transport()
    receipt=load(task) or {'schema':'retacred-reference-archive.v1','status':'partial',
        'source_identity':source_identity(),'exchange_url':url,
        'target_dir':f'q-attention/retacred-reference-suite/{suite.name}/{task.name}',
        'entries':{},'removed':[]}
    if receipt['exchange_url']!=url:raise ValueError('archive origin changed')
    helper.create_directory(url,token,receipt['target_dir'],context=context)
    for path in paths:
        path=regular(path)
        if not path.is_relative_to(task):raise ValueError('archive path not inside task')
        key=path.relative_to(task).as_posix();helper.safe_relative(key,label='artifact')
        if not path.is_file():
            _,_,_,entry=locate(path)
            if key not in receipt['entries'] or entry!=receipt['entries'][key]:raise ValueError('foreign archived path')
            continue
        sha=digest(path);size=path.stat().st_size
        old=receipt['entries'].get(key)
        if old:
            if old['sha256']!=sha or old['size_bytes']!=size:raise ValueError('immutable archived artifact changed')
        else:
            parts=[];whole=hashlib.sha256();total=0
            # One <=512MiB spool at a time, not another entire tensor/checkpoint.
            if size<SHARD:
                remote=key
                helper._upload_file(url,token,path,receipt['target_dir']+'/'+remote,size,sha,context=context)
                total=readback(url,token,opener,receipt['target_dir']+'/'+remote,sha,size,whole)
                parts=[{'path':remote,'size_bytes':size,'sha256':sha}]
            else:
                spool=suite/'storage'/'spool';spool.mkdir(parents=True,exist_ok=True);regular(spool)
                with path.open('rb') as source:
                    index=0
                    while total<size:
                        length=min(SHARD,size-total)
                        with write_reservation(spool/'part',length):
                            fd,name=tempfile.mkstemp(prefix='part-',dir=spool)
                            temporary=Path(name)
                            try:
                                h=hashlib.sha256();written=0
                                with os.fdopen(fd,'wb') as out:
                                    while written<length:
                                        block=source.read(min(1024**2,length-written))
                                        if not block:raise ValueError('archive source shortened')
                                        out.write(block);h.update(block);written+=len(block)
                                    out.flush();os.fsync(out.fileno())
                                remote=key+f'.part{index:04d}';part_sha=h.hexdigest()
                                helper._upload_file(url,token,temporary,receipt['target_dir']+'/'+remote,length,part_sha,context=context)
                                total+=readback(url,token,opener,receipt['target_dir']+'/'+remote,part_sha,length,whole)
                                parts.append({'path':remote,'size_bytes':length,'sha256':part_sha});index+=1
                            finally:temporary.unlink(missing_ok=True)
            if total!=size or whole.hexdigest()!=sha or digest(path)!=sha:raise ValueError('archive whole-file or concurrent source drift')
            receipt['entries'][key]={'sha256':sha,'size_bytes':size,'parts':parts,
                'readback_sha256':whole.hexdigest(),'readback_size_bytes':total}
            receipt['catalog_sha256']=fingerprint(receipt['entries'])
            durable_json(receipt_path(task),receipt)
        if remove:
            # Intent is persisted first. A crash on either side of unlink is idempotent.
            if key not in receipt['removed']:receipt['removed'].append(key)
            receipt['catalog_sha256']=fingerprint(receipt['entries']);durable_json(receipt_path(task),receipt)
            if digest(path)!=receipt['entries'][key]['sha256']:raise ValueError('cleanup identity changed')
            path.unlink();fsync_parent(path)
    return receipt


def archive_traces(task, traces, *, directory='case_study'):
    import copy
    from .reference_execution import schedule_archive
    frozen=copy.deepcopy(traces)
    return schedule_archive(lambda:_archive_traces(task,frozen,directory=directory))


def _archive_traces(task, traces, *, directory='case_study'):
    if owner(task) is None:return
    paths=[];certificates={}
    for event in traces:
        path=regular(Path(task)/directory/event.get('path',event.get('name')))
        if digest(path)!=event['sha256']:raise ValueError('committed trace manifest drift')
        payload=json.loads(path.read_text())
        if 'tensor_artifact' in payload:
            tensor=path.parent/payload['tensor_artifact'];verify_available(tensor,payload['sha256'],payload['size_bytes'])
            paths.extend([path,tensor])
        else:
            from .ctl_drp_trace import validate as ctl_validate
            required=payload['task_contract']['required_value_representations']
            layers=sum(name.startswith('layer') and name.endswith('.output') for name in required)
            ctl_validate(path,layers=layers)
            certificates[path.relative_to(task).as_posix()]={'kind':'ctl_full_value_lineage',
                'layers':layers,'trace_sha256':digest(path),'source_identity':source_identity()}
            for rep in payload['samples'][0]['representations']:
                artifact=rep['artifact_refs'][0];tensor=path.parent/artifact['path']
                verify_available(tensor,artifact['sha256'],artifact['bytes']);paths.append(tensor)
            paths.append(path)
    paths=list(dict.fromkeys(paths))
    receipt=archive_files(task,paths)
    for key,certificate in certificates.items():receipt['entries'][key]['validation']=certificate
    receipt['catalog_sha256']=fingerprint(receipt['entries']);durable_json(receipt_path(task),receipt)
    # Keep compact producer JSON local; reclaim full values after durable receipt.
    archive_files(task,[p for p in paths if p.suffix=='.pt'],remove=True)


def archive_orphans(task):
    """Uncommitted attempts are not promoted to case evidence; keep a private hash archive."""
    import re
    folder=Path(task)/'orphaned_trace_attempts'
    if owner(task) is None or not folder.exists():return
    paths=[]
    for p in folder.iterdir():
        regular(p)
        if not re.fullmatch(r'\d+-(train|test)-(?:update\d+-event\d+|index\d+-update\d+)\.(json|pt)',p.name):
            raise ValueError('foreign orphan file preserved')
        paths.append(p)
    archive_files(task,paths)
    archive_files(task,[p for p in paths if p.suffix=='.pt'],remove=True)


def archive_completed(task):
    task=regular(task)
    from .reference_complete import validate_task
    summary=validate_task(task)
    if owner(task) is None:raise ValueError('completed archive requires active suite owner')
    # Only the declared task's evidence and checkpoints, never raw datasets/logs.
    archive_traces(task,summary['case_study']['traces'])
    paths=[]
    for directory in ('case_study','auxiliary_case_study'):
        folder=task/directory
        if folder.exists():
            for p in sorted(folder.iterdir()):
                regular(p)
                if not p.is_file() or p.suffix not in ('.pt','.json'):raise ValueError('foreign case artifact')
                paths.append(p)
    for folder in [task/'checkpoints',task/'target/checkpoints']:
        if folder.exists():
            for p in sorted(folder.rglob('*')):
                regular(p)
                if p.is_file():
                    if p.suffix not in ('.pt','.json'):raise ValueError('foreign checkpoint artifact')
                    paths.append(p)
    receipt=archive_files(task,paths)
    # Checkpoints are only reclaimed once all case files and metrics are sealed.
    receipt.update(status='complete',summary_sha256=digest(task/'run_summary.json'),
        validated_trace_identity=fingerprint(summary['case_study']['traces']))
    durable_json(receipt_path(task),receipt)
    reclaim=[p for p in paths if p.suffix=='.pt']
    archive_files(task,reclaim,remove=True)
    print(f'[storage] {task.name}: complete values/checkpoints verified on exchange; local payloads reclaimed',flush=True)
    return load(task)


def ctl_archived_validation(path, layers):
    """Reuse an actual full-value validation only for the identical manifest/bytes/code."""
    path=regular(path);payload=json.loads(path.read_text())
    artifacts={a['path']:a for rep in payload['samples'][0]['representations'] for a in rep['artifact_refs']}
    if all((path.parent/name).is_file() for name in artifacts):return False
    _,receipt,key,entry=locate(path)
    expected={'kind':'ctl_full_value_lineage','layers':layers,'trace_sha256':digest(path),'source_identity':source_identity()}
    if entry['sha256']!=digest(path) or entry.get('validation')!=expected:
        raise ValueError('archived CTL trace has no matching full-value lineage certificate')
    for name,spec in artifacts.items():verify_available(path.parent/name,spec['sha256'],spec['bytes'])
    return True


@contextmanager
def materialize(path):
    """Coordinate readers fetch full originals, not reconstructed or truncated values."""
    path=regular(path)
    if path.is_file():yield path;return
    task,receipt,key,entry=locate(path)
    url,token,_,_,opener=transport()
    if url!=receipt['exchange_url']:raise ValueError('artifact reader origin drift')
    cache=task/'storage_restore';cache.mkdir(exist_ok=True);regular(cache)
    with write_reservation(cache/path.name,entry['size_bytes']):
        fd,name=tempfile.mkstemp(prefix=path.stem+'-',suffix=path.suffix,dir=cache)
        temporary=Path(name)
        try:
            whole=hashlib.sha256();size=0
            with os.fdopen(fd,'wb') as sink:
                for part in entry['parts']:
                    size+=readback(url,token,opener,receipt['target_dir']+'/'+part['path'],part['sha256'],part['size_bytes'],whole,sink)
                sink.flush();os.fsync(sink.fileno())
            if size!=entry['size_bytes'] or whole.hexdigest()!=entry['sha256']:raise ValueError('restored artifact whole hash drift')
            yield temporary
        finally:
            # Linux mmap tensors keep their mapping after unlink; no persistent cache copy.
            temporary.unlink(missing_ok=True)


def aggregate(run,jobs):
    receipts=[archive_completed(j['directory']) for j in jobs]
    files=[];identities={}
    for job,r in zip(jobs,receipts):
        for name,entry in r['entries'].items():
            identities[job['id']+'/'+name]=entry['sha256']
            for part in entry['parts']:
                files.append({'path':job['id']+'/'+part['path'],'size_bytes':part['size_bytes'],
                    'sha256':part['sha256'],'uploaded':True})
        # The catalog is required by readers to reconstruct original large files.
        helper=uploader();url,token,_,_,_=transport()
        helper.upload_declared_files(base_url=url,token=token,target_dir=r['target_dir'],
            files=[(receipt_path(job['directory']),'reference_archive.json')],ca_file=CA)
        files.append({'path':job['id']+'/reference_archive.json','size_bytes':receipt_path(job['directory']).stat().st_size,
            'sha256':digest(receipt_path(job['directory'])),'uploaded':True})
    if len({r['exchange_url'] for r in receipts})!=1:raise ValueError('mixed archive origins')
    result={'schema_version':1,'status':'complete','exchange_url':receipts[0]['exchange_url'],
        'target_dir':f'q-attention/retacred-reference-suite/{run.name}','files':files,'artifact_identity':identities,
        'storage_policy':'full_values_and_latest_best_archived; originals_recoverable_by_catalog'}
    durable_json(run/'exchange_manifest.json',result)
    return result


def main(argv=None):
    import argparse
    from .reference_storage import configure,pause_cli
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore',type=Path,required=True,help='exact original artifact path in a suite task')
    a=p.parse_args(argv)
    task,receipt,key,entry=locate(a.restore)
    if task.parent.name!='tasks':raise ValueError('restore requires a suite task path')
    configure(task.parent.parent)
    def restore():
        if a.restore.exists():verify_available(a.restore,entry['sha256'],entry['size_bytes']);return 0
        a.restore.parent.mkdir(parents=True,exist_ok=True)
        with materialize(a.restore) as temporary:
            os.replace(temporary,a.restore);fsync_parent(a.restore)
        verify_available(a.restore,entry['sha256'],entry['size_bytes'])
        print('[storage] exact original restored: '+str(a.restore),flush=True);return 0
    return pause_cli(restore)


if __name__=='__main__':raise SystemExit(main())
