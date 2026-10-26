"""Stream the explicit closed artifact inventory into bounded verified parts."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tarfile
from urllib.parse import quote
from urllib.request import Request,urlopen
from . import rmea_protocol as p

PART_BYTES=64*1024**2

def helper():
    spec=importlib.util.spec_from_file_location('rmea_exchange_transport',p.ROOT/'scripts/collab/lib/exchange_upload.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def preflight():
    # Reuse the already deployed HTTPS/namespace health check, without writes.
    spec=importlib.util.spec_from_file_location('rmea_exchange_check',p.ROOT/'scripts/check_qattn_exchange.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.check()

def upload(run,closed,*,transport=None):
    run=Path(run);closed=Path(closed);p.audit(closed)
    module=transport or helper();base=os.environ.get('PROJECT_EXCHANGE_URL','').rstrip('/');token=os.environ.get('PROJECT_EXCHANGE_TOKEN','')
    if module.parse_base_url(base)[0]!='https' or not token:raise ValueError('Configured authenticated HTTPS exchange required')
    context=module.ssl_context_for('https',module.DEFAULT_CA_FILE)
    fingerprint=p.digest(closed/'RUN_COMPLETE');target=module.safe_relative('q-attention/q_rmea/'+run.name+'/'+closed.name,label='target')
    module.create_directory(base,token,target,context=context)
    state=run/'export-state'/closed.name;state.mkdir(parents=True,exist_ok=True)
    receipt_path=state/'exchange.json';previous=p.read(receipt_path) if receipt_path.exists() else None
    if previous and (previous['closed_sha256']!=fingerprint or previous['target_dir']!=target):raise ValueError('Exchange identity differs')
    receipt=dict(schema='q-rmea.exchange.v1',status='pending',closed_sha256=fingerprint,target_dir=target,parts=[])
    previous_parts={r['name']:r for r in previous['parts']} if previous else {}
    p.atomic_json(receipt_path,receipt)
    archive_hash=hashlib.sha256();archive_bytes=0
    class Sink:
        def __init__(self):self.file=(state/'part.pending').open('wb');self.size=0;self.index=0
        def write(self,data):
            nonlocal archive_bytes
            archive_hash.update(data);archive_bytes+=len(data);remaining=memoryview(data)
            while remaining:
                size=min(len(remaining),PART_BYTES-self.size);self.file.write(remaining[:size]);remaining=remaining[size:];self.size+=size
                if self.size==PART_BYTES:self.flush_part()
            return len(data)
        def flush_part(self):
            if not self.size:return
            self.file.flush();os.fsync(self.file.fileno());self.file.close()
            path=state/'part.pending';sha=p.digest(path);name=f'cases.part{self.index:05d}'
            expected=dict(name=name,sha256=sha,size_bytes=self.size,verified=True)
            if name in previous_parts:
                if previous_parts[name]!=expected:raise ValueError('Cached exchange part differs')
                request=Request(base+'/upload/'+quote(target+'/'+name,safe='/._-'),headers=module._auth_headers(token),method='HEAD')
                with urlopen(request,context=context,timeout=30) as response:
                    if int(response.headers['Content-Length'])!=self.size or response.headers.get('ETag')!='"sha256:'+sha+'"':raise ValueError('Remote part absent/changed')
            else:module._upload_file(base,token,path,target+'/'+name,self.size,sha,context=context)
            receipt['parts'].append(expected);p.atomic_json(receipt_path,receipt)
            path.unlink();self.size=0;self.index+=1;self.file=path.open('wb')
        def close(self):
            try:self.flush_part()
            finally:
                self.file.close()
                if (state/'part.pending').exists():(state/'part.pending').unlink()
    sink=Sink()
    names=sorted(set(p.read(closed/'RUN_COMPLETE')['artifacts'])|{'RUN_COMPLETE'})
    try:
        with tarfile.open(fileobj=sink,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for name in names:
                module.safe_relative(name,label='closed artifact');path=closed/name
                info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o600;info.mtime=0
                with path.open('rb') as handle:archive.addfile(info,handle)
        sink.close()
        if previous and previous.get('archive_sha256',archive_hash.hexdigest())!=archive_hash.hexdigest():raise ValueError('Archive differs on retry')
        if len(receipt['parts'])<len(previous_parts):raise ValueError('Cached extra parts')
        p.audit(closed)
        receipt.update(status='uploaded',archive_sha256=archive_hash.hexdigest(),archive_bytes=archive_bytes)
        p.atomic_json(receipt_path,receipt);return receipt
    finally:
        if not sink.file.closed:sink.file.close()
