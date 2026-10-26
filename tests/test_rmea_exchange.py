"""Deterministic closed-file serialization and verified receipt retry."""
import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
from baselines.retacred import rmea_protocol as p
from baselines.retacred import rmea_exchange as e

class Exchange(unittest.TestCase):
    def test_stream_parts_match_receipts_and_retry_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'20261026T000000Z_seed13';closed=run/'discovery';closed.mkdir(parents=True)
            (closed/'tensor.npy').write_bytes(b'0123456789'*300);p.seal(closed);remote={};uploads=[]
            def upload(base,token,path,name,size,digest,**kw):
                raw=path.read_bytes();self.assertEqual(len(raw),size);self.assertEqual(hashlib.sha256(raw).hexdigest(),digest)
                remote[name]=raw;uploads.append(name);return {'sha256':digest,'size_bytes':size}
            module=types.SimpleNamespace(parse_base_url=lambda b:('https',),ssl_context_for=lambda *x:None,DEFAULT_CA_FILE=None,
                safe_relative=lambda s,**kw:s,create_directory=lambda *a,**k:None,_upload_file=upload,_auth_headers=lambda t:{})
            class Response:
                def __init__(self,name):self.headers={'Content-Length':str(len(remote[name])),'ETag':'"sha256:'+hashlib.sha256(remote[name]).hexdigest()+'"'}
                def __enter__(self):return self
                def __exit__(self,*a):pass
            with patch.dict(os.environ,PROJECT_EXCHANGE_URL='https://exchange.invalid',PROJECT_EXCHANGE_TOKEN='test-only'),patch.object(p,'audit',return_value=None),patch.object(e,'PART_BYTES',1024):
                first=e.upload(run,closed,transport=module);count=len(uploads)
                joined=b''.join(remote[first['target_dir']+'/'+r['name']] for r in first['parts'])
                self.assertEqual(hashlib.sha256(joined).hexdigest(),first['archive_sha256'])
                with tarfile.open(fileobj=io.BytesIO(joined)) as archive:
                    self.assertEqual(set(archive.getnames()),{'tensor.npy','RUN_COMPLETE'})
                    self.assertEqual(archive.extractfile('tensor.npy').read(),(closed/'tensor.npy').read_bytes())
                # A failed early HEAD must retain the later verified receipts.
                with patch.object(e,'urlopen',side_effect=OSError('HEAD interrupted')):
                    with self.assertRaisesRegex(OSError,'HEAD interrupted'):e.upload(run,closed,transport=module)
                pending=p.read(run/'export-state/discovery/exchange.json')
                self.assertEqual(pending['status'],'pending');self.assertEqual(pending['parts'],first['parts'])
                self.assertEqual(pending['archive_sha256'],first['archive_sha256'])
                with patch.object(e,'urlopen',side_effect=lambda request,**kw:Response(request.full_url.split('/upload/',1)[1])):
                    second=e.upload(run,closed,transport=module)
                self.assertEqual(first,second);self.assertEqual(len(uploads),count)
    def test_upload_failure_preserves_pending_receipt_without_closing(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run';closed=run/'discovery';closed.mkdir(parents=True);(closed/'x').write_bytes(b'x');p.seal(closed)
            module=types.SimpleNamespace(parse_base_url=lambda b:('https',),ssl_context_for=lambda *x:None,DEFAULT_CA_FILE=None,safe_relative=lambda s,**kw:s,create_directory=lambda *a,**k:None,
                _upload_file=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('transport failed')))
            with patch.dict(os.environ,PROJECT_EXCHANGE_URL='https://exchange.invalid',PROJECT_EXCHANGE_TOKEN='test-only'),patch.object(p,'audit',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'transport failed'):e.upload(run,closed,transport=module)
            self.assertEqual(p.read(run/'export-state/discovery/exchange.json')['status'],'pending')

if __name__=='__main__':unittest.main()
