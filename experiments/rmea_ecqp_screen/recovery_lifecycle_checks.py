"""Composition tests for restore.run; no real transport or admission.

The real plan validator and numerical extractor have separate fixtures. These
tests replace those two boundaries to exercise lock/seal/cleanup ordering using
tiny files. They do not validate the six real chunks or authenticate COS bytes.
"""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def checks(restore,folder):
    folder=Path(folder);folder.mkdir()
    sidecars=folder/'sidecars';sidecars.mkdir();(sidecars/'sidecar.bin').write_bytes(b'original-sidecar')
    def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    chunks=[dict(id=f'seed29:{i}',stage='seed29',index=i,bytes=3,
        sha256=hashlib.sha256(bytes([i])*3).hexdigest()) for i in [0,1]]
    plan=dict(fixture_only=True,chunks=chunks,trusted_sidecars={
        'sidecar.bin':dict(bytes=16,sha256=fingerprint(sidecars/'sidecar.bin'))})
    plan_path=folder/'plan.json';plan_path.write_text(json.dumps(plan),encoding='utf8')
    urls=folder/'urls.json';urls.write_text('{}',encoding='utf8');calls=[]
    def validate(observed):assert observed.get('fixture_only') is True
    def download(root,observed,unused):
        calls.append('download')
        for c in observed['chunks']:
            p=root/'chunks'/c['stage']/f"{c['index']}.bin";p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(bytes([c['index']])*3)
        return dict(chunks={c['id']:dict(requests=1,bytes=c['bytes']) for c in chunks})
    def extract(root,observed,unused):
        calls.append('extract');assets=root/'assets';assets.mkdir();files={}
        for name in ['train-h.npy','train-offsets.npy','train-roles.npy','train-targets.npy','metadata.json']:
            p=assets/name;p.write_bytes(name.encode());files[name]=dict(bytes=p.stat().st_size,sha256=fingerprint(p))
        return dict(files=files,prefix_examples=2,prefix_tokens=3,fixture_only=True)
    def invoke(root):return restore.run(plan_path,fingerprint(plan_path),urls,sidecars,root)
    def rejects(fn):
        try:fn()
        except (AssertionError,FileExistsError,RuntimeError,OSError):return
        raise AssertionError('Expected preservation failure')
    with patch.object(restore,'validate_plan',validate),patch.object(restore,'download',download),patch.object(restore,'extract',extract):
        root=folder/'clean';first=invoke(root)
        assert first['status']=='PASS_TRAIN_PREFIX_RECOVERY' and first['reproducible_chunks_removed']
        assert calls==['download','extract'] and not (root/'recovery.lock').exists()
        assert all(not(root/'chunks'/c['stage']/f"{c['index']}.bin").exists() for c in chunks)
        second=invoke(root)
        assert second['new_GET_requests']==0 and calls==['download','extract']
        # A completed receipt whose assets changed must fail before a new
        # download or acceptance replacement.
        sealed=(root/'prefix-acceptance.json').read_bytes()
        (root/'assets/train-h.npy').write_bytes(b'tampered')
        rejects(lambda:invoke(root))
        assert (root/'prefix-acceptance.json').read_bytes()==sealed and calls==['download','extract']
        blocked=folder/'locked';blocked.mkdir();(blocked/'recovery.lock').write_bytes(b'other-attempt')
        rejects(lambda:invoke(blocked))
        assert (blocked/'recovery.lock').read_bytes()==b'other-attempt' and calls==['download','extract']
        partial=folder/'unsealed';(partial/'assets').mkdir(parents=True)
        (partial/'assets/KEEP').write_bytes(b'unsealed-preserve')
        rejects(lambda:invoke(partial))
        assert (partial/'assets/KEEP').read_bytes()==b'unsealed-preserve' and calls==['download','extract']
        # Inject failure after the receipt was durably written but before any
        # chunk unlink. Recovery must use that receipt and make zero new GETs.
        failed=folder/'cleanup-failure';original_unlink=Path.unlink
        def fail_cleanup(path,*args,**kwargs):
            if path.suffix=='.bin' and 'chunks' in path.parts:raise OSError('injected cleanup failure')
            return original_unlink(path,*args,**kwargs)
        with patch.object(Path,'unlink',fail_cleanup):rejects(lambda:invoke(failed))
        accepted=json.loads((failed/'prefix-acceptance.json').read_bytes())
        assert accepted['status']=='PREFIX_VERIFIED_CLEANUP_PENDING'
        assert len(calls)==4 and all((failed/'chunks'/c['stage']/f"{c['index']}.bin").exists() for c in chunks)
        repaired=invoke(failed)
        assert repaired['status']=='PASS_TRAIN_PREFIX_RECOVERY' and repaired['new_GET_requests']==0 and len(calls)==4
        assert not(failed/'recovery.lock').exists()
    return ['whole_run_receipt_before_chunk_cleanup','sealed_reuse_zero_download',
        'tampered_asset_existing_lock_and_unsealed_output_preserved',
        'injected_postseal_cleanup_failure_repaired_without_GET']
