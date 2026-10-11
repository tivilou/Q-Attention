"""Tiny filesystem/Torch/transport fixtures only; never formal data training."""
import hashlib
import io
import json
import types
from pathlib import Path
import pytest
from baselines.retacred import reference_storage as storage, reference_archive as archive
from baselines.retacred.common import atomic_json,digest,fingerprint


@pytest.fixture
def scope(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'ROOT',tmp_path)
    suite=tmp_path/'runs'/'fixture';suite.mkdir(parents=True)
    monkeypatch.setenv(storage.ENV,str(suite))
    task=suite/'tasks'/'typed_marker_seed13';task.mkdir(parents=True)
    monkeypatch.setattr(archive,'source_identity',lambda:'fixture-exact-code')
    return suite,task


@pytest.fixture
def exchange(monkeypatch):
    files={};fail={'readback':False};counts={'uploads':0}
    class Response(io.BytesIO):
        status=200
        def __init__(self,data):
            super().__init__(data);self.headers={'Content-Length':str(len(data))}
    class Opener:
        def open(self,request,timeout):
            from urllib.parse import unquote
            name=unquote(request.full_url.split('/files/',1)[1]);data=files[name]
            if fail['readback']:data=bytes([data[0]^1])+data[1:]
            return Response(data)
    class Helper:
        def safe_relative(self,name,**kw):
            if name.startswith('/') or any(x.startswith('.') or not x for x in name.split('/')):raise ValueError('unsafe')
            return name
        def create_directory(self,*a,**kw):return {}
        def _upload_file(self,url,token,path,remote,size,sha,**kw):
            value=path.read_bytes();assert len(value)==size and hashlib.sha256(value).hexdigest()==sha
            if remote in files and files[remote]!=value:raise ValueError('immutable remote conflict')
            files[remote]=value;counts['uploads']+=1
            return {'path':remote,'size_bytes':size,'sha256':sha,'status':'uploaded'}
        def upload_declared_files(self,*,base_url,token,target_dir,files,ca_file):
            for path,name in files:self._upload_file(base_url,token,path,target_dir+'/'+name,path.stat().st_size,digest(path))
            return {'status':'uploaded'}
    helper=Helper()
    monkeypatch.setattr(archive,'transport',lambda:('https://fixture.invalid','fixture-credential',helper,None,Opener()))
    monkeypatch.setattr(archive,'uploader',lambda:helper)
    monkeypatch.setattr(archive,'SHARD',4096)
    return files,fail,counts


def test_latest_write_pressure_preserves_original_and_returns75(scope,monkeypatch):
    import torch
    from baselines.retacred.train import save_torch
    suite,task=scope;path=task/'checkpoints/latest.pt';path.parent.mkdir()
    torch.save({'cursor':7,'model':torch.ones(3)},path);before=digest(path)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:types.SimpleNamespace(free=1))
    assert storage.pause_cli(lambda:save_torch(path,{'cursor':8,'model':torch.zeros(3)}))==75
    assert digest(path)==before and not path.with_name('latest.pt.tmp').exists()
    assert json.loads((suite/'STORAGE_PAUSED.json').read_text())['status']=='paused'


def test_simultaneous_writer_reservations_do_not_overcommit(scope,monkeypatch):
    suite,task=scope
    monkeypatch.setattr(storage,'FLOOR',0)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:types.SimpleNamespace(free=100))
    with storage.write_reservation(task/'one.pt',70):
        with pytest.raises(storage.DiskPressure):
            with storage.write_reservation(task/'two.pt',40):pass
    assert not list((suite/'storage').glob('claim-*.json'))
    with storage.write_reservation(task/'two.pt',40):pass


def test_first_checkpoint_disk_pause_is_reconstructible_without_losing_resume(scope,monkeypatch):
    from test_retacred_reference_complete import Tiny,row
    from baselines.retacred.reference_worker import seed_all,train_standard
    from baselines.retacred.reference_recipes import recipe
    _,task=scope;task.rmdir()
    data={s:[row(i) for i in range(4)] for s in ('train','valid','test')}
    r=dict(recipe('typed_marker'),epochs=1,logical_batch=2,selection='final_epoch')
    original=storage.shutil.disk_usage
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:types.SimpleNamespace(free=1))
    seed_all(13);a=Tiny('cpu')
    with pytest.raises(storage.DiskPressure):train_standard(a,data,task,r,13,1,case=False)
    assert (task/'INITIALIZATION_PENDING.json').exists() and not (task/'checkpoints/latest.pt').exists()
    monkeypatch.setattr(storage.shutil,'disk_usage',original)
    seed_all(13);b=Tiny('cpu')
    _,state=train_standard(b,data,task,r,13,1,resume=True,case=False)
    assert state['cursor']==2 and not (task/'INITIALIZATION_PENDING.json').exists()


def test_exact_checkpoint_and_tensor_bytes_survive_sharded_offload_and_mmap(scope,exchange):
    import torch
    suite,task=scope;path=task/'checkpoints/latest.pt';path.parent.mkdir()
    expected=torch.arange(4000,dtype=torch.float32)
    torch.save({'model':expected,'optimizer':{'moment':expected*2},'cursor':9,'rng':torch.get_rng_state()},path)
    sha=digest(path);receipt=archive.archive_files(task,[path],remove=True)
    assert not path.exists() and receipt['entries']['checkpoints/latest.pt']['sha256']==sha
    assert len(receipt['entries']['checkpoints/latest.pt']['parts'])>1
    archive.verify_available(path,sha)
    with archive.materialize(path) as temporary:
        assert digest(temporary)==sha
        restored=torch.load(temporary,weights_only=True,mmap=True)
    assert not temporary.exists() and torch.equal(restored['model'],expected)
    assert restored['cursor']==9 and torch.equal(restored['optimizer']['moment'],expected*2)
    archive.archive_files(task,[path],remove=True)  # restart after already-unlinked file


def test_bad_readback_preserves_only_valid_local_copy_and_retry_is_idempotent(scope,exchange):
    suite,task=scope;files,fail,_=exchange
    path=task/'case_study/event.pt';path.parent.mkdir();path.write_bytes(b'x'*18000);sha=digest(path)
    fail['readback']=True
    with pytest.raises(ValueError,match='readback'):archive.archive_files(task,[path],remove=True)
    assert path.exists() and digest(path)==sha and not archive.receipt_path(task).exists()
    fail['readback']=False
    archive.archive_files(task,[path],remove=True)
    assert not path.exists() and archive.load(task)['entries']['case_study/event.pt']['readback_sha256']==sha
    assert not list((suite/'storage/spool').iterdir())


def test_cleanup_after_ledger_commit_can_resume_without_reupload(scope,exchange,monkeypatch):
    suite,task=scope;_,_,counts=exchange
    path=task/'case_study/event.pt';path.parent.mkdir();path.write_bytes(b'x'*18000)
    archive.archive_files(task,[path]);uploaded=counts['uploads']
    original=Path.unlink
    def interrupted(p,*a,**kw):
        if p==path:raise OSError('injected crash after cleanup intent')
        return original(p,*a,**kw)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError):archive.archive_files(task,[path],remove=True)
    assert path.exists() and 'case_study/event.pt' in archive.load(task)['removed']
    monkeypatch.setattr(Path,'unlink',original)
    archive.archive_files(task,[path],remove=True)
    assert not path.exists() and counts['uploads']==uploaded


def test_archive_ledger_tampering_and_symlinks_rejected(scope,exchange):
    suite,task=scope;outside=suite.parent/'outside.pt';outside.write_bytes(b'outside')
    path=task/'evil.pt';path.symlink_to(outside)
    with pytest.raises(ValueError,match='regular'):archive.archive_files(task,[path],remove=True)
    assert outside.read_bytes()==b'outside'
    good=task/'good.pt';good.write_bytes(b'good');archive.archive_files(task,[good],remove=True)
    r=json.loads(archive.receipt_path(task).read_text());r['entries']['good.pt']['sha256']='0'*64
    atomic_json(archive.receipt_path(task),r)
    with pytest.raises(ValueError,match='catalog'):archive.verify_available(good,'0'*64)


def test_verified_asset_shards_only_reclaimed_after_whole_sha_pass(tmp_path):
    final=tmp_path/'model.bin';parts=tmp_path/'parts';parts.mkdir()
    final.write_bytes(b'abcdef');a=parts/'model.bin.part0000';a.write_bytes(b'abcdef')
    foreign=parts/'other-run.part0000';foreign.write_bytes(b'preserve')
    row={'size_bytes':6,'sha256':digest(final),'parts':[{'name':a.name,'size_bytes':6,'sha256':digest(a)}]}
    bad=dict(row,sha256='0'*64)
    with pytest.raises(ValueError):storage.clean_verified_parts(final,bad,parts)
    assert a.exists()
    storage.clean_verified_parts(final,row,parts)
    assert not a.exists() and final.read_bytes()==b'abcdef' and foreign.read_bytes()==b'preserve'


def test_only_owned_abandoned_ctl_temporaries_are_reclaimed(tmp_path):
    run=tmp_path/'run';directory=run/'checkpoints';directory.mkdir(parents=True)
    owned=directory/'.generation-owned';owned.mkdir()
    contract={'seed':13,'cursor':0}
    atomic_json(owned/'storage-owner.json',{'schema':'ctl-owned-stage.v1','contract_sha256':fingerprint(contract)})
    (owned/'state.pt.part').write_bytes(b'abandoned partial state')
    unknown=directory/'.generation-unknown';unknown.mkdir();(unknown/'state.pt').write_bytes(b'preserve')
    live=directory/'update00000001-latest';live.mkdir();(live/'state.pt').write_bytes(b'latest')
    storage.clean_abandoned_ctl_stages(run,contract)
    assert not owned.exists() and (unknown/'state.pt').read_bytes()==b'preserve' and (live/'state.pt').read_bytes()==b'latest'


def test_disk_budget_is_shared_and_concurrency_bounded_not15_times_seed_count(monkeypatch):
    from baselines.retacred.reference_complete import model_ids
    monkeypatch.setattr(storage,'asset_rows',lambda names:[])
    a=storage.disk_plan(model_ids('all'),[13],slots=1)
    b=storage.disk_plan(model_ids('all'),[13,29,53],slots=1)
    c=storage.disk_plan(model_ids('all'),[13],slots=3)
    assert a['required_additional_bytes']==b['required_additional_bytes']
    assert a['required_additional_bytes']<c['required_additional_bytes']<352*storage.GIB
    assert storage.task_peak('curriculum')>storage.task_peak('typed_marker')


def test_selected_actual_tensor_values_remain_readable_after_active_archive(scope,exchange):
    import torch
    from test_retacred_reference_complete import Tiny,row
    from baselines.retacred.reference_worker import seed_all,train_standard,evaluate
    from baselines.retacred.reference_recipes import recipe
    from baselines.retacred.reference_trace import validate,ValueReader
    _,task=scope
    r=dict(recipe('typed_marker'),epochs=1,logical_batch=2,selection='final_epoch')
    data={s:[row(i) for i in range(4)] for s in ('train','valid','test')}
    # train_standard expects a fresh directory.
    task.rmdir();seed_all(13);a=Tiny('cpu');features,state=train_standard(a,data,task,r,13,1)
    first=task/'case_study'/state['traces'][0]['path'];meta=json.loads(first.read_text())
    assert not (first.parent/meta['tensor_artifact']).exists()
    validate(task,state['traces']+[dict(state['traces'][0],phase='test')])
    reader=ValueReader(first)
    key=next(k for k,v in reader.values.items() if v.ndim==1 and v.numel()>0)
    assert reader.read(key,window=[(0,1)])==reader.values[key][:1].tolist()
    assert (task/'checkpoints/latest.pt').exists()  # live optimizer is not evicted


def test_curriculum_ledger_skips_completed_reviewers_without_checkpoint_reload(tmp_path,monkeypatch):
    from baselines.retacred import reference_worker as w
    from baselines.retacred.reference_models import SyntaxCurriculum
    from baselines.retacred.reference_recipes import recipe
    from test_retacred_reference_complete_models import EVIDENCE,small_config,record,parser
    monkeypatch.delenv(storage.ENV,raising=False)
    monkeypatch.setattr(w,'make',lambda *a,**k:SyntaxCurriculum(EVIDENCE/'tokenizer',parser(),device='cpu',config=small_config()))
    data={s:[record(i) for i in range(6)] for s in ('train','valid','test')}
    args=types.SimpleNamespace(run_dir=tmp_path,seed=13,micro_batch=1,device='cpu',pause_after=None)
    r=dict(recipe('curriculum'),reviewer_epochs=1,logical_batch=2)
    expected=w.cross_review(args,{},data,r)
    for path in (tmp_path/'reviewers').rglob('*.pt'):path.unlink()
    monkeypatch.setattr(w,'make',lambda *a,**k:(_ for _ in ()).throw(AssertionError('reviewer retrained')))
    assert w.cross_review(args,{},data,r)==expected


def test_native_ctl_exact_resume_with_one_local_latest_and_archived_case_pins(scope,exchange):
    import torch
    from test_retacred_ctl_drp_runner import make,load_latest
    from baselines.retacred import ctl_drp_runner as native
    from baselines.retacred.ctl_drp_neural import tokenizer_from_cache,descriptions_from_native
    import os
    assets=(tokenizer_from_cache(Path(os.environ['CTL_DRP_BPE_CACHE'])),descriptions_from_native(Path(os.environ['CTL_DRP_NATIVE'])))
    suite,task=scope;task.rmdir()
    a=make(suite/'synthetic-data',assets)
    paused=native.run(a,task,pause_after=1)
    assert paused['status']=='paused'
    b=make(suite/'synthetic-data',assets)
    done=native.run(b,task,resume=True)
    assert done['status']=='diagnostic_complete'
    local=list((task/'checkpoints').rglob('*.pt'))
    assert 1<=len(local)<=2 and load_latest(task)['cursor']==2
    r=archive.load(task)
    assert any(k.startswith('checkpoints/') and k.endswith('/state.pt') for k in r['entries'])
    assert any(k.startswith('case_study/') and k.endswith('.pt') for k in r['entries'])
    native.validate_complete(task,b)


def test_completed_archive_export_retry_never_retrains_or_requires_local_tensors(scope,exchange,monkeypatch):
    import torch
    from baselines.retacred import reference_complete as complete
    suite,task=scope
    for folder in ('case_study','checkpoints'):(task/folder).mkdir()
    events=[]
    for phase in ('train','test'):
        tensor=task/'case_study'/f'{phase}.pt';torch.save({'embedding':torch.arange(8)},tensor)
        meta=task/'case_study'/f'{phase}.json'
        atomic_json(meta,{'status':'complete','tensor_artifact':tensor.name,'sha256':digest(tensor),'size_bytes':tensor.stat().st_size,
            'representations':[{'value_status':'complete_values'}]})
        events.append({'phase':phase,'path':meta.name,'sha256':digest(meta)})
    torch.save({'cursor':2,'optimizer':{'moment':torch.ones(8)}},task/'checkpoints/latest.pt')
    atomic_json(task/'run_summary.json',{'case_study':{'traces':events}})
    monkeypatch.setattr(complete,'validate_task',lambda p:json.loads((p/'run_summary.json').read_text()))
    first=archive.archive_completed(task)
    assert first['status']=='complete' and not (task/'checkpoints/latest.pt').exists()
    second=archive.archive_completed(task)
    assert first==second and all(not p.exists() for p in [task/'case_study/train.pt',task/'case_study/test.pt'])
    jobs=[{'id':task.name,'directory':task}]
    a=archive.aggregate(suite,jobs);b=archive.aggregate(suite,jobs)
    assert a==b and a['status']=='complete'
    assert any(f['path'].endswith('reference_archive.json') for f in a['files'])
