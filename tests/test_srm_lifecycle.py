"""Unit contracts plus separately enabled real-parent canary/recovery tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from q_attention.experiments import srm_protocol as p, srm_runner as runner, cfra_checkpoint as cp
from q_attention.analysis.srm_trace import Trace, values, vector, state_sha, validate_trace
from q_attention.analysis.ecpa_trace import read_values
from q_attention.adapters.q_srm import SRMModel, objective
from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig
from q_attention.tasks.relation import RelationRecord,RelationDataset,build_vocab,build_label_map,collate_relation_batch


def trace_fixture(tmp_path,mode):
    rows=[RelationRecord(tokens=['Alice','joined','Lab'],subject=(0,1),object=(2,3),label='member',metadata={}),
          RelationRecord(tokens=['Bob','left','Team','today'],subject=(0,1),object=(2,3),label='former',metadata={})]
    vocab=build_vocab(rows);labels=build_label_map(rows)
    ds=RelationDataset(rows,vocab,labels);batch=collate_relation_batch([ds[0]])
    torch.manual_seed(3)
    base=RelationExtractionModel(RelationTransformerConfig(vocab_size=len(vocab),num_labels=len(labels),dim=16,
        num_heads=2,num_layers=2,ff_dim=32,dropout=0,max_length=8))
    model=SRMModel(base,mode)
    names=[dict(name=n,shape=list(v.shape),numel=v.numel()) for n,v in model.named_parameters() if v.requires_grad]
    writer=Trace(tmp_path,mode,dict(seed=13,model_identity=mode,profile='unit_fixture',run_id='unit',
        dataset='synthetic unit fixture',code_revision='unit',config_sha256=p.identity({'fixture':1})),labels,names)
    for split in ('train','valid','test'):
        for checkpoint in ('initial','best','final'):
            model.zero_grad(set_to_none=True);before=vector(model)
            loss,_=objective(model(**runner.Worker.inputs(batch),capture=True),batch['labels']);loss.backward()
            writer.add(rows[0],split,checkpoint,'replay',values(model,batch['labels'],loss,before,vector(model,True),before),0,state_sha(model))
    before=vector(model);identity=state_sha(model);gradient=vector(model,True)
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=.001);optimizer.step()
    writer.add(rows[0],'train','update0','training',values(model,batch['labels'],loss,before,gradient,vector(model)),0,identity)
    writer.payload['trace_completeness']='complete';writer.save()
    return tmp_path/'sample_trace.json'


@pytest.mark.parametrize('mode',p.MODES)
def test_complete_unit_trace_and_independent_circuit_replay(tmp_path,mode):
    path=trace_fixture(tmp_path,mode);report=validate_trace(path)
    assert report['samples']==10 and report['full_value_artifacts']>=250
    doc=json.loads(path.read_text());sample=doc['samples'][0]
    name='attention';rep=next(r for r in sample['representations'] if r['id']==name)
    a=np.load(path.parent/rep['artifact_refs'][0]['path'],allow_pickle=False)
    coords=tuple(n//2 for n in a.shape)
    assert read_values(path,sample['sample_id'],name,coords)==a[coords]


def test_semantics_rejects_refreshed_hash_for_changed_null(tmp_path):
    path=trace_fixture(tmp_path,'quantum');doc=json.loads(path.read_text())
    rep=next(r for r in doc['samples'][0]['representations'] if r['id']=='null_probability')
    file=path.parent/rep['artifact_refs'][0]['path'];a=np.load(file,allow_pickle=False);a[:]=0
    np.save(file,a,allow_pickle=False);digest=p.sha(file)
    rep['artifact_refs'][0]['sha256']=digest
    rep['value_capture']['artifact_refs'][0]['sha256']=digest
    rep['value_capture']['reader_projection']['source_sha256']=[digest]
    p.atomic_json(path,doc)
    with pytest.raises(ValueError,match='normalization'):validate_trace(path)


def test_trace_rejects_missing_inventory_and_value_tamper(tmp_path):
    path=trace_fixture(tmp_path,'diagonal');doc=json.loads(path.read_text())
    ref=doc['samples'][0]['representations'][0]['artifact_refs'][0]
    file=path.parent/ref['path'];file.write_bytes(file.read_bytes()+b'broken')
    with pytest.raises(ValueError,match='mutated'):validate_trace(path)


def test_protocol_frozen_scope_and_cli():
    assert p.config('screen')['expected_selected']==dict(train=630,valid=631,test=587)
    assert p.config('screen')['seeds']==[13,29,53,71,97]
    assert p.config('canary')['expected_selected']==dict(train=12,valid=6,test=6)
    with pytest.raises(ValueError):p.config('formal')
    with pytest.raises(SystemExit):runner.parser().parse_args(['--output-dir','/tmp/example','--profile','formal'])
    assert runner.child_spec('/tmp/example',13,'quantum')['argv'][1]==str(p.ROOT/'experiments/run_srm_screen.py')


def test_outcome_blind_cursor_and_first_case():
    for epoch in range(2):
        batches=[p.update_indexes(12,4,u,13) for u in range(epoch*3,(epoch+1)*3)]
        assert sorted(sum(batches,[]))==list(range(12))
        if epoch==0:assert batches[0][0]==0


def test_review_binding_fails_for_stale_source(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    path=tmp_path/'docs/validation/q_srm_review.json'
    path.parent.mkdir(parents=True)
    p.atomic_json(path,dict(review_scope='mechanism',verdict='PASS',must_fix=[],source_sha256='wrong'))
    with pytest.raises(ValueError,match='review scope'):p.validate_review()


def _equal(left,right):
    if isinstance(left,torch.Tensor):assert torch.equal(left,right)
    elif isinstance(left,np.ndarray):assert np.array_equal(left,right)
    elif isinstance(left,dict):
        assert set(left)==set(right)
        for key in left:_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):_equal(a,b)
    else:assert left==right


@pytest.mark.skipif(os.environ.get('QSRM_RUN_LIFECYCLE')!='1',reason='requires exact-code scope-both review before real-parent canary')
def test_real_parent_resume_export_failure_and_tamper(tmp_path):
    direct,resumed=tmp_path/'direct',tmp_path/'resumed'
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def launch(root,*extra,expected=0):
        args=[sys.executable,str(p.ROOT/'experiments/run_srm_screen.py'),'--profile','canary','--device','cpu','--output-dir',str(root),*extra]
        result=subprocess.run(args,cwd=p.ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==expected,result.stdout+'\n'+result.stderr
    launch(direct)
    launch(resumed,'--stop-after','2',expected=85)
    interrupted=cp.load(resumed/'seed13/quantum');assert interrupted['next_update']==2
    launch(resumed,'--resume')
    for mode in ('warmup',*p.MODES):
        a,b=cp.load(direct/f'seed13/{mode}'),cp.load(resumed/f'seed13/{mode}')
        for key in ('model','optimizer','next_update','selected_through','best_model','best_metric','best_update','rng','next_indexes'):_equal(a[key],b[key])
        assert p.read(direct/f'seed13/{mode}/summary.json')==p.read(resumed/f'seed13/{mode}/summary.json')
    manifest=p.read(resumed/'run_manifest.json')
    before={str(f.relative_to(resumed)):p.sha(f) for f in resumed.rglob('*') if f.is_file()}
    failed=tmp_path/'failed_export'
    with pytest.raises(RuntimeError,match='injected'):runner.export(resumed,failed,inject_failure=True)
    assert not failed.exists() and not list(tmp_path.glob('.failed_export-*'))
    runner.export(resumed,failed)
    after={str(f.relative_to(resumed)):p.sha(f) for f in resumed.rglob('*') if f.is_file()}
    assert before==after
    assert {f.name for f in failed.iterdir()}=={'summary.json','manifest.json','trace_validation.json','receipt.json'}
    launch(resumed,'--resume','--export-only')
    assert before=={str(f.relative_to(resumed)):p.sha(f) for f in resumed.rglob('*') if f.is_file()}
    summary=p.read(resumed/'summary.json')
    assert summary['decision']['classification']=='canary_only'
    assert len(summary['conditions'])==5
    assert sum(x['trace']['samples'] for x in summary['conditions'])==50
    job=resumed/'seed13/quantum';pointer=p.read(job/'latest.json');generation=job/'checkpoints'/pointer['latest']['file']
    generation.write_bytes(b'damaged')
    with pytest.raises(ValueError,match='hash mismatch'):runner.completed(job,manifest,13,'quantum')
    p.atomic_json(tmp_path/'lifecycle_result.json',dict(status='passed',conditions=5,traces=50,
        full_arrays=sum(x['trace']['full_value_artifacts'] for x in summary['conditions']),exact_batch_resume=True,
        export_failure_cleanup=True,immutable_retry=True,checkpoint_tamper_rejected=True))
