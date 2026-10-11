"""Bounded real tensor and subprocess fixtures; never a formal result."""
import json,types,subprocess,sys
from pathlib import Path
import pytest
from baselines.retacred import reference_complete as suite,reference_worker as worker
from baselines.retacred.common import atomic_json,digest

def test_writer_lock_rejects_concurrent_parent(tmp_path):
    from baselines.retacred.reference_worker import exclusive
    with exclusive(tmp_path/'run'):
        with pytest.raises(ValueError,match='writer'): 
            with exclusive(tmp_path/'run'):pass

def test_dynamic_queue_fault_cancel_and_resume(tmp_path,monkeypatch):
    root=tmp_path/'suite';root.mkdir()
    jobs=[{'id':f'm{i}','model':'fixture','seed':13,'directory':root/f'm{i}'} for i in range(3)]
    def command(job,*a):
        code="import pathlib,sys; p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);(p/'RUN_COMPLETE').touch()"
        return [sys.executable,'-c',code,str(job['directory'])]
    monkeypatch.setattr(suite,'command',command)
    monkeypatch.setattr(suite,'validate_task',lambda p: {} if (p/'RUN_COMPLETE').exists() else (_ for _ in ()).throw(ValueError('missing')))
    original_sleep=suite.time.sleep;monkeypatch.setattr(suite.time,'sleep',lambda _:original_sleep(.02))
    args=types.SimpleNamespace(run_dir=root)
    assert suite.queue(jobs,['GPU-fixture-a','GPU-fixture-b'],args,None,None)==0
    assert all((j['directory']/'RUN_COMPLETE').exists() for j in jobs)
    monkeypatch.setattr(suite,'command',lambda job,*a:[sys.executable,'-c','raise SystemExit(3)'])
    failed={'id':'failure','model':'fixture','seed':13,'directory':root/'failure'}
    assert suite.queue([failed],['GPU-fixture-a'],args,None,None)==1
    assert json.loads((root/'RUN_FAILED.json').read_text())['failures'][0]['exit_code']==3

def test_disk_estimator_retains_reviewers_and_atomic_generations():
    from baselines.retacred.reference_preflight import disk_requirement
    from baselines.retacred.reference_storage import task_peak
    assert task_peak('curriculum')>task_peak('typed_marker')
    assert disk_requirement(suite.model_ids('all'),[13])<352*1024**3

def test_nli_actual_tiny_batch_pause_resume_and_grouped_trace(tmp_path,monkeypatch):
    import torch
    from transformers import RobertaForSequenceClassification,RobertaTokenizerFast
    from baselines.retacred import metaentail_neural,reference_nli
    from baselines.retacred.metaentail_neural import encoded_group,collate_pairs
    from baselines.retacred.metaentail import grouped_prediction
    from baselines.retacred.reference_recipes import recipe
    from test_retacred_reference_complete_models import EVIDENCE,small_config,record
    class Foundation:
        def __init__(self,checkpoint,device,checkpointing):
            self.device=device;self.nli_order=('contradiction','neutral','entailment')
            self.tokenizer=RobertaTokenizerFast.from_pretrained(EVIDENCE/'tokenizer',local_files_only=True,model_max_length=512,add_prefix_space=False)
            config=small_config(labels=3);config._attn_implementation='eager'
            self.model=RobertaForSequenceClassification(config).to(device)
        def encode(self,row,f,*,split,max_length):return encoded_group(row,f,self.tokenizer,self.nli_order,split=split,max_length=max_length)
        def batch(self,features):return {k:torch.tensor(v,device=self.device) for k,v in collate_pairs(features).items()}
        def forward(self,b):
            out=self.model(**b);return out.loss,out.logits,out
        def identity(self):return {'synthetic_initialization':True,'architecture':'actual_Roberta_three_way_NLI'}
        def decide(self,g,p):return grouped_prediction(g['group'],p,nli_order=self.nli_order)
    monkeypatch.setattr(metaentail_neural,'MetaEntailFoundation',Foundation)
    rows=[record(0),dict(record(1),relation='no_relation')];data={s:rows for s in ('train','valid','test')}
    source=tmp_path/'data';source.mkdir();atomic_json(source/'data_manifest.json',{'splits':{'synthetic':{'records':2}},'synthetic':True})
    r=dict(recipe('metaentail'),epochs=1,logical_batch=2)
    def args(run,resume=False,pause=None):return types.SimpleNamespace(seed=13,device='cpu',model='metaentail',run_dir=run,micro_batch=1,resume=resume,pause_after=pause,diagnostic=True,diagnostic_updates=2,data_dir=source)
    assert reference_nli.run(args(tmp_path/'full'),{'mnli':'fixture'},data,r)==0
    assert reference_nli.run(args(tmp_path/'resume',pause=1),{'mnli':'fixture'},data,r)==75
    assert reference_nli.run(args(tmp_path/'resume',resume=True),{'mnli':'fixture'},data,r)==0
    a=torch.load(tmp_path/'full/checkpoints/latest.pt',weights_only=False);b=torch.load(tmp_path/'resume/checkpoints/latest.pt',weights_only=False)
    assert a['state']['cursor']==b['state']['cursor']==2
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    summary=json.loads((tmp_path/'resume/run_summary.json').read_text())
    for event in summary['case_study']['traces']:
        trace=json.loads((tmp_path/'resume/case_study'/event['path']).read_text())
        assert digest(tmp_path/'resume/case_study'/event['path'])==event['sha256']
        if event['phase']=='test':assert 'grouped_decision' in trace and trace['samples'][0]['stages'][-1]['output_refs']==['final_relation']
    assert summary['splits']['test']['items']==2 and summary['splits']['test']['loss_items']==4
    assert not (tmp_path/'resume/RUN_COMPLETE').exists()

def test_curriculum_selected_sample_actual_heldout_reviewer_tensors(tmp_path,monkeypatch):
    import types
    from baselines.retacred.reference_models import SyntaxCurriculum
    from baselines.retacred.reference_recipes import recipe
    from test_retacred_reference_complete_models import EVIDENCE,small_config,record,parser
    monkeypatch.setattr(worker,'make',lambda *a,**k:SyntaxCurriculum(EVIDENCE/'tokenizer',parser(),device='cpu',config=small_config()))
    data={s:[record(i) for i in range(6)] for s in ('train','valid','test')}
    args=types.SimpleNamespace(run_dir=tmp_path,seed=13,micro_batch=1,device='cpu',pause_after=None)
    r=dict(recipe('curriculum'),reviewer_epochs=1,logical_batch=2)
    difficulty=worker.cross_review(args,{},data,r)
    review=json.loads((tmp_path/'cross_review.private.json').read_text())
    assert len(difficulty)==6 and len(review['traces'])==2
    for event in review['traces']:
        path=tmp_path/'auxiliary_case_study'/event['path'];trace=json.loads(path.read_text())
        assert digest(path)==event['sha256'] and trace['samples'][0]['lane']=='auxiliary_review'
        assert trace['samples'][0]['split']=='train'
        assert (path.parent/trace['tensor_artifact']).is_file()
