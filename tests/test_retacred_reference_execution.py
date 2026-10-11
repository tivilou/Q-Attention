"""Finite OOM transitions, full-state batch recovery and evidence, bounded data."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.retacred import reference_execution as execution


def arguments(path, micro='auto'):
    return SimpleNamespace(run_dir=path, micro_batch=micro, seed=13, checkpoint_seconds=120.)


def test_actual_oom_is_the_only_tier_trigger_and_resume_preserves_it(tmp_path):
    import torch
    args=arguments(tmp_path/'job')
    calls=[]
    def failure():
        calls.append(args.micro_batch)
        raise torch.cuda.OutOfMemoryError('injected CUDA allocation failure')
    assert execution.adaptive_cli(args,'metaentail',32,failure)==76
    assert calls==[32]
    restored=arguments(tmp_path/'job')
    execution.prepare(restored,'metaentail',32)
    assert restored.micro_batch==16 and restored.execution_state['index']==1
    with pytest.raises(ValueError,match='ordinary'):
        execution.adaptive_cli(arguments(tmp_path/'job'),'metaentail',32,lambda:(_ for _ in ()).throw(ValueError('ordinary failure')))
    assert json.loads(execution.ledger_path(tmp_path/'job').read_text())['index']==1


def test_finite_exhaustion_and_h3_padding_guard(tmp_path):
    import torch
    assert execution.tiers('sequence_routing_h3',24)==[
        {'micro':1,'checkpointing':False},{'micro':1,'checkpointing':True}]
    assert [x['micro'] for x in execution.tiers('metaentail',32,'8')]==[8,4,2,1,1]
    with pytest.raises(ValueError,match='H3 requires micro-batch1'):execution.tiers('sequence_routing_h3',24,'2')
    operation=lambda:(_ for _ in ()).throw(torch.cuda.OutOfMemoryError('test'))
    assert execution.adaptive_cli(arguments(tmp_path/'job'),'sequence_routing_h3',24,operation)==76
    with pytest.raises(torch.cuda.OutOfMemoryError):
        execution.adaptive_cli(arguments(tmp_path/'job'),'sequence_routing_h3',24,operation)


def test_checkpoint_cadence_and_background_failure_propagation():
    now=[0.];cadence=execution.Cadence({'checkpoint_seconds':120},clock=lambda:now[0])
    now[0]=119;assert not cadence.due()
    now[0]=120;assert cadence.due()
    cadence.saved(7);assert cadence.cursor==7 and not cadence.due()
    queue=execution.ArchiveQueue()
    queue.submit(lambda:(_ for _ in ()).throw(ValueError('upload failed')))
    with pytest.raises(ValueError,match='upload failed'):queue.close()


def test_periodic_checkpoint_pause_then_next_update_is_exact(tmp_path):
    import torch
    from transformers import RobertaConfig
    from baselines.retacred.common import ROOT
    from baselines.retacred.reference_models import SequenceRouting
    from baselines.retacred.reference_worker import train_standard,seed_all
    from baselines.retacred.reference_recipes import recipe
    evidence=ROOT/'.external/retacred/reference-cohort-completion-20261009T'
    config=RobertaConfig(vocab_size=50265,hidden_size=16,num_hidden_layers=2,
        num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,pad_token_id=1)
    records=[dict(id='bounded-'+str(i),token=['Alice','joined','Acme','.'],subj_start=0,subj_end=0,
        obj_start=2,obj_end=2,subj_type='PERSON',obj_type='ORGANIZATION',relation='per:employee_of') for i in range(6)]
    def adapter():
        seed_all(13)
        return SequenceRouting(evidence/'tokenizer',ROOT/'.external/reference_sources/sequence_routing',
            evidence/'resources',device='cpu',config=copy.deepcopy(config))
    r=dict(recipe('sequence_routing_h3'),epochs=1,logical_batch=2,selection='final_epoch')
    data={s:records for s in ('train','valid','test')}
    a=adapter();a.execution_policy={'checkpoint_seconds':120.}
    _,state=train_standard(a,data,tmp_path/'whole',r,13,1,case=False)
    b=adapter();b.execution_policy={'checkpoint_seconds':120.}
    assert train_standard(b,data,tmp_path/'split',r,13,1,case=False,pause_after=1) is None
    saved=torch.load(tmp_path/'split/checkpoints/latest.pt',weights_only=False)
    assert saved['state']['cursor']==1
    c=adapter();c.execution_policy={'checkpoint_seconds':120.}
    _,restored=train_standard(c,data,tmp_path/'split',r,13,1,case=False,resume=True)
    assert state['cursor']==restored['cursor']==3
    for key,value in a.model.state_dict().items():assert torch.equal(value,c.model.state_dict()[key]),key


def test_h3_skips_dead_mlm_but_forward_and_gradients_match():
    import torch
    from transformers import RobertaConfig,RobertaForMaskedLM
    torch.manual_seed(13)
    config=RobertaConfig(vocab_size=71,hidden_size=16,num_hidden_layers=2,
        num_attention_heads=2,intermediate_size=32,pad_token_id=1)
    a=RobertaForMaskedLM(config).train();b=copy.deepcopy(a)
    ids=torch.tensor([[0,9,12,5,2]])
    outputs=[]
    for model,dead in ((a,True),(b,False)):
        torch.manual_seed(29)
        result=(model if dead else model.roberta)(input_ids=ids,output_hidden_states=True)
        stack=torch.stack(result.hidden_states,dim=-2)
        loss=stack.square().mean();loss.backward();outputs.append(stack)
    assert torch.equal(outputs[0],outputs[1])
    for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert (pa.grad is None)==(pb.grad is None),name
        if pa.grad is not None:assert torch.equal(pa.grad,pb.grad),name


def test_oom_partition_change_restores_optimizer_and_never_skips_input(tmp_path):
    import torch
    from baselines.retacred.reference_worker import train_standard,seed_all
    from baselines.retacred.reference_recipes import recipe
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.classifier=torch.nn.Linear(2,40)
        def forward(self,input_ids,labels):
            scores=self.classifier(input_ids.float())
            return torch.nn.functional.cross_entropy(scores,labels),scores
    def adapter():
        seed_all(13);obj=SimpleNamespace(model=Model())
        obj.encode=lambda records:[{'input_ids':r['input_ids'],'labels':r['label']} for r in records]
        obj.finalize_vocabulary=lambda:None
        obj.batch=lambda rows:{'input_ids':torch.tensor([r['input_ids'] for r in rows]),'labels':torch.tensor([r['labels'] for r in rows])}
        obj.forward=lambda batch:obj.model(**batch)
        return obj
    records=[{'id':str(i),'input_ids':[i%3,1],'label':i%2} for i in range(8)]
    data={s:records for s in ('train','valid','test')}
    r=dict(recipe('entity_marker'),epochs=2,logical_batch=4,selection='final_epoch')
    a=adapter();train_standard(a,data,tmp_path/'reference',r,13,4,case=False)
    b=adapter();args=arguments(tmp_path/'recovered')
    execution.prepare(args,'entity_marker',4);execution.attach(b,args)
    assert train_standard(b,data,args.run_dir,r,13,4,case=False,pause_after=1) is None
    def fail():
        raise torch.cuda.OutOfMemoryError('allocation during second batch')
    assert execution.adaptive_cli(arguments(args.run_dir),'entity_marker',4,fail)==76
    c=adapter();restored=arguments(args.run_dir)
    execution.prepare(restored,'entity_marker',4);execution.attach(c,restored)
    _,state=train_standard(c,data,args.run_dir,r,13,restored.micro_batch,case=False,resume=True)
    assert restored.micro_batch==2 and state['cursor']==4
    for key,value in a.model.state_dict().items():torch.testing.assert_close(value,c.model.state_dict()[key],rtol=1e-5,atol=1e-7)


def test_queue_oom_retry_stays_on_its_original_physical_gpu(tmp_path,monkeypatch):
    from baselines.retacred import reference_complete as suite
    root=tmp_path/'suite';root.mkdir()
    jobs=[dict(id=n,model='fixture',seed=13,directory=root/n) for n in ('a','b')]
    launches=[];counts={}
    class Process:
        def __init__(self,argv,**kwargs):
            name=argv[0];gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']
            launches.append((name,gpu));counts[name]=counts.get(name,0)+1
            self.rc=76 if name=='b' and counts[name]==1 else 0
            if self.rc==76:
                from baselines.retacred.common import atomic_json
                atomic_json(execution.ledger_path(root/name),{'index':1,'events':[{'to_tier':1}]})
        def poll(self):return self.rc
    monkeypatch.delenv('QATTN_REFERENCE_STORAGE_ROOT',raising=False)
    monkeypatch.setattr(suite,'command',lambda job,*_: [job['id']])
    monkeypatch.setattr(suite.subprocess,'Popen',Process)
    monkeypatch.setattr(suite,'validate_task',lambda _: {})
    monkeypatch.setattr(suite.time,'sleep',lambda _:None)
    assert suite.queue(jobs,['GPU-a','GPU-b'],SimpleNamespace(run_dir=root),None,None)==0
    assert launches==[('a','GPU-a'),('b','GPU-b'),('b','GPU-b')]


def test_encoding_audit_cache_is_complete_and_bound_to_originals(tmp_path,monkeypatch):
    import os
    from baselines.retacred import ctl_drp_task as task
    from baselines.retacred.ctl_drp_runner import synthetic_receipt
    from baselines.retacred.ctl_drp_neural import tokenizer_from_cache,descriptions_from_native
    data=task.PhysicalSplits(synthetic_receipt(tmp_path/'inputs'))
    tokenizer=tokenizer_from_cache(os.environ['CTL_DRP_BPE_CACHE'])
    descriptions=descriptions_from_native(os.environ['CTL_DRP_NATIVE'])
    path=tmp_path/'audit.json'
    initial=task.cached_encoding_audit(data,tokenizer,descriptions,path)
    monkeypatch.setattr(task,'audit_encodings',lambda *_:(_ for _ in ()).throw(AssertionError('repeated full encoding audit')))
    assert task.cached_encoding_audit(data,tokenizer,descriptions,path)==initial
    changed=dict(descriptions);changed[next(iter(changed))]+=' changed'
    with pytest.raises(ValueError,match='identity drift'):task.cached_encoding_audit(data,tokenizer,changed,path)
    value=json.loads(path.read_text());value['audit']['train']['rows_dropped']=1
    from baselines.retacred.common import atomic_json
    atomic_json(path,value)
    with pytest.raises(ValueError,match='integrity drift'):task.cached_encoding_audit(data,tokenizer,descriptions,path)


def test_real_worker_cli_diagnostic_retains_full_values_without_train_reevaluation(tmp_path,monkeypatch):
    import torch
    from transformers import RobertaConfig
    from baselines.retacred.common import ROOT,atomic_json
    from baselines.retacred import reference_worker as worker
    from baselines.retacred.reference_models import SequenceRouting
    from baselines.retacred.reference_recipes import recipe
    evidence=ROOT/'.external/retacred/reference-cohort-completion-20261009T'
    config=RobertaConfig(vocab_size=50265,hidden_size=16,num_hidden_layers=2,
        num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,pad_token_id=1)
    def factory(*args,device,checkpointing,**kwargs):
        adapter=SequenceRouting(evidence/'tokenizer',ROOT/'.external/reference_sources/sequence_routing',
            evidence/'resources',device=device,config=copy.deepcopy(config),checkpointing=checkpointing)
        return adapter
    monkeypatch.setattr(worker,'make',factory)
    r=dict(recipe('sequence_routing_h3'),epochs=1,logical_batch=1,selection='final_epoch')
    monkeypatch.setattr(worker,'recipe',lambda _:r)
    source=tmp_path/'inputs';source.mkdir()
    rows=[dict(id='bounded-'+str(i),token=['Alice','joined','Acme','.'],subj_start=0,subj_end=0,
        obj_start=2,obj_end=2,subj_type='PERSON',obj_type='ORGANIZATION',relation='per:employee_of') for i in range(2)]
    for name in ('train','dev','test'):atomic_json(source/(name+'.json'),rows)
    atomic_json(source/'data_manifest.json',{'synthetic':True,'splits':{'synthetic':{'records':2}}})
    assets=tmp_path/'assets.json';atomic_json(assets,{})
    run=tmp_path/'run';device='cuda:0' if torch.cuda.is_available() else 'cpu'
    assert worker.main(['--model','sequence_routing_h3','--assets',str(assets),'--data-dir',str(source),
        '--run-dir',str(run),'--device',device,'--diagnostic','--diagnostic-updates','2'])==0
    summary=json.loads((run/'run_summary.json').read_text())
    assert summary['diagnostic'] and summary['updates']==2
    assert set(summary['splits'])=={'valid','test'}
    assert {t['phase'] for t in summary['case_study']['traces']}=={'train','test'}
    for event in summary['case_study']['traces']:
        trace=json.loads((run/'case_study'/event['path']).read_text())
        assert (run/'case_study'/trace['tensor_artifact']).is_file()


def test_real_parent_child_cuda_oom_restarts_from_committed_update(tmp_path,monkeypatch):
    import torch,textwrap,sys
    if not torch.cuda.is_available():pytest.skip('requires bounded CUDA diagnostic')
    from baselines.retacred import reference_complete as suite
    from baselines.retacred.common import atomic_json
    source=tmp_path/'inputs';source.mkdir()
    rows=[dict(id='bounded-'+str(i),token=['Alice','joined','Acme','.'],subj_start=0,subj_end=0,
        obj_start=2,obj_end=2,subj_type='PERSON',obj_type='ORGANIZATION',relation='per:employee_of') for i in range(2)]
    for name in ('train','dev','test'):atomic_json(source/(name+'.json'),rows)
    atomic_json(source/'data_manifest.json',{'synthetic':True,'splits':{'synthetic':{'records':2}}})
    assets=tmp_path/'assets.json';atomic_json(assets,{})
    root=tmp_path/'suite';root.mkdir()
    job=dict(id='sequence_routing_h3_seed13',model='sequence_routing_h3',seed=13,directory=root/'task')
    # The fixture only substitutes a tiny model/budget and requests diagnostic
    # mode. Production argv, parser, locks, execution wrapper, train/save/load,
    # parent process handling and GPU isolation all execute normally.
    code=textwrap.dedent('''
        import copy,runpy,sys,torch
        from transformers import RobertaConfig
        from baselines.retacred import reference_models as models,reference_recipes as recipes
        from baselines.retacred.common import ROOT
        frozen=recipes.recipe('sequence_routing_h3')
        recipes.recipe=lambda _:dict(frozen,epochs=1,logical_batch=1,selection='final_epoch')
        def make(*args,device,checkpointing,**kwargs):
            config=RobertaConfig(vocab_size=50265,hidden_size=16,num_hidden_layers=2,
                num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,pad_token_id=1)
            evidence=ROOT/'.external/retacred/reference-cohort-completion-20261009T'
            adapter=models.SequenceRouting(evidence/'tokenizer',ROOT/'.external/reference_sources/sequence_routing',
                evidence/'resources',device=device,checkpointing=checkpointing,config=config)
            forward=adapter.forward;calls=[0]
            def observed(batch):
                if not checkpointing and calls[0]==1:
                    # A real CUDA allocation fails after update1 is committed.
                    oversized=torch.cuda.get_device_properties(0).total_memory+2**30
                    torch.empty(oversized,dtype=torch.uint8,device=device)
                calls[0]+=1
                return forward(batch)
            adapter.forward=observed
            return adapter
        models.make=make
        sys.argv=['reference_worker',*sys.argv[1:]]
        runpy.run_module('baselines.retacred.reference_worker',run_name='__main__')
    ''')
    original=suite.command
    def command(j,args,a,data):
        argv=original(j,args,a,data)
        return [sys.executable,'-u','-c',code,*argv[4:],'--diagnostic','--diagnostic-updates','2']
    def validate(path):
        summary=json.loads((path/'run_summary.json').read_text())
        assert summary['diagnostic'] and summary['updates']==2
        assert {e['phase'] for e in summary['case_study']['traces']}=={'train','test'}
        return summary
    monkeypatch.delenv('QATTN_REFERENCE_STORAGE_ROOT',raising=False)
    monkeypatch.setattr(suite,'command',command);monkeypatch.setattr(suite,'validate_task',validate)
    original_sleep=suite.time.sleep;monkeypatch.setattr(suite.time,'sleep',lambda _:original_sleep(.05))
    args=SimpleNamespace(run_dir=root,micro_batch='auto',checkpoint_seconds=120.)
    assert suite.queue([job],['0'],args,assets,source)==0
    state=json.loads(execution.ledger_path(job['directory']).read_text())
    assert state['index']==1 and len(state['events'])==1
    checkpoint=torch.load(job['directory']/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert checkpoint['state']['cursor']==2
    summary=validate(job['directory'])
    event=next(t for t in summary['case_study']['traces'] if t['phase']=='train')
    trace=json.loads((job['directory']/'case_study'/event['path']).read_text())
    assert trace['optimizer_update']==1
    log=(root/'logs'/f"{job['id']}.log").read_text()
    assert '[OOM] tier 0 -> 1' in log
