from pathlib import Path
from types import SimpleNamespace
import pytest
from baselines.retacred.ctl_drp_ddp import partition
from baselines.retacred.reference_mixed_queue import allocation,group_size
from baselines.retacred.reference_complete import command


@pytest.mark.parametrize('size',[2,3,6])
@pytest.mark.parametrize('count',[1,3,511,512])
def test_partition_preserves_every_sentence_and_sample(count,size):
    indices=list(range(count));parts=partition(indices,size,selected=count-1)
    assert sorted(i for p in parts for i in p)==indices
    assert count-1 in parts[0]


def test_six_gpu_atomic_two_two_one_one():
    devices=list(map(str,range(6)));width=group_size('auto',devices)
    models=['ctl_drp_supervised','ctl_drp_unsupervised','ptr','sequence_routing_h3']
    groups=[]
    for model in models:
        g=allocation({'model':model},devices,width);groups.append(g)
        devices=[d for d in devices if d not in g]
    assert groups==[['0','1'],['2','3'],['4'],['5']]
    assert group_size('auto',['0'])==1
    assert group_size('auto',['0','1','2'])==2
    assert allocation({'model':'ctl_drp_supervised'},['2'],2) is None
    assert allocation({'model':'ctl_drp_supervised','retry_gpu_ids':['0','1']},['1','2'],2) is None
    with pytest.raises(ValueError):group_size('3',['0','1'])


def test_exact_ddp_parent_argv(tmp_path):
    args=SimpleNamespace(micro_batch='auto',checkpoint_seconds=120,review_receipt=tmp_path/'review')
    job=dict(model='ctl_drp_supervised',seed=13,directory=tmp_path/'task',gpu_ids=['2','4'])
    argv=command(job,args,tmp_path/'assets',tmp_path/'data')
    assert argv[3]=='baselines.retacred.ctl_drp_ddp'
    assert argv[4:7]==['--world-size','2','--']
    assert '--ctl-worker' in argv and '--micro-batch' in argv
    assert 'baselines.retacred.reference_complete' not in argv[7:]
    job['gpu_ids']=['2'];assert command(job,args,None,None)[3]=='baselines.retacred.reference_complete'


def test_oom_tiers_start_at_actual_rank_capacity_without_duplicate_levels(tmp_path):
    from baselines.retacred.reference_execution import prepare
    args=SimpleNamespace(run_dir=tmp_path/'task',micro_batch='auto',seed=13,
        checkpoint_seconds=120.,physical_sentence_cap=256)
    state=prepare(args,'ctl_drp_supervised',512)
    assert args.micro_batch==256
    assert state['policy']['logical_batch']==512
    assert state['policy']['tiers'][:2]==[dict(micro=256,checkpointing=False),dict(micro=248,checkpointing=False)]


def test_resume_keeps_original_group_when_an_earlier_task_is_already_done(tmp_path,monkeypatch):
    from baselines.retacred import reference_complete as suite,reference_archive,reference_mixed_queue
    from baselines.retacred.common import atomic_json
    job=dict(id='remaining_CTL',model='ctl_drp_unsupervised',seed=13,directory=tmp_path/'task')
    log=tmp_path/'logs/remaining_CTL.command.json';log.parent.mkdir()
    atomic_json(log,dict(gpu_ids=['2','3'],world_size=2))
    launched=[]
    class Process:
        def __init__(self,argv,**kwargs):launched.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.delenv('QATTN_REFERENCE_STORAGE_ROOT',raising=False)
    monkeypatch.setattr(suite,'command',lambda *_:['bounded-fixture'])
    monkeypatch.setattr(suite,'validate_task',lambda *_:{})
    monkeypatch.setattr(reference_archive,'archive_completed',lambda *_:None)
    monkeypatch.setattr(reference_mixed_queue.subprocess,'Popen',Process)
    monkeypatch.setattr(reference_mixed_queue.time,'sleep',lambda *_:None)
    args=SimpleNamespace(run_dir=tmp_path,ctl_gpus_per_task='auto',resume=tmp_path)
    assert reference_mixed_queue.queue([job],['0','1','2','3'],args,None,None)==0
    assert launched==['2,3']


def test_ddp_dashboard_counts_tasks_instead_of_ranks():
    import io
    from baselines.retacred.reference_terminal import Dashboard
    stream=io.StringIO()
    rows=[dict(gpu=str(i),task='CTL' if i<2 else 'single',world_size=2 if i<2 else 1,rank=i if i<2 else 0) for i in range(3)]
    Dashboard(['0','1','2'],15,stream=stream).render([],rows,[])
    assert '2 running' in stream.getvalue() and 'DDP1/2' in stream.getvalue() and 'DDP2/2' in stream.getvalue()
