"""Actual execution controls, pinned source upgrade and rejection boundaries."""
import copy
import io
import json
from types import SimpleNamespace

import pytest

from baselines.retacred import reference_execution as ex,execution_compat as compat
from baselines.retacred.common import atomic_json,fingerprint


def arguments(run,micro='auto'):
    return SimpleNamespace(run_dir=run,micro_batch=micro,seed=13,
        checkpoint_seconds=120.,physical_sentence_cap=256)


@pytest.fixture
def historical(tmp_path,monkeypatch):
    monkeypatch.delenv(compat.ORIGIN_ENV,raising=False)
    manifest=tmp_path/'run_manifest.json'
    atomic_json(manifest,{'source':compat.binding()['origin_source']})
    compat.activate_suite(manifest)
    return tmp_path


def legacy(run,index=8,micro='auto'):
    policy=dict(schema='reference-execution-tiers.v1',model='ctl_drp_supervised',seed=13,
        requested_micro=micro,tiers=ex.tiers('ctl_drp_supervised',256,micro,linear=False),
        checkpoint_seconds=120.,guarantee='full_state_restore; tier_changes_not_bitwise_identical',
        logical_batch=512,per_rank_sentence_cap=256)
    state=dict(policy=policy,index=index,events=[dict(from_tier=i,to_tier=i+1,trigger='cuda_oom') for i in range(index)])
    atomic_json(ex.ledger_path(run),state)
    return copy.deepcopy(state)


def test_fixed_absolute_steps_then_all_small_widths():
    table=ex.tiers('ctl_drp_supervised',256)
    widths=[r['micro'] for r in table[:-1]]
    assert widths[:3]==[256,248,240]
    assert all(a-b==8 for a,b in zip(widths,widths[1:]) if a>16)
    assert widths[-16:]==list(range(16,0,-1))
    assert len(widths)==len(set(widths))
    assert table[-1]==dict(micro=1,checkpointing=True)
    assert [r['micro'] for r in ex.tiers('ptr',32,'5')]==[5,4,3,2,1,1]


def test_exact_release_binding_and_unknown_source_rejection(tmp_path,monkeypatch):
    monkeypatch.delenv(compat.ORIGIN_ENV,raising=False)
    compat.verify_release()
    source=copy.deepcopy(compat.binding()['origin_source']);source['ctl_drp_neural.py']='0'*64
    manifest=tmp_path/'run_manifest.json';atomic_json(manifest,dict(source=source))
    with pytest.raises(ValueError,match='pinned a90af7d'):compat.activate_suite(manifest)
    assert not compat.active()


def test_no_blanket_source_or_model_bypass(historical,monkeypatch):
    from baselines.retacred import reference_complete as suite
    current=suite.identity();current['ctl_drp_neural.py']='0'*64
    monkeypatch.setattr(suite,'identity',lambda:current)
    with pytest.raises(ValueError,match='scientific or unapproved'):compat.verify_release()


def test_upgrade_probe_once_preserves_original_policy_and_monotonic_tier(historical):
    import torch
    run=historical/'job';before=legacy(run)
    args=arguments(run);state=ex.prepare(args,'ctl_drp_supervised',512)
    assert args.micro_batch==2 and not args.activation_checkpointing
    assert state['policy']==before['policy'] and state['index']==8
    assert compat.source_map()['ctl_drp_neural.py']==compat.binding()['origin_source']['ctl_drp_neural.py']
    args=arguments(run)
    def failed():raise torch.cuda.OutOfMemoryError('actual CUDA OOM category')
    assert ex.adaptive_cli(args,'ctl_drp_supervised',512,failed)==76
    state=json.loads(ex.ledger_path(run).read_text())
    assert state['index']==9 and state['linear_upgrade']['cursor']==1
    assert state['events'][-1]['from_micro']==2 and state['events'][-1]['to_micro']==1
    restored=arguments(run);ex.prepare(restored,'ctl_drp_supervised',512)
    assert restored.micro_batch==1  # Repeated resumes never repeat the +1 probe.
    with pytest.raises(ValueError,match='ordinary'):
        ex.adaptive_cli(arguments(run),'ctl_drp_supervised',512,lambda:(_ for _ in ()).throw(ValueError('ordinary')))
    assert json.loads(ex.ledger_path(run).read_text())['index']==9


def test_explicit_cap_and_recomputation_are_not_silently_relaxed(historical):
    run=historical/'explicit';before=legacy(run,index=0,micro='1')
    args=arguments(run,micro='1');ex.prepare(args,'ctl_drp_supervised',512)
    assert args.micro_batch==1 and args.execution_policy==before['policy']
    run=historical/'recompute';legacy(run,index=9)
    args=arguments(run);ex.prepare(args,'ctl_drp_supervised',512)
    assert args.micro_batch==1 and args.activation_checkpointing


@pytest.mark.parametrize('field,value',[('seed',29),('checkpoint_seconds',10.),('physical_sentence_cap',171)])
def test_historical_scientific_options_still_rejected(historical,field,value):
    run=historical/field;legacy(run)
    args=arguments(run);setattr(args,field,value)
    with pytest.raises(ValueError,match='policy drift'):ex.prepare(args,'ctl_drp_supervised',512)


def test_mixed_parent_accepts_exactly_one_transition_and_keeps_rank_order(historical,monkeypatch):
    import torch
    from baselines.retacred import reference_complete as suite,reference_mixed_queue as mixed,reference_archive
    run=historical/'job';legacy(run)
    log=historical/'logs/job.command.json';log.parent.mkdir()
    atomic_json(log,dict(gpu_ids=['2','3'],world_size=2))
    launches=[]
    class Process:
        def __init__(self,argv,**kwargs):
            launches.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            opts=arguments(run)
            if len(launches)==1:
                self.rc=ex.adaptive_cli(opts,'ctl_drp_supervised',512,
                    lambda:(_ for _ in ()).throw(torch.cuda.OutOfMemoryError('test')))
            else:
                ex.prepare(opts,'ctl_drp_supervised',512);assert opts.micro_batch==1;self.rc=0
        def poll(self):return self.rc
        def wait(self):return self.rc
    monkeypatch.delenv('QATTN_REFERENCE_STORAGE_ROOT',raising=False)
    monkeypatch.setattr(suite,'command',lambda *_:['fixture'])
    monkeypatch.setattr(suite,'validate_task',lambda *_:{})
    monkeypatch.setattr(reference_archive,'archive_completed',lambda *_:None)
    monkeypatch.setattr(mixed.subprocess,'Popen',Process)
    monkeypatch.setattr(mixed.time,'sleep',lambda *_:None)
    args=SimpleNamespace(run_dir=historical,ctl_gpus_per_task='auto',resume=historical)
    job=dict(id='job',model='ctl_drp_supervised',seed=13,directory=run)
    assert mixed.queue([job],['0','1','2','3'],args,None,None)==0
    assert launches==['2,3','2,3']


def test_completed_archive_origin_is_preserved(historical):
    from baselines.retacred.reference_archive import source_identity
    assert source_identity()==fingerprint(compat.binding()['origin_source'])
    compat.record_upgrade(historical)
    receipt=json.loads((historical/'execution_upgrade.json').read_text())
    assert receipt['executing_source']!=receipt['origin_source']
    compat.record_upgrade(historical)


def test_dashboard_distinguishes_end_allocation_from_update_peak(monkeypatch):
    from baselines.retacred import reference_terminal as terminal
    monkeypatch.setattr(terminal.shutil,'get_terminal_size',lambda *_:SimpleNamespace(columns=260))
    stream=io.StringIO();gib=1024**3
    terminal.Dashboard(['0'],1,stream=stream).render([], [dict(gpu='0',task='CTL',batch=3,batches=10,
        memory_allocated_bytes=6.67*gib,memory_peak_allocated_bytes=14*gib,
        memory_peak_reserved_bytes=15*gib,memory_total_bytes=16*gib)], [])
    text=stream.getvalue();assert 'VRAM 6.67G peak 14.00/16.0G reserve 15.00G' in text
