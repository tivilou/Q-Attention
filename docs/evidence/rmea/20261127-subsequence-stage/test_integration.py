"""Actual selection hook and admission/gating test with no model computation."""
from pathlib import Path
import hashlib,json,tempfile,subprocess,sys
import numpy as np
import goal_aligned_stage as stage
import goal_aligned_runner as runner
H=Path(__file__).resolve().parent
def expect_failure(fn,kind=AssertionError):
    try:fn()
    except kind:return
    raise AssertionError('Negative control unexpectedly accepted')

class FakeStatefulHead:
    def __init__(self):self.training=True;self.step=0
    def state_dict(self):return {'step':self.step}
    def eval(self):self.training=False
    def train(self,training):self.training=training

with tempfile.TemporaryDirectory(prefix='qattn-stage-integration-') as tmp:
    f=Path(tmp);models={k:FakeStatefulHead() for k in runner.TAGS}
    dirs={k:f/k for k in models}
    trace=[];after=[]
    def process(model,mode,step):
        assert not model.training and model.state_dict()=={'step':step}
        assert (dirs[mode]/f'selection-{step}.begin.json').exists()
        assert stage.sha(dirs[mode]/f'selection-{step}.private.pt')==hashlib.sha256(str(step).encode()).hexdigest()
        trace.append((mode,step))
        c=np.eye(40)/40
        return dict(weighted_macro_f1=.8 if step in (320,480) else .6,weighted_confusion=c.tolist()),'fixture-predictions'
    hook=stage.attach_selection_progress(lambda m,s,**v:after.append((m,s,v['step'])),models=models,score_selection=process,folders=dirs,contract_sha='fixture',atomic_torch=lambda p,v:p.write_text(str(v['step'])),cpu_tree=lambda x:x,state_identity=lambda x:x['step'])
    for m,model in models.items():
        for i in range(1,641):
            model.step=i;hook(m,'training',step=i);assert model.training
        rows=stage.SelectionJournal(dirs[m],m,'fixture').receipts(require_complete=True)
        assert stage.choose(rows)['step']==320
        expect_failure(lambda:hook(m,'training',step=640))
    assert trace==[(m,i) for m in runner.TAGS for i in stage.CHECKPOINTS]
    assert len(after)==2560

q,*controls=runner.TAGS
curves={k:{480:.5,640:.5} for k in runner.TAGS}
good=dict(point_deltas={k:.02 for k in controls},minimum_delta_quantiles_05_95=[.01,.03])
gaps=dict(path_D=.006,terminal_D=.006)
assert runner.gate_decision(good,curves,q,controls,gaps)[1]=='promote_full_single_seed'
assert runner.gate_decision(good,curves,q,controls,dict(path_D=.01,terminal_D=0))[1]=='stop_exact_configuration'
assert runner.gate_decision(good,curves,q,controls,dict(path_D=0,terminal_D=.01))[1]=='stop_exact_configuration'
curve2={k:dict(v) for k,v in curves.items()};curve2[q][480]=.4
assert runner.gate_decision(good,curve2,q,controls,gaps)[1]=='defer_uncertain_or_cost'
assert runner.gate_decision({**good,'minimum_delta_quantiles_05_95':[-.001,.03]},curves,q,controls,gaps)[1]=='defer_uncertain_or_cost'
bind={'order_sha256':'order'}
rec=dict(status='ADMITTED_MEASURED_GOAL_ALIGNED_SCREEN',attempt='attempt',spec_sha256='spec',stage_bundle_sha256='scope',cohort_sha256=runner.COHORT,order_sha256='order',technical_updates=4,previous_campaign_technical_updates=4,source_review_verdict='PASS',semantic_preflight='PASS',native_observability_preflight='PASS',measured_forecast='PASS',existing_gpu_count=2,first_request_at=100,science_request_at=200,technical_allocated_gpu_seconds=100,max_science_elapsed_seconds=1600,predicted_science_gpu_seconds=3000,predicted_science_elapsed_seconds=1450,reserved_closeout_seconds=40)
kw=dict(spec_sha='spec',stage_bundle_sha='scope',attempt='attempt',devices=['0','1'],spec_count=2,now=250)
assert runner.require_admission(rec,bind,**kw)==1510
for key,v in [('source_review_verdict','PASS_WITH_RISK'),('native_observability_preflight','NOT_RUN'),('measured_forecast','NOT_RUN'),('attempt','wrong'),('stage_bundle_sha256','wrong'),('predicted_science_gpu_seconds',3500),('technical_updates',5)]:
    expect_failure(lambda:runner.require_admission({**rec,key:v},bind,**kw))
expect_failure(lambda:runner.require_admission(rec,bind,**{**kw,'now':1801}),TimeoutError)
expect_failure(lambda:runner.require_admission({**rec,'predicted_science_elapsed_seconds':1550},bind,**kw),TimeoutError)
expect_failure(lambda:runner.require_admission({**rec,'max_science_elapsed_seconds':100},bind,**kw),TimeoutError)
expect_failure(lambda:runner.require_admission({**rec,'predicted_science_gpu_seconds':3040},bind,**kw),TimeoutError)
expect_failure(lambda:runner.require_admission(rec,bind,**{**kw,'now':270}),TimeoutError)
expect_failure(lambda:runner.require_admission(rec,bind,**{**kw,'spec_count':1}),ValueError)
expect_failure(lambda:runner.require_admission(rec,bind,**{**kw,'devices':['0','0']}),ValueError)
assert runner.require_device_count(rec,2,['0','1'],2)==2
expect_failure(lambda:runner.require_device_count(rec,2,['0','1'],1),ValueError)
o=subprocess.run([sys.executable,'-X','utf8','-B','-O',str(H/'goal_aligned_runner.py'),'--help'],capture_output=True,text=True)
assert o.returncode!=0 and 'Unoptimized Python required' in o.stderr
with tempfile.TemporaryDirectory(prefix='qattn-claimed-') as tmp:
    root=Path(tmp);claim=runner.claim_once(root/'claim',dict(attempt='one',output='out-a',admission_sha256='a'))
    expect_failure(lambda:runner.claim_once(root/'claim',dict(attempt='two',output='out-b',admission_sha256='b')),FileExistsError)
    (root/'qualification.private.npz').write_bytes(b'stub')
    (root/'manifest.json').write_text(json.dumps(dict(admission_sha256='a',campaign_claim_sha256=claim)))
    (root/'summary.json').write_text(json.dumps(dict(admission_sha256='a',qualification_sha256=runner.digest(root/'qualification.private.npz'))))
    (root/'all_fits_selection_frozen.json').write_text('{}')
    chain=runner.complete_chain(root,'a',claim)
    assert chain['admission_sha256']=='a' and len(chain['artifacts_sha256'])==4
    expect_failure(lambda:runner.complete_chain(root,'b',claim))
    (root/'qualification.private.npz').write_bytes(b'changed')
    expect_failure(lambda:runner.complete_chain(root,'a',claim))

result=dict(status='PASS_CURRENT_STAGE_INTEGRATION_HOOK_AND_SEAL',mock_committed_update_notifications=2560,
    actual_selection_hook_passes=16,duplicate_selection_rejected=True,earliest_exact_ties=True,
    independent_ablation_gates=True,selection_stability_and_interval_gates=True,missing_or_mismatched_admission_rejected=True,
    optimize_mode_at_entry_rejected=True,late_forecast_rejected=True,cross_output_campaign_claim_rejected=True,
    changed_qualification_and_admission_rejected=True,one_card_spec_and_short_CUDA_inventory_rejected=True,
    reviewed_source_stage_sha256=stage.sha(H/'goal_aligned_stage.py'),runner_sha256=stage.sha(H/'goal_aligned_runner.py'),
    real_model_created=False,new_forwards=0,new_optimizer_updates=0)
if '--copy-stdout-only' not in sys.argv:
    with (H/'integration-tests-r3.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
