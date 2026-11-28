"""Synthetic coverage of the changed prescription and lifecycle arithmetic."""
import json
from pathlib import Path
import numpy as np

import goal_aligned_runner as r
from presence_endpoints import DIRECTION_GATES, task_errors, task_prediction

q,*controls=r.TAGS
y=np.array([0,0,1,1,2,3,4,39]); w=np.array([.20,.20,.05,.05,.1,.1,.1,.2])
z=np.zeros((8,40));z[1,1]=1;z[2,1]=1;z[3,0]=2;z[3,1]=1;z[4,2]=1;z[5,1:5]=1
z[6,0]=2;z[6,1:4]=1;z[7,39]=1
a=task_errors(z,y,w)
assert [a[k]['count'] for k in ('null_false_positive','positive_correct','positive_top3','positive_to_null','positive_to_wrong_positive')]==[1,3,5,2,1]
assert abs(a['positive_correct']['weighted_mass']-.35)<1e-14
assert abs(a['positive_top3']['weighted_mass']-.5)<1e-14
better=z.copy();better[1,1]=0;better=task_errors(better,y,w)
test={k:a for k in controls};test[q]=better
t=task_prediction(test,q,controls);assert all(all(v.values()) for v in t.values())
interval={'point_deltas':{c:.011 for c in controls},'minimum_delta_quantiles_05_95':[.001,.020]}
curves={k:{480:.4,640:.43} for k in r.TAGS};g={'block_D':.005}
assert r.gate_decision(interval,curves,q,controls,g,t)[1]=='promote_full_single_seed'
assert r.gate_decision(interval,curves,q,controls,{'block_D':.004999},t)[1]=='stop_exact_configuration'
for k in DIRECTION_GATES:
    t[controls[0]][k]=False
    assert r.gate_decision(interval,curves,q,controls,g,t)[1]=='stop_exact_configuration'
    t[controls[0]][k]=True
curves[q][480]=.39
assert r.gate_decision(interval,curves,q,controls,g,t)[1]=='defer_uncertain_or_cost'
for false in ({},float('nan')):
    try:
        if isinstance(false,dict):r.gate_decision(interval,curves,q,controls,g,false)
        else:r.gate_decision(interval,curves,q,controls,{'block_D':false},t)
    except AssertionError:pass
    else:raise AssertionError('Malformed adjudication was accepted')

binding={'order_sha256':'order'}
args=dict(spec_sha='spec',stage_bundle_sha='bundle',attempt='once',devices=['existing'],spec_count=1,now=1100.)
record=dict(status='ADMITTED_MEASURED_GOAL_ALIGNED_SCREEN',attempt='once',spec_sha256='spec',stage_bundle_sha256='bundle',
    cohort_sha256=r.COHORT,order_sha256='order',technical_updates=4,previous_campaign_technical_updates=8,prior_reserved_gpu_seconds=343.895,
    source_review_verdict='PASS',source_review_required_fixes=[],semantic_preflight='PASS',native_observability_preflight='PASS',measured_forecast='PASS',
    existing_gpu_count=1,first_request_at=1000.,science_request_at=1090.,technical_allocated_gpu_seconds=100.,
    max_science_elapsed_seconds=800.,predicted_science_gpu_seconds=600.,predicted_science_elapsed_seconds=600.,reserved_closeout_seconds=10.)
assert r.require_admission(record,binding,**args)==780
for mutation in (dict(previous_campaign_technical_updates=4),dict(source_review_required_fixes=['required']),dict(prior_reserved_gpu_seconds=0.),dict(predicted_science_gpu_seconds=3157.)):
    try:r.require_admission({**record,**mutation},binding,**args)
    except AssertionError:pass
    else:raise AssertionError('Bad receipt accepted')
for bad in (dict(spec_count=2),dict(devices=['x','x'])):
    try:r.require_admission(record,binding,**{**args,**bad})
    except ValueError:pass
    else:raise AssertionError('Bad device lane receipt accepted')
assert r.digest(Path(__file__).with_name('profile.json'))==r.PROFILE
print(json.dumps(dict(status='PASS_STAGE_DIRECTIONS_TIES_AND_ADMISSION_GUARDS',task_forwards=0,optimizer_updates=0,gpu_allocations=0,checks=17,runner_sha256=r.digest(Path(r.__file__)))))
