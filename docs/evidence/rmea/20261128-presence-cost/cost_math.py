"""Read sealed aggregate timing and produce source-bounded sensitivity arithmetic."""
from pathlib import Path
import hashlib
import itertools
import json
import statistics
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_bytes())


def near(a,b):
    if abs(a-b)>1e-8:
        raise AssertionError((a,b))


def lanes(cost):
    vals=list(cost.values())
    fixed=[0.,0.]
    for n in vals:
        fixed[min(range(2),key=fixed.__getitem__)]+=n
    ideal=min(max(sum(v for v,t in zip(vals,s) if t),sum(v for v,t in zip(vals,s) if not t))
              for s in itertools.product((False,True),repeat=len(vals)))
    return {'single_gpu_s':sum(vals),'fixed_two_gpu_allocated_s':2*max(fixed),
            'ideal_two_gpu_allocated_s':2*ideal,
            'ideal_two_gpu_wall_s':ideal,'fixed_two_gpu_wall_s':max(fixed)}


def calc(t,raw,x,f):
    p=t['projection']; assert raw['projection']==p and raw['updates']==4
    assert x['source_hashes']==f['source_hashes']
    assert all(x[k]==0 for k in ('new_task_forwards','new_optimizer_updates','new_gpu_allocations'))
    assert f['planned_repetitions']==len(x['alternating_repetitions'])==7
    a=list(p['per_arm_base_seconds'])
    ug={k:raw['results'][k]['update_seconds'] for k in a}; es=raw['eval_seconds_fit_batch_only']
    assert set(es)==set(a)|{'block_D'}
    sh=x['fit_and_score_shapes']; ratio={k:sh[k]['mean_padded_sites_per_batch']/sh['first_fit']['mean_padded_sites_per_batch'] for k in ('fit','selection','qualification')}
    overhead={}
    for k in a:
        m=16*es['block_D'] if k==a[0] else 0.
        overhead[k]=(p['per_arm_base_seconds'][k]-640*ug[k]-48*es[k]-m)/640
    h=statistics.mean(overhead.values())
    for v in overhead.values():near(v,h)
    assert h>0
    margin=p['uncertainty_multiplier']; assert margin==1.1
    def scenario(delta,scale_train,scale_eval,keep_overhead):
        fit=ratio['fit'] if scale_train else 1.
        s=ratio['selection'] if scale_eval else 1.
        q=ratio['qualification'] if scale_eval else 1.
        flow={k:640*(max(0.,ug[k]-delta[i])*fit+(h if keep_overhead else 0.))+
                es[k]*(32*s+16*q)+(16*q*es['block_D'] if i==0 else 0.) for i,k in enumerate(a)}
        return lanes({k:v*margin for k,v in flow.items()})
    ds=[[v['crash_safe_34']-v['atomic_1'] for v in rep] for rep in x['alternating_repetitions']]
    peak=[max(row[i] for row in ds) for i in range(4)]
    mids=[statistics.median(row[i] for row in ds) for i in range(4)]
    base=scenario([0.]*4,False,False,True)
    near(base['single_gpu_s'],p['single_gpu_compute_seconds']); near(base['fixed_two_gpu_allocated_s'],p['two_gpu_allocated_compute_gpu_seconds'])
    scenarios={'sealed':base,'journal_delta_medians_only':scenario(mids,False,False,True),
               'largest_observed_journal_delta_each_arm_and_length_proportional':scenario(peak,True,True,True),
               'same_optimistic_case_with_zero_collation_checkpoint_proxy':scenario(peak,True,True,False)}
    cap=t['remaining_candidate_gpu_seconds'];wall=t['remaining_candidate_wall_seconds']
    assert cap==3048.543 and wall<2400
    for v in scenarios.values():
        v.update(single_passes=v['single_gpu_s']<=min(wall,cap), even_ideal_two_passes=v['ideal_two_gpu_wall_s']<=wall and v['ideal_two_gpu_allocated_s']<=cap)
    actual=x['delta_seconds']
    near(statistics.median(sum(row) for row in ds),actual['median'])
    zero=scenarios['same_optimistic_case_with_zero_collation_checkpoint_proxy']
    assert not any(s[key] for s in scenarios.values() for key in ('single_passes','even_ideal_two_passes'))
    return dict(status='COST_DEFER_RETAINED_BY_SOURCE_AND_HOST_DIAGNOSTIC',arm_order=a,
        measured_seconds_for_timed_canary_update=ug,timed_canary_journal_calls_per_arm=34,
        measured_joint_host_journal_delta_s=actual,arm_max_host_delta_s=dict(zip(a,peak)),
        fixed_proxy_s_per_step=h,budget=dict(gpu_s=cap,wall_s=wall),
        scale_if_entire_runtime_proportional_to_padding=ratio,scenarios=scenarios,
        min_unexplained_serial_gap_even_optimistic=zero['single_gpu_s']-cap,
        min_unexplained_two_gpu_gap_even_optimistic=zero['ideal_two_gpu_allocated_s']-cap,
        scope='Same-volume synthetic journal and padded sites only. Scenario subtraction and proportional scaling are generous sensitivity cases, not validated wall-time forecasts or lower bounds. CUDA wait, cold-start, overlap, and task performance remain unmeasured.',
        new_task_forwards=0,new_optimizer_updates=0,new_gpu_allocations=0,
        old_charge_preserved=True,scientific_training_admitted=False)


if __name__=='__main__':
    terminal,raw,host,frozen=map(Path,sys.argv[1:5])
    t=load(terminal);assert t['pinned_sha256']['summary']==sha(raw)
    print(json.dumps(calc(t,load(raw),load(host),load(frozen)),indent=2,allow_nan=False))
