"""Conditional cost breakpoints for unchanged C/Gcov, no CPU-to-GPU projection."""
import itertools
import hashlib
import json
from pathlib import Path
import sys

ARMS=('phase_presence_q','phase_presence_block_d','phase_presence_real16','gaussian_covariance')
PINS={
    'terminal':'2b55e6c7e0aeea5da72a5520697e9ccfa775cbe641c4309ba5482d51debb5afa',
    'sensitivity':'9f270c80ca95e873a28fc7e0c75c7cbc0dffcc73a9262b4b8ee488955b28748a',
    'profile':'0b2da30bf1c0ca6b8809220c2fa72b598835c6e37c84b40c3c610ee4281e7e25',
}

def load(p):
    return json.loads(Path(p).read_bytes())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fits_at_share(weights, counts, membership, deadline):
    cap=1.
    for bit in (0,1):
        base=sum(v for v,s,x in zip(weights,membership,counts) if s==bit and not x)
        scaled=sum(v for v,s,x in zip(weights,membership,counts) if s==bit and x)
        if base>deadline:return None
        if scaled:cap=min(cap,(deadline-base)/scaled)
    return cap


def breakpoints(weights,gpu,wall):
    q,d,c,g=weights
    effective=min(gpu,wall)
    one=None if c+g>effective else min(1.,(effective-c-g)/(q+d))
    device=min(gpu/2,wall)
    # Native ordered two-lane fit dispatch starts Q and D. D is cheaper,
    # so starts C. Q completes before D+C for all common Q/D ratios 0..1.
    assert 0<q-d<c
    fixed=(0,1,1,0)
    shares=[fits_at_share(weights,(1,1,0,0),bits,device) for bits in itertools.product((0,1),repeat=4)]
    ideal=max((s for s in shares if s is not None),default=None)
    pattern=fits_at_share(weights,(1,1,0,0),fixed,device)
    return dict(single_lane_max_share=one,fixed_two_lane_max_share=pattern,optimistic_two_lane_max_share=ideal,
                required_reductions_percent=[100*(1-x) if x is not None else None for x in (one,pattern,ideal)],
                scopes_feasible_even_if_qd_free=[v is not None for v in (one,pattern,ideal)])


if __name__=='__main__':
    t_path,z_path,raw_path,profile_path=sys.argv[1:5]
    assert dict(terminal=sha(t_path),sensitivity=sha(z_path),profile=sha(profile_path))==PINS
    t,z,raw,profile=map(load,(t_path,z_path,raw_path,profile_path))
    assert t['status']=='VERIFIED_SINGLE_TERMINAL_NATIVE_CANARY' and t['classification']=='DEFER_MEASURED_COST_ENVELOPE'
    assert t['allocation']=='released' and t['generation']==130 and t['science_updates']==0
    assert t['pinned_sha256']['summary']==sha(raw_path) and raw['profile_sha256']==PINS['profile']
    p=t['projection'];assert p==raw['projection'] and profile['data']['sizes']==dict(fit=4096,selection=1024,qualification=2048)
    assert profile['training']['max_updates_per_arm']==p['fit_updates_per_arm']==640
    assert profile['training']['logical_batch']==128 and profile['training']['checkpoint_selection_updates']==[160,320,480,640]
    assert p['regular_score_batches_per_arm']==48 and p['fixed_Q_batches_each']==16
    assert set(p['per_arm_base_seconds'])==set(raw['results'])==set(z['arm_max_host_delta_s'])==set(ARMS)
    max_gpu=profile['budget']['max_new_gpu_seconds_all_arms_and_evaluations']
    max_wall=profile['budget']['max_campaign_elapsed_seconds']
    assert round(max_gpu-t['cumulative_reserved_gpu_seconds'],3)==z['budget']['gpu_s']==t['remaining_candidate_gpu_seconds']
    assert abs(max_wall-t['request_to_release_seconds']-z['budget']['wall_s'])<1e-9
    u=[raw['results'][a]['update_seconds'] for a in ARMS];excess=[z['arm_max_host_delta_s'][a] for a in ARMS]
    shape=z['scale_if_entire_runtime_proportional_to_padding'];sc=p['uncertainty_multiplier'];es=raw['eval_seconds_fit_batch_only']
    sealed=[sc*p['per_arm_base_seconds'][a] for a in ARMS]
    generous=[sc*(640*max(y-x,0.)*shape['fit']+es[a]*(32*shape['selection']+16*shape['qualification'])+
                  (16*shape['qualification']*es['block_D'] if i==0 else 0.))
              for i,(a,x,y) in enumerate(zip(ARMS,excess,u))]
    assert abs((sum(generous)-z['budget']['gpu_s'])-z['min_unexplained_serial_gap_even_optimistic'])<1e-7
    budget=z['budget'];s=breakpoints(sealed,budget['gpu_s'],budget['wall_s']);g=breakpoints(generous,budget['gpu_s'],budget['wall_s'])
    assert breakpoints([5.,4.,2.,2.],4.,4.)['single_lane_max_share']==0.
    assert breakpoints([5.,4.,2.,2.],3.,3.)['single_lane_max_share'] is None
    print(json.dumps(dict(status='PASS_CANDIDATE_REQUIRED_GPU_REDUCTION_ONLY',sealed=s,previous_optimistic_sensitivity=g,
                         pinned_sources=PINS,arm_order=ARMS,cohort_sizes=profile['data']['sizes'],updates_per_arm=640,
                         measured_new_gpu_time=False,scientific_training_admitted=False,
                         scope='Common Q/D reduction applied to entire estimated Q/D arm work. C and Gcov held fixed. Excludes further delay/charge; also optimistic about separate stage barriers. Diagnostics only.'),indent=2))
