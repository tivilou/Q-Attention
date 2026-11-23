"""Independently check the new group gradients, directions and primary gate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--reference',type=Path,required=True)
    a=ap.parse_args();r=a.output;cfg=json.loads((r.parent/'design.json').read_bytes())
    s=json.loads((r/'summary.json').read_bytes());seal=json.loads((r/'RUN_COMPLETE').read_bytes())
    for n,h in seal['files'].items():assert sha(r/n)==h
    assert s['optimizer_updates']==0 and sha(r.parent/'design.json')==s['design_sha256']
    cache=Path(cfg['completed_direction_control'])/'output/diagnostic.private.npz';assert sha(cache)==cfg['completed_direction_sha256']
    old=dict(np.load(cache,allow_pickle=False));q=dict(np.load(r/'diagnostic.private.npz',allow_pickle=False))
    assert all(np.isfinite(v).all() for v in q.values() if v.dtype.kind in 'biufc')
    assert all(np.array_equal(q[k],old[k]) for k in ('logits','labels','calibration','groups'))
    assert q['logit_derivative'].shape==(3,8192,40) and q['directions'].shape==(3,40,2,3,2)
    assert np.array_equal(q['logit_derivative'][0],old['logit_derivative'])
    assert np.array_equal(q['directions'],np.load(r/'directions.private.npy',allow_pickle=False))
    assert s['directions_frozen']['directions_sha256']==sha(r/'directions.private.npy')
    assert s['directions_frozen']['chosen_at']<=s['assessment_started_at']
    y=q['labels'];cal=q['calibration'];counts=[int((cal&(y==0)).sum()),int((cal&(y!=0)).sum()),int((~cal&(y==0)).sum()),int((~cal&(y!=0)).sum())]
    assert counts==cfg['group_counts'] and not set(q['groups'][cal])&set(q['groups'][~cal])
    g0=q['gcal_zero'];gp=q['gcal_positive'];ga0=q['gassessment_zero'];gap=q['gassessment_positive']
    assert max(np.max(np.abs((counts[0]*g0+counts[1]*gp)/4192-old['gcal'])),np.max(np.abs((counts[2]*ga0+counts[3]*gap)/4000-old['gassessment'])))<1e-12
    # Independently minimize the scalar quadratic over endpoints and its
    # stationary point, then compare the selected direction and lambda.
    delta=g0-gp;den=float(np.dot(delta.ravel(),delta.ravel()));scale=max(1.,float(np.vdot(g0,g0)),float(np.vdot(gp,gp)))
    if den<=1e-24*scale:lam=None;common=np.zeros_like(g0)
    else:
        stationary=float(np.clip(-np.dot(gp.ravel(),delta.ravel())/den,0.,1.))
        candidates=[0.,1.,stationary]
        lam=min(candidates,key=lambda x:np.linalg.norm(x*g0+(1-x)*gp)**2)
        common=lam*g0+(1-lam)*gp
    recorded=s['selection']['lambda_class0']
    assert (recorded is None and lam is None) or abs(recorded-lam)<1e-10
    raw=[old['gcal'],(g0+gp)/2,common];z=q['logits']
    prob=np.exp(z-np.logaddexp.reduce(z,axis=1)[:,None]);prob[np.arange(8192),y]-=1
    maxerr=0.;worst={}
    for j,name in enumerate(cfg['modes']):
        norm=np.linalg.norm(raw[j]);valid=norm>1e-12 and (name!='common' or lam is not None)
        assert bool(valid)==s['selection']['directions'][name]['valid']
        if not valid:
            assert s['results'][name]['status']=='unresolved' and not np.any(q['directions'][j]);continue
        assert np.max(np.abs(q['directions'][j]+raw[j]/norm))<1e-12
        ce=np.sum(prob*q['logit_derivative'][j],axis=1)
        for part,keep,grads in (('calibration',cal,(g0,gp)),('assessment',~cal,(ga0,gap))):
            means=[]
            for iszero,g,key in zip((True,False),grads,('class0','positive')):
                mask=keep&((y==0) if iszero else (y!=0));mean=float(ce[mask].mean());inner=float(np.vdot(g,q['directions'][j]))
                maxerr=max(maxerr,abs(mean-inner));assert abs(mean-inner)<1e-9
                assert abs(mean-s['results'][name]['groups'][part][key])<1e-10;means.append(mean)
            assert abs(max(means)-s['results'][name]['groups'][part]['worst'])<1e-10
            if part=='assessment':worst[name]=max(means)
        if name=='common':
            assert max(float(np.vdot(g0,q['directions'][j])),float(np.vdot(gp,q['directions'][j])))<=-norm+1e-9*max(1.,norm)
    tau=1e-8*max(1.,float(np.linalg.norm(ga0)),float(np.linalg.norm(gap)))
    if 'common' not in worst or 'balanced' not in worst:gate='UNRESOLVED_CLOSE_EXACT_PROFILE'
    else:gate='PASS_LOCAL_DIRECTION_ONLY' if worst['common'] < -tau and worst['balanced']-worst['common']>tau else 'FAIL_CLOSE_EXACT_PROFILE'
    assert gate==s['gate'] and abs(tau-s['assessment_threshold'])<1e-12
    print(json.dumps(dict(status='PASS',gate=gate,group_gradient_jvp_max_error=maxerr,
        cached_frequency_direction_and_jvp_reused=True,all_group_counts_seals_selection_and_primary_recomputed=True,
        descriptive_tables_reuse='Previously accepted native tables helper; primary/group fields independently recomputed',
        optimizer_updates=0,diagnostic_sha256=sha(r/'diagnostic.private.npz'))))


if __name__=='__main__':main()
