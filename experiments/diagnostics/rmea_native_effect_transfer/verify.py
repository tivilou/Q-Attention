"""Independent immutable-result verification of direction, CE and margin tables."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--reference',type=Path,required=True)
    a=ap.parse_args();r=a.output;s=json.loads((r/'summary.json').read_bytes());seal=json.loads((r/'RUN_COMPLETE').read_bytes())
    for n,h in seal['files'].items():assert digest(r/n)==h
    assert s['optimizer_updates']==0
    if s['status']=='no_direction':
        assert s['gradient_norm']<=1e-12
        print(json.dumps(dict(status='PASS',no_direction=True,optimizer_updates=0)));return
    q=dict(np.load(r/'diagnostic.private.npz',allow_pickle=False))
    assert all(np.isfinite(v).all() for v in q.values() if v.dtype.kind in 'biufc')
    z=q['logits'];dz=q['logit_derivative'];y=q['labels'];cal=q['calibration'];pred=z.argmax(-1)
    assert z.shape==dz.shape==(8192,40) and int(cal.sum())==4192
    prior=dict(np.load(a.reference/'output/diagnostic.private.npz',allow_pickle=False))
    assert np.array_equal(y,prior['labels']) and np.array_equal(cal,prior['calibration'])
    assert np.array_equal(q['groups'],prior['groups']) and not set(q['groups'][cal])&set(q['groups'][~cal])
    assert np.array_equal(pred,prior['logits'].argmax(-1)) and np.max(np.abs(z-prior['logits']))<1e-8
    gc=q['gcal'];ga=q['gassessment'];d=q['direction'];norm=np.linalg.norm(gc)
    assert gc.shape==ga.shape==d.shape==(40,2,3,2)
    assert np.max(np.abs(d+gc/norm))<1e-12 and abs(np.linalg.norm(d)-1)<1e-12
    assert np.array_equal(np.load(r/'direction.private.npy',allow_pickle=False),d)
    assert s['direction_frozen']['direction_sha256']==digest(r/'direction.private.npy')
    assert s['direction_frozen']['chosen_at']<=s['assessment_started_at']
    coeff=np.exp(z-np.logaddexp.reduce(z,axis=1)[:,None]);coeff[np.arange(8192),y]-=1
    ce=np.einsum('ir,ir->i',coeff,dz);margin=[]
    for i in range(8192):
        competitors=[j for j in range(40) if j!=int(y[i])];best=max(z[i,j] for j in competitors)
        tied=[j for j in competitors if z[i,j]==best]
        margin.append(dz[i,y[i]]-max(dz[i,j] for j in tied))
    margin=np.array(margin)
    assert np.max(np.abs(ce-q['ce_derivative']))<1e-10
    assert np.max(np.abs(margin-q['margin_derivative']))<1e-10
    rowmean=float(ce[~cal].mean());inner=float(np.vdot(ga,d));cal_error=abs(float(ce[cal].mean())+norm)
    assert abs(inner-rowmean)<1e-9 and cal_error<1e-9
    assert abs(rowmean-s['primary_assessment_ce_derivative'])<1e-10
    tau=1e-8*max(1.,float(np.linalg.norm(ga)))
    sign='negative' if rowmean < -tau else 'positive' if rowmean>tau else 'unresolved'
    assert sign==s['primary_sign'] and abs(tau-s['primary_numerical_threshold'])<1e-12
    for part,keep in dict(all=np.ones(8192,bool),calibration=cal,assessment=~cal).items():
        masks=dict(all=keep,native_errors=keep&(pred!=y),native_correct=keep&(pred==y),
            positive_to_zero=keep&(y!=0)&(pred==0),zero_to_positive=keep&(y==0)&(pred!=0),
            wrong_positive=keep&(pred!=y)&(y!=0)&(pred!=0))
        tolerance=1e-8*np.maximum(1,np.max(np.abs(dz),axis=1))
        for name,m in masks.items():
            row=s['partitions'][part][name]
            assert row['n']==int(m.sum())
            if not m.any():
                assert row['ce_directional_derivative'] is None and row['margin_derivative_mean'] is None
                assert row['margin_increases']==row['margin_decreases']==row['margin_unresolved']==0
                continue
            assert abs(row['ce_directional_derivative']-float(ce[m].mean()))<1e-10
            assert abs(row['margin_derivative_mean']-float(margin[m].mean()))<1e-10
            assert row['margin_increases']==int((m&(margin>tolerance)).sum())
            assert row['margin_decreases']==int((m&(margin < -tolerance)).sum())
            assert row['margin_unresolved']==int((m&(np.abs(margin)<=tolerance)).sum())
    print(json.dumps(dict(status='PASS',optimizer_updates=0,primary_sign=sign,
        assessment_ce_derivative=rowmean,gradient_inner_product_error=abs(inner-rowmean),
        calibration_identity_error=cal_error,all8192_tables_prior_split_and_seals_recomputed=True,
        direction_is_calibration_only_rule=True,diagnostic_sha256=digest(r/'diagnostic.private.npz'))))


if __name__=='__main__':main()
