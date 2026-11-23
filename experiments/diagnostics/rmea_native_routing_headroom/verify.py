"""Independent aggregate recomputation and Cholesky-based real-fixture check."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--reference',type=Path,required=True);args=ap.parse_args();root=args.output
    summary=json.loads((root/'summary.json').read_bytes());seal=json.loads((root/'RUN_COMPLETE').read_bytes())
    for n,h in seal['files'].items():assert sha(root/n)==h
    q=dict(np.load(root/'diagnostic.private.npz',allow_pickle=False))
    prior=dict(np.load(args.reference/'output/diagnostic.private.npz',allow_pickle=False))
    z=q['logits']; b=q['spectral']; w=q['replay']; c=q['convex'];y=q['labels'];cal=q['calibration'];tol=q['tolerance']
    assert z.shape==(8192,40) and b.shape==w.shape==c.shape==(8192,40,2)
    assert all(np.isfinite(v).all() for v in q.values() if v.dtype.kind in 'biufc')
    assert np.array_equal(y,prior['labels']) and np.array_equal(cal,prior['calibration'])
    assert np.array_equal(q['groups'],prior['groups'])
    assert not set(q['groups'][cal])&set(q['groups'][~cal])
    assert np.all(np.max(np.abs(z-prior['logits']),-1)<tol)
    assert np.array_equal(z.argmax(-1),prior['logits'].argmax(-1))
    assert np.all(np.max(np.abs(b-w),axis=(1,2))<tol)
    assert np.all(np.max(b[:,:,0]-z,-1)<tol) and np.all(np.max(z-b[:,:,1],-1)<tol)
    assert np.all(np.max(c[:,:,0]-b[:,:,0],-1)<tol)
    assert np.all(np.max(b[:,:,1]-c[:,:,1],-1)<tol)
    partitions=dict(all=np.ones(8192,bool),calibration=cal,assessment=~cal)
    pred=np.argmax(z,axis=1);correct=pred==y
    recomputed={}
    for partition,keep in partitions.items():
        masks=dict(all=keep,native_errors=keep&~correct,
                   positive_to_zero=keep&(y!=0)&(pred==0),
                   zero_to_positive=keep&(y==0)&(pred!=0),
                   wrong_positive=keep&(y!=0)&(pred!=0)&~correct)
        table={}
        for name,mask in masks.items():
            row={'n':int(mask.sum())}
            for family in ('born_all_rows','born_class0_fixed','born_true_row_only','convex_all_rows'):
                counts=[0,0,0]
                for i in np.flatnonzero(mask):
                    target=int(y[i]);other=[r for r in range(40) if r!=target]
                    if family=='convex_all_rows':
                        estimate=c[i,target,1]-max(c[i,r,0] for r in other);witness=estimate
                    elif family=='born_true_row_only':
                        estimate=b[i,target,1]-max(z[i,r] for r in other)
                        witness=w[i,target,1]-max(z[i,r] for r in other)
                    else:
                        fixed=family=='born_class0_fixed'
                        hi=z[i,0] if fixed and target==0 else b[i,target,1]
                        whi=z[i,0] if fixed and target==0 else w[i,target,1]
                        estimate=hi-max(z[i,0] if fixed and r==0 else b[i,r,0] for r in other)
                        witness=whi-max(z[i,0] if fixed and r==0 else w[i,r,0] for r in other)
                    index=0 if witness>tol[i] else 1 if estimate < -tol[i] else 2
                    counts[index]+=1
                keys=('positive_margin','negative_numerical_margin_not_certificate','unresolved') if family=='convex_all_rows' else ('explicit_witness_positive','negative_numerical_margin_not_certificate','unresolved')
                row[family]=dict(zip(keys,counts))
            table[name]=row
        recomputed[partition]=table
    assert recomputed==summary['partitions']
    assert recomputed['assessment']['native_errors']['n']==478
    # Independent matrix construction: explicit ket-bra token loop and
    # Cholesky congruence instead of production eigendecomposition whitening.
    fixtures=dict(np.load(root/'algebra-fixtures.private.npz',allow_pickle=False));maxerr=0.
    native_error=0.;eps=1e-6
    for i in range(8):
        ps=fixtures[f'{i}_psi'];u=fixtures[f'{i}_utilities'];phi=fixtures[f'{i}_phi']
        bias=fixtures[f'{i}_bias'];n=len(ps)
        rho=np.array([np.outer(v,v.conj()) for v in ps]);s=sum(rho)/n+eps*np.eye(8)
        l=np.linalg.cholesky(s);inv=np.linalg.solve(l,np.eye(8))
        for r in range(40):
            br=sum(u[r,t]*rho[t] for t in range(n))/n+eps*float(u[r].mean())*np.eye(8)
            transformed=inv@br@inv.conj().T
            eig=np.linalg.eigvalsh((transformed+transformed.conj().T)/2)
            exact=np.array([eig[0],eig[-1]])+bias[r]
            maxerr=max(maxerr,float(np.max(np.abs(exact-b[i,r]))))
            weights=np.array([abs(np.vdot(phi[r],v))**2+eps for v in ps])
            logits=bias[r]+np.dot(weights,u[r])/sum(weights)
            native_error=max(native_error,abs(logits-z[i,r]))
            directions=fixtures[f'{i}_extremizer_phi'][r]
            assert directions.shape==(8,2)
            for endpoint in (0,1):
                weights=np.array([abs(np.vdot(directions[:,endpoint],v))**2+eps for v in ps])
                witness=bias[r]+np.dot(weights,u[r])/sum(weights)
                assert abs(witness-w[i,r,endpoint]) < tol[i]
        assert maxerr < max(tol[:i+1]) and native_error < max(tol[:i+1])
    result=dict(status='PASS',all_partition_tables_recomputed=True,all8192_shapes_finite_prior_split_and_seal_verified=True,
                independent_first8_cholesky_extreme_error=maxerr,independent_first8_direct_native_error=native_error,
                exact_impossibility_certification=False,optimizer_updates=summary['optimizer_updates'],
                seconds=summary['seconds'],diagnostic_sha256=sha(root/'diagnostic.private.npz'))
    assert result['optimizer_updates']==0
    print(json.dumps(result))


if __name__=='__main__':main()
