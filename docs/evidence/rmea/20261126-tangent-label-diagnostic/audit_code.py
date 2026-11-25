"""CPU audit of already-produced private arrays; no task cache/head/checkpoint."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
root=Path('/hy-tmp/Q-Attention/control/tangent-label-diagnostic-20261126-v2');out=root/'output'
summary=json.loads((out/'summary.json').read_bytes());trace=json.loads((out/'sample-trace.v2.private.json').read_bytes())
expected=json.load(sys.stdin)
for n,row in expected.items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==row['sha256']
names=['g','variance','self','upper'];audited={};eta={};case=[];trace_values=0
with np.load(out/'population.private.npz',allow_pickle=False) as z:
    for split in ('fit','assessment'):
        scores=z[split+'_scores'];logits=z[split+'_logits'];y=z[split+'_target'];n=len(y)
        correct=logits.argmax(1)==y;delta=np.empty((n,4));tol=1e-10*np.maximum(1,np.abs(scores).max(1))
        # Scalar loop provides an independent rival and per-row indexing check.
        for i in range(n):
            rival=max((c for c in range(40) if c!=y[i]),key=lambda c:(logits[i,c],-c))
            assert rival==int(z[split+'_rival'][i]);delta[i]=scores[i,y[i]]-scores[i,rival]
        np.testing.assert_allclose(delta,z[split+'_delta'],atol=0,rtol=0)
        for k,name in enumerate(names):
            row=summary['metrics'][split][name]
            for label,take in [('all',np.ones(n,bool)),('error',~correct),('correct',correct)]:
                expected_row=row[label];a=delta[take,k];t=tol[take,k]
                assert int(take.sum())==expected_row['n']
                assert int((a>t).sum())==expected_row['positive_count']
                assert int((a<-t).sum())==expected_row['negative_count']
            for c in range(40):
                take=y==c;a=delta[take,k];t=tol[take,k];rowc=row['per_gold_class'][c]
                assert int(take.sum())==rowc['n'] and int((a>t).sum())==rowc['positive_count'] and int((a<-t).sum())==rowc['negative_count']
        upper=scores[:,:,3];g=scores[:,:,0];defined=upper>0;ratio=g[defined]/upper[defined]
        assert not np.any(g[~defined]>1e-8) and ratio.min()>=0 and ratio.max()<=1+1e-8
        eta[split]=dict(defined=int(defined.sum()),undefined=int((~defined).sum()),minimum=float(ratio.min()),maximum=float(ratio.max()),mean=float(ratio.mean()))
        cm=np.bincount(40*y+logits.argmax(1),minlength=1600).reshape(40,40);den=cm.sum(0)+cm.sum(1)
        f1=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den>0)
        pair={}
        for k,name in enumerate(names[1:],1):
            both=2*(delta[:,0]>tol[:,0])+(delta[:,k]>tol[:,k])
            pair[name]={g:np.bincount(both[take].astype(int),minlength=4).tolist() for g,take in [('error',~correct),('correct',correct)]}
        audited[split]=dict(rows=n,errors=int((~correct).sum()),correct=int(correct.sum()),original_native_all40_macro_f1=float(f1.mean()),paired_positive_codes=pair)
    for sample in trace['samples']:
        split=sample['split'];index=sample['roles']['source_sample']['cache_index'];pos=np.flatnonzero(z[split+'_indices']==index);assert len(pos)==1;pos=int(pos[0])
        ref=sample['representations'][0]['artifact_refs'][0]
        with np.load(out/ref['path'],allow_pickle=False) as a:
            assert int(a['target'][0])==z[split+'_target'][pos];np.testing.assert_array_equal(a['logits'],z[split+'_logits'][pos])
            for k,name in enumerate(names):np.testing.assert_array_equal(a[name],z[split+'_scores'][pos,:,k])
            psi,phi,u=a['psi'],a['phi'],a['utilities'];p=abs(phi.conj()@psi.T)**2;raw=p+1e-6;alpha=raw/raw.sum(1,keepdims=True)
            np.testing.assert_allclose(p,a['probability'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(alpha,a['alpha'],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose((alpha*u).sum(1)+a['bias'],a['logits'],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(a['values']@a['decoder'].T,u.T,atol=1e-10,rtol=1e-10)
            assert a['h'].shape[0]==sample['roles']['source_sample']['metadata']['words']
            trace_values+=sum(a[k].size for k in a.files)
            gold=int(a['target'][0]);rival=int(z[split+'_rival'][pos]);pred=int(a['logits'].argmax());
            case.append(dict(split=split,ordinal=int(sample['sample_id'].split(':')[-1]),words=len(psi),gold=gold,prediction=pred,rival=rival,
                original_correct=bool(pred==gold),gold_minus_rival_slope={name:float(a[name][gold]-a[name][rival]) for name in names},
                raw_tokens_available='tokens' in sample['roles']['source_sample']['metadata']))
g=summary['metrics']['assessment']['g'];best=max(summary['metrics']['assessment'][k]['error']['positive_fraction'] for k in names[1:])
adv=g['error']['positive_fraction']-best
assert abs(adv-summary['error_advantage'])<1e-15 and adv<.02 and summary['decision']=='STOP_CURRENT_LABEL_DIRECTION'
assert summary['native_head_forwards']==192 and summary['native_batched_circuit_calls']==384 and summary['optimizer_updates']==0
print(json.dumps(dict(status='PASS_SAVED_OUTPUT_AUDIT',audited=audited,eta=eta,cases=case,trace_elements=trace_values,
    source_linked_trace_cases=len(case),error_advantage=adv,decision=summary['decision'],raw_tokens_available_cases=sum(c['raw_tokens_available'] for c in case),
    new_native_calls=0,new_feature_or_checkpoint_reads=0,new_label_cache_reads=0,private_arrays_downloaded=False),allow_nan=False))
