"""Five frozen Q and real heads: original validation development, no new test pass."""
import argparse,hashlib,importlib.util,json,os,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'rmea_original_scores'))
import job as original
from common import guard,sha,read,complete
from filter import route,projector,unit_rows,density
spec=importlib.util.spec_from_file_location('r66_probe',HERE.parent/'rmea_evidence_dispersion/job.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def measured(y,p):
    assert y.shape==p.shape and len(y)>0
    cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,p),1)
    tp=np.diag(cm)[1:];den=cm.sum(0)[1:]+cm.sum(1)[1:]
    f=np.divide(2*tp,den,out=np.zeros(39,np.float64),where=den>0)
    return dict(items=len(y),macro_f1=float(f.mean()),accuracy=float((y==p).mean()),confusion=cm.tolist())


def changes(y,base,new):
    return dict(changed=int((base!=new).sum()),
        correct_positive_recoveries=int(((y>0)&(base==0)&(new==y)).sum()),
        newly_lost_correct_positives=int(((y>0)&(base==y)&(new!=y)).sum()),
        new_true_null_false_positives=int(((y==0)&(base==0)&(new!=0)).sum()),
        repaired_true_null_false_positives=int(((y==0)&(base!=0)&(new==0)).sum()),
        new_wrong_positive_labels=int(((y>0)&(new>0)&(new!=y)&(new!=base)).sum()))


def state_vectors(torch,head,h,roles,detail,kind):
    if kind=='quantum':return detail['psi'].numpy(),detail['phi'].numpy()
    keys=head.key(torch.cat((h,roles),-1))
    # Match native F.normalize; route handles degenerate zero norms explicitly.
    return torch.nn.functional.normalize(keys,dim=-1).numpy(),torch.nn.functional.normalize(head.query,dim=-1).numpy()


def apply(torch,head,detail,states,effects,roles,mask):
    arrays={k:[] for k in ['identity','protected','unprotected']};stats=[]
    for i in range(len(states)):
        a,s=route(states[i],effects,roles[i],mask[i],detail['alpha'][i].numpy())
        for k in arrays:arrays[k].append(a[k])
        stats.append(s)
    scores={k:(torch.from_numpy(np.stack(v))*detail['utilities']).sum(-1).add(head.bias).numpy() for k,v in arrays.items()}
    return scores,stats


def unit(contract):
    rng=np.random.default_rng(20261118)
    states=np.eye(8,dtype=np.complex128)[[0,1,2,3]]
    effects=unit_rows(rng.normal(size=(40,8))+1j*rng.normal(size=(40,8)))
    roles=np.zeros((4,4));roles[0,2]=1;roles[1,3]=1;mask=np.ones(4,bool)
    from filter import attention
    alpha=attention(states,effects,mask)
    result,stats=route(states,effects,roles,mask,alpha)
    assert stats['rank']==2 and stats['reason']=='ok'
    assert stats['protected_mass_delta_mean']>=-1e-9
    assert not np.array_equal(result['protected'],result['identity'])
    missing=roles.copy();missing[:,3]=0
    r,s=route(states,effects,missing,mask,alpha)
    assert s['reason']=='missing_role' and np.array_equal(r['protected'],alpha)
    r,s=route(states,effects,roles,np.array([1,1,0,0],bool),alpha)
    assert s['reason']=='no_context' and np.array_equal(r['unprotected'],alpha)
    near=states.copy();near[1]=unit_rows((states[0]+1e-6*states[1])[None])[0]
    _,reason,_=projector(near,roles,mask);assert reason=='union_rank_boundary'
    # Exact tied role densities protect the entire eight-dimensional space.
    tie=np.tile(np.eye(8,dtype=np.complex128),(2,1));rr=np.zeros((16,4));rr[:8,2]=1;rr[8:,3]=1
    P,reason,s=projector(tie,rr,np.ones(16,bool));assert reason=='ok' and s['rank']==8 and np.allclose(P,np.eye(8),atol=1e-12)
    # Unitary basis covariance: no phase convention or eigenvector ordering dependence.
    U=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))[0]
    rotated=states@U.T;rot_effects=effects@U.T
    r,s=route(rotated,rot_effects,roles,mask,alpha)
    assert np.allclose(r['protected'],result['protected'],atol=2e-7,rtol=2e-6)
    torch,Head=original.torch_setup(contract['original_contract']);torch.manual_seed(20261118)
    with torch.inference_mode():
        for kind in contract['heads']:
            head=Head(kind).eval().requires_grad_(False);h=torch.randn(2,4,1024)
            r=torch.from_numpy(np.tile(roles,(2,1,1)).astype(np.float32));m=torch.ones(2,4,dtype=torch.bool)
            d=head(h,r,m,details=True);v,e=state_vectors(torch,head,h,r,d,kind)
            outputs,_=apply(torch,head,d,v,e,r.numpy(),m.numpy())
            assert np.allclose(outputs['identity'],d['logits'].numpy(),atol=5e-5,rtol=3e-4)
    return dict(status='PASS_UNIT',checks=['protected_mass_and_filter_spectrum','missing_role_identity','no_context_identity',
        'near_parallel_union_fallback','full_top_eigenspace_ties','unitary_basis_covariance','quantum_and_real_native_identity'],head_training=False,probe_fit=False,cos_requests=0)


def run(args,c,preflight):
    torch,Head=original.torch_setup(c['original_contract']);data,expected=original.pinned_inputs(args,c['original_contract'])
    data=data['valid'];metadata=read(args.inputs/'assets/shared/cache/metadata.json')['valid']
    hashes=[hashlib.sha256(('rmea-dispersion-probe-v1:'+x['id_hash']).encode()).hexdigest() for x in metadata]
    assert len(set(hashes))==len(hashes)
    evaluate=np.array([int(x[:8],16)%3==0 for x in hashes]);cells=[];anchor_errors={};started=time.monotonic()
    with torch.inference_mode():
        for seed in c['seeds']:
            for kind in c['heads']:
                head=Head(kind).eval().requires_grad_(False)
                head.load_state_dict(torch.load(args.inputs/f'assets/seed{seed}/checkpoints/{kind}-selected.pt',map_location='cpu',weights_only=True),strict=True)
                anchor_errors[f'{seed}:{kind}']=original.anchors(torch,head,args.inputs,seed,kind,c['original_contract'])
                if preflight:continue
                collected={k:[] for k in ['baseline','identity','protected','unprotected']};gold=[];reason=Counter();ranks=Counter();deltas=[];l1=[]
                for start in range(0,len(data['targets']),64):
                    h,roles,mask,y=original.batch(torch,data,start,64);detail=head(h,roles,mask,details=True)
                    v,e=state_vectors(torch,head,h,roles,detail,kind)
                    output,stats=apply(torch,head,detail,v,e,roles.numpy(),mask.numpy())
                    collected['baseline'].append(detail['logits'].numpy());gold.append(y)
                    for k,z in output.items():collected[k].append(z)
                    for s in stats:
                        reason[s['reason']]+=1
                        if s['rank'] is not None:
                            ranks[str(s['rank'])]+=1;deltas.append(s['protected_mass_delta_mean']);l1.append(s['attention_l1_mean'])
                arrays={k:np.concatenate(v) for k,v in collected.items()};Y=np.concatenate(gold);S=arrays['baseline'];base=S.argmax(1)
                counts=original.count_batch(S,Y);ref=expected['cells'][f'{seed}:{kind}:valid']
                assert counts['total']==ref['items'] and counts['confusion']==ref['confusion'],'Original confusion bridge mismatch'
                assert np.allclose(arrays['identity'],S,atol=5e-5,rtol=3e-4),'Numerical identity changed logits beyond frozen tolerance'
                assert np.array_equal(arrays['identity'].argmax(1),base),'Numerical identity changed predictions'
                null=base==0;fit=null&~evaluate;test=null&evaluate;assert fit.sum()>=100 and test.sum()>=100
                F=np.column_stack((S.astype(np.float64),S.astype(np.float64)**2))
                q,fitted=probe.fit_predict(F[fit],(Y[fit]>0).astype(np.float64),F[test],.01)
                score_control=base.copy();score_control[test]=np.where(q>.5,S[test,1:].argmax(1)+1,0)
                predictions={k:z.argmax(1) for k,z in arrays.items()};predictions['score_quadratic']=score_control
                # Every policy has the same ALL-example evaluation denominator.
                metrics={k:measured(Y[evaluate],p[evaluate]) for k,p in predictions.items()}
                cost={k:changes(Y[evaluate],base[evaluate],p[evaluate]) for k,p in predictions.items() if k!='baseline'}
                deltas_f1={k:m['macro_f1']-metrics['baseline']['macro_f1'] for k,m in metrics.items() if k!='baseline'}
                cell=dict(seed=seed,head=kind,original_confusion_exact=True,identity_argmax_exact=True,
                    identity_max_abs=float(abs(arrays['identity']-S).max()),evaluation=metrics,changes=cost,macro_deltas=deltas_f1,
                    score_probe_fit=fitted,score_probe_binary=probe.metrics((Y[test]>0).astype(float),q),
                    routing=dict(reasons=dict(reason),rank_histogram=dict(ranks),protected_mass_delta_mean=float(np.mean(deltas)) if deltas else None,
                        attention_l1_mean=float(np.mean(l1)) if l1 else None),
                    protected_beats_both=metrics['protected']['macro_f1']>max(metrics['unprotected']['macro_f1'],metrics['score_quadratic']['macro_f1'],metrics['baseline']['macro_f1']))
                cells.append(cell);print(json.dumps(dict(seed=seed,head=kind,macro_deltas=deltas_f1,protected_beats_both=cell['protected_beats_both'])),flush=True)
    if preflight:return dict(status='PASS_FROZEN_INPUT_PREFLIGHT',heads=10,anchors=anchor_errors,anchor_count=40,head_training=False,probe_fit=False)
    qcells=[x for x in cells if x['head']=='quantum'];qualified=all(x['protected_beats_both'] for x in qcells)
    return dict(schema='qattn.role-protected.development.v1',status='PASS_ROLE_FILTER_DIAGNOSTIC',cells=cells,anchors=anchor_errors,
        criterion='ALL5_PROTECTED_BEATS_CONTROLS' if qualified else 'FIXED_INTERVENTION_CRITERION_NOT_MET',
        head_training=False,probe_fit=True,original_test_evaluation=False,cos_requests=0,elapsed_seconds=time.monotonic()-started,
        candidate='P1-ROLE-PROTECTED-NUISANCE-FILTER',L5='NOT_ACCEPTED',
        limits=['Original heads selected on whole validation; evaluation fold is development only, shared across seeds.',
            'No new full test predictions. Existing two test anchors per head only.',
            'Protected and unprotected filters share all numerical/eligibility fallbacks. Identity control isolates unit-normalization/float64 rewrite.',
            'Real-overlap uses its own trained head: comparator of generic filtering benefit, not isolated quantum attribution.',
            'Input-dependent effects may be outside original circuit image; no preparation, hardware, novelty or inherited L4 claim.',
            'Failure closes this exact fixed-head intervention, not all trainable variants. No coefficient, rank-rule or threshold search.'])


def main():
    guard();p=argparse.ArgumentParser();p.add_argument('--action',choices=['unit','preflight','diagnose'],required=True)
    for name in ['inputs','plan','expected','preflight-acceptance']:p.add_argument('--'+name,type=Path)
    p.add_argument('--preflight-acceptance-sha256');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    c=read(HERE/'contract.json')
    for name,digest in c['dependency_sources'].items():assert sha(ROOT/name)==digest
    assert c['original_contract']==read(HERE.parent/'rmea_original_scores/contract.json')
    if args.action=='diagnose':
        assert sha(args.preflight_acceptance)==args.preflight_acceptance_sha256
        a=read(args.preflight_acceptance)
        assert a['status']=='ACCEPTED_ORIGINAL_INPUT_PREFLIGHT_ONLY' and a['contract_sha256']==sha(HERE/'contract.json')
        assert a['task_state']=='complete' and a['quiescent'] and a['returncode']==0
        assert a['owner_binding_exact_restored'] and a['worker_binding_exact_restored']
        assert a['report']['status']=='PASS_FROZEN_INPUT_PREFLIGHT' and a['report']['anchor_count']==40
    result=unit(c) if args.action=='unit' else run(args,c,args.action=='preflight')
    result.update(action=args.action,contract_sha256=sha(HERE/'contract.json'),environment=c['environment'],
        python_execution=dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ))
    complete(args.output,args.action,result);print(json.dumps(dict(status=result['status'],action=args.action,head_training=False)))
if __name__=='__main__':main()
