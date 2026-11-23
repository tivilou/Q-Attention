"""R0: observe fixed complete-model positive/null readout on the sealed VALID32."""
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import json
import sys
import time
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def save(p,v):
    q=p.with_suffix('.pending');q.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());q.replace(p)


def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();r=a.root;cfg=json.loads(a.design.read_bytes())
    assert not sys.flags.optimize and sha(__file__)==cfg['code_sha256']
    for n,h in cfg['input_hashes'].items():assert sha(r/n)==h,n
    assert not a.output.exists();a.output.mkdir(parents=True)
    save(a.output/'started.json',dict(attempt=a.attempt,design_sha256=sha(a.design)));start=time.monotonic()
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available(),'CPU allocation must hide all CUDA devices'
    sys.path.insert(0,str(r/'source'));from baselines.retacred.rmea import RMEAHead
    base=r/'control/full-model-error-structure-20261123/semantic'
    truth=json.loads((base/'selection.private.json').read_bytes());assert len(truth)==32
    sem={q['case_id']:q for q in json.loads((base/'semantic-results.private.json').read_bytes())}
    words=json.loads((r/'inputs/native-word-metadata-20261123/valid.private.json').read_bytes())
    tm=json.loads((r/'control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
    vm=json.loads((r/'control/task-bottleneck-discovery-20261123/validation-cases.private.json').read_bytes())['metadata']
    labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
    with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as z:fit=z['calibration'].copy()
    support=defaultdict(set);seen=set()
    for i in np.flatnonzero(fit):g,st,ot=tm[i];support[st,ot].add(g);seen.add(g)
    ss=hashlib.sha256(json.dumps(sorted((list(k),sorted(v)) for k,v in support.items()),separators=(',',':')).encode()).hexdigest()
    assert ss==cfg['frozen_support_sha256']
    cache=r/'inputs/native-baseline/discovery/cache';data={k:np.load(cache/('valid-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    provenance=json.loads((r/'control/historical-type-replay-20261123/model-provenance.json').read_bytes())
    rows=[];arrays={};parity=identity=decomposition=0.;cursor=0
    tol=cfg['numeric_tolerance']
    with torch.inference_mode():
        for mode in cfg['modes']:
            prior=r/'control/historical-type-replay-20261123';model=provenance['models'][mode]
            state=torch.load(prior/model['file'],map_location='cpu',weights_only=True)
            assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
            head=RMEAHead(mode).float();head.load_state_dict(state,strict=True);head.eval()
            previous=np.load(prior/'output'/(mode+'-logits.private.npy'),mmap_mode='r',allow_pickle=False)
            for rec in truth:
                i=rec['row'];case=rec['case_id'];w=words[i];g=int(data['targets'][i]);assert labels[g]==rec['gold'] and vm[i][0]==g
                left,right=map(int,data['offsets'][i:i+2]);h=np.array(data['h'][left:right],np.float32);roles=np.array(data['roles'][left:right],np.float32)
                assert len(h)==len(w['tokens'])
                for span,col in ((w['subject'],2),(w['object'],3)):
                    expected=np.zeros(len(h),bool);expected[slice(*span)]=True;assert np.array_equal(expected,roles[:,col].astype(bool))
                d=head(torch.from_numpy(h)[None],torch.from_numpy(roles)[None],torch.ones((1,len(h)),dtype=torch.bool),details=True)
                z=d['logits'][0].numpy().astype(float);alpha=d['alpha'][0].numpy().astype(float);u=d['utilities'][0].numpy().astype(float);bias=head.bias.numpy().astype(float)
                err=float(np.max(np.abs(z-previous[i])));parity=max(parity,err);assert err<tol,(mode,case,'parity',err)
                err=float(np.max(np.abs((alpha*u).sum(-1)+bias-z)));identity=max(identity,err);assert err<tol
                allow=np.ones(40,bool);pair=tuple(vm[i][1:])
                if pair in support:
                    allow[:]=False;allow[0]=True
                    for k in support[pair]|(set(range(40))-seen):allow[k]=True
                prediction=int(np.where(allow,z,-np.inf).argmax());old_prediction=int(np.where(allow,previous[i],-np.inf).argmax());assert prediction==old_prediction
                if mode=='quantum':assert labels[prediction]==rec['predicted']
                target=g if g>0 else int(np.where(allow[1:],previous[i,1:],-np.inf).argmax()+1)
                rival=z.copy();rival[~allow]=-np.inf;rival[target]=-np.inf;best=float(rival.max())
                s=sem[case];cue=np.array(s['evidence_intersection'],dtype=int)
                eligible=bool(g>0 and s['confident_consensus'] and s['consensus']==labels[g] and len(cue)>0)
                ag,an,ug,un=alpha[target],alpha[0],u[target],u[0]
                value_term=(ag+an)/2*(ug-un);routing_term=(ag-an)*(ug+un)/2;bias_term=float(bias[target]-bias[0])
                error=abs(float(value_term.sum()+routing_term.sum()+bias_term-(z[target]-z[0])));decomposition=max(decomposition,error);assert error<2*tol
                row=dict(case_id=case,mode=mode,family=rec['family'],gold=labels[g],target=labels[target],prediction=labels[prediction],tokens=len(h),
                    semantic_gold_supported=eligible,semantic_cue_size=int(len(cue)) if eligible else 0,
                    target_minus_null=float(z[target]-z[0]),target_minus_best_rival=float(z[target]-best),bias_difference=bias_term,
                    common_attention_relative_value=float(value_term.sum()),routing_difference_common_value=float(routing_term.sum()),
                    all_token_target_row_upper_margin=float(ug.max()+bias[target]-best),
                    semantic_cue_target_row_upper_margin=float(ug[cue].max()+bias[target]-best) if eligible else None,
                    semantic_cue_uniform_margin=float(ug[cue].mean()+bias[target]-best) if eligible else None,
                    semantic_cue_target_attention_mass=float(ag[cue].sum()) if eligible else None,
                    semantic_cue_null_attention_mass=float(an[cue].sum()) if eligible else None,
                    semantic_cue_target_attention_enrichment=float(ag[cue].sum()/(len(cue)/len(h))) if eligible else None,
                    semantic_cue_value_term=float(value_term[cue].sum()) if eligible else None,
                    semantic_cue_routing_term=float(routing_term[cue].sum()) if eligible else None)
                rows.append(row);prefix=mode+'__'+case+'__'
                for k in ('logits','alpha','utilities','values','probability','psi'):
                    if k in d:arrays[prefix+k]=d[k][0].numpy()
                arrays[prefix+'bias']=head.bias.numpy().copy();arrays[prefix+'roles']=roles;arrays[prefix+'cue_indices']=cue
                cursor+=1
                save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='fixed_complete_model_readout',current=cursor,total=len(truth)*len(cfg['modes']),unit='example',observed_at=time.time(),heartbeat_at=time.time()))
    np.savez(a.output/'details.private.npz',**arrays);save(a.output/'cases.private.json',rows)
    groups={}
    for mode in cfg['modes']:
        groups[mode]={}
        for family in ('positive_missed','positive_correct','null_false_positive','null_correct'):
            rr=[q for q in rows if q['mode']==mode and q['family']==family];ee=[q for q in rr if q['semantic_gold_supported']]
            groups[mode][family]=dict(rows=len(rr),semantic_gold_supported=len(ee),
                all_token_bound_cannot_cross=sum(q['all_token_target_row_upper_margin'] < -tol for q in rr),
                supported_all_token_bound_cannot_cross=sum(q['all_token_target_row_upper_margin'] < -tol for q in ee),
                supported_cue_bound_can_cross=sum(q['semantic_cue_target_row_upper_margin'] > tol for q in ee),
                supported_uniform_cue_can_cross=sum(q['semantic_cue_uniform_margin'] > tol for q in ee),
                supported_value_term_negative=sum(q['common_attention_relative_value'] < -tol for q in ee),
                supported_bias_negative=sum(q['bias_difference'] < -tol for q in ee),
                supported_cue_enrichment=[q['semantic_cue_target_attention_enrichment'] for q in ee],
                supported_cue_upper_margins=[q['semantic_cue_target_row_upper_margin'] for q in ee])
    report=dict(profile=cfg['profile'],attempt=a.attempt,rows=len(rows),cases=len(truth),groups=groups,seconds=time.monotonic()-start,
        replay_max_abs=parity,logit_identity_max_abs=identity,margin_decomposition_max_abs=decomposition,
        optimizer_updates=0,gpu_compute=0,test_rows=0,cos_writes=0,
        private_details_sha256=sha(a.output/'details.private.npz'),private_case_numbers_sha256=sha(a.output/'cases.private.json'),
        bound='Known convex weighted-average bound, unchanged values/bias/every rival, only one target row relaxed. Not shared-circuit feasibility, retraining impossibility, a new theorem or deployed repair.',
        scope='Fixed32 reused VALID from original complete selected checkpoints on rebuiltcache. Descriptive readout, same-model semantic citations uncertain; no prevalence, intervention utility, quantum attribution or L5. Full-case intermediate arrays stay private compute.')
    save(a.output/'summary.json',report);(a.output/'RUN_COMPLETE').write_bytes(b'completed\n');print(json.dumps(report))


if __name__=='__main__':main()
