"""Read-only32case native decomposition; semantic-cue convex bounds, not a method."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    q=p.with_suffix('.pending');q.write_bytes(json.dumps(v,indent=2,allow_nan=False).encode()+bytes([10]));q.replace(p)

def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes());r=a.root
    assert sha(__file__)==cfg['code_sha256']
    for n,h in cfg['input_hashes'].items():assert sha(r/n)==h,n
    assert not a.output.exists();a.output.mkdir(parents=True)
    start=time.monotonic();save(a.output/'started.json',dict(attempt=a.attempt,design_sha256=sha(a.design)))
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(r/'source'));from baselines.retacred.rmea import RMEAHead
    head=RMEAHead('quantum').double();head.load_state_dict(torch.load(r/'inputs/native-checkpoint/native-final.pt',map_location='cpu',weights_only=True));head.eval()
    base=r/'control/semantic-problem-audit-20261123';truth=json.loads((base/'selection.private.json').read_bytes());sem=json.loads((base/'semantic-results.private.json').read_bytes());sem={v['case_id']:v for v in sem}
    words=json.loads((r/'inputs/native-word-metadata-20261123/train.private.json').read_bytes());types=json.loads((r/'control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
    labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
    with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as z:
        native=z['logits'].copy();fit=z['calibration'].copy();gold=z['labels'].copy()
    support={};seen=set()
    for i in np.flatnonzero(fit):
        g,st,ot=types[i];support.setdefault((st,ot),set()).add(g);seen.add(g)
    cache=r/'inputs/native-baseline/discovery/cache';data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    rows=[];arrays={};parity=identity=0.
    with torch.inference_mode():
        for count,rec in enumerate(truth):
            i=rec['row'];key=rec['case_id'];w=words[i];assert not fit[i] and len(w['tokens'])==int(data['offsets'][i+1]-data['offsets'][i])
            left,right=map(int,data['offsets'][i:i+2]);h=np.array(data['h'][left:right],np.float64);roles=np.array(data['roles'][left:right],np.float64);mask=np.ones((1,len(h)),bool)
            for span,col in ((w['subject'],2),(w['object'],3)):
                expected=np.zeros(len(h),bool);expected[slice(*span)]=True;assert np.array_equal(expected,roles[:,col].astype(bool))
            d=head(torch.from_numpy(h)[None],torch.from_numpy(roles)[None],torch.from_numpy(mask),details=True)
            z=d['logits'][0].numpy();alpha=d['alpha'][0].numpy();u=d['utilities'][0].numpy();bias=head.bias.numpy();g=int(gold[i]);assert data['targets'][i]==g
            err=float(np.max(np.abs(z-native[i])));parity=max(parity,err);assert err<1e-8
            err=float(np.max(np.abs((alpha*u).sum(-1)+bias-z)));identity=max(identity,err);assert err<1e-10
            allow=np.ones(40,bool);pair=tuple(types[i][1:])
            if pair in support:
                allow[:]=False;allow[0]=True
                for k in support[pair]|(set(range(40))-seen):allow[k]=True
            assert allow[g];pred=int(np.where(allow,z,-np.inf).argmax());assert labels[pred]==rec['predicted']
            rivals=z.copy();rivals[~allow]=-np.inf;rivals[g]=-np.inf;best=float(rivals.max());s=sem[key]
            cue=np.array(s['evidence_intersection'],dtype=int)
            eligible=bool(g>0 and s['confident_consensus'] and s['consensus']==labels[g] and len(cue)>0)
            row=dict(case_id=key,family=rec['family'],gold_label=labels[g],typed_prediction=labels[pred],tokens=len(h),
                original_gold_margin=float(z[g]-best),all_token_single_row_upper_margin=float(u[g].max()+bias[g]-best),
                semantic_gold_cue_eligible=eligible,semantic_gold_cue_size=int(len(cue)) if eligible else 0,
                semantic_gold_cue_upper_margin=float(u[g,cue].max()+bias[g]-best) if eligible else None,
                semantic_gold_cue_uniform_margin=float(u[g,cue].mean()+bias[g]-best) if eligible else None,
                semantic_gold_cue_attention_mass=float(alpha[g,cue].sum()) if eligible else None,
                semantic_gold_cue_token_share=float(len(cue)/len(h)) if eligible else None,
                semantic_gold_cue_attention_enrichment=float(alpha[g,cue].sum()/(len(cue)/len(h))) if eligible else None)
            rows.append(row)
            for n in ('logits','alpha','utilities','probability','psi','values'):arrays[key+'_'+n]=d[n][0].numpy()
            arrays[key+'_roles']=roles;arrays[key+'_cue_indices']=cue
            save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='fixed_native_semantic_decomposition',current=count+1,total=len(truth),unit='example',observed_at=time.time(),heartbeat_at=time.time()))
    np.savez(a.output/'details.private.npz',**arrays);save(a.output/'cases.private.json',rows)
    groups={}
    for family in ('positive_missed','positive_correct','null_false_positive','null_correct'):
        rr=[x for x in rows if x['family']==family];eligible=[x for x in rr if x['semantic_gold_cue_eligible']]
        groups[family]=dict(rows=len(rr),semantic_gold_cue_eligible=len(eligible),
            all_token_bound_cannot_cross=int(sum(x['all_token_single_row_upper_margin']<=-1e-8 for x in rr)),
            semantic_cue_bound_cannot_cross=int(sum(x['semantic_gold_cue_upper_margin']<=-1e-8 for x in eligible)),
            semantic_cue_bound_can_cross=int(sum(x['semantic_gold_cue_upper_margin']>1e-8 for x in eligible)),
            uniform_cue_can_cross=int(sum(x['semantic_gold_cue_uniform_margin']>1e-8 for x in eligible)),
            cue_attention_enrichment=[x['semantic_gold_cue_attention_enrichment'] for x in eligible],
            cue_upper_margins=[x['semantic_gold_cue_upper_margin'] for x in eligible])
    report=dict(profile=cfg['profile'],attempt=a.attempt,rows=len(rows),strata=groups,seconds=time.monotonic()-start,
        native_replay_max_abs=parity,logit_identity_max_abs=identity,optimizer_updates=0,gpu_compute=0,test_rows=0,cos_writes=0,
        private_details_sha256=sha(a.output/'details.private.npz'),private_case_numbers_sha256=sha(a.output/'cases.private.json'),
        bound='With unchanged per-token utility, class bias, and every rival logit, any normalized nonnegative attention supported on E has target logit <=max_E utility+bias. Negative upper-margin excludes repair by changing this one target attention row on E. No claim about jointly changed rival rows, retrained values, shared circuit or global model impossibility.',
        scope='Frozen32outcome-stratified exposedTRAIN examples. Consensus cues from two same-model readers are uncertain semantic evidence; counts are not prevalence or independent utility. No actual oracle repair, trained candidate, causal attribution beyond this fixed arithmetic, novelty or L5.')
    save(a.output/'summary.json',report);(a.output/'RUN_COMPLETE').write_bytes(b'completed\n');print(json.dumps(report))

if __name__=='__main__':main()
