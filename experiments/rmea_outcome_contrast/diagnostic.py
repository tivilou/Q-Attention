"""One frozen, class-stratified validation replay; zero optimizer/test access."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from legacy.rmea import RMEAHead
from contrast import intervention,VARIANTS
from math_checks import check

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')

def detail(head,h,roles,mask):
    with torch.inference_mode():
        x=head(torch.from_numpy(h).float(),torch.from_numpy(roles).float(),torch.from_numpy(mask),details=True)
    return dict({k:v.detach().cpu().numpy() for k,v in x.items()},mask=mask,bias=head.bias.detach().cpu().numpy())

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    contract=json.loads((args.inputs/'input-contract.json').read_text())
    assert sha(args.inputs/'input-contract.json')=='d58c65a23f0a452ebc4c4f2bb89fef85fbc0e4559a60a8ba3cd0ed1811f968bd'
    for name,digest in contract['files'].items():assert sha(args.inputs/name)==digest,name
    selection=json.loads((args.inputs/'selection.json').read_text())
    assert selection['diagnostic_count']==240 and len(selection['items'])<=242
    assert selection['source_revision']=='f3f22660b609811e6fb824f7d19e9cafd0c1e446'
    assert selection['features_sha256']==sha(args.inputs/'selected-valid.npz')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    head=RMEAHead('quantum').cpu()
    head.load_state_dict(torch.load(args.inputs/'checkpoints/quantum-selected.pt',weights_only=True,map_location='cpu'),strict=True);head.eval()
    algebra=check();parity=[]
    for i in (0,1):
        with np.load(args.inputs/f'trace/quantum-selected-valid-{i}.npz',allow_pickle=False) as f:old={k:f[k] for k in f.files}
        new=detail(head,old['h'],old['roles'],old['mask'])
        for key in ('angles','effect_angles','psi','phi','alpha','utilities','logits'):
            np.testing.assert_allclose(new[key],old[key],atol=1e-4,rtol=3e-4)
        parity.append(float(np.abs(new['logits']-old['logits']).max()))
    with np.load(args.inputs/'selected-valid.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    assert len(data['offsets'])==len(selection['items'])+1 and data['offsets'][-1]==len(data['h'])==len(data['roles'])
    cases=[]
    for i,item in enumerate(selection['items']):
        if not item['diagnostic']:continue
        lo,hi=map(int,data['offsets'][i:i+2]);assert hi>lo
        saved=detail(head,data['h'][lo:hi].astype(np.float32)[None],data['roles'][lo:hi][None],np.ones((1,hi-lo),bool))
        weights,logits,trace=intervention(saved)
        # Predictions and evidence ranges above are label free. Gold below is ORACLE DIAGNOSTIC ONLY.
        target=int(data['targets'][i]);assert target==item['target'] and 0<=target<40
        u=saved['utilities'][0].astype(float);bias=saved['bias'].astype(float)
        upper=u.max(-1)+bias;lower=u.min(-1)+bias
        oracle_scores=lower.copy();oracle_scores[target]=upper[target]
        oracle_margin=float(oracle_scores[target]-np.delete(oracle_scores,target).max())
        cases.append(dict(cache_index=item['cache_index'],id_hash=item['id_hash'],target=target,
            predictions={k:int(logits[k][0].argmax()) for k in VARIANTS},
            convex_oracle_prediction=int(oracle_scores.argmax()),convex_oracle_margin=oracle_margin,
            convex_oracle_strictly_correct=oracle_margin>1e-8,**trace))
    assert len(cases)==240
    confusions={};metrics={}
    for k in VARIANTS:
        cm=np.zeros((40,40),np.int64)
        for c in cases:cm[c['target'],c['predictions'][k]]+=1
        denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denom,out=np.zeros(40),where=denom>0)
        metrics[k]=dict(correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/240),stratified_macro_f1=float(f1.mean()),
            corrected=sum(c['predictions']['static']!=c['target']==c['predictions'][k] for c in cases),
            harmed=sum(c['predictions']['static']==c['target']!=c['predictions'][k] for c in cases),
            changed=sum(c['predictions']['static']!=c['predictions'][k] for c in cases))
        confusions[k]=cm.tolist()
    assert metrics['static']['correct']==136
    assert np.array(confusions['static']).sum(1).tolist()==[6]*40
    headroom=dict(kind='Unrestricted independent-relation convex-attention oracle; GOLD LABELS, not executable inference',
        strictly_feasible=sum(c['convex_oracle_strictly_correct'] for c in cases),
        incorrect_but_strictly_feasible=sum(c['predictions']['static']!=c['target'] and c['convex_oracle_strictly_correct'] for c in cases),
        limit='Finite 240-case witness only. Does not bound full-population Macro or learned Born representability. Shared parameters and EPS constraints are relaxed; no predicted gain.')
    report=dict(status='PASS',kind='Frozen-head exploratory diagnostic on REUSED validation subset',cases=240,classes=40,
        metrics=metrics,confusion_matrices=confusions,algebra=algebra,numeric_parity_max=parity,headroom=headroom,
        input_contract_sha256=sha(args.inputs/'input-contract.json'),source_sha256=sha(Path(__file__)),
        selection=selection['selection'],original_valid_count=selection['original_valid_count'],
        reused_validation=True,training_runs=0,formal_runs=0,
        claim_ceiling='Post-selection exploratory frozen intervention, no independent training, test, generalization estimate, utility tier or L3',
        direct_screen_gate='Positive subset Macro delta AND corrections>harms allows separate bounded-training design only. Otherwise stop this direct insertion; no gate/scale/seed rescue.')
    save(args.output/'report.json',report);save(args.output/'cases.json',dict(cases=cases))
    print(json.dumps({k:v for k,v in report.items() if k!='confusion_matrices'}))

if __name__=='__main__':main()
