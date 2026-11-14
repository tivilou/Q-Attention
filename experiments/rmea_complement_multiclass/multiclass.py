"""Frozen-head CPU validation diagnostic; never fits or selects a model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from legacy.rmea import RMEAHead
from intervention import intervention

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')

def detail(head,h,roles,mask):
    with torch.inference_mode():
        values=head(torch.from_numpy(h).float(),torch.from_numpy(roles).float(),torch.from_numpy(mask),details=True)
    saved={k:v.detach().cpu().numpy() for k,v in values.items()}
    saved.update(mask=mask,bias=head.bias.detach().cpu().numpy())
    return saved

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    contract=json.loads((args.inputs/'input-contract.json').read_text(encoding='utf8'))
    for name,digest in contract['files'].items():assert sha(args.inputs/name)==digest,name
    selection=json.loads((args.inputs/'selection.json').read_text(encoding='utf8'))
    assert selection['source_revision']=='f3f22660b609811e6fb824f7d19e9cafd0c1e446'
    assert selection['features_sha256']==sha(args.inputs/'selected-valid.npz')
    assert selection['diagnostic_count']<=240 and len(selection['items'])<=242
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    head=RMEAHead('quantum').cpu()
    head.load_state_dict(torch.load(args.inputs/'checkpoints/quantum-selected.pt',map_location='cpu',weights_only=True),strict=True);head.eval()
    parity=[]
    with np.load(args.inputs/'selected-valid.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    assert data['h'].dtype==np.float16 and data['roles'].dtype==np.float32
    assert len(data['offsets'])==len(selection['items'])+1 and data['offsets'][-1]==len(data['h'])==len(data['roles'])
    for original in (0,1):
        with np.load(args.inputs/f'trace/quantum-selected-valid-{original}.npz',allow_pickle=False) as f:old={k:f[k] for k in f.files}
        fresh=detail(head,old['h'],old['roles'],old['mask'])
        errors={}
        for name in ('angles','effect_angles','psi','phi','alpha','utilities','logits'):
            np.testing.assert_allclose(fresh[name],old[name],atol=1e-4,rtol=3e-4)
            errors[name]=float(np.abs(fresh[name]-old[name]).max())
        local=next(i for i,x in enumerate(selection['items']) if x['cache_index']==original)
        lo,hi=map(int,data['offsets'][local:local+2]);mask=old['mask'][0].astype(bool)
        np.testing.assert_array_equal(data['h'][lo:hi].astype(np.float32),old['h'][0,mask])
        np.testing.assert_array_equal(data['roles'][lo:hi],old['roles'][0,mask])
        parity.append({'cache_index':original,'max_errors':errors})
    variants=('static','complement','dephased_pair');cases=[]
    for i,item in enumerate(selection['items']):
        if not item['diagnostic']:continue
        lo,hi=map(int,data['offsets'][i:i+2]);h=data['h'][lo:hi].astype(np.float32)[None];roles=data['roles'][lo:hi][None]
        saved=detail(head,h,roles,np.ones((1,hi-lo),dtype=bool))
        weights,logits,d=intervention(saved)
        # All interventions and logits are fixed before posthoc target access.
        target=int(data['targets'][i]);assert target==item['target'] and 0<=target<40
        predictions={k:int(logits[k][0].argmax()) for k in variants}
        cases.append({'cache_index':item['cache_index'],'id_hash':item['id_hash'],'target':target,'predictions':predictions,
            'pair_mass_mean':d['pair_mass_mean'],'pair_mass_max':d['pair_mass_max'],
            'complement_logit_linf':float(np.abs(logits['complement']-logits['static']).max()),
            'complement_vs_dephased_logit_linf':float(np.abs(logits['complement']-logits['dephased_pair']).max())})
    assert len(cases)==selection['diagnostic_count']
    counts={};confusions={}
    for k in variants:
        cm=np.zeros((40,40),np.int64)
        for c in cases:cm[c['target'],c['predictions'][k]]+=1
        support=cm.sum(1);supported=support>0;tp=np.diag(cm)
        f1=np.divide(2*tp,support+cm.sum(0),out=np.zeros(40,float),where=(support+cm.sum(0))>0)
        counts[k]={'correct':int(tp.sum()),'accuracy':float(tp.sum()/len(cases)),
            'stratified_macro_f1_over_supported_classes':float(f1[supported].mean()),
            'changed_from_static':sum(c['predictions'][k]!=c['predictions']['static'] for c in cases),
            'corrected':sum(c['predictions']['static']!=c['target'] and c['predictions'][k]==c['target'] for c in cases),
            'harmed':sum(c['predictions']['static']==c['target'] and c['predictions'][k]!=c['target'] for c in cases)}
        confusions[k]=cm.tolist()
    report={'status':'PASS','kind':'stratified fixed-checkpoint validation diagnostic','cases':len(cases),
        'target_classes':len({c['target'] for c in cases}),'original_valid_count':selection['original_valid_count'],
        'selection':selection['selection'],'support':np.array(confusions['static']).sum(1).tolist(),'metrics':counts,'confusion_matrices':confusions,
        'numeric_parity':parity,'mean_pair_mass':float(np.mean([c['pair_mass_mean'] for c in cases])),
        'max_logit_delta':max(c['complement_logit_linf'] for c in cases),
        'complement_vs_dephased_prediction_changes':sum(c['predictions']['complement']!=c['predictions']['dephased_pair'] for c in cases),
        'input_contract_sha256':sha(args.inputs/'input-contract.json'),'source_sha256':sha(Path(__file__)),
        'claim_ceiling':'Frozen-model reliance/action/risk on a class-stratified validation subset. No validation-population estimate, independent training, significance, novelty, formal utility tier or L3.',
        'training_runs':0,'formal_runs':0,'checkpoint_reload':True}
    save(args.output/'report.json',report);save(args.output/'cases.json',{'cases':cases})
    print(json.dumps({k:v for k,v in report.items() if k not in ('confusion_matrices','numeric_parity','support')}))

if __name__=='__main__':main()
