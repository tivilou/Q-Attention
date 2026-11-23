"""Frozen training-cohort mechanism diagnostic; no fits or held-out utility claim."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/hy-tmp/Q-Attention/source-protected-context-gradient-20261122')
from baselines.retacred.rmea import RMEAHead,EPS
from baselines.retacred.rmea_protected_context import ProtectedContextHead

a=argparse.ArgumentParser();a.add_argument('--output',required=True,type=Path);a.add_argument('--baseline-run',required=True,type=Path);a.add_argument('--pins',required=True,type=Path);args=a.parse_args()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
cache=args.baseline_run/'cache';pins=json.loads(args.pins.read_bytes())
for n,row in pins.items():
 p=cache/n
 if p.is_symlink() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('Pinned input differs:'+n)
cp=args.baseline_run/'checkpoints/native-final.pt'
if digest(cp)!='2da52b1ce826046a6f443269f47f49e4e9fa63e10f22f9ebdcde6f890e12ce51':raise ValueError('Native checkpoint differs')
data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')};metadata=json.loads((cache/'metadata.json').read_bytes())['train']
assert len(metadata)==len(data['targets'])==58465
# Freeze membership before model outputs: same class quotas as historical valid96,
# now a new TRAIN cohort; all models already trained on this split.
selected=[]
for label in range(40):
 indices=sorted(np.flatnonzero(data['targets']==label).tolist(),key=lambda i:(metadata[i]['id_hash'],i))
 selected+=indices[:18 if label==0 else 2]
assert len(selected)==96 and len(set(selected))==96
population=[dict(index=i,id_hash=metadata[i]['id_hash'],target=int(data['targets'][i])) for i in selected]
contract=dict(profile='training96_frozen_mass_held_discovery_v1',population=population,selection='train: smallest ID hashes,18no_relation+2each39positive; before outputs',checkpoint_sha256=digest(cp),split='train',primary_population='78positive-label cases',primary_gate='mean mass-held matched-native>1e-6 AND mean mass-held matched-DFT>1e-6',selection_bias='Checkpoint trained on this split; discovery mechanism only, no held-out utility or independent confirmation',optimizer_updates=0,cos_operations=0)
if (args.output/'RUN_COMPLETE').exists():raise ValueError('No reanalysis of a completed output')
atomic(args.output/'population-contract.json',contract)
state=torch.load(cp,map_location='cpu',weights_only=True);native=RMEAHead('quantum').double().eval();native.load_state_dict(state,strict=True)
heads={k:ProtectedContextHead(k).double().eval() for k in ('protected_context','context_scrambled','unprotected_context')}
for h in heads.values():h.load_state_dict(state,strict=True)
rows=[];max_sum_error=0.;max_mention_error=0.
with torch.no_grad():
 for i in selected:
  start,stop=map(int,data['offsets'][i:i+2]);h=torch.from_numpy(np.array(data['h'][start:stop],dtype=np.float64,copy=True))[None];roles=torch.from_numpy(np.array(data['roles'][start:stop],dtype=np.float64,copy=True))[None];mask=torch.ones(1,stop-start,dtype=torch.bool)
  d0=native(h,roles,mask,details=True);logits0=d0['logits'][0];alpha0=d0['alpha'][0];utilities=d0['utilities'][0];mention=((roles[0,:,2]!=0)|(roles[0,:,3]!=0));context=~mention;target=int(data['targets'][i]);competitor=int(logits0.masked_fill(torch.arange(40)==target,-torch.inf).argmax());m0=alpha0[:,context].sum(-1,keepdim=True)
  base_margin=float(logits0[target]-logits0[competitor]);values={}
  for kind,head in heads.items():
   d=head(h,roles,mask,details=True);alpha=d['alpha'][0]
   assert torch.equal(d['utilities'],d0['utilities']) and float((d['phi']-d0['phi']).abs().max())<1e-12
   held=alpha0.clone()
   if context.any():
    raw=d['raw'][0][:,context];held[:,context]=raw/raw.sum(-1,keepdim=True)*m0
   else:assert torch.equal(held,alpha0)
   max_sum_error=max(max_sum_error,float((held.sum(-1)-1).abs().max()));max_mention_error=max(max_mention_error,float((held[:,mention]-alpha0[:,mention]).abs().max()))
   assert max_sum_error<1e-12 and max_mention_error==0.
   logits=(held*utilities).sum(-1)+head.bias
   actual=d['logits'][0]
   values[kind]=dict(held_margin_delta=float(logits[target]-logits[competitor])-base_margin,actual_margin_delta=float(actual[target]-actual[competitor])-base_margin,held_correct=int(logits.argmax())==target,actual_correct=int(actual.argmax())==target,held_attention_change=float((held-alpha0).abs().sum(-1).mean()))
  rows.append(dict(index=i,id_hash=metadata[i]['id_hash'],target=target,competitor=competitor,base_margin=base_margin,native_correct=int(logits0.argmax())==target,protected_rank=int((d['singular_values'][0]>1e-10*d['singular_values'][0,0]).sum()),contexts=int(context.sum()),conditions=values))

def summarize(rows):
 def mean(f):return sum(f(r) for r in rows)/len(rows)
 result={k:dict(held_margin_delta=mean(lambda r:r['conditions'][k]['held_margin_delta']),actual_margin_delta=mean(lambda r:r['conditions'][k]['actual_margin_delta']),held_correct=sum(r['conditions'][k]['held_correct'] for r in rows),actual_correct=sum(r['conditions'][k]['actual_correct'] for r in rows)) for k in heads}
 matched=result['protected_context']['held_margin_delta'];dft=result['context_scrambled']['held_margin_delta'];return dict(n=len(rows),native_correct=sum(r['native_correct'] for r in rows),conditions=result,held_matched_minus_dft=matched-dft,primary_gate=matched>1e-6 and matched-dft>1e-6,actual_matched_margin_positive=result['protected_context']['actual_margin_delta']>0)
summary=dict(profile=contract['profile'],status='complete',checkpoint_sha256=contract['checkpoint_sha256'],population_sha256=digest(args.output/'population-contract.json'),primary=summarize([r for r in rows if r['target']>0]),all96=summarize(rows),checks=dict(max_held_sum_error=max_sum_error,max_held_mention_error=max_mention_error),rows=rows,optimizer_updates=0,cos_operations=0,L5='NOT_ACCEPTED',limitations=contract['selection_bias'])
atomic(args.output/'redistribution-summary.json',summary);atomic(args.output/'RUN_COMPLETE',dict(artifacts={n:digest(args.output/n) for n in ('population-contract.json','redistribution-summary.json')}));print(json.dumps({k:summary[k] for k in ('status','primary','all96','checks','optimizer_updates','cos_operations')}))
