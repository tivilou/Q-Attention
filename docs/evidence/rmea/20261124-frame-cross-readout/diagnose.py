"""Four preregistered frozen readout swaps; no model fitting or selection."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np

r=Path(__file__).resolve().parent
cfg=json.loads((r/'design.json').read_bytes())
a=json.loads((r/'launch.json').read_bytes())['allocation']
assert len(a['gpu_uuids'])==1
os.environ['CUDA_VISIBLE_DEVICES']=a['gpu_uuids'][0]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
source=Path(cfg['source']);trained=Path(cfg['trained_root']);o=trained/'output'
sys.path[:0]=[str(source),str(trained)]
from router import FrameHead
from baselines.retacred.rmea_task_runtime import score_batch
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(o/'summary.json')==cfg['summary_sha256']
assert sha(o/'diagnostic.private.npz')==cfg['private_predictions_sha256']
for mode,h in cfg['weight_hashes'].items():assert sha(o/mode/'final.pt')==h
out=r/'output';out.mkdir(exist_ok=False)
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
with np.load(o/'diagnostic.private.npz',allow_pickle=False) as v:
 indices=v['assessment_indices'].copy();y=v['labels'][indices].copy()
 stored={k:v[k+'_logits'].copy() for k in ('quantum','quantum_frame','dephased','dephased_frame')}
assert len(indices)==4000
cache=Path(cfg['inputs'])/'discovery/cache'
data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
assert np.array_equal(y,data['targets'][indices])
def batch(ids):
 lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
 h=np.zeros((len(ids),max(lengths),1024),np.float64)
 roles=np.zeros(h.shape[:2]+(4,),np.float64);mask=np.zeros(h.shape[:2],bool)
 for j,i in enumerate(ids):
  lo,hi=map(int,data['offsets'][i:i+2]);n=hi-lo
  h[j,:n]=data['h'][lo:hi];roles[j,:n]=data['roles'][lo:hi];mask[j,:n]=True
 return tuple(torch.from_numpy(v).cuda() for v in (h,roles,mask))
def measure(z):
 pred=z.argmax(1);cm=np.bincount(40*y+pred,minlength=1600).reshape(40,40)
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
 s=z-z.max(1,keepdims=True)
 return dict(all40_macro_f1=float(f.mean()),positive39_macro_f1=float(f[1:].mean()),ce=float(np.mean(np.log(np.exp(s).sum(1))-s[np.arange(len(y)),y])),correct=int(np.trace(cm)),positive_to_null=int(((y>0)&(pred==0)).sum()),null_false_positive=int(((y==0)&(pred!=0)).sum()),confusion=cm.tolist())
summary_old=json.loads((o/'summary.json').read_bytes())
rows={};logits={};started=time.time()
for k,z in stored.items():
 row=measure(z)
 assert abs(row['all40_macro_f1']-summary_old['results'][k]['all40_macro_f1'])<1e-12
 rows[k]=row
for j,contrast in enumerate(cfg['contrasts']):
 model=FrameHead(contrast['readout']).double()
 state=torch.load(o/contrast['weights']/'final.pt',map_location='cpu',weights_only=True)
 model.load_state_dict(state);model.cuda().eval()
 z=[]
 with torch.no_grad():
  for lo in range(0,len(indices),32):
   values=score_batch(model,batch(indices[lo:lo+32]))
   assert torch.isfinite(values).all();z.append(values.cpu().numpy())
 z=np.concatenate(z);name=contrast['name'];logits[name]=z
 row=measure(z);old=stored[contrast['reference']].argmax(1);new=z.argmax(1)
 row.update(reference=contrast['reference'],delta_all40_macro_f1=row['all40_macro_f1']-rows[contrast['reference']]['all40_macro_f1'],repairs=int(((old!=y)&(new==y)).sum()),damages=int(((old==y)&(new!=y)).sum()),changed=int((old!=new).sum()))
 rows[name]=row
 write(r/'progress.json',dict(schema='research.progress.v1',attempt=a['runner_attempt_id'],phase=name,current=j+1,total=4,unit='frozen_forward_pass',heartbeat_at=time.time(),observed_at=time.time()))
 del model;torch.cuda.empty_cache()
np.savez(out/'diagnostic.private.npz',labels=y,indices=indices,**logits)
summary=dict(status='completed',seconds=time.time()-started,results=rows,fit_updates=0,assessment_items=4000,valid_read=False,test_read=False,cos_writes=0,design_sha256=sha(r/'design.json'),source_weights_sha256=cfg['weight_hashes'],private_diagnostic_sha256=sha(out/'diagnostic.private.npz'),scope='ExposedTRAIN fixed-state intervention; learned-state and readout sensitivity only, not semantic causality, independently trained utility or promotion.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
write(out/'summary.json',summary);write(out/'RUN_COMPLETE',{'summary.json':sha(out/'summary.json')})
print(json.dumps(dict(status='completed',seconds=summary['seconds'],results={k:{n:v[n] for n in ('all40_macro_f1','ce','correct')} for k,v in rows.items()})))
