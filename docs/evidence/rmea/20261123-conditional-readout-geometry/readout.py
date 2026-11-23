"""R0: original frozen routing, exact effective linear readout and TRAIN gradient."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,threading,time
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
 assert sha(__file__)==cfg['code_sha256']
 for n,h in cfg['input_hashes'].items():assert sha(r/n)==h,n
 assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic()
 import torch
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 assert torch.cuda.device_count()==2
 import sys;sys.path.insert(0,str(r/'source'))
 from baselines.retacred.rmea import RMEAHead
 cache=r/'inputs/native-baseline/discovery/cache';data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
 n=len(data['targets']);assert n==58465
 counts=np.bincount(data['targets'],minlength=40);lock=threading.Lock();cursors={m:0 for m in cfg['modes']}
 labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
 def progress(mode,current):
  with lock:
   cursors[mode]=current;save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='fixed_readout_gradient',current=sum(cursors.values()),total=n*len(cursors),unit='example',children=cursors.copy(),observed_at=time.time(),heartbeat_at=time.time()))
 def worker(mode,lane):
  device=torch.device('cuda',lane);folder=r/'control/historical-type-replay-20261123'
  with torch.cuda.device(device),torch.inference_mode():
   state=torch.load(folder/'checkpoints'/(mode+'-selected.pt'),map_location='cpu',weights_only=True)
   head=RMEAHead(mode).float();head.load_state_dict(state);head.to(device).eval()
   c=head.decoder.detach().double();p=head.value.weight.detach().double();q=head.value.bias.detach().double();b=head.bias.detach().double()
   w=c@p;bias=c@q+b;sv=torch.linalg.svdvals(c)
   gw=torch.zeros_like(w);gb=torch.zeros_like(bias);ce=0.;maxerr=0.;meanerr=0.;batches=0
   previous=np.load(r/'control/full-model-error-structure-20261123/output'/(mode+'-logits.private.npy'),mmap_mode='r',allow_pickle=False)
   replayerr=0.
   for lo in range(0,n,cfg['batch_size']):
    hi=min(n,lo+cfg['batch_size']);ids=range(lo,hi);lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
    h=np.zeros((hi-lo,max(lengths),1024),np.float32);roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
    for j,i in enumerate(ids):
     left,right=map(int,data['offsets'][i:i+2]);h[j,:right-left]=data['h'][left:right];roles[j,:right-left]=data['roles'][left:right];mask[j,:right-left]=True
    hh=torch.from_numpy(h).to(device);rr=torch.from_numpy(roles).to(device);mm=torch.from_numpy(mask).to(device)
    details=head(hh,rr,mm,details=True);alpha=details['alpha'].double();z=details['logits'].double()
    # The observed route is a constant for this conditional derivative.
    pool=torch.einsum('brt,bth->brh',alpha,hh.double());collapsed=(pool*w[None]).sum(-1)+bias
    err=(collapsed-z).abs();maxerr=max(maxerr,float(err.max()));meanerr+=float(err.sum());batches+=1
    replayerr=max(replayerr,float(np.abs(z.cpu().numpy()-previous[lo:hi]).max()))
    target=torch.as_tensor(np.array(data['targets'][lo:hi]),dtype=torch.long,device=device)
    probability=z.softmax(-1);delta=probability.clone();delta[torch.arange(hi-lo,device=device),target]-=1
    gw+=torch.einsum('br,brh->rh',delta,pool);gb+=delta.sum(0)
    ce+=float((-z[torch.arange(hi-lo,device=device),target]+z.logsumexp(-1)).sum())
    progress(mode,hi)
   gw/=n;gb/=n;gp=c.T@gw;gq=c.T@gb;gc=gw@p.T+gb[:,None]*q[None,:]
   directnorm=float(gw.norm());factornorm=float(gp.norm());lower=float(sv.min())*directnorm
   assert maxerr<cfg['parity_tolerance'] and replayerr<cfg['parity_tolerance']
   assert factornorm+1e-12>=lower
   # Folded representation can be embedded back into the original64-wide head exactly.
   embedding=torch.zeros((40,64),dtype=torch.float64,device=device);embedding[:,:40]=torch.eye(40,dtype=torch.float64,device=device)
   projected=torch.zeros((64,1024),dtype=torch.float64,device=device);projected[:40]=w
   embedding_error=float((embedding@projected-w).abs().max());assert embedding_error<1e-12
   gradients=dict(w=gw.cpu().numpy(),bias=gb.cpu().numpy(),factor_p=gp.cpu().numpy(),factor_q=gq.cpu().numpy(),factor_c=gc.cpu().numpy())
   np.savez(a.output/(mode+'-gradient.private.npz'),**gradients)
   return dict(mode=mode,rows=n,raw_train_ce=ce/n,collapse_max_abs=maxerr,collapse_mean_abs=meanerr/(40*n),replay_max_abs=replayerr,
    decoder_singular_values=list(map(float,sv.cpu().numpy())),decoder_condition=float(sv.max()/sv.min()),decoder_rank_numerical=int((sv>1e-10*sv.max()).sum()),
    direct_w_gradient_frobenius=directnorm,effective_bias_gradient_norm=float(gb.norm()),factor_p_gradient_frobenius=factornorm,factor_c_gradient_frobenius=float(gc.norm()),factor_q_gradient_norm=float(gq.norm()),
    factor_gradient_lower_bound=lower,embedding_error=embedding_error,
    classes=[dict(label=labels[k],support=int(counts[k]),effective_w_gradient_norm=float(gw[k].norm()),effective_bias_gradient=float(gb[k])) for k in range(40)],
    gradient_sha256=sha(a.output/(mode+'-gradient.private.npz')))
 with ThreadPoolExecutor(max_workers=2) as pool:
  pending=[pool.submit(worker,m,i) for i,m in enumerate(cfg['modes'])];rows=[p.result() for p in pending]
 report=dict(profile=cfg['profile'],attempt=a.attempt,seconds=time.monotonic()-start,results={q['mode']:q for q in rows},optimizer_updates=0,valid_forwards=0,test_rows=0,cos_writes=0,
  scope='Current-cache complete TRAIN conditional derivative, not historical optimizer convergence, a loss-gap certificate, a finite update or validation utility. Cachedfeatures differ historical bytes. Known linear factorization/convexity facts are not L5.',
  code_sha256=cfg['code_sha256'])
 save(a.output/'summary.json',report);(a.output/'RUN_COMPLETE').write_bytes(b'completed\n');print(json.dumps(report))
if __name__=='__main__':main()
