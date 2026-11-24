"""Descriptive pullbacks of existing selected states; no optimizer, fitting or claims."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
 ap=argparse.ArgumentParser()
 for name in ('root','design','output','progress'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
 assert sha(__file__)==cfg['code_sha256']
 source=a.root/cfg['source'];completed=a.root/cfg['completed']
 for n,h in cfg['source_hashes'].items():assert sha(source/n)==h
 pin=cfg['selected'];path=completed/pin['path'];assert sha(path)==pin['sha256']
 sys.path.insert(0,str(source));import torch
 from baselines.retacred.rmea import circuit,EPS
 from baselines.retacred.rmea_matched_routers import Head
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 assert not torch.cuda.is_available();assert not a.output.exists();a.output.mkdir(parents=True)
 model=Head('quantum').double();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
 cache=a.root/cfg['cache'];data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets')}
 # Prior exact replay already verified the immutable full cache. Recheck small
 # offset/role identities, mmap only the first8 original stored examples.
 for n in ('train-roles.npy','train-offsets.npy'):assert sha(cache/n)==cfg['train_cache'][n]['sha256']
 states=[]
 with torch.no_grad():
  for i in range(8):
   lo,hi=map(int,data['offsets'][i:i+2]);x=torch.from_numpy(np.concatenate((data['h'][lo:hi],data['roles'][lo:hi]),-1).astype(np.float64))
   states.append(circuit((torch.pi*torch.tanh(model.head.prepare(x))).reshape(-1,2,3,2)).numpy())
 start=time.monotonic();rows=[];fd_errors=[];norm_errors=[]
 for r in range(40):
  raw=model.head.effect[r].detach().clone().requires_grad_()
  def packed(x):
   z=circuit(torch.pi*torch.tanh(x));return torch.cat((z.real,z.imag))
  out=packed(raw);phi=(out[:8]+1j*out[8:]).detach().numpy()
  jac=torch.autograd.functional.jacobian(packed,raw).reshape(16,12).detach().numpy();j=jac[:8]+1j*jac[8:]
  projection=j.conj().T@phi
  g=4*np.real(j.conj().T@j-np.outer(projection,projection.conj()));g=(g+g.T)/2
  eig,vec=np.linalg.eigh(g);threshold=1e-10*max(1.,eig[-1]);active=eig>threshold
  assert eig[0]>-1e-10 and active.any()
  w=vec[:,active]/np.sqrt(eig[active]);ratios=[];null_residuals=[];normalizers=[]
  for psi in states:
   amplitude=psi@phi.conj();q=np.abs(amplitude)**2
   dq=2*np.real(amplitude.conj()[:,None]*(psi@j.conj()))
   s=(q+EPS).sum();alpha=(q+EPS)/s;da=(dq-alpha[:,None]*dq.sum(0))/s
   f=da.T@(da/alpha[:,None]);ratios.append(float(np.linalg.eigvalsh(w.T@f@w)[-1]))
   null_residuals.append(float(np.linalg.norm(f@vec[:,~active])) if (~active).any() else 0.)
   normalizers.append(float(s))
  # One frozen derivative direction per class, full12coordinates covered by r%12.
  k=r%12;delta=torch.zeros_like(raw);delta.reshape(-1)[k]=1e-6
  fd=(packed(raw+delta)-packed(raw-delta)).detach().numpy()/(2e-6)
  fd_errors.append(float(np.max(np.abs(fd-jac[:,k]))))
  norm_errors.append(float(abs(np.vdot(phi,phi)-1)))
  rows.append(dict(relation=r,metric_rank=int(active.sum()),state_qfi_eigenvalues=eig.tolist(),
   positive_condition=float(eig[-1]/eig[active][0]),
   max_routing_to_state_qfi=float(max(ratios)),median_routing_to_state_qfi=float(np.median(ratios)),
   metric_nullspace_residual=max(null_residuals),min_normalizer=min(normalizers)))
  save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='effect_geometry',current=r+1,total=40,unit='relation',observed_at=time.time(),heartbeat_at=time.time()))
 assert max(fd_errors)<1e-8 and max(norm_errors)<1e-12
 summary=dict(profile=cfg['profile'],rows=rows,seconds=time.monotonic()-start,classes=40,train_contexts=8,
  maximum_derivative_check_error=max(fd_errors),maximum_state_norm_error=max(norm_errors),
  rank_counts={str(k):sum(v['metric_rank']==k for v in rows) for k in sorted({v['metric_rank'] for v in rows})},
  median_condition=float(np.median([v['positive_condition'] for v in rows])),
  median_max_routing_to_state_qfi=float(np.median([v['max_routing_to_state_qfi'] for v in rows])),
  maximum_routing_to_state_qfi=max(v['max_routing_to_state_qfi'] for v in rows),
  training_updates=0,new_valid=0,new_test=0,cos_writes=0,
  scope='Known Fubini-Study/QFI and categorical Fisher pullbacks of ALL40 selectedQ effects on first8TRAIN contexts. Numerical spectra, not a cause of training failure, optimizer utility, new theory, quantum necessity or L3/L4-T. QNG is prior art. Thresholded positive spectrum is not certified exact rank.')
 save(a.output/'summary.json',summary);save(a.output/'RUN_COMPLETE',{'summary.json':sha(a.output/'summary.json')})
 print(json.dumps({k:summary[k] for k in ('seconds','rank_counts','median_condition','median_max_routing_to_state_qfi','maximum_routing_to_state_qfi','maximum_derivative_check_error')}))

if __name__=='__main__':main()
