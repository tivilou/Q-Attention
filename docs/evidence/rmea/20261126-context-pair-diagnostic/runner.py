"""One no-update pass over the exact existing gen98 cohort and four frozen heads."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,signal,sys,time
import numpy as np
from pair_math import compute,NAMES
from pair_statistics import summarize

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def loadmodule(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser()
 for k in ('root','design','output','progress'):ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--attempt',required=True);args=ap.parse_args();began=time.monotonic()
 cfg=json.loads(args.design.read_bytes());spec=cfg['identity_spec'];assert cfg['wall_seconds']==600 and cfg['batch_size']==32
 def expired(*_):raise TimeoutError('Frozen600-second single diagnostic expired; no automatic retry')
 signal.signal(signal.SIGALRM,expired);signal.alarm(600)
 assert not args.output.exists();args.output.mkdir();cursor=0;head_calls=circuit_calls=0
 def progress(phase):save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase=phase,current=cursor,total=4096,
  unit='assessment_example',observed_at=time.time(),heartbeat_at=time.time()))
 progress('verify_fixed_inputs')
 for n,h in cfg['code_hashes'].items():assert digest(args.design.parent/n)==h,n
 for n,h in cfg['input_hashes'].items():
  assert Path(n).resolve().is_relative_to(args.root.resolve()) and digest(n)==h,n
 cohort=json.loads(Path(spec['cohort']).read_bytes());idx=np.asarray(cohort['assessment'],np.int64);fit=cohort['fit']
 assert len(idx)==4096 and len(fit)==8192 and not set(fit)&set(idx.tolist())
 cache=Path(next(p for p in spec['inputs'] if p.endswith('train-h.npy'))).parent
 data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
 metadata=json.loads((cache/'metadata.json').read_bytes())['train'];assert len(metadata)==58465
 assert data['offsets'][-1]==len(data['h'])==len(data['roles'])
 archives={}
 for group in ('distinct-token-evidence-20261125','connected-value-pooling-20261125'):
  with np.load(args.root/'control'/group/'output/diagnostic.private.npz',allow_pickle=False) as z:
   assert np.array_equal(z['assessment_indices'],idx) and np.array_equal(z['labels'],data['targets'])
   assert np.array_equal(np.flatnonzero(z['calibration']),np.sort(fit))
   archives[group]={mode:z[mode+'_logits'].copy() for mode,m in spec['models'].items() if m['source_group']==group}
 old={k:v for group in archives.values() for k,v in group.items()}
 assert all(a.shape==(4096,40) for a in old.values())
 sys.path.insert(0,spec['source']);sys.path.insert(0,str(args.root/'control/distinct-token-evidence-20261125'))
 import torch
 from baselines.retacred.rmea_token_transport_study import state_identity
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0)
 assert torch.cuda.device_count()==1;device=torch.device('cuda',0)
 qplugin=loadmodule('existing_distinct_plugin',args.root/'control/distinct-token-evidence-20261125/plugin.py')
 gplugin=loadmodule('existing_gcov_plugin',args.root/'control/connected-value-pooling-20261125/plugin.py')
 models={};before={}
 for mode,m in spec['models'].items():
  plugin=gplugin if mode=='gaussian_covariance' else qplugin
  model=plugin.make_head(mode).float();state=torch.load(m['checkpoint'],map_location='cpu',weights_only=True)
  model.load_state_dict(state,strict=True);assert sum(p.numel() for p in model.parameters())==m['parameters']
  before[mode]=state_identity(model.state_dict());model.to(device).eval().requires_grad_(False);models[mode]=model
 import moment_head
 original=moment_head.circuit;active='';captured=[];key=[]
 def instrumented(angles):
  nonlocal circuit_calls
  out=original(angles);circuit_calls+=1
  if active=='quantum_distinct':captured.append((angles,out))
  return out
 moment_head.circuit=instrumented
 hook=models['gaussian_covariance'].head.key.register_forward_hook(lambda _m,_i,out:key.append(out))
 replay={mode:[] for mode in models};ss=[];trace={};maxerr=0.;argmax_mismatches=0;real16=0.;normerr=0.
 reasons=dict(short_rows=0,zero_norm_rows=0,zero_mass_Q=0,zero_mass_D=0,zero_mass_G=0)
 with torch.inference_mode():
  for lo in range(0,4096,32):
   rows=idx[lo:lo+32];length=[int(data['offsets'][i+1]-data['offsets'][i]) for i in rows];width=max(length)
   hh=np.zeros((len(rows),width,1024),np.float32);rr=np.zeros((len(rows),width,4),np.float32);mm=np.zeros((len(rows),width),bool)
   for j,i in enumerate(rows):
    a,b=map(int,data['offsets'][i:i+2]);assert metadata[i]['words']==b-a
    hh[j,:b-a]=data['h'][a:b];rr[j,:b-a]=data['roles'][a:b];mm[j,:b-a]=True
   inputs=tuple(torch.from_numpy(x).to(device) for x in (hh,rr,mm));h,roles,mask=inputs
   captured.clear();key.clear();batch_logits={}
   for mode,model in models.items():
    active=mode;z=model(*inputs);head_calls+=1;arr=z.cpu().numpy();assert np.isfinite(arr).all()
    expected=old[mode][lo:lo+len(rows)];maxerr=max(maxerr,float(np.max(np.abs(arr-expected)/(1+np.abs(expected)))))
    argmax_mismatches+=int(np.sum(arr.argmax(-1)!=expected.argmax(-1)));replay[mode].append(arr);batch_logits[mode]=arr
   assert len(captured)==2 and len(key)==1
   angles,psi=captured[0];assert psi.shape==h.shape[:2]+(8,)
   scale=(key[0].new_tensor([3-math.sqrt(2),3+math.sqrt(2)]*6)/24).sqrt();scaled=key[0]*scale
   scores,detail=compute(h,mask,psi,scaled)
   real16=max(real16,float(detail['real16_error']));normerr=max(normerr,float((psi.abs().square().sum(-1)-1).abs().masked_fill(~mask,0).max()))
   reasons['short_rows']+=int(detail['short_rows'].sum());reasons['zero_norm_rows']+=int(detail['zero_norm_rows'].sum())
   for axis,name in enumerate(('Q','D','G')):reasons['zero_mass_'+name]+=int((detail['masses'][:,axis]<=0).sum())
   ss.append(scores.cpu().numpy())
   if lo==0:
    # Four fixed first assessment cases; no outcome selection. Complete cached
    # inputs and newly computed kernels/statistics, not an encoder/training trace.
    for j in range(4):
     n=length[j];prefix='case'+str(j)+'_'
     trace.update({prefix+k:v for k,v in dict(h=hh[j,:n],roles=rr[j,:n],mask=mm[j,:n],
      angles=angles[j,:n].cpu().numpy(),psi=psi[j,:n].cpu().numpy(),scaled_gaussian_key=scaled[j,:n].cpu().numpy(),
      **{k:detail[k][j,:n,:n].cpu().numpy() for k in ('distance','KQ','KD','KG','valid')},scores=scores[j].cpu().numpy(),
      logits=np.stack([batch_logits[k][j] for k in models]),sealed_logits=np.stack([old[k][j] for k in models]),
      row_index=np.array(idx[j]),target=np.array(data['targets'][idx[j]])).items()})
   cursor+=len(rows);progress('assessment')
   if lo%1024==0:print(json.dumps(dict(phase='assessment',current=cursor,total=4096)),flush=True)
 hook.remove();moment_head.circuit=original
 scores=np.concatenate(ss);logits={k:np.concatenate(v) for k,v in replay.items()};labels=np.asarray(data['targets'][idx],np.int64)
 after={k:state_identity(v.state_dict()) for k,v in models.items()};assert before==after
 assert head_calls==512 and circuit_calls==512 and cursor==4096
 numerical=maxerr<=2e-4 and argmax_mismatches==0 and real16<=1e-12
 predictions={k:v.argmax(-1) for k,v in old.items()};baselines={};metric_error=0.
 for mode,pred in predictions.items():
  cm=np.zeros((40,40),np.int64);np.add.at(cm,(labels,pred),1)
  denominator=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(40),where=denominator!=0)
  macro=float(f1.mean());metric_error=max(metric_error,abs(macro-spec['models'][mode]['all40_macro_f1']))
  baselines[mode]=dict(all40_macro_f1=macro,confusion_matrix=cm.tolist(),correct=int(np.trace(cm)),parameters=spec['models'][mode]['parameters'])
 numerical=numerical and metric_error<=1e-12
 result=summarize(scores,labels,predictions,numerical)
 # Independent NumPy reference for every pair in all four fixed actual cases.
 reference_error=0.
 for j in range(4):
  p='case'+str(j)+'_';xx=trace[p+'h'].astype(np.float64);ps=trace[p+'psi'].astype(np.complex128);gg=trace[p+'scaled_gaussian_key'].astype(np.float64)
  uu=xx/np.linalg.norm(xx,axis=-1,keepdims=True);d=1-uu@uu.T;K=np.abs(ps.conj()@ps.T)**2;pop=np.abs(ps)**2
  kernels=(K,pop@pop.T,np.exp(-np.square(gg[:,None]-gg[None,:]).sum(-1)))
  select=np.triu_indices(len(xx),1);av=[]
  for name,kernel in zip(('KQ','KD','KG'),kernels):
   reference_error=max(reference_error,float(np.max(np.abs(kernel-trace[p+name]))));mass=kernel[select].sum()
   av.append(float(kernel[select]@d[select]/mass-d[select].mean()))
  expected=np.array([av[0]-av[2],av[1]-av[2],len(xx),K[select].mean(),*av[:1],av[2],av[1]])
  reference_error=max(reference_error,float(np.max(np.abs(expected-trace[p+'scores']))),float(np.max(np.abs(d-trace[p+'distance']))))
 numerical=numerical and reference_error<=1e-10
 if not numerical:result=summarize(scores,labels,predictions,False)
 for mode,m in spec['models'].items():assert digest(m['checkpoint'])==m['checkpoint_sha256']
 np.savez_compressed(args.output/'population.private.npz',scores=scores,score_names=np.asarray(NAMES),indices=idx,labels=labels,**{k+'_predictions':v for k,v in predictions.items()})
 np.savez_compressed(args.output/'trace-cases.private.npz',**trace)
 summary=dict(schema='qattn.context-pair-diagnostic.v1',status='COMPLETE',**result,
  elapsed_seconds=time.monotonic()-began,head_forwards=head_calls,native_batched_circuit_calls=circuit_calls,optimizer_updates=0,
  task_rows=4096,all_relations=40,all_four_models_retained=True,retained_baselines=baselines,undefined_reasons=reasons,method_selected=False,goal='ACTIVE')
 verification=dict(status='PASS' if numerical else 'FAIL_NUMERIC',sealed_logit_max_scaled_error=maxerr,argmax_mismatches=argmax_mismatches,
  real16_kernel_error=real16,native_state_norm_error=normerr,independent_numpy_max_error=reference_error,sealed_macro_error=metric_error,
  state_before=before,state_after=after,input_files_verified=len(cfg['input_hashes']),model_checkpoints={k:m['checkpoint_sha256'] for k,m in spec['models'].items()},
  frozen_hypothesis_sha256=cfg['hypothesis_sha256'],private_population_sha256=digest(args.output/'population.private.npz'),
  private_trace_sha256=digest(args.output/'trace-cases.private.npz'),
  trace='First4 assessment cases, full cached h/roles/mask/angle/psi/scaledGkey/allpairkernels/statistics/4modelold-newlogits. Upstream encoder and training trajectories not available; same-agent NumPy reference, not external audit.')
 save(args.output/'summary.json',summary);save(args.output/'verification.json',verification)
 save(args.output/'RUN_COMPLETE',{n:digest(args.output/n) for n in ('summary.json','verification.json')})
 progress('completed');signal.alarm(0);print(json.dumps(dict(status='COMPLETE',decision=result['decision'],auc=result['auc'],gates=result['gates'],seconds=summary['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
