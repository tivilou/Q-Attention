"""One frozen no-fit native-Q TRAIN pass, all40 relations, no changed weights."""
from pathlib import Path
import argparse,hashlib,json,signal,sys,time
import numpy as np
from tangent_math import slopes_torch,operator_numpy,statistics,decide,NAMES
import trace_io
from independent_reference import circuit

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
    ap=argparse.ArgumentParser()
    for k in ('root','design','output','progress'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--attempt',required=True);args=ap.parse_args();began=time.monotonic()
    cfg=json.loads(args.design.read_bytes());f=cfg['fixed_inputs']
    assert cfg['batch_size']==64 and cfg['wall_seconds']==600
    def expired(*_):raise TimeoutError('Frozen600-second attempt expired; no automatic retry')
    signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    assert not args.output.exists();args.output.mkdir();cursor=forwards=0
    def progress(phase):
        save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase=phase,current=cursor,total=12288,
            unit='example_pass',observed_at=time.time(),heartbeat_at=time.time()))
    progress('verify_fixed_inputs')
    for n,h in cfg['code_hashes'].items():assert digest(args.design.parent/n)==h,n
    for n,h in cfg['input_hashes'].items():assert Path(n).resolve().is_relative_to(args.root.resolve()) and digest(n)==h,n
    cohort=json.loads(Path(f['cohort_path']).read_bytes());fit,assessment=cohort['fit'],cohort['assessment']
    assert (len(fit),len(assessment))==(8192,4096) and len(set(fit+assessment))==12288
    assert not {cohort['selected_groups'][str(i)] for i in fit}&{cohort['selected_groups'][str(i)] for i in assessment}
    completed=args.root/f['completed_relative'];cache=completed/'cache'
    metadata=json.loads((cache/'metadata.json').read_bytes())['train'];assert len(metadata)==58465
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    assert len(data['offsets'])==58466 and data['offsets'][-1]==len(data['h'])==len(data['roles']) and len(data['targets'])==58465
    sys.path.insert(0,str(args.root/f['source_relative']))
    import torch
    from baselines.retacred.rmea_matched_routers import Head
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==1;device=torch.device('cuda',0)
    state=torch.load(completed/f['checkpoint_relative'],map_location='cpu',weights_only=True)
    assert state_identity(state)==f['state_sha256']
    model=Head('quantum').double().to(device);model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    before=state_identity(model.state_dict());head=model.head
    errors={k:0. for k in ('psi_norm','phi_norm','alpha_sum','logits','self_identity','bound_violation','numpy_slope','numpy_residual','circuit_psi','circuit_phi')}
    samples=[];results={};private_results={};max_scale=1.
    with torch.inference_mode():
        for split,indices in [('fit',fit),('assessment',assessment)]:
            ss=[];ll=[];yy=[]
            for lo in range(0,len(indices),64):
                idx=indices[lo:lo+64];lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in idx];n=max(lengths)
                hh=np.zeros((len(idx),n,1024),np.float64);rr=np.zeros((len(idx),n,4),np.float64);mm=np.zeros((len(idx),n),bool)
                for j,i in enumerate(idx):
                    a,b=map(int,data['offsets'][i:i+2]);assert metadata[i]['words']==b-a
                    hh[j,:b-a]=data['h'][a:b];rr[j,:b-a]=data['roles'][a:b];mm[j,:b-a]=True
                h,roles,mask=(torch.from_numpy(x).to(device) for x in (hh,rr,mm))
                d=head(h,roles,mask,True);forwards+=1;s=slopes_torch(d,mask)
                assert all(torch.isfinite(v).all() for v in list(d.values())+list(s.values()) if torch.is_tensor(v))
                def err(k,v):errors[k]=max(errors[k],float(v.abs().max().item()))
                err('psi_norm',d['psi'].abs().square().sum(-1)-1);err('phi_norm',d['phi'].abs().square().sum(-1)-1)
                err('alpha_sum',d['alpha'].sum(-1)-1);err('logits',s['z']+head.bias-d['logits'])
                assert float(d['probability'].min())>=-1e-12 and float(d['probability'].max())<=1+1e-12
                self_identity=2*((d['utilities']-s['z'][:,:,None]).square()*d['probability']*(1-d['probability'])*mask[:,None]).sum(-1)/d['normalizer'].squeeze(-1)
                err('self_identity',(s['self']-self_identity)/(1+s['self'].abs()))
                err('bound_violation',torch.clamp(s['g']-s['upper'],min=0)/(1+s['upper']))
                max_scale=max(max_scale,float(s['g'].max()),float(d['utilities'].abs().max()))
                # Label decode follows native prediction AND all classwise slopes.
                labels=np.asarray(data['targets'][idx],dtype=np.int64);assert ((labels>=0)&(labels<40)).all()
                scores=torch.stack([s[k] for k in NAMES],-1).cpu().numpy();logits=d['logits'].cpu().numpy()
                ss.append(scores);ll.append(logits);yy.append(labels)
                if lo==0:
                    for j in range(4):
                        m=lengths[j];arrays={k:d[k][j,:,:m].cpu().numpy().copy() for k in ('utilities','probability','alpha')}
                        arrays.update({k:d[k][j,:m].cpu().numpy().copy() for k in ('angles','psi','values')})
                        arrays.update({k:d[k].cpu().numpy().copy() for k in ('effect_angles','phi')})
                        arrays.update(normalizer=d['normalizer'][j].cpu().numpy().copy(),logits=logits[j].copy(),h=hh[j,:m].copy(),roles=rr[j,:m].copy(),mask=mm[j,:m].copy(),
                                      decoder=head.decoder.cpu().numpy().copy(),bias=head.bias.cpu().numpy().copy(),target=np.asarray([labels[j]],np.int64))
                        arrays.update({k:v[j].cpu().numpy().copy() for k,v in s.items()})
                        reference=operator_numpy(arrays['psi'],arrays['phi'],arrays['utilities'])
                        for k,rkey in [('g','numpy_slope'),('residual','numpy_residual')]:
                            errors[rkey]=max(errors[rkey],float(np.max(abs(arrays[k]-reference[k])/(1+abs(reference[k])))))
                        pp=np.stack([circuit(a) for a in arrays['angles']]);qq=np.stack([circuit(a) for a in arrays['effect_angles']])
                        errors['circuit_psi']=max(errors['circuit_psi'],float(np.max(abs(pp-arrays['psi']))))
                        errors['circuit_phi']=max(errors['circuit_phi'],float(np.max(abs(qq-arrays['phi']))))
                        samples.append(trace_io.capture(args.output,split,j,idx[j],metadata[idx[j]],arrays,f['checkpoint_sha256']))
                cursor+=len(idx);progress(split)
                if lo%1024==0:print(json.dumps(dict(phase=split,current=cursor,total=12288)),flush=True)
            scores,logits,labels=map(np.concatenate,(ss,ll,yy));metrics,aux=statistics(scores,logits,labels);results[split]=metrics
            private_results.update({split+'_'+k:v for k,v in dict(scores=scores,logits=logits,target=labels,indices=np.asarray(indices),**aux).items()})
    assert forwards==192 and cursor==12288
    after=state_identity(model.state_dict());assert before==after and digest(completed/f['checkpoint_relative'])==f['checkpoint_sha256']
    np.savez_compressed(args.output/'population.private.npz',**private_results)
    trace=trace_io.close(args.output,samples,cfg,digest(args.design),args.design.parent/'source')
    numerical=all(v<=1e-8 for v in errors.values())
    decision=decide(results['assessment'],numerical)
    summary=dict(schema='qattn.tangent-label-summary.v1',status='COMPLETE',**decision,metrics=results,
        elapsed_seconds=time.monotonic()-began,native_head_forwards=forwards,native_batched_circuit_calls=2*forwards,optimizer_updates=0,
        task_rows=cursor,all_relations=40,exposure='Full TRAIN-trained teacher; no independent confirmation',
        trace=trace,old431='NOT_ASSESSED: exact certificate IDs not mapped',model_admission=False,L3_L4T='NOT_ACCEPTED')
    verification=dict(status='PASS' if numerical else 'FAIL_NUMERIC',errors=errors,state_before=before,state_after=after,stored_state_sha256=f['state_sha256'],
        checkpoint_sha256=f['checkpoint_sha256'],input_files_verified=len(cfg['input_hashes']),trace=trace,independent_implementation='same agent dense A/B/C + Kronecker circuit; no external review')
    save(args.output/'summary.json',summary);save(args.output/'verification.json',verification)
    save(args.output/'RUN_COMPLETE',dict(summary_sha256=digest(args.output/'summary.json'),verification_sha256=digest(args.output/'verification.json')))
    progress('completed');signal.alarm(0);print(json.dumps(dict(status='COMPLETE',**decision,elapsed_seconds=summary['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
