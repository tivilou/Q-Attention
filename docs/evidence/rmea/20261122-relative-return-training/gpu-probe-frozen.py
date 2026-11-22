"""Bounded GPU feasibility: stable task microbatch, fresh native state, three readouts."""
import argparse, hashlib, importlib.util, json, os, signal, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()

def save(path, value):
    temp=path.with_suffix('.pending')
    temp.write_bytes((json.dumps(value,indent=2)+'\n').encode())
    temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','inputs','design','output','progress'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--attempt',required=True);a=p.parse_args()
    cfg=json.loads(a.design.read_bytes());signal.alarm(cfg['timeout_seconds']);start=time.time()
    assert not a.output.exists();a.output.mkdir()
    for name,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/name)==h
    for name,h in cfg['source_hashes'].items():assert digest(a.source/name)==h
    assert digest(a.inputs/'discovery/manifest.json')==cfg['native_manifest_sha256']
    cache=a.inputs/'discovery/cache'
    for name,v in cfg['cache_hashes'].items():
        f=cache/name;assert not f.is_symlink() and f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    sys.path.insert(0,str(a.source))
    spec=importlib.util.spec_from_file_location('baselines.retacred.rmea_relative_return',Path(__file__).parent/'rmea_relative_return.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from baselines.retacred.rmea import RMEAHead, circuit, EPS
    from baselines.retacred.rmea_task_runtime import backward, score_batch
    from baselines.retacred.rmea_token_transport_study import state_identity
    from experiments.diagnostics.rmea_relative_return.relative_return import RelativeReturnHead as OldHead
    assert torch.cuda.device_count()==2
    torch.set_num_threads(2);torch.manual_seed(13);torch.use_deterministic_algorithms(True)
    initial={n:v.clone() for n,v in RMEAHead('quantum').double().state_dict().items()}
    initial_sha=state_identity(initial)
    assert initial_sha=='037232c406d5cb1b23cea0ee289b7ab2d82366a2febfcedfdeb6fc69ec8da369'
    data={n:np.load(cache/('train-'+n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('h','roles','offsets','targets')}
    count=len(data['targets']);assert count==58465
    ids=np.random.default_rng(13).permutation(count)[:32].tolist()
    worst=np.argsort(-np.diff(data['offsets']),kind='stable')[:4].tolist()
    def batch(indices,device):
        lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in indices];n=max(lengths)
        h=np.zeros((len(indices),n,1024),np.float64);roles=np.zeros((len(indices),n,4),np.float64);mask=np.zeros((len(indices),n),bool)
        for j,i in enumerate(indices):
            lo,hi=map(int,data['offsets'][i:i+2]);h[j,:hi-lo]=data['h'][lo:hi];roles[j,:hi-lo]=data['roles'][lo:hi];mask[j,:hi-lo]=True
        return tuple(torch.from_numpy(x).to(device) for x in (h,roles,mask)),torch.from_numpy(np.array(data['targets'][indices],dtype=np.int64)).to(device)
    def worker(device_index,modes):
        device=torch.device('cuda',device_index);rows={}
        with torch.cuda.device(device):
            for mode in modes:
                head=module.RelativeReturnHead(mode).double();head.load_state_dict(initial,strict=True)
                assert state_identity(head.state_dict())==initial_sha and sum(p.numel() for p in head.parameters())==81028
                head=head.to(device);inputs,y=batch(ids,device);before={n:v.detach().clone() for n,v in head.named_parameters()}
                opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
                with torch.no_grad():
                    x=tuple(v[:2] for v in inputs)
                    got=head(*x,details=True)
                    if mode=='native_square':
                        parity=float((got['probability']-got['native']['probability'].square()).abs().max())
                    else:
                        native=RMEAHead('quantum').double();native.load_state_dict(initial);native=native.to(device)
                        old=OldHead(native,circuit,{'return_coherent':'coherent','return_dephased':'dephased'}[mode],EPS)
                        parity=float((got['logits']-old(*x)[0]).abs().max());del old,native
                    norm_error=float((got['alpha'].sum(-1)-1).abs().max())
                    assert parity<1e-10 and norm_error<1e-10 and torch.isfinite(got['logits']).all()
                    del got,x
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);t=time.time()
                opt.zero_grad(set_to_none=True);loss=backward(head,inputs,y)
                norms={n:float(p.grad.norm()) for n,p in head.named_parameters()}
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()) and all(v>0 for v in norms.values())
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
                torch.cuda.synchronize(device);elapsed=time.time()-t
                with torch.no_grad():after=float(torch.nn.functional.cross_entropy(score_batch(head,inputs),y))
                change=max(float((v-before[n]).abs().max()) for n,v in head.named_parameters())
                assert torch.isfinite(loss) and np.isfinite(after) and change>0
                first_peak=torch.cuda.max_memory_allocated(device);del inputs,y,before
                inputs,y=batch(worst,device);opt.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);t=time.time()
                worst_loss=backward(head,inputs,y)
                worst_norms={n:float(p.grad.norm()) for n,p in head.named_parameters() if p.grad is not None}
                assert torch.isfinite(worst_loss) and len(worst_norms)==len(list(head.parameters())) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()) and all(v>0 for v in worst_norms.values())
                torch.cuda.synchronize(device);worst_seconds=time.time()-t
                worst_peak=torch.cuda.max_memory_allocated(device)
                rows[mode]=dict(device=device_index,parameters=81028,flat_initial_state_sha256=initial_sha,parity_error=parity,
                    attention_normalization_error=norm_error,gradient_norms=norms,optimizer_updates=1,
                    train_ce_before=float(loss),train_ce_after=after,parameter_max_update=change,
                    batch32_update_seconds=elapsed,batch32_peak_bytes=first_peak,
                    worst4_backward_seconds=worst_seconds,worst4_peak_bytes=worst_peak,
                    worst4_max_length=inputs[0].shape[1],worst4_optimizer_updates=0,worst4_gradient_norms=worst_norms,
                    checkpoint_saved=False)
                del head,opt,inputs,y,loss,worst_loss
                torch.cuda.empty_cache()
        return rows
    save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='gpu_batch_feasibility',current=0,total=3,unit='condition',observed_at=time.time(),heartbeat_at=time.time()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,0,('return_coherent','native_square')),pool.submit(worker,1,('return_dephased',))]
        rows={}
        for f in futures:rows.update(f.result())
    assert set(rows)==set(cfg['modes'])
    result=dict(status='PASS',profile=cfg['profile'],design_sha256=digest(a.design),
        initial_sha256=initial_sha,batch32_ids=ids,worst4_ids=worst,results=rows,seconds=time.time()-start,
        scope='Two allocated GPUs; one real batch32 update/arm and worst-length microbatch4 backward only. No trained utility or full-study admission, heldout use, saved checkpoints or COS writes.',L5='NOT_ACCEPTED')
    save(a.output/'summary.json',result);save(a.output/'RUN_COMPLETE',dict(attempt=a.attempt,summary_sha256=digest(a.output/'summary.json')))
    save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='complete',current=3,total=3,unit='condition',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps(result))
if __name__=='__main__':main()
