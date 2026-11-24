"""Scoped technical check of the new query-state transport; no utility screen."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    assert sha(__file__)==cfg['code_sha256']
    source=Path(cfg['source']);cache=Path(cfg['cache']);plugin=a.design.parent/'rmea_query_transport.py'
    assert sha(plugin)==cfg['plugin_sha256']
    for n,h in cfg['source_hashes'].items():assert sha(source/n)==h,n
    for n in ('train-roles.npy','train-offsets.npy'):assert sha(cache/n)==cfg['train_cache'][n]['sha256']
    assert (cache/'train-h.npy').stat().st_size==cfg['train_cache']['train-h.npy']['bytes']
    sys.path.insert(0,str(source));import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import RMEAHead,circuit
    from baselines.retacred.rmea_entity_query import real_circuit
    from baselines.retacred import rmea_matched_routers as native
    name='baselines.retacred.rmea_query_transport';spec=importlib.util.spec_from_file_location(name,plugin)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic()
    torch.manual_seed(13);angles=torch.randn(8,12,dtype=torch.float64)
    fixtures={}
    for real in (False,True):
        state=torch.zeros(8,8,dtype=torch.float64 if real else torch.complex128);state[:,0]=1
        out=module.transport(angles,state,real=real)
        expected=real_circuit(angles) if real else circuit(angles.reshape(8,2,3,2))
        error=float((out-expected).abs().max());assert error<1e-12
        fixtures['real' if real else 'complex']=dict(basis_circuit_error=error,norm_error=float((out.abs().square().sum(-1)-1).abs().max()))
    data={n:np.load(cache/('train-'+n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('h','roles','offsets','targets')}
    assert sha(cache/'train-targets.npy')==cfg['train_targets_sha256']
    lengths=np.diff(data['offsets'][:9]);h=np.zeros((8,int(lengths.max()),1024),np.float32)
    r=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
    for i in range(8):
        left,right=map(int,data['offsets'][i:i+2]);n=right-left
        h[i,:n]=data['h'][left:right];r[i,:n]=data['roles'][left:right];mask[i,:n]=True
    x=tuple(map(torch.from_numpy,(h,r,mask)));y=torch.tensor(np.array(data['targets'][:8]),dtype=torch.long)
    torch.manual_seed(13);initial=RMEAHead('quantum').float().state_dict()
    train=dict(seed=13,training_dtype='float32',parameter_schemas={m:native.parameter_schema(module.native_mode(m)) for m in module.MODES})
    rows={}
    for kind in module.MODES:
        model=module.initialize_head(kind,initial,train);module.validate_initialized_head(model,kind,initial,train)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        key,q1=model.states(*x);_,q0=model.states(*x,conditioned=False)
        g=lambda q:torch.einsum('brd,bsd->brs',q.conj(),q)
        gram_error=float((g(q1)-g(q0)).abs().max());assert gram_error<3e-6
        norm_error=float((q1.abs().square().sum(-1)-1).abs().max());assert norm_error<3e-6
        z=model(*x);assert torch.isfinite(z).all()
        padded=tuple(F.pad(v,(0,0,0,3)) if v.ndim==3 else F.pad(v,(0,3)) for v in x)
        padding_error=float((z-model(*padded)).abs().max());assert padding_error<2e-5
        parity=None
        if kind!='real_circuit_transport':
            old=native.initialize_head(module.native_mode(kind),initial,train)
            parity=float((model(*x,conditioned=False)-old(*x)).abs().max());assert parity<2e-5
        loss=F.cross_entropy(z,y);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
        rows[kind]=dict(parameters=sum(v.numel() for v in model.parameters()),gram_error=gram_error,norm_error=norm_error,padding_error=padding_error,disabled_native_error=parity,initial_train_ce=float(loss.detach()),finite_gradients=True,real_train_updates=1)
        save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='query_transport_changed_plugin_technical',current=len(rows),total=3,unit='condition',observed_at=time.time(),heartbeat_at=time.time()))
    result=dict(gate=True,plugin_sha256=sha(plugin),code_sha256=sha(__file__),fixtures=fixtures,conditions=rows,seconds=time.monotonic()-start,test_model_rows=0,test_target_reads=0,valid_model_rows=0,cos_writes=0,utility_screen=False,source_sha256=cfg['input_hashes']['source'],limitations=['Technical readiness only; no heldout utility, novelty, L3 or T claim.','Core multiGPU/resume/COS capabilities are reused without a lifecycle matrix.'])
    save(a.output/'summary.json',result);save(a.output/'RUN_COMPLETE',dict(summary_sha256=sha(a.output/'summary.json'),code_sha256=cfg['code_sha256']))
    print(json.dumps(result))
if __name__=='__main__':main()
