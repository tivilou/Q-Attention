from pathlib import Path
import argparse,json,sys,time
"""Independent operator and NumPy chain-rule checks for the new kernel only."""
import math
import numpy as np


def main():
    if sys.flags.optimize:raise ValueError('Technical checks require assertions enabled')
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=True)
    assert not (args.output/'technical-started.json').exists(),'Retain first execution record'
    sys.path.insert(0,str(args.source))
    from baselines.retacred import rmea_corroboration_protocol as p
    start=time.time();p.atomic_json(args.output/'technical-started.json',dict(time=start,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract())))
    cfg=p.config('discovery');cache=args.inputs/'cache';manifest=args.inputs/'manifest.json'
    assert p.digest(manifest)==p.BASE_MANIFEST
    pins=p.read(manifest)['cache']
    def pinned(name):
        f=cache/name;pin=pins[name]
        assert not f.is_symlink() and f.is_file() and f.stat().st_size==pin['bytes'] and p.digest(f)==pin['sha256']
        return f
    import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_corroboration_factory import initialize_head,validate_initialized_head
    from baselines.retacred.rmea_matched_routers import initialize_head as native_initialize
    from baselines.retacred.rmea_frozen_training import FrozenFeatureTraining
    from baselines.retacred import rmea_corroboration_runtime as runtime
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    math=dict(reused_exact_screen_plugin_sha256='6174cb5aba15c106ed432dbe08c6f750dae6b70499a32be34797e9d744f0b131',new_scope='full-factory and schema integration only')
    torch.manual_seed(13);initial=RMEAHead('quantum').float().state_dict()
    roles=np.load(pinned('train-roles.npy'),mmap_mode='r',allow_pickle=False)
    offsets=np.load(pinned('train-offsets.npy'),mmap_mode='r',allow_pickle=False)
    features=np.load(pinned('train-h.npy'),mmap_mode='r',allow_pickle=False)
    targets=np.load(pinned('train-targets.npy'),mmap_mode='r',allow_pickle=False)
    lengths=np.diff(offsets[:9]);n=int(max(lengths))
    h=np.zeros((8,n,1024),np.float32);r=np.zeros((8,n,4),np.float32);mask=np.zeros((8,n),bool)
    for i,length in enumerate(lengths):
        a,b=map(int,offsets[i:i+2]);h[i,:length]=features[a:b];r[i,:length]=roles[a:b];mask[i,:length]=True
    x=tuple(torch.from_numpy(a).to('cuda:0') for a in (h,r,mask));y=torch.tensor(np.array(targets[:8]),device='cuda:0',dtype=torch.long)
    rows={};init_states=[]
    for kind in p.MODES:
        model=initialize_head(kind,initial,cfg);validate_initialized_head(model,kind,initial,cfg);model.cuda()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        if init_states:assert all(torch.equal(v,init_states[0][k]) for k,v in before.items())
        else:init_states.append(before)
        assert sum(v.numel() for v in model.parameters())==81028
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])
        z=model(*x);assert torch.isfinite(z).all()
        padded=tuple(F.pad(a,(0,0,0,3)) if a.ndim==3 else F.pad(a,(0,3)) for a in x)
        error=float((z-model(*padded)).abs().max());assert error<2e-5
        loss=F.cross_entropy(z,y);assert torch.isfinite(loss)
        loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_norm']);optimizer.step()
        changed=any(not torch.equal(v,before[k]) for k,v in model.state_dict().items());assert changed
        rows[kind]=dict(parameters=81028,padding_error=error,initial_train_ce=float(loss.detach()),
            finite_gradients=True,real_train_updates=1,state_changed=True)
        del model,optimizer,z,loss,before
    row=dict(gate=True,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()),config_sha256=p.identity(cfg),
        input_manifest_sha256=p.digest(manifest),math=math,conditions=rows,test_model_rows=0,test_target_reads=0,real_train_updates_per_condition=1,
        environment=FrozenFeatureTraining(p,runtime).execution_environment(),seconds=time.time()-start,
        scope='Changed formulas/model only; no VALID/TEST reads/inference, utility gate or lifecycle recertification')
    p.atomic_json(args.output/'unit-validation.json',row)
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={'unit-validation.json':p.digest(args.output/'unit-validation.json')}))
    print(json.dumps(dict(gate=True,conditions=list(rows),seconds=row['seconds'],math=math,test_model_rows=0)))

if __name__=='__main__':main()
