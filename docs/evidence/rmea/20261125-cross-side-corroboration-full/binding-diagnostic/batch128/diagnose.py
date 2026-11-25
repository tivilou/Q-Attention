"""Fixed TRAIN context-summary permutation; immutable screened models, no fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path, value):
    pending=path.with_suffix('.pending')
    pending.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
    pending.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('design','allocation','output','progress'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--attempt',required=True)
    args=ap.parse_args();cfg=json.loads(args.design.read_bytes())
    a=json.loads(args.allocation.read_bytes())['allocation']
    assert a['runner_attempt_id']==args.attempt and len(set(a['gpu_uuids']))==2
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(a['gpu_uuids'])
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    signal.alarm(cfg['timeout_seconds']);started=time.monotonic()
    for n,h in cfg['code_hashes'].items():assert sha(Path(__file__).parent/n)==h
    for path,h in cfg['input_hashes'].items():assert sha(path)==h
    args.output.mkdir(exist_ok=True)
    assert not (args.output/'started.json').exists(), 'No repeated scientific attempt'
    save(args.output/'started.json',dict(attempt=args.attempt,design_sha256=sha(args.design)))
    import numpy as np
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,cfg['source'])
    sys.path.insert(0,str(Path(__file__).parent))
    import plugin
    from baselines.retacred.rmea import circuit, EPS
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2
    parent=Path(cfg['parent']);cache=Path(cfg['cache'])
    with np.load(parent/'output/diagnostic.private.npz',allow_pickle=False) as z:
        indices=z['assessment_indices'].copy();groups=z['groups'][indices].copy()
    assert len(indices)==4096 and len(set(map(int,indices)))==4096
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
          for k in ('h','roles','offsets')}
    side_counts=[]
    for i in indices:
        lo,hi=map(int,data['offsets'][i:i+2]);roles=data['roles'][lo:hi]
        s=np.abs(roles[:,0])<=np.abs(roles[:,1])
        side_counts.append([int(s.sum()),int((~s).sum())])
    side_counts=np.asarray(side_counts)
    eligible=np.flatnonzero(side_counts.min(1)>0)
    assert len(eligible)>=cfg['minimum_eligible_fraction']*len(indices)
    digest=lambda b:hashlib.sha256(b).hexdigest()
    order=np.array(sorted(eligible,key=lambda i:digest((cfg['permutation_prefix']+str(int(indices[i]))).encode())))
    donors=np.arange(len(indices))
    for shift in range(1,len(order)):
        proposed=np.roll(order,shift)
        if np.all(groups[order]!=groups[proposed]):
            donors[order]=proposed;break
    else:raise AssertionError('No outcome-blind different-sentence derangement')
    assert np.array_equal(np.sort(donors),np.arange(len(indices)))
    assert np.all(donors[eligible]!=eligible) and np.all(groups[eligible]!=groups[donors[eligible]])
    assert np.array_equal(np.sort(donors[eligible]),np.sort(eligible))
    # Freeze the exact intervention BEFORE labels or new model outputs are read.
    np.save(args.output/'donors.private.npy',donors,allow_pickle=False)
    mapping=dict(items=4096,eligible=int(len(eligible)),ineligible_identity=int(len(indices)-len(eligible)),
        shift=int(shift),donor_sha256=sha(args.output/'donors.private.npy'),label_blind=True,
        same_sentence_donors_eligible=int((groups[eligible]==groups[donors[eligible]]).sum()),
        same_sentence_donors_all=int((groups==groups[donors]).sum()),
        mean_abs_object_count_delta=float(np.abs(side_counts[:,1]-side_counts[donors,1]).mean()),
        matching='Global exact-sentence derangement, not length/type matched; OOD dependence diagnostic only.')
    assert mapping['donor_sha256']==cfg['expected_donor_sha256']
    save(args.output/'mapping.json',mapping)
    targets=np.load(cache/'train-targets.npy',allow_pickle=False)[indices]
    assert min(np.bincount(targets,minlength=40))>0
    parent_summary=json.loads((parent/'output/summary.json').read_bytes())
    with np.load(parent/'output/diagnostic.private.npz',allow_pickle=False) as z:
        prior_logits={k:z[k+'_logits'].copy() for k in cfg['modes']}

    def batch(positions,device):
        ids=indices[positions]
        lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
        h=np.zeros((len(ids),max(lengths),1024),np.float32)
        roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
        for j,i in enumerate(ids):
            lo,hi=map(int,data['offsets'][i:i+2]);n=hi-lo
            h[j,:n]=data['h'][lo:hi];roles[j,:n]=data['roles'][lo:hi];mask[j,:n]=True
        return tuple(torch.from_numpy(v).to(device) for v in (h,roles,mask))

    def intervene(model,rec,ref):
        h,roles,mask=rec;dh,dr,dm=ref;a=model.head
        rawkey=a.key(torch.cat((h,roles),-1));draw=a.key(torch.cat((dh,dr),-1))
        k,q,dkey=plugin.scaled_key(rawkey),plugin.scaled_key(a.query),plugin.scaled_key(draw)
        base=torch.exp(-(k[:,None]-q[None,:,None]).square().sum(-1))
        subject,own_obj=plugin.sides(roles,mask);_,obj=plugin.sides(dr,dm)
        count=obj.sum(-1).clamp_min(1).to(h.dtype)
        if model.kind=='quantum_corroboration':
            psi=circuit((torch.pi*torch.tanh(rawkey)).reshape(h.shape[:2]+(2,3,2)))
            other=circuit((torch.pi*torch.tanh(draw)).reshape(dh.shape[:2]+(2,3,2)))
            rho=torch.einsum('bti,btj,bt->bij',other,other.conj(),obj.to(other.dtype))/count[:,None,None]
            gate=torch.einsum('bti,bij,btj->bt',psi.conj(),rho,psi).real.clamp_min(0)
        else:
            assert model.kind=='gaussian_corroboration'
            distance=(k.square().sum(-1)[:,:,None]+dkey.square().sum(-1)[:,None,:]
                      -2*torch.einsum('btd,bsd->bts',k,dkey)).clamp_min(0)
            gate=(torch.exp(-distance)*obj[:,None]).sum(-1)/count[:,None]
        gate=torch.where(subject & own_obj.any(-1)[:,None] & obj.any(-1)[:,None],gate,torch.ones_like(gate))
        raw=(base*gate[:,None]+EPS)*mask[:,None]
        alpha=raw/raw.sum(-1,keepdim=True)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

    lock=threading.Lock();cursors={k:0 for k in cfg['modes']}
    def progress(kind,n):
        with lock:
            cursors[kind]=n
            save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,
                phase='context_permutation',current=sum(cursors.values()),total=8192,unit='evaluated_example',
                heartbeat_at=time.time(),observed_at=time.time(),children={k:dict(current=v,total=4096) for k,v in cursors.items()}))

    def worker(kind,gpu):
        device=torch.device('cuda',gpu);checkpoint=parent/'output'/kind/'final.pt'
        assert sha(checkpoint)==parent_summary['results'][kind]['final_sha256']
        model=plugin.make_head(kind).float()
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(weights);model.to(device);model.eval()
        actual=[];shuffled=[];identity_error=0.
        with torch.cuda.device(device),torch.inference_mode():
            for lo in range(0,4096,cfg['batch_size']):
                pos=np.arange(lo,min(lo+cfg['batch_size'],4096));rec=batch(pos,device);ref=batch(donors[pos],device)
                z=model(*rec);identity=intervene(model,rec,rec);perm=intervene(model,rec,ref)
                assert torch.isfinite(z).all() and torch.isfinite(identity).all() and torch.isfinite(perm).all()
                error=float((z-identity).abs().max());identity_error=max(identity_error,error)
                assert torch.allclose(z,identity,atol=cfg['replay_atol'],rtol=cfg['replay_rtol'])
                actual.append(z.cpu().numpy());shuffled.append(perm.cpu().numpy());progress(kind,int(pos[-1])+1)
        actual=np.concatenate(actual);shuffled=np.concatenate(shuffled)
        replay_stats=dict(max_abs_error=float(np.max(np.abs(actual-prior_logits[kind]))),argmax_mismatches=int((actual.argmax(-1)!=prior_logits[kind].argmax(-1)).sum()),batch_size=cfg['batch_size'])
        save(args.output/('replay-'+kind+'.json'),replay_stats)
        assert np.allclose(actual,prior_logits[kind],atol=cfg['replay_atol'],rtol=cfg['replay_rtol']), replay_stats
        assert np.array_equal(actual.argmax(-1),prior_logits[kind].argmax(-1))
        assert sha(checkpoint)==parent_summary['results'][kind]['final_sha256']
        return dict(actual=actual,shuffled=shuffled,identity_max_error=identity_error,
                    prior_max_error=float(np.max(np.abs(actual-prior_logits[kind]))))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={k:pool.submit(worker,k,i) for i,k in enumerate(cfg['modes'])}
        results={k:v.result() for k,v in futures.items()}
    def metrics(z):
        p=z.argmax(-1);cm=np.bincount(40*targets+p,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        ce=float(F.cross_entropy(torch.from_numpy(z),torch.tensor(targets)).item())
        return dict(macro=float(f.mean()),ce=ce,confusion_matrix=cm.tolist(),correct=int((p==targets).sum()))
    rows={k:dict(original=metrics(v['actual']),permuted=metrics(v['shuffled']),
                 identity_max_error=v['identity_max_error'],prior_max_error=v['prior_max_error']) for k,v in results.items()}
    for kind,row in rows.items():
        original=row['original'];pinned=parent_summary['results'][kind]
        assert original['confusion_matrix']==pinned['confusion_matrix']
        assert original['correct']==pinned['assessment_correct']
        cm=original['confusion_matrix']
        assert sum(cm[y][0] for y in range(1,40))==pinned['positive_to_null']
        assert sum(cm[0][1:])==pinned['null_false_positive']
        assert abs(original['macro']-pinned['all40_macro_f1'])<=cfg['macro_replay_atol']
        assert abs(original['ce']-pinned['ce'])<=cfg['ce_replay_atol']
    drops={k:v['original']['macro']-v['permuted']['macro'] for k,v in rows.items()}
    q,g=cfg['modes'];interaction=drops[q]-drops[g]
    gates=dict(quantum_macro_depends_on_original_context=drops[q]>0,
               quantum_relative_advantage_depends_on_original_context=interaction>0)
    np.savez(args.output/'diagnostic.private.npz',indices=indices,donors=donors,labels=targets,
             **{k+'_'+which:v[which] for k,v in results.items() for which in ('actual','shuffled')})
    summary=dict(status='completed',attempt=args.attempt,profile=cfg['profile'],seconds=time.monotonic()-started,
        results=rows,macro_drops=drops,quantum_minus_classical_drop=interaction,gates=gates,
        gate='PASS_FOR_FULL_SINGLE_SEED_PREPARATION' if all(gates.values()) else 'DEFER_EXACT_BINDING_PROFILE',
        mapping=mapping,code_hashes=cfg['code_hashes'],input_hashes=cfg['input_hashes'],design_sha256=sha(args.design),
        private_diagnostic_sha256=sha(args.output/'diagnostic.private.npz'),optimizer_updates=0,cos_writes=0,
        original_metrics_replay_verified=True,
        valid_evaluated=False,test_evaluated=False,L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',
        limits='Global donor summary is OOD; this measures routing dependence, not semantic binding causality or theory novelty. All4096 retained, immutable screened checkpoints, no new model or seed.')
    save(args.output/'summary.json',summary)
    save(args.output/'RUN_COMPLETE',{n:sha(args.output/n) for n in ('summary.json','mapping.json')})
    save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='completed',
        current=8192,total=8192,unit='evaluated_example',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps(dict(gate=summary['gate'],macro_drops=drops,interaction=interaction,seconds=summary['seconds'])),flush=True)


if __name__=='__main__':main()
