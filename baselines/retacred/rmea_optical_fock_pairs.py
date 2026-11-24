"""Post-selection outcome counts; no fitting, selection or raw-data publication."""
from pathlib import Path
import numpy as np

def paired_counts(output,cfg,results,native,native_manifest):
    import torch
    from . import rmea_optical_fock_protocol as p
    from .rmea_optical_fock import initialize_head
    from .rmea_matched_routers import initialize_head as native_initialize
    from .rmea import RMEAHead
    from .rmea_token_transport_study import state_identity
    from .rmea_fair_reference_runtime import require
    output=Path(output)
    # This is the pre-existing accepted full comparator's local realization.
    # Its selected states remain dependencies of this campaign and must be
    # restored from that run's pinned recovery before final analysis if lost.
    old=Path('/hy-tmp/Q-Attention/owner/attempts/9a28320fd4174c8f95544c9ba96eb1ae/output/20261123T221843Z_seed13/fair-reference-seed13-20261123T221843Z/discovery')
    cache=output/'cache';data={k:np.load(cache/('test-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    n=cfg['counts']['test'];gold=np.asarray(data['targets'])
    require(len(gold)==n and len(data['offsets'])==n+1,'Aligned TEST identity differs')
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed']);initial=RMEAHead('quantum').float().state_dict()
    decisions={};identities={}
    with torch.no_grad():
        for name,row in list(results.items())+[('native/'+k,v) for k,v in native['results'].items()]:
            is_native=name.startswith('native/');kind=name.split('/')[-1]
            root=old if is_native else output;chosen=row['evaluation_checkpoint']
            path=root/chosen['path']
            require(chosen['path']=='checkpoints/'+kind+'-selected.pt' and path.is_file() and not path.is_symlink()
                and p.digest(path)==chosen['sha256'],'Paired analysis selected checkpoint differs')
            state=torch.load(path,map_location='cpu',weights_only=True)
            require(state_identity(state)==chosen['state_sha256'],'Paired analysis state differs')
            model=(native_initialize(kind,initial,native_manifest['config']) if is_native else initialize_head(kind,initial,cfg))
            model.load_state_dict(state);model.cuda().eval();pred=[]
            for start in range(0,n,cfg['eval_batch_size']):
                end=min(start+cfg['eval_batch_size'],n);lengths=np.diff(data['offsets'][start:end+1]);width=int(max(lengths))
                h=np.zeros((end-start,width,1024),np.float32);r=np.zeros((end-start,width,4),np.float32);mask=np.zeros((end-start,width),bool)
                for j,i in enumerate(range(start,end)):
                    a,b=map(int,data['offsets'][i:i+2]);length=b-a
                    h[j,:length]=data['h'][a:b];r[j,:length]=data['roles'][a:b];mask[j,:length]=True
                logits=model(*(torch.from_numpy(a).cuda() for a in (h,r,mask)))
                require(bool(torch.isfinite(logits).all()),'Nonfinite selected analysis logits')
                pred.extend(logits.argmax(-1).tolist())
            decisions[name]=np.asarray(pred,dtype=np.int64)==gold
            identities[name]=dict(sha256=chosen['sha256'],state_sha256=chosen['state_sha256'],epoch=chosen['epoch'])
            del model,state
    q=decisions.pop('fock1');counts={}
    for kind,c in decisions.items():
        row=dict(repaired=int(np.sum(q&~c)),harmed=int(np.sum(~q&c)),both_correct=int(np.sum(q&c)),both_wrong=int(np.sum(~q&~c)))
        require(sum(row.values())==n,'Paired partition incomplete');counts[kind]=row
    require(len(counts)==9,'All nine outcome comparisons required')
    return dict(split='test',rows=n,phase='Post-selection descriptive replay; no reselection or updates',
        selected_states=identities,comparisons=counts,raw_predictions_published=False)
