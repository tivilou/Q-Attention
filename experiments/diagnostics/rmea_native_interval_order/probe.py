"""Fixed-decoder, rebiasing-robust shared-pooling decision-order feasibility."""
import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)


def match(lower,upper,gold,strict_correct,gap):
    """Stored-order positives; lowest eligible null L; null indices consumed globally."""
    null=np.flatnonzero((gold==0)&strict_correct);used=np.zeros(len(gold),dtype=bool)
    lists={r:null[np.lexsort((null,lower[null,r-1]))] for r in range(1,40)}
    pairs=[]
    for i in np.flatnonzero((gold!=0)&strict_correct):
        r=int(gold[i]);ordered=lists[r]
        start=int(np.searchsorted(lower[ordered,r-1],upper[i,r-1]+gap,side='left'))
        for j in ordered[start:]:
            if not used[j]:
                assert i!=j and lower[j,r-1]-upper[i,r-1]>=gap
                used[j]=True;pairs.append(dict(positive=int(i),null=int(j),relation=r,
                    positive_upper=float(upper[i,r-1]),null_lower=float(lower[j,r-1]),
                    gap=float(lower[j,r-1]-upper[i,r-1])))
                break
    return pairs


def math_fixture():
    lo=np.zeros((6,39));hi=np.ones((6,39));y=np.array([1,2,0,0,1,0]);ok=np.ones(6,dtype=bool)
    lo[2,0]=2;lo[3,0]=2;lo[5,0]=3;lo[3,1]=2;hi=np.maximum(hi,lo)
    p=match(lo,hi,y,ok,.05)
    assert [(x['positive'],x['null'],x['relation']) for x in p]==[(0,2,1),(1,3,2),(4,5,1)]
    assert not match(lo,np.full((6,39),10.),y,ok,.05)
    # The same arbitrary bias shifts both intervals; a positive fixed gap survives.
    assert (2.+1234.)-(1.+1234.)==1.
    return dict(status='PASS',positive_pairs=3,global_disjoint=True,wide_interval_rejected=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','checkpoint','design','output','progress'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--attempt',required=True);args=p.parse_args();signal.alarm(1800)
    cfg=json.loads(args.design.read_bytes());args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'summary.json').exists(),'No repeat, reselection or rescue'
    assert digest(args.checkpoint)==cfg['checkpoint_sha256']
    assert digest(args.inputs/'discovery/manifest.json')==cfg['native_manifest_sha256']
    assert digest(Path(__file__))==cfg['code_sha256']
    for name,sha in cfg['source_hashes'].items():assert digest(args.source/name)==sha
    import torch
    sys.path.insert(0,str(args.source));from baselines.retacred.rmea import RMEAHead
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=RMEAHead('quantum').double().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint,map_location='cuda',weights_only=True));model.requires_grad_(False)
    cache=args.inputs/'discovery/cache'
    data={split:{k:np.load(cache/(split+'-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
          for k in ('h','roles','offsets','targets')} for split in ('train','valid')}
    assert len(data['train']['targets'])==58465 and len(data['valid']['targets'])==19584
    started=time.time();processed=0;replay_error=0.;fixture_error=0.;bounds={};valid_logits=[]

    def batch(split,indices):
        d=data[split];lengths=[int(d['offsets'][i+1]-d['offsets'][i]) for i in indices]
        h=torch.zeros(len(indices),max(lengths),1024,dtype=torch.float64,device='cuda')
        roles=torch.zeros(len(indices),max(lengths),4,dtype=torch.float64,device='cuda')
        mask=torch.zeros(len(indices),max(lengths),dtype=torch.bool,device='cuda')
        for j,(i,n) in enumerate(zip(indices,lengths)):
            a,b=map(int,d['offsets'][i:i+2])
            h[j,:n]=torch.from_numpy(np.array(d['h'][a:b],dtype=np.float64,copy=True)).cuda()
            roles[j,:n]=torch.from_numpy(np.array(d['roles'][a:b],dtype=np.float64,copy=True)).cuda();mask[j,:n]=True
        return h,roles,mask

    with torch.inference_mode():
        math_check=math_fixture()
        for split,count in [('train',cfg['train_fixture_items']),('valid',19584)]:
            low=[];high=[]
            for lo in range(0,count,16):
                indices=list(range(lo,min(lo+16,count)));h,roles,mask=batch(split,indices)
                detail=model(h,roles,mask,details=True);u=detail['utilities']
                contrast=u[:,1:,:]-u[:,:1,:]
                lower=contrast.masked_fill(~mask[:,None,:],float('inf')).amin(-1)
                upper=contrast.masked_fill(~mask[:,None,:],-float('inf')).amax(-1)
                replay=(detail['alpha']*u).sum(-1)+model.bias
                replay_error=max(replay_error,float((replay-detail['logits']).abs().max()))
                assert torch.isfinite(lower).all() and torch.isfinite(upper).all() and torch.isfinite(replay).all()
                if split=='train':
                    direct=torch.einsum('btd,rd->brt',model.value(h),model.decoder[1:]-model.decoder[:1])
                    fixture_error=max(fixture_error,float((direct[mask[:,None,:].expand_as(direct)]-
                                        contrast[mask[:,None,:].expand_as(direct)]).abs().max()))
                else:
                    low.append(lower.cpu().numpy());high.append(upper.cpu().numpy());valid_logits.append(replay.cpu().numpy())
                processed+=len(indices)
                write(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,
                    phase='interval_order_'+split,current=processed,total=cfg['train_fixture_items']+19584,
                    unit='example',observed_at=time.time(),heartbeat_at=time.time()))
            if split=='train':assert replay_error<=1e-5 and fixture_error<=1e-8
            else:bounds=dict(lower=np.concatenate(low),upper=np.concatenate(high))
        logits=np.concatenate(valid_logits);gold=np.array(data['valid']['targets'],copy=True)
        pred=logits.argmax(1);sorted_logits=np.sort(logits,axis=1)
        correct=(pred==gold)&(sorted_logits[:,-1]>sorted_logits[:,-2])
        pairs=match(bounds['lower'],bounds['upper'],gold,correct,cfg['gap'])
        # Direct value-only recomputation for every counted pair; no second native classifier pass.
        direct_errors=[];invalid=[]
        for pair in pairs:
            i,j,r=pair['positive'],pair['null'],pair['relation']
            extremes=[]
            for idx,take_max in [(i,True),(j,False)]:
                h,_,mask=batch('valid',[idx]);values=model.value(h)
                c=torch.einsum('btd,d->bt',values,model.decoder[r]-model.decoder[0])[0,mask[0]]
                extremes.append(float(c.max() if take_max else c.min()))
            error=max(abs(extremes[0]-pair['positive_upper']),abs(extremes[1]-pair['null_lower']))
            gap=extremes[1]-extremes[0];direct_errors.append(error)
            if error>1e-8 or gap<cfg['gap']:invalid.append((i,j))
            pair['direct_gap']=gap;pair['direct_error']=error
        assert not invalid,'Invalid candidate certificate; no drop-and-reselect'
    counts=np.bincount([x['relation'] for x in pairs],minlength=40)
    n=len(pairs);diverse=int(np.count_nonzero(counts[1:]>=cfg['minimum_pairs_per_relation']))
    gate=n>=cfg['minimum_pairs'] and diverse>=cfg['minimum_relations']
    assert len({x['positive'] for x in pairs})==n and len({x['null'] for x in pairs})==n
    np.savez(args.output/'bounds.private.npz',**bounds,gold=gold,prediction=pred,strict_correct=correct)
    write(args.output/'pairs.private.json',pairs)
    summary=dict(status='completed',promotion_gate=gate,pairs=n,fraction_of_all_valid=n/19584,
        per_relation_pairs=counts.tolist(),relations_with_minimum_pairs=diverse,
        required_pairs=cfg['minimum_pairs'],required_relations=cfg['minimum_relations'],
        minimum_pairs_per_relation=cfg['minimum_pairs_per_relation'],gap_threshold=cfg['gap'],
        valid_items=19584,valid_positive=int(np.count_nonzero(gold)),strict_native_correct=int(correct.sum()),
        native_accuracy=float((pred==gold).mean()),train_fixture_items=cfg['train_fixture_items'],
        native_replay_max_error=replay_error,one_hot_contrast_max_error=fixture_error,
        direct_pair_max_error=max(direct_errors,default=0.),math_fixture=math_check,
        numerical_scope='Float64 stored-checkpoint coefficient certificates with1e-8 replay check; not interval/exact-real circuit proof',
        seconds=time.time()-started,checkpoint_sha256=cfg['checkpoint_sha256'],design_sha256=digest(args.design),
        source_hashes=cfg['source_hashes'],code_sha256=cfg['code_sha256'],attempt=args.attempt,
        private_hashes={n:digest(args.output/n) for n in ('bounds.private.npz','pairs.private.json')},
        optimizer_updates=0,test_read=False,cos_writes=0,L5='NOT_ACCEPTED',
        claim_scope='Finite cohort/fixed decoder: every arbitrary shared-token pooling with arbitrary global class biases loses at leastK native-correct decisions in row-disjoint pairs. Not net error/Macro loss, Bornnecessity, fresh5L4inheritance, novelty orL5.')
    assert replay_error<=1e-5 and fixture_error<=1e-8
    write(args.output/'summary.json',summary);write(args.output/'RUN_COMPLETE',dict(summary_sha256=digest(args.output/'summary.json')))
    write(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='completed',current=processed,
        total=processed,unit='example',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
