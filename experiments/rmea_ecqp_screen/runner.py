"""Fixed ECQP development-screen wiring; no real-data CLI in this revision.

Unit execution is separately reviewed/admitted. Real input recovery and training
still require a complete source-bound launch contract and finite traffic budget.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
from experiments.rmea_ecqp.head import ECQPHead,cohort_derangement


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def state_fingerprint(state):
    value=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        a=tensor.detach().cpu().contiguous().numpy()
        value.update(json.dumps([name,str(a.dtype),list(a.shape)],separators=(',',':')).encode())
        value.update(a.tobytes())
    return value.hexdigest()


class TrainingModel(nn.Module):
    """The actual future checkpoint path must use this containing module."""
    def __init__(self,condition,native):
        super().__init__();self.condition=condition
        if condition['id']=='native_quantum':self.head=copy.deepcopy(native)
        else:
            self.head=ECQPHead(routing=condition['routing'],feature=condition['feature'])
            self.head.load_native(native)

    def forward(self,h,roles,mask,donor_summary=None,details=False):
        if self.condition['id']=='native_quantum':return self.head(h,roles,mask,details=details)
        return self.head(h,roles,mask,anchor_override=donor_summary,details=details)


def read_training_prefix(assets,receipt,cohort_path,cohort_sha256,donors_path,donors_sha256,
                         *,expected_receipt_sha256,expected_plan_sha256):
    """Never discover input paths by globbing or fall back to valid/test."""
    assets=Path(assets).resolve(strict=True)
    assert Path(receipt).resolve(strict=True).parent==assets.parent
    assert digest(receipt)==expected_receipt_sha256
    accepted=json.loads(Path(receipt).read_bytes())
    assert accepted['status']=='PASS_TRAIN_PREFIX_RECOVERY' and accepted['prefix_examples']==4096
    assert accepted['plan_sha256']==expected_plan_sha256
    assert not accepted['full_original_H_hash_recomputed'] and accepted['sidecar_original_hashes_verified']
    assert set(accepted['files'])=={'train-h.npy','train-offsets.npy','train-roles.npy','train-targets.npy','metadata.json'}
    for name,entry in accepted['files'].items():
        p=assets/name;assert not p.is_symlink() and p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256']
    assert digest(cohort_path)==cohort_sha256 and digest(donors_path)==donors_sha256
    cohort=json.loads(Path(cohort_path).read_bytes())['rows'];donors=json.loads(Path(donors_path).read_bytes())
    metadata=json.loads((assets/'metadata.json').read_bytes())['train'];ids=[r['id_hash'] for r in metadata]
    assert ids==[r['id_hash'] for r in cohort] and len(set(ids))==4096
    assert [r['source_index'] for r in cohort]==list(range(4096))
    offsets=np.load(assets/'train-offsets.npy',mmap_mode='r',allow_pickle=False)
    assert offsets.dtype==np.dtype('<i8'), 'Offsets must be little-endian int64'
    h=np.load(assets/'train-h.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(assets/'train-roles.npy',mmap_mode='r',allow_pickle=False)
    targets=np.load(assets/'train-targets.npy',mmap_mode='r',allow_pickle=False)
    assert offsets.shape==(4097,) and offsets[0]==0 and offsets[-1]==151571 and np.all(offsets[1:]>offsets[:-1])
    assert h.shape==(151571,1024) and h.dtype==np.dtype('<f2')
    assert roles.shape==(151571,4) and roles.dtype==np.dtype('<f4')
    assert targets.shape==(4096,) and targets.dtype==np.dtype('<i8') and np.all((targets>=0)&(targets<40))
    assert all(int(offsets[i+1]-offsets[i])==r['words'] for i,r in enumerate(metadata))
    lookup={sid:i for i,sid in enumerate(ids)};splits={}
    for split,expected in [('fit',3124),('eval',972)]:
        selected=[r for r in cohort if r['split']==split];assert len(selected)==expected
        regenerated=cohort_derangement([r['id_hash'] for r in selected],[r['group'] for r in selected],split)
        assert regenerated==donors[split]
        splits[split]=np.array([lookup[r['id_hash']] for r in selected],dtype=np.int64)
    assert set(splits['fit']).isdisjoint(splits['eval'])
    assert {r['group'] for r in cohort if r['split']=='fit'}.isdisjoint({r['group'] for r in cohort if r['split']=='eval'})
    summaries=np.empty((4096,1024),dtype=np.float32)
    for i in range(4096):
        lo,hi=map(int,offsets[i:i+2]);a=np.asarray(h[lo:hi],dtype=np.float32);r=roles[lo:hi]
        assert np.isfinite(a).all() and np.isfinite(r).all() and np.all((r[:,2:]==0)|(r[:,2:]==1))
        flags=r[:,2:];counts=flags.sum(0);assert np.all(counts>0)
        # Use the source head's float32 pooling implementation for donors;
        # NumPy matrix multiplication need not share its reduction behavior.
        _,_,entities=ECQPHead.checked_inputs(torch.from_numpy(a)[None],
            torch.from_numpy(np.array(r,copy=True))[None],torch.ones(1,hi-lo,dtype=torch.bool))
        summaries[i]=entities.mean(1)[0].numpy()
    return dict(h=h,roles=roles,offsets=offsets,targets=targets,metadata=metadata,cohort=cohort,
        splits=splits,donors=donors,lookup=lookup,summaries=summaries,receipt_sha256=digest(receipt))


def batch(data,indices,split,donor=False):
    assert set(map(int,indices))<=set(map(int,data['splits'][split]))
    lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in indices]
    h=np.zeros((len(indices),max(lengths),1024),dtype=np.float32)
    roles=np.zeros((len(indices),max(lengths),4),dtype=np.float32);mask=np.zeros(h.shape[:2],dtype=bool)
    for j,(i,length) in enumerate(zip(indices,lengths)):
        lo,hi=map(int,data['offsets'][i:i+2]);h[j,:length]=data['h'][lo:hi];roles[j,:length]=data['roles'][lo:hi];mask[j,:length]=True
    target=torch.tensor(np.array(data['targets'][indices],dtype=np.int64))
    summary=None
    if donor:
        donor_indices=[data['lookup'][data['donors'][split][data['metadata'][i]['id_hash']]] for i in indices]
        assert set(donor_indices)<=set(map(int,data['splits'][split]))
        summary=torch.from_numpy(data['summaries'][donor_indices].copy())
    return tuple(torch.from_numpy(x) for x in (h,roles,mask))+(target,summary)


@torch.no_grad()
def evaluate(model,data,indices,split,force_mismatch=False):
    model.eval();pred=[];loss=0.;logits=[]
    for start in range(0,len(indices),32):
        ids=indices[start:start+32];h,r,mask,y,summary=batch(data,ids,split,
            donor=model.condition['donor'] or force_mismatch)
        out=model(h,r,mask,summary)[0];assert torch.isfinite(out).all()
        loss+=float(F.cross_entropy(out,y,reduction='sum'));pred.extend(out.argmax(-1).tolist());logits.append(out.numpy())
    return dict(prediction=np.asarray(pred,dtype=np.int64),ce=loss/len(indices),logits=np.concatenate(logits))


@torch.no_grad()
def intervention(model,data):
    """Descriptive fixed-model changes, not a causal/semantic attribution."""
    assert model.condition['id']=='ecqp_quantum'
    model.eval();totals={};counts={};maxima={}
    for start in range(0,len(data['splits']['eval']),32):
        ids=data['splits']['eval'][start:start+32]
        h,r,mask,_,summary=batch(data,ids,'eval',True)
        actual=model(h,r,mask,details=True)
        mismatched=model(h,r,mask,summary,details=True)
        for name in ['prepare_raw','psi','alpha','logits']:
            delta=(actual[name]-mismatched[name]).abs()
            if name in ['prepare_raw','psi']:delta=delta[mask]
            elif name=='alpha':delta=delta.transpose(1,2)[mask]
            assert torch.isfinite(delta).all()
            totals[name]=totals.get(name,0.)+float(delta.double().sum())
            counts[name]=counts.get(name,0)+delta.numel()
            maxima[name]=max(maxima.get(name,0.),float(delta.max()))
    return {n:dict(mean_abs=totals[n]/counts[n],max_abs=maxima[n]) for n in totals}


def train_conditions(data,design,work):
    """Synthetic unit or separately admitted fixed train-only development job."""
    expected=dict(seed=20261118,epochs=4,batch_size=32,optimizer='AdamW',lr=.001,weight_decay=.01,
        clip_norm=1.,threads=1,device='cpu',independent_initializations=1)
    assert all(design['training'][n]==v for n,v in expected.items())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261118)
    native=RMEAHead('quantum');native_state=copy.deepcopy(native.state_dict())
    initial_shared=state_fingerprint(native_state)
    rng=np.random.default_rng(20261118)
    orders=[rng.permutation(data['splits']['fit']) for _ in range(4)]
    results={};work=Path(work);assert not work.exists();work.mkdir()
    for condition in design['conditions']:
        native.load_state_dict(native_state);model=TrainingModel(condition,native)
        shared={n:v for n,v in model.head.state_dict().items() if n not in ['B','projection']}
        assert state_fingerprint(shared)==initial_shared
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        epochs=[];gradient_max={n:0. for n,_ in model.named_parameters()}
        assert sum(p.numel() for p in model.parameters())==condition['params']
        for epoch,order in enumerate(orders,1):
            model.train();loss_sum=0.
            for start in range(0,len(order),32):
                selected=order[start:start+32];h,r,mask,y,summary=batch(data,selected,'fit',condition['donor'])
                optimizer.zero_grad(set_to_none=True);out=model(h,r,mask,summary)[0]
                loss=F.cross_entropy(out,y);assert torch.isfinite(loss)
                loss.backward()
                for name,param in model.named_parameters():
                    assert param.grad is not None and torch.isfinite(param.grad).all()
                    gradient_max[name]=max(gradient_max[name],float(param.grad.abs().max()))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                optimizer.step();loss_sum+=float(loss.detach())*len(selected)
            evaluated=evaluate(model,data,data['splits']['eval'],'eval')
            epochs.append(dict(epoch=epoch,fit_ce=loss_sum/len(order),evaluation_ce=evaluated['ce']))
        # Read only final weights, no validation-selected checkpoint.
        state_path=work/(condition['id']+'-final.pt');torch.save(model.state_dict(),state_path)
        replacement=TrainingModel(condition,native);replacement.load_state_dict(torch.load(state_path,weights_only=True))
        replay=evaluate(replacement,data,data['splits']['eval'],'eval')
        assert np.array_equal(replay['logits'],evaluated['logits'])
        result=dict(final=replay,epochs=epochs,checkpoint_sha256=digest(state_path),
            parameters=sum(p.numel() for p in model.parameters()),gradient_max_abs=gradient_max,
            initial_shared_state_sha256=initial_shared,final_epoch=4)
        if condition['id']=='ecqp_quantum':
            result['postfit_mismatch']=evaluate(replacement,data,data['splits']['eval'],'eval',True)
            result['B_norm']=float(replacement.head.B.norm())
            result['intervention']=intervention(replacement,data)
        results[condition['id']]=result
    assert len(results)==9
    return results


def summarize(data,results,metrics_module):
    """Emit compact paired results without serializing raw predictions/text."""
    ids=data['splits']['eval'];gold=np.asarray(data['targets'][ids],dtype=np.int64)
    groups=[data['cohort'][int(i)]['group'] for i in ids]
    metrics={name:metrics_module.from_confusion(metrics_module.confusion(gold,row['final']['prediction']))
        |dict(ce=row['final']['ce']) for name,row in results.items()}
    uncertainty=metrics_module.paired_group_bootstrap(gold,
        {name:row['final']['prediction'] for name,row in results.items()},groups)
    fit_support=np.bincount(np.asarray(data['targets'][data['splits']['fit']],dtype=np.int64),minlength=40)
    interval=uncertainty['intervals']['ecqp_quantum']['delta_vs_native_quantiles']
    primary=metrics['ecqp_quantum']['macro_f1']-metrics['native_quantum']['macro_f1']
    ecqp=results['ecqp_quantum'];mechanism=(ecqp['gradient_max_abs']['head.B']>0 and ecqp['B_norm']>0
        and ecqp['intervention']['logits']['max_abs']>0)
    return dict(status='TRAIN_ONLY_EXPLORATORY_SCREEN',decision=metrics_module.decision(metrics,interval,fit_support),
        metrics=metrics,primary_macro_delta=primary,uncertainty=uncertainty,fit_support=fit_support.tolist(),
        mechanism_status='NUMERIC_ENTITY_PATH_ACTIVE' if mechanism else 'ENTITY_PATH_NOT_SUPPORTED',
        mechanism_scope='Numerical fixed-model donor perturbation; not semantic causality or quantum necessity.',
        postfit_mismatch=metrics_module.from_confusion(metrics_module.confusion(gold,ecqp['postfit_mismatch']['prediction']))
            |dict(ce=ecqp['postfit_mismatch']['ce']),
        conditions={name:{k:v for k,v in row.items() if k not in ['final','postfit_mismatch']} for name,row in results.items()},
        utility_scope='One initialization, original-train-only entity-pair-group development screen.',
        L5='NOT_ACCEPTED',formal_handoff=False)
