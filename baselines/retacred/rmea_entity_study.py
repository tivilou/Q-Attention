"""Frozen single-seed entity study. Full training is collaborator-only."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys

from . import rmea_entity_study_protocol as p


def train(cfg, output, labels, device='cuda', trace=True):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .common import metrics
    from .rmea_entity import make_head, entity_inputs, mismatched_pairs
    from .rmea_entity_study_trace import capture, close
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cache=output/'cache'
    metadata=p.read(cache/'metadata.json')
    data={s:{k:np.load(cache/f'{s}-{k}.npy',mmap_mode='r',allow_pickle=False)
             for k in ('h','roles','offsets','targets')} for s in cfg['counts']}

    def batch(split, indices, targets=True):
        d=data[split];lengths=[int(d['offsets'][i+1]-d['offsets'][i]) for i in indices]
        n=max(lengths)
        h=np.zeros((len(indices),n,1024),np.float32)
        roles=np.zeros((len(indices),n,4),np.float32);mask=np.zeros((len(indices),n),bool)
        for j,(i,length) in enumerate(zip(indices,lengths)):
            a,b=map(int,d['offsets'][i:i+2]);h[j,:length]=d['h'][a:b]
            roles[j,:length]=d['roles'][a:b];mask[j,:length]=True
        inputs=tuple(torch.from_numpy(a).to(device) for a in (h,roles,mask))
        y=torch.from_numpy(np.array(d['targets'][indices],dtype=np.int64)).to(device) if targets else None
        return inputs,y

    def batches(split):
        for start in range(0,cfg['counts'][split],cfg['batch_size']):
            yield list(range(start,min(start+cfg['batch_size'],cfg['counts'][split])))

    # No label is read for calibration or the whole-split donor permutation.
    total=torch.zeros(2,1028,dtype=torch.float64)
    with torch.no_grad():
        for indices in batches('train'):
            inputs,_=batch('train',indices,False)
            total+=entity_inputs(*inputs).double().cpu().sum(0)
    means=(total/cfg['counts']['train']).float().to(device)
    np.savez(output/'train-calibration.npz',role_means=means.cpu().numpy(),train_count=cfg['counts']['train'])
    donors={s:mismatched_pairs(torch.arange(n),[m['id_hash'] for m in metadata[s]]).tolist()
            for s,n in cfg['counts'].items()}

    def overrides(split,indices,inputs):
        pair=entity_inputs(*inputs)
        donor_inputs,_=batch(split,[donors[split][i] for i in indices],False)
        return {'mean':means[None].expand(len(indices),-1,-1),
                'swapped':pair.flip(1),'mismatch':entity_inputs(*donor_inputs)}

    def evaluate(head,split,diagnosis=False):
        head.eval();gold=[];pred=[];loss=0.
        sums={k:0. for k in ('mean','swapped','mismatch')}
        with torch.no_grad():
            for indices in batches(split):
                inputs,y=batch(split,indices)
                logits,alpha=head(*inputs)
                if not torch.isfinite(logits).all() or not torch.isfinite(alpha).all():raise ValueError('Nonfinite forward')
                if not torch.allclose(alpha.sum(-1),torch.ones_like(alpha.sum(-1)),atol=2e-5):raise ValueError('Weight normalization')
                loss+=F.cross_entropy(logits,y,reduction='sum').item()
                gold+=y.tolist();pred+=logits.argmax(-1).tolist()
                if diagnosis:
                    for name,pair in overrides(split,indices,inputs).items():
                        other=head(*inputs,entity_override=pair)[0]
                        sums[name]+=F.cross_entropy(other,y,reduction='sum').item()
        result=metrics(gold,pred,labels,loss/len(gold))
        if diagnosis:result['branch_diagnosis']={k:dict(ce=v/len(gold),minus_actual_ce=(v-loss)/len(gold),items=len(gold)) for k,v in sums.items()}
        return result

    checkpoints=output/'checkpoints';checkpoints.mkdir();samples=[];results={};common={}
    for mode in p.MODES:
        torch.manual_seed(cfg['seed']);head=make_head(mode).to(device)
        if mode=='no_content':
            head.fit_role_means((batch('train',indices,False)[0] for indices in batches('train')),split='train')
            torch.testing.assert_close(head.role_means,means,atol=0,rtol=0)
        common[mode]=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in
            (head.prepare.weight,head.prepare.bias,head.effect,head.value.weight,head.value.bias,head.decoder,head.bias))).hexdigest()
        torch.save(head.state_dict(),checkpoints/f'{mode}-initial.pt')
        if trace:
            for split in cfg['counts']:samples+=capture(output,head,split,'initial',batch,metadata,overrides)
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        best=None;history=[];best_epoch=None
        for epoch in range(cfg['epochs']):
            head.train();order=torch.randperm(cfg['counts']['train'],generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
            for start in range(0,len(order),cfg['batch_size']):
                indices=order[start:start+cfg['batch_size']];inputs,y=batch('train',indices)
                optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(head(*inputs)[0],y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();first=epoch==0 and start==0
                if first:
                    update={'before/'+k:v.detach().cpu().numpy().copy() for k,v in head.named_parameters()}
                    update.update({'gradient/'+k:v.grad.detach().cpu().numpy().copy() for k,v in head.named_parameters()})
                    if not all(np.isfinite(a).all() for a in update.values()):raise ValueError('Nonfinite gradient')
                    update['cache_indices']=np.array(indices);update['loss']=np.array(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(head.parameters(),cfg['clip_norm'],error_if_nonfinite=True);optimizer.step()
                if first:
                    update.update({'after/'+k:v.detach().cpu().numpy().copy() for k,v in head.named_parameters()})
                    np.savez(output/f'{mode}-first-update.npz',**update);del update
            valid=evaluate(head,'valid');history.append(valid)
            if best is None or valid['macro_f1']>best['macro_f1']:
                best=valid;best_epoch=epoch+1;torch.save(head.state_dict(),checkpoints/f'{mode}-selected.pt')
            print(p.json.dumps(dict(stage='train',condition=mode,epoch=epoch+1,valid_macro_f1=valid['macro_f1'])),flush=True)
        head.load_state_dict(torch.load(checkpoints/f'{mode}-selected.pt',map_location=device,weights_only=True));head.eval()
        if trace:
            for split in cfg['counts']:samples+=capture(output,head,split,'selected',batch,metadata,overrides)
        record=dict(valid_history=history,selected_epoch=best_epoch,selected_valid=best,
            parameter_count=sum(v.numel() for v in head.parameters()),
            updates=cfg['epochs']*((cfg['counts']['train']+cfg['batch_size']-1)//cfg['batch_size']))
        if 'test' in cfg['counts']:record['test']=evaluate(head,'test',mode=='ordered')
        elif mode=='ordered':record['valid_branch_diagnosis']=evaluate(head,'valid',True)['branch_diagnosis']
        results[mode]=record
        del head,optimizer;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    if len(set(common.values()))!=1:raise ValueError('Common initial parameters differ')
    readiness=close(output,samples,cfg,metadata,donors) if trace else dict(status='not_applicable',reason='bounded train/valid learning pilot; end-to-end trace checked separately')
    if not trace:p.atomic_json(output/'trace_validation.json',readiness)
    split='test' if 'test' in cfg['counts'] else 'selected_valid'
    delta=results['ordered'][split]['macro_f1']-results['static'][split]['macro_f1']
    summary=dict(schema='q-rmea.entity.results.v1',seed=cfg['seed'],results=results,labels=labels,
        common_initialization=common,common_initialization_equal=True,trace=readiness,
        scope='formal_single_seed_pending_owner_audit' if cfg['profile']=='discovery' else 'technical_only',
        full_data_training=cfg['profile']=='discovery',test_selection=False,
        gate_candidates=dict(ordered_minus_static_macro_f1=delta,utility_positive=delta>0),
        limitations=['No automatic L3, novelty, quantum exclusivity or matched-capacity claim.',
         'Single seed cannot establish reproducible utility or causal attribution.',
         'Official split overlap and previous test exposure remain; no document/entity/OOD independence.',
         'Historical benchmark baselines are not rerun; static is the integrated disabled-branch comparator.',
         'Role-swap is branch-only intervention, not a new task or relabeling of examples.'])
    p.atomic_json(output/'summary.json',summary)
    return summary


FIXTURE_SHA='4ca46403baf14d86293d3c0eb78b24c1f5f6a43841597c535b9dfc58baba9f2b'


def fixture(archive,output):
    import tarfile
    if p.digest(archive)!=FIXTURE_SHA:raise ValueError('Only pinned historical tiny fixture allowed')
    with tarfile.open(archive,'r:gz') as t:
        manifest=p.json.load(t.extractfile('canary/manifest.json'))
        if manifest['config']['counts']!={'train':16,'valid':8,'test':8}:raise ValueError('Not a tiny canary')
        for member in t.getmembers():
            if member.name.startswith('canary/cache/'):
                rel=Path(member.name).relative_to('canary')
                target=output/rel
                if not member.isfile() or not target.resolve().is_relative_to(output.resolve()):raise ValueError('Unsafe cache member')
                target.parent.mkdir(parents=True,exist_ok=True)
                with t.extractfile(member) as src,target.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
    return manifest


def run(args):
    import torch
    cfg=p.config(args.profile);p.review_gate(formal=args.profile=='discovery')
    if p.git('status','--porcelain'):raise ValueError('Clean exact source required')
    if args.profile=='discovery':
        if os.environ.get('QATTN_COLLABORATOR_FORMAL')!='1' or p.git('branch','--show-current')!='1.1':raise ValueError('Full formal experiment is collaborator-only via parent')
    if args.fixture_cache and args.profile!='canary':raise ValueError('Fixture can only certify a bounded canary')
    if not args.fixture_cache and (not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1):raise ValueError('Exactly one assigned GPU required')
    output=args.output.resolve()
    if output.is_relative_to(p.ROOT) or output.exists():raise ValueError('Fresh private output outside checkout required')
    output.mkdir(parents=True)
    with p.lock(output/'.run.lock'):
        if args.fixture_cache:
            old=fixture(args.fixture_cache,output);labels=p.read(output/'cache/metadata.json')
            labels=sorted(p.read(p.ROOT/'docs/contracts/q_rmea_entity_labels.json'))
            environment=dict(python=sys.version.split()[0],torch=torch.__version__,gpu='CPU cached historical canary',fixture_sha256=FIXTURE_SHA)
        else:
            from .rmea_run import extract
            labels,_=extract(args,cfg,output)
            environment=dict(python=sys.version.split()[0],torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
        manifest=dict(schema='q-rmea.entity.run.v1',config=cfg,source_sha256=p.identity(p.sources()),git_revision=p.git('rev-parse','HEAD'),
            data={s:dict(source_sha256=p.EXPECTED[s][1],source_count=p.EXPECTED[s][0],selected_count=n) for s,n in cfg['counts'].items()},environment=environment)
        p.atomic_json(output/'manifest.json',manifest)
        train(cfg,output,labels,device='cpu' if args.fixture_cache else 'cuda')
        if not args.fixture_cache:
            for split in cfg['counts']:
                if p.digest(args.data_dir/(split+'.jsonl'))!=p.EXPECTED[split][1]:raise ValueError('Data changed during run')
        p.seal(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('canary','discovery'),required=True)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture-cache',type=Path)
    run(parser.parse_args())


if __name__=='__main__':main()
