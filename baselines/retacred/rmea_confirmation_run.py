"""One frozen-profile child. Complete-data discovery is collaborator-only."""
import gc
import hashlib
import heapq
import os
from pathlib import Path
import sys
from . import rmea_confirmation_protocol as p

def extract(args,cfg,output):
    import numpy as np
    import torch
    from transformers import RobertaModel,RobertaTokenizerFast
    from scripts.audit_rmea_inputs import inspect
    from scripts.audit_rmea_tokenization import marked_words
    # Check the full benchmark identity, including official split limitations.
    inspection=inspect(args.data_dir)
    if not inspection['dataset_identity_matches_reference']:raise ValueError('Wrong benchmark')
    labels=inspection['train_derived_labels'];label_to_id={v:i for i,v in enumerate(labels)}
    assets=p.read(p.ROOT/'baselines/retacred/assets.lock.json')
    for row in assets['files']:
        f=args.assets/row['name']
        if f.is_symlink() or f.stat().st_size!=row['size_bytes'] or p.digest(f)!=row['sha256']:raise ValueError('Pinned asset mismatch')
    tokenizer=RobertaTokenizerFast.from_pretrained(args.assets,add_prefix_space=True,local_files_only=True)
    encoder=RobertaModel.from_pretrained(args.assets,local_files_only=True,add_pooling_layer=False,torch_dtype=torch.float16).to('cuda').eval()
    encoder.requires_grad_(False);metadata={}
    cache=output/'cache';cache.mkdir()
    for split,count in cfg['counts'].items():
        def rows():
            with (args.data_dir/(split+'.jsonl')).open(encoding='utf8') as handle:
                for line in handle:
                    row=p.json.loads(line);sid=str(row['metadata']['id'])
                    yield hashlib.sha256(('rmea-sample-v1:'+split+':'+sid).encode()).hexdigest(),row
        selected=[row for _,row in heapq.nsmallest(count,rows(),key=lambda pair:pair[0])]
        if len(selected)!=count:raise ValueError('Wrong selected count')
        offsets=np.array([0]+list(np.cumsum([len(row['tokens']) for row in selected])),dtype=np.int64)
        h=np.lib.format.open_memmap(cache/(split+'-h.npy'),mode='w+',dtype=np.float16,shape=(int(offsets[-1]),1024))
        role=np.lib.format.open_memmap(cache/(split+'-roles.npy'),mode='w+',dtype=np.float32,shape=(int(offsets[-1]),4))
        targets=[];metadata[split]=[]
        for start in range(0,count,4):
            batch=selected[start:start+4];marked=[marked_words(row) for row in batch]
            encoded=tokenizer([x[0] for x in marked],is_split_into_words=True,padding=True,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>512:raise ValueError('No truncation permitted')
            with torch.inference_mode():hidden=encoder(**{k:v.cuda() for k,v in encoded.items()}).last_hidden_state.float()
            for i,row in enumerate(batch):
                j=start+i;n=len(row['tokens']);lo,hi=map(int,offsets[j:j+2]);mapping=encoded.word_ids(i)
                vectors=[];word_pieces=[]
                for word in range(n):
                    pieces=[k for k,w in enumerate(mapping) if w is not None and marked[i][1][w]==word]
                    if not pieces:raise ValueError('Original word missing')
                    word_pieces.append(pieces);vectors.append(hidden[i,pieces].mean(0))
                h[lo:hi]=torch.stack(vectors).half().cpu().numpy()
                positions=np.arange(n,dtype=np.float32);s,o=row['subject'],row['object']
                role[lo:hi]=np.stack([np.clip((positions-(s[0]+s[1]-1)/2)/32,-1,1),np.clip((positions-(o[0]+o[1]-1)/2)/32,-1,1),
                    ((positions>=s[0])&(positions<s[1])).astype(np.float32),((positions>=o[0])&(positions<o[1])).astype(np.float32)],-1)
                targets.append(label_to_id[row['label']])
                meta=dict(id_hash=hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest(),words=n)
                if j<2:
                    meta.update(tokens=row['tokens'],subject=s,object=o,input_ids=encoded['input_ids'][i].tolist(),
                                attention_mask=encoded['attention_mask'][i].tolist(),word_pieces=word_pieces)
                    np.save(cache/f'{split}-encoder-{j}.npy',hidden[i].cpu().numpy())
                metadata[split].append(meta)
            if start%1024==0:print(p.json.dumps(dict(stage='features',split=split,done=min(start+4,count),total=count)),flush=True)
        h.flush();role.flush();del h,role,selected
        np.save(cache/(split+'-offsets.npy'),offsets);np.save(cache/(split+'-targets.npy'),np.array(targets,dtype=np.int64))
    p.atomic_json(cache/'metadata.json',metadata)
    del encoder,hidden;gc.collect();torch.cuda.empty_cache()
    return labels,inspection

def train(args,cfg,output,labels):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .rmea import RMEAHead,replay
    from .rmea_confirmation_trace import capture,close
    from .common import metrics
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    device='cuda';cache=output/'cache';metadata=p.read(cache/'metadata.json')
    data={s:{k:np.load(cache/(s+'-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')} for s in cfg['counts']}
    def batch(split,indices):
        d=data[split];offset=d['offsets'];lengths=[int(offset[i+1]-offset[i]) for i in indices];n=max(lengths)
        h=np.zeros((len(indices),n,1024),np.float32);r=np.zeros((len(indices),n,4),np.float32);mask=np.zeros((len(indices),n),bool)
        for j,(i,length) in enumerate(zip(indices,lengths)):
            a,b=map(int,offset[i:i+2]);h[j,:length]=d['h'][a:b];r[j,:length]=d['roles'][a:b];mask[j,:length]=True
        return tuple(torch.from_numpy(x).to(device) for x in (h,r,mask,np.array(d['targets'][indices],dtype=np.int64)))+([metadata[split][i]['id_hash'] for i in indices],)
    def evaluate(head,split,diagnosis=False):
        head.eval();gold=[];pred=[];loss=0.;shuffled_loss=0.;tied_loss=0.;eligible=0;max_residual=0.;cm_shuffle=[]
        with torch.no_grad():
            for start in range(0,cfg['counts'][split],cfg['batch_size']):
                h,r,mask,y,ids=batch(split,list(range(start,min(start+cfg['batch_size'],cfg['counts'][split]))))
                d=head(h,r,mask,True);logits=d['logits']
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
                loss+=F.cross_entropy(logits,y,reduction='sum').item();gold+=y.tolist();pred+=logits.argmax(-1).tolist()
                if diagnosis:
                    intervention=replay(head,d,mask,r,ids);shuffled=intervention['shuffled_logits']
                    shuffled_loss+=F.cross_entropy(shuffled,y,reduction='sum').item();tied_loss+=F.cross_entropy(intervention['tied_logits'],y,reduction='sum').item()
                    eligible+=int((intervention['eligible_context_mask'].sum(-1)>=2).sum());cm_shuffle+=shuffled.argmax(-1).tolist()
                    max_residual=max(max_residual,float(intervention['identity_residual'].abs().max()))
                    if head.mode=='uniform' and not torch.allclose(logits,shuffled,atol=2e-5,rtol=0):raise ValueError('Uniform shuffle must be invariant')
        result=metrics(gold,pred,labels,loss/len(gold))
        if diagnosis:result['diagnosis']=dict(shuffle_ce=shuffled_loss/len(gold),shuffle_minus_actual_ce=(shuffled_loss-loss)/len(gold),
            tied_ce=tied_loss/len(gold),eligible_context_samples=eligible,total_samples=len(gold),max_identity_residual=max_residual,
            shuffled=metrics(gold,cm_shuffle,labels,shuffled_loss/len(gold)))
        return result
    checkpoints=output/'checkpoints';checkpoints.mkdir();traces=[];results={};common={}
    for mode in p.MODES:
        torch.manual_seed(cfg['seed']);head=RMEAHead(mode).to(device)
        common[mode]=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in (head.value.weight,head.value.bias,head.decoder,head.bias))).hexdigest()
        torch.save(head.state_dict(),checkpoints/(mode+'-initial.pt'))
        for split in cfg['counts']:traces+=capture(output,mode,'initial',split,[0,1],batch,metadata,head,'checkpoints/'+mode+'-initial.pt')
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        history=[];best=None;best_epoch=None
        for epoch in range(cfg['epochs']):
            head.train();order=torch.randperm(cfg['counts']['train'],generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
            for start in range(0,len(order),cfg['batch_size']):
                indices=order[start:start+cfg['batch_size']];h,r,mask,y,_=batch('train',indices)
                optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(head(h,r,mask)[0],y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward()
                first=epoch==0 and start==0
                if first:
                    update={'before/'+k:v.detach().cpu().numpy().copy() for k,v in head.named_parameters()}
                    update.update({'gradient/'+k:v.grad.detach().cpu().numpy().copy() for k,v in head.named_parameters()})
                    update['cache_indices']=np.array(indices);update['loss']=np.array(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(head.parameters(),cfg['clip_norm'],error_if_nonfinite=True);optimizer.step()
                if first:
                    update.update({'after/'+k:v.detach().cpu().numpy().copy() for k,v in head.named_parameters()})
                    np.savez(output/(mode+'-first-update.npz'),**update);del update
            valid=evaluate(head,'valid');history.append(valid)
            if best is None or valid['macro_f1']>best['macro_f1']:
                best=valid;best_epoch=epoch+1;torch.save(head.state_dict(),checkpoints/(mode+'-selected.pt'))
            print(p.json.dumps(dict(stage='train',condition=mode,epoch=epoch+1,valid_macro_f1=valid['macro_f1'])),flush=True)
        head.load_state_dict(torch.load(checkpoints/(mode+'-selected.pt'),map_location=device,weights_only=True));head.eval()
        for split in cfg['counts']:traces+=capture(output,mode,'selected',split,[0,1],batch,metadata,head,'checkpoints/'+mode+'-selected.pt')
        results[mode]=dict(valid_history=history,selected_epoch=best_epoch,selected_valid=best,test=evaluate(head,'test',True),
            parameter_count=sum(v.numel() for v in head.parameters()),updates=cfg['epochs']*((cfg['counts']['train']+cfg['batch_size']-1)//cfg['batch_size']))
        del head,optimizer;gc.collect();torch.cuda.empty_cache()
    if len(set(common.values()))!=1:raise ValueError('Common initialization differs')
    report=close(output,traces)
    summary=dict(schema='q-rmea.results.v1',scope='technical_only' if cfg['profile']=='canary' else 'formal_single_seed_pending_owner_audit',
        results=results,common_initialization=common,common_initialization_equal=True,labels=labels,trace=report,
        full_data_training=cfg['profile']=='discovery',test_selection=False,seed=cfg['seed'],
        gate_candidates=dict(utility_positive=results['quantum']['test']['macro_f1']>results['uniform']['test']['macro_f1'],
            mechanism_ce_positive=results['quantum']['test']['diagnosis']['shuffle_minus_actual_ce']>0),
        limitations=['No automatic L2/L3 or novelty claim.','Historical references are context, not matched controls.','Official split overlaps remain; no document/entity/OOD independence.','No quantum or effective-capacity advantage claimed.'])
    p.atomic_json(output/'summary.json',summary)

def run(args):
    import torch
    cfg=p.config(args.profile,args.seed);p.review_gate()
    if args.profile=='discovery':
        if os.environ.get('QATTN_COLLABORATOR_FORMAL')!='1' or p.git('branch','--show-current')!='1.1':raise ValueError('Full formal run is collaborator-only via handoff')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:raise ValueError('Exactly one assigned GPU required')
    output=Path(args.output).resolve()
    if output.is_relative_to(p.ROOT) or output.exists():raise ValueError('Fresh private output outside checkout required; no partial training resume')
    output.mkdir(parents=True)
    with p.lock(output/'.run.lock'):
        labels,inspection=extract(args,cfg,output)
        manifest=dict(schema='q-rmea.run.v1',config=cfg,source_sha256=p.identity(p.sources()),git_revision=p.git('rev-parse','HEAD'),
            data={s:dict(source_sha256=p.EXPECTED[s][1],source_count=p.EXPECTED[s][0],selected_count=n) for s,n in cfg['counts'].items()},
            environment=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=__import__('importlib.metadata',fromlist=['version']).version('numpy'),transformers=__import__('importlib.metadata',fromlist=['version']).version('transformers'),gpu=torch.cuda.get_device_name(0)),
            data_limitations=inspection.get('limitations',[]))
        p.atomic_json(output/'manifest.json',manifest);train(args,cfg,output,labels)
        for split in cfg['counts']:
            if p.digest(args.data_dir/(split+'.jsonl'))!=p.EXPECTED[split][1]:raise ValueError('Data changed during run')
        p.seal(output)

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=p.SEEDS,required=True)
    parser.add_argument('--profile',choices=('canary','discovery'),required=True)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())

if __name__=='__main__':main()
