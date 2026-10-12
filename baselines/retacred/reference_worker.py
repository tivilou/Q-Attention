"""One reference/seed with full-state recovery and execution-only OOM tiers."""
from __future__ import annotations
import argparse,copy,json,math,random,signal,time,fcntl
from contextlib import contextmanager

@contextmanager
def exclusive(path):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_name('.'+path.name+'.writer.lock')
    if lock.is_symlink():raise ValueError('symlinked writer lock')
    with lock.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('another writer owns this run') from None
        yield

from pathlib import Path
from contextlib import nullcontext
from .common import PACKAGE,ROOT,atomic_json,digest,fingerprint,metrics
from .reference_models import make,LABELS,head_gradients
from .reference_mechanisms import reviewer_partition,curriculum_order,resampled_order,classification_with_abstention
from .reference_recipes import recipe
from .reference_trace import Capture,emit,validate,head_snapshot
from .train import rng_state,restore_rng,save_torch
from .reference_execution import Cadence,attach,metadata,poll_archives

def seed_all(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def optimizer_for(adapter,r,total):
    import torch
    if r['model']=='ptr':return adapter.make_updates(total//r['epochs'],r['epochs'])
    groups=[];decay=r.get('layer_decay',1.)
    parameters={}
    layers=getattr(getattr(adapter.model,'encoder',None),'config',None)
    depth=layers.num_hidden_layers if layers else 0
    for name,p in adapter.model.named_parameters():
        if not p.requires_grad:continue
        level=depth+1
        if name.startswith('encoder.embeddings'):level=0
        if 'encoder.encoder.layer.' in name:level=int(name.split('encoder.encoder.layer.')[1].split('.')[0])+1
        rate=r['learning_rate']*decay**(depth+1-level)
        wd=0. if name.endswith('bias') or 'LayerNorm' in name else r['weight_decay']
        parameters.setdefault((rate,wd),[]).append(p)
    groups=[dict(params=ps,lr=lr,weight_decay=wd) for (lr,wd),ps in parameters.items()]
    cls=torch.optim.Adam if r['optimizer']=='torch_Adam' else torch.optim.AdamW
    opt=cls(groups,lr=r['learning_rate'],betas=tuple(r['betas']),eps=r['eps'],foreach=False)
    if r['scheduler'].startswith('source_LinearLR'):
        sched=torch.optim.lr_scheduler.LinearLR(opt,total_iters=5)
    else:
        from transformers import get_linear_schedule_with_warmup
        sched=get_linear_schedule_with_warmup(opt,int(total*r['warmup_ratio']),total)
    return opt,sched

def orders_for(model,features,seed,epochs,difficulty=None):
    orders=[]
    for ep in range(epochs):
        if model=='luke_resample':order=resampled_order([f['labels'] for f in features],seed+ep)
        elif difficulty is not None:order=curriculum_order(difficulty,ep,epochs,seed)
        else:order=list(range(len(features)));random.Random(seed+ep).shuffle(order)
        orders.append(order)
    return orders

def evaluate(adapter,features,records,r,micro,*,case_index=None,run=None,step=0,events=None):
    import torch
    adapter.model.eval();labels=getattr(adapter,'labels',LABELS);gold=[];pred=[];loss_total=0.;loss_ok=True;traces=[]
    for start in range(0,len(features),micro):
        part=features[start:start+micro];batch=adapter.batch(part)
        rows=[case_index-start] if case_index is not None and start<=case_index<start+len(part) else []
        capture=Capture(adapter.model,rows,len(part)) if rows else None
        with torch.no_grad(),capture if capture else nullcontext():
            loss,logits=adapter.forward(batch)
            if capture:capture.snapshot_mechanism()
        guessed=logits.argmax(-1).tolist();true=[f['labels'] for f in part]
        gold.extend(true);pred.extend(guessed)
        if loss is None:loss_ok=False
        else:loss_total+=float(loss)*len(part)
        if capture:
            traces.append(emit(adapter,batch,[records[case_index]],capture,phase='test',step=step,event=len(events or [])+len(traces),
                directory=run/'case_study',loss=loss,logits=logits,extra={'predicted_relations':[labels[g] for g in guessed],
                'target_relations':[labels[g] for g in true],'dev_coefficients':getattr(adapter,'coefficients',None)}))
    result=metrics(gold,pred,labels,loss_total/len(gold) if loss_ok else None)
    result['loss_scope']=getattr(adapter,'loss_scope','model_training_objective_mean_per_sentence')
    if not loss_ok:result['loss_status']='not_defined_for_nonprobabilistic_corrected_scores'
    return result,traces

def train_standard(adapter,data,run,r,seed,micro,*,resume=False,pause_after=None,stage='main',difficulty=None,case=True,max_updates=None):
    import torch
    from .execution_compat import source_map
    from .reference_execution import reset_update_peak,memory_metadata
    memory_device=next(adapter.model.parameters()).device
    features={split:adapter.encode(rows) for split,rows in data.items()};adapter.finalize_vocabulary()
    orders=orders_for(r['model'],features['train'],seed,r['epochs'],difficulty)
    plan=[(ep,order[s:s+r['logical_batch']]) for ep,order in enumerate(orders) for s in range(0,len(order),r['logical_batch'])]
    budget=len(plan);opt,sched=optimizer_for(adapter,r,budget)
    policy=getattr(adapter,'execution_policy',None)
    contract={'schema':'reference-execution-contract.v3','recipe':r,'seed':seed,'physical_micro':policy or micro,
        'asset_identity':getattr(adapter,'initialization_identity',{}),
        'data':{s:fingerprint(rows) for s,rows in data.items()},'order':fingerprint(orders),'stage':stage,
        'source':source_map([p.name for p in sorted(PACKAGE.glob('reference_*.py'))]),
        'runtime':{'torch':torch.__version__,'transformers':__import__('transformers').__version__},
        'budget':budget,'topology':'one_process_one_gpu_logical_identity_independent_of_physical_GPU'}
    path=run/'checkpoints'/'latest.pt';state={'cursor':0,'traces':[],'best_dev':-1.,'best_cursor':None,'best_sha256':None,'best_file':None}
    def save():
        save_torch(path,dict(contract=contract,model=adapter.model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(),state=state))
    from .reference_storage import initialize_checkpoint,initial_save_pending
    if resume and not path.is_file() and initial_save_pending(run,contract):
        initialize_checkpoint(run,path,contract,save)
    if resume:
        if not path.is_file():raise ValueError('explicit resume checkpoint missing')
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if saved['contract']!=contract:raise ValueError('immutable recipe/data/source/physical partition drift')
        adapter.model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        state=saved['state'];restore_rng(saved['rng']);del saved
        if state.get('best_sha256') and digest(run/'checkpoints'/state['best_file'])!=state['best_sha256']:raise ValueError('best checkpoint hash drift')
        declared={e['path'] for e in state['traces']}|{Path(e['path']).with_suffix('.pt').name for e in state['traces']}
        case_dir=run/'case_study'
        for p in list(case_dir.iterdir()) if case_dir.exists() else []:
            if p.name not in declared:
                if p.is_symlink() or p.suffix not in ('.json','.pt'):raise ValueError('foreign trace attempt')
                quarantine=run/'orphaned_trace_attempts';quarantine.mkdir(exist_ok=True);p.rename(quarantine/(str(time.time_ns())+'-'+p.name))
        if r['model']!='curriculum':
            from .reference_archive import archive_orphans,archive_traces
            archive_orphans(run)
            if state['traces']:archive_traces(run,state['traces'])
    else:
        if run.exists() and any(run.iterdir()):raise ValueError('existing worker run needs resume')
        run.mkdir(parents=True,exist_ok=True);initialize_checkpoint(run,path,contract,save)
    cadence=Cadence(policy,cursor=state['cursor'])
    selected=next((i for i,f in enumerate(features['train']) if f['labels']!=0),0)
    pause=[False];handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for s in handlers:signal.signal(s,lambda *_:pause.__setitem__(0,True))
    start=time.monotonic();initial=state['cursor']
    try:
        while state['cursor']<budget:
            poll_archives()
            if pause[0] or pause_after is not None and state['cursor']>=pause_after or max_updates is not None and state['cursor']>=max_updates:
                if state['cursor']!=cadence.cursor:save();cadence.saved(state['cursor'])
                atomic_json(run/'RUN_PAUSED.json',{'cursor':state['cursor'],'phase':stage});return None
            ep,indices=plan[state['cursor']];adapter.model.train();opt.zero_grad(set_to_none=True);loss_value=0.;pending=[]
            reset_update_peak(memory_device)
            for s in range(0,len(indices),micro):
                ids=indices[s:s+micro];batch=adapter.batch([features['train'][i] for i in ids])
                rows=[ids.index(selected)] if case and selected in ids and not state['traces'] else []
                capture=Capture(adapter.model,rows,len(ids)) if rows else None
                before=head_snapshot(adapter) if rows else None
                with capture if capture else nullcontext():
                    loss,logits=adapter.forward(batch);scaled=loss*len(ids)/len(indices)
                    if capture:capture.snapshot_mechanism()
                    if not torch.isfinite(scaled):raise ValueError('nonfinite loss before optimizer update')
                scaled.backward();loss_value+=float(scaled.detach())
                if capture:pending.append((batch,capture,loss.detach(),logits.detach(),before))
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(),r['max_grad_norm']);opt.step()
            if not r['scheduler'].startswith('source_LinearLR'):sched.step()
            for batch,capture,loss,logits,before in pending:
                state['traces'].append(emit(adapter,batch,[data['train'][selected]],capture,phase='train',step=state['cursor']+1,
                    event=len(state['traces']),directory=run/'case_study',loss=loss,logits=logits,parameter_before=before))
            state['cursor']+=1
            update_memory=memory_metadata(memory_device)
            end_epoch=state['cursor']==budget or plan[state['cursor']][0]!=ep
            if end_epoch:
                if r['scheduler'].startswith('source_LinearLR'):sched.step()
                dev,_=evaluate(adapter,features['valid'],data['valid'],r,micro)
                if dev['micro_f1']>state['best_dev']:
                    state['best_dev']=dev['micro_f1'];state['best_cursor']=state['cursor'];state['best_file']=f"best-u{state['cursor']}.pt"
                    save_torch(run/'checkpoints'/state['best_file'],adapter.model.state_dict())
                    state['best_sha256']=digest(run/'checkpoints'/state['best_file'])
                atomic_json(run/f'dev_epoch{ep}.json',dev)
            # Sample events and epoch-selection state are committed immediately.
            # Ordinary updates use the declared time window, not full writes per step.
            if cadence.due() or pending or end_epoch:
                save();cadence.saved(state['cursor'])
            if pending and r['model']!='curriculum':
                from .reference_archive import archive_traces
                archive_traces(run,state['traces'])
            for old in (run/'checkpoints').glob('best-u*.pt'):
                if old.name!=state['best_file'] and old.name[6:-3].isdigit() and not old.is_symlink():old.unlink()
            elapsed=time.monotonic()-start;speed=(state['cursor']-initial)/max(elapsed,1e-9)
            hb={'phase':stage,'batch':state['cursor'],'batches':budget,'batches_per_second':speed,'eta_seconds':(budget-state['cursor'])/max(speed,1e-9),'loss':loss_value,
                **metadata(adapter,state['cursor'],cadence.cursor)}
            hb.update(update_memory)
            atomic_json(run/'heartbeat.json',hb);print(f"[{r['model']} {stage}] {state['cursor']}/{budget} | {speed:.2f} batch/s | loss {loss_value:.5f}",flush=True)
        if state['cursor']!=cadence.cursor:save();cadence.saved(state['cursor'])
        if r['selection']!='final_epoch':adapter.model.load_state_dict(torch.load(run/'checkpoints'/state['best_file'],map_location='cpu',weights_only=True,mmap=True),strict=True)
        return features,state
    finally:
        for s,h in handlers.items():signal.signal(s,h)

def reviewer_storage(task,folder):
    from .reference_storage import owner
    if owner(task) is None:return
    from .reference_archive import archive_files
    paths=[folder/'review-complete.private.json']+sorted((folder/'checkpoints').glob('*.pt'))
    archive_files(task,paths)
    archive_files(task,[p for p in paths if p.suffix=='.pt'],remove=True)

def cross_review(args,assets,data,r):
    """Durable reviewer votes allow optimizer offload without retraining a reviewer."""
    from .reference_archive import source_identity, verify_available
    from .reference_storage import durable_json
    root=args.run_dir/'reviewers';folds=reviewer_partition(len(data['train']),args.seed,r['reviewer_count'])
    votes={str(i):[] for i in range(len(data['train']))};review_traces=[]
    selected=next((i for i,row in enumerate(data['train']) if row['relation']!='no_relation'),0)
    for k,own in enumerate(folds):
        folder=root/str(k);held=[i for i in range(len(data['train'])) if i not in set(own)]
        ledger=folder/'review-complete.private.json'
        identity=fingerprint({'own':own,'held':held,'seed':args.seed+1000+k,'micro':getattr(args,'execution_policy',args.micro_batch),
            'recipe':r,'data':fingerprint(data['train']),'assets':assets.get('verified_assets',{}),'source':source_identity()})
        if ledger.exists():
            completed=json.loads(ledger.read_text())
            if completed.get('identity')!=identity or set(completed['votes'])!={str(i) for i in held}:
                raise ValueError('completed reviewer contract drift')
            for event in completed['traces']:
                path=args.run_dir/'auxiliary_case_study'/event['path']
                if digest(path)!=event['sha256']:raise ValueError('completed reviewer trace drift')
                payload=json.loads(path.read_text());verify_available(path.parent/payload['tensor_artifact'],payload['sha256'],payload['size_bytes'])
            reviewer_storage(args.run_dir,folder)
        else:
            seed_all(args.seed+1000+k);adapter=make('curriculum',assets,data['train'],device=args.device,checkpointing=getattr(args,'activation_checkpointing',True))
            attach(adapter,args)
            train=[data['train'][i] for i in own]
            reviewer_recipe={**r,'model':'curriculum_reviewer','epochs':r['reviewer_epochs'],'selection':'final_epoch'}
            result=train_standard(adapter,{'train':train,'valid':train},folder,reviewer_recipe,args.seed+1000+k,args.micro_batch,
                resume=(folder/'checkpoints/latest.pt').is_file(),stage=f'reviewer{k}',case=False)
            if result is None:return None
            import torch
            adapter.model.eval();features=adapter.encode([data['train'][i] for i in held]);own_votes={};events=[]
            # A crash before this ledger commits may repeat evaluation, not optimizer updates.
            for path in list((args.run_dir/'auxiliary_case_study').glob('*')):
                if f'-event{100000+k}' in path.stem:
                    from .reference_storage import regular
                    regular(path)
                    quarantine=args.run_dir/'orphaned_trace_attempts';quarantine.mkdir(exist_ok=True)
                    path.rename(quarantine/(str(time.time_ns())+'-'+path.name))
            with torch.no_grad():
                for s in range(0,len(held),args.micro_batch):
                    ids=held[s:s+args.micro_batch];batch=adapter.batch(features[s:s+args.micro_batch])
                    cap=Capture(adapter.model,[ids.index(selected)],len(ids)) if selected in ids else None
                    with cap if cap else nullcontext():
                        _,logits=adapter.forward(batch)
                        if cap:cap.snapshot_mechanism()
                    if cap:events.append(emit(adapter,batch,[data['train'][selected]],cap,phase='test',step=result[1]['cursor'],event=100000+k,
                        directory=args.run_dir/'auxiliary_case_study',loss=None,logits=logits,
                        extra={'stage':'curriculum_review','source_split':'train','reviewer':k,'own_subset_training':True,'heldout_selected_index':selected}))
                    for i,pred in zip(ids,logits.argmax(-1).tolist()):
                        own_votes[str(i)]={'reviewer':k,'predicted':pred,'correct':pred==LABELS.index(data['train'][i]['relation'])}
            completed={'identity':identity,'votes':own_votes,'traces':events,'optimizer_updates_complete':result[1]['cursor']}
            durable_json(ledger,completed)
            reviewer_storage(args.run_dir,folder)
            del adapter
            if torch.cuda.is_available():torch.cuda.empty_cache()
        for i,v in completed['votes'].items():votes[i].append(v)
        review_traces+=completed['traces']
    difficulty=[sum(not x['correct'] for x in votes[str(i)]) for i in range(len(data['train']))]
    receipt={'folds':folds,'votes':votes,'difficulty':difficulty,'traces':review_traces,'training_membership':'own_subset_only','test_access':False,'seed':args.seed}
    durable_json(args.run_dir/'cross_review.private.json',receipt);return difficulty

def tune(adapter,features,records,r,micro):
    import torch
    adapter.model.eval();parts=[];gold=[];masks=[]
    with torch.no_grad():
        for s in range(0,len(features),micro):
            batch=adapter.batch(features[s:s+micro]);adapter.forward(batch);gold.extend(f['labels'] for f in features[s:s+micro])
            last=adapter.model.last
            if r['model']=='luke_core':
                parts.append(tuple(last[k].cpu() for k in ('factual_probability','entity_only_probability','blank_probability')));masks.append(batch['candidate_mask'].cpu())
            else:parts.append(tuple(last[k].cpu() for k in ('factual_logits','counterfactual_logits')))
    arrays=[torch.cat([p[i] for p in parts]) for i in range(len(parts[0]))];best=-1.;chosen=None
    grid=[(a,b) for a in r['core_grid'] for b in r['core_grid']] if r['model']=='luke_core' else r['alpha_grid']
    for coefficients in grid:
        from .reference_mechanisms import core_scores
        scores=core_scores(*arrays,coefficients,torch.cat(masks)) if r['model']=='luke_core' else arrays[0]-coefficients*arrays[1]
        f1=metrics(gold,scores.argmax(-1).tolist(),LABELS,None)['micro_f1']
        if f1>best:best=f1;chosen=coefficients
    adapter.coefficients=chosen
    if r['model']=='luke_cfie':adapter.model.alpha=chosen
    return {'fit_split':'valid','coefficients':chosen,'micro_f1':best,'grid_size':len(grid),'test_access':False}

def _parse_and_run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--seed',type=int,default=13)
    for name in ('assets','data-dir','run-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--micro-batch',default='auto');p.add_argument('--checkpoint-seconds',type=float,default=120.)
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--pause-after',type=int);p.add_argument('--diagnostic',action='store_true');p.add_argument('--diagnostic-updates',type=int,default=2)
    args=p.parse_args(argv)
    if args.seed<0:raise ValueError('nonnegative seed required')
    from .reference_execution import adaptive_cli
    with exclusive(args.run_dir):
        return adaptive_cli(args,args.model,recipe(args.model)['logical_batch'],lambda:run_worker(args))

def run_worker(args):
    assets=json.loads(args.assets.read_text());r=recipe(args.model)
    data={s:json.loads((args.data_dir/f).read_text()) for s,f in [('train','train.json'),('valid','dev.json'),('test','test.json')]}
    if not args.diagnostic and [len(data[s]) for s in ('train','valid','test')]!=[58465,19584,13418]:raise ValueError('complete canonical splits required')
    if args.diagnostic and (max(len(rows) for rows in data.values())>64 or not 1<=args.diagnostic_updates<=2):raise ValueError('diagnostic is at most64 rows and2 updates')
    if args.model.startswith('ctl_drp_'):
        raise ValueError('CTL uses native ctl_drp_runner through reference_complete dispatcher')
    if (args.run_dir/'RUN_COMPLETE').exists():
        from .reference_complete import validate_task
        validate_task(args.run_dir);return 0
    if args.model=='metaentail':
        from .reference_nli import run
        return run(args,assets,data,r)
    if (args.run_dir/'RUN_COMPLETE').exists():
        summary=json.loads((args.run_dir/'run_summary.json').read_text());validate(args.run_dir,summary['case_study']['traces']);return 0
    difficulty=None
    if args.model=='curriculum':
        evidence=args.run_dir/'cross_review.private.json'
        if evidence.exists():difficulty=json.loads(evidence.read_text())['difficulty']
        else:
            # Stage directories are separate from the target model's checkpoint.
            args.run_dir.mkdir(parents=True,exist_ok=True);difficulty=cross_review(args,assets,data,r)
            if difficulty is None:return 75
    seed_all(args.seed);adapter=make(args.model,assets,data['train'],device=args.device,checkpointing=getattr(args,'activation_checkpointing',True))
    attach(adapter,args)
    adapter.initialization_identity=assets.get('verified_assets',{});adapter.model_id=args.model
    target=args.run_dir/'target' if args.model=='curriculum' else args.run_dir
    if target!=args.run_dir and (args.run_dir/'case_study').exists() and not (target/'case_study').exists():
        (args.run_dir/'case_study').rename(target/'case_study')
    result=train_standard(adapter,data,target,r,args.seed,args.micro_batch,resume=args.resume or (target/'checkpoints/latest.pt').exists(),
        pause_after=args.pause_after,difficulty=difficulty,max_updates=args.diagnostic_updates if args.diagnostic else None)
    if result is None:return 75
    features,state=result
    if args.model in ('luke_core','luke_cfie'):atomic_json(args.run_dir/'dev_tuning.json',tune(adapter,features['valid'],data['valid'],r,args.micro_batch))
    splits={};traces=state['traces']
    # Final train-split inference does not select/check this reference. Training
    # loss and sampled gradients are already recorded; keep full dev/test only.
    for split in ('valid','test'):
        report,events=evaluate(adapter,features[split],data[split],r,args.micro_batch,case_index=0 if split=='test' else None,run=target,step=state['cursor'],events=traces)
        splits[split]=report;traces+=events
    if target!=args.run_dir:
        # Do not copy large numeric tensors. Manifest paths are relative to the
        # one task; exporter accepts the explicitly linked target directory.
        import os
        for filename in ('case_study','contract.json'):
            source=target/filename
            if filename=='case_study':
                destination=args.run_dir/filename
                if not destination.exists():source.rename(destination)
            else:atomic_json(args.run_dir/filename,json.loads(source.read_text()))
    if args.model=='curriculum':
        import os
        review=json.loads((args.run_dir/'cross_review.private.json').read_text())
        for event in review['traces']:
            for name in (event['path'],Path(event['path']).with_suffix('.pt').name):
                source=args.run_dir/'auxiliary_case_study'/name;destination=args.run_dir/'case_study'/name
                if destination.exists():
                    if digest(destination)!=digest(source):raise ValueError('auxiliary trace identity drift')
                else:os.link(source,destination) # full values, no duplicate weight-sized allocation
            traces.append(event)
        selected=next((i for i,row in enumerate(data['train']) if row['relation']!='no_relation'),0)
        atomic_json(args.run_dir/'case_study'/'curriculum-flow.json',{'schema':'curriculum-sample-flow.v1','sample_id':data['train'][selected]['id'],
            'source_split':'train','selected_index':selected,'heldout_reviewer_events':review['traces'],
            'post_prediction_votes':review['votes'][str(selected)],'difficulty':review['difficulty'][selected],
            'pacing':r['curriculum_pacing'],'target_events':[e for e in traces if e not in review['traces']],
            'purpose':'同一个训练样本经独立reviewer真实前向评分、预测后比对得到难度，再参与分阶段target训练；全部观测张量在链接的私有事件中。'})
    from .reference_execution import drain_archives
    drain_archives()
    validate(args.run_dir,traces)
    from .reference_complete import finish_summary
    finish_summary(args.run_dir,args.model,args.seed,r,splits,state['cursor'],traces,args.data_dir,assets,diagnostic=args.diagnostic)
    return 0

def main(argv=None):
    from .reference_storage import pause_cli
    return pause_cli(lambda:_parse_and_run(argv))

if __name__=='__main__':raise SystemExit(main())
