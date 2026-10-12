"""Expanded-pair NLI training, grouped inference and honest abstention metrics."""
from __future__ import annotations
import json,random,signal,time,math
from pathlib import Path
from .common import atomic_json,digest,fingerprint,PACKAGE
from .reference_models import LABELS,head_gradients
from .reference_mechanisms import classification_with_abstention
from .reference_trace import Capture,emit,validate
from .train import save_torch,rng_state,restore_rng
from .reference_worker import seed_all,optimizer_for
from .metaentail import fit_type_filter,nli_target
from .reference_execution import Cadence,attach,metadata,poll_archives

def run(args,assets,data,r):
    import torch
    from .execution_compat import source_map
    from .reference_execution import reset_update_peak,memory_metadata
    from .metaentail_neural import MetaEntailFoundation
    seed_all(args.seed);adapter=MetaEntailFoundation(assets['mnli'],device=args.device,checkpointing=getattr(args,'activation_checkpointing',True))
    attach(adapter,args)
    adapter.model_id='metaentail';adapter.parameter_trace=lambda:head_gradients(adapter.model)
    feasibility=fit_type_filter(data['train'],split='train')
    groups={s:[adapter.encode(row,feasibility,split='dev' if s=='valid' else s,max_length=512) for row in rows] for s,rows in data.items()}
    pairs=[f for g in groups['train'] for f in g['features']]
    selected=next((i for i,row in enumerate(data['train']) if row['relation']!='no_relation'),0)
    selected_id=data['train'][selected]['id'];required={f['candidate_relation'] for f in groups['train'][selected]['features']}
    orders=[]
    for ep in range(r['epochs']):order=list(range(len(pairs)));random.Random(args.seed+ep).shuffle(order);orders.append(order)
    plan=[(ep,order[i:i+r['logical_batch']]) for ep,order in enumerate(orders) for i in range(0,len(order),r['logical_batch'])]
    contract={'schema':'reference-metaentail-execution.v2','recipe':r,'seed':args.seed,'physical_pairs':getattr(args,'execution_policy',args.micro_batch),
        'data':{s:fingerprint(rows) for s,rows in data.items()},'filter':feasibility.to_dict(),'pairs':len(pairs),
        'orders':fingerprint(orders),'budget':len(plan),'assets':adapter.identity(),
        'source':source_map([p.name for p in sorted(PACKAGE.glob('reference_*.py'))]),
        'runtime':{'torch':torch.__version__,'transformers':__import__('transformers').__version__}}
    run=Path(args.run_dir);path=run/'checkpoints/latest.pt';opt,sched=optimizer_for(adapter,r,len(plan))
    state={'cursor':0,'traces':[],'captured_candidates':[],'best_dev':-1.,'best_cursor':None,'best_sha256':None,'best_file':None}
    def save():save_torch(path,dict(contract=contract,model=adapter.model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(),state=state))
    from .reference_storage import initialize_checkpoint,initial_save_pending
    if args.resume and not path.is_file() and initial_save_pending(run,contract):
        initialize_checkpoint(run,path,contract,save)
    if args.resume:
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if saved['contract']!=contract:raise ValueError('immutable NLI pair/filter/data/recipe drift')
        adapter.model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);state=saved['state'];restore_rng(saved['rng']);del saved
        if state['best_sha256'] and digest(run/'checkpoints'/state['best_file'])!=state['best_sha256']:raise ValueError('NLI best checkpoint drift')
        declared={e['path'] for e in state['traces']}|{Path(e['path']).with_suffix('.pt').name for e in state['traces']}
        for orphan in list((run/'case_study').iterdir()) if (run/'case_study').exists() else []:
            if orphan.name not in declared:
                if orphan.is_symlink() or orphan.suffix not in ('.json','.pt'):raise ValueError('foreign NLI trace attempt')
                quarantine=run/'orphaned_trace_attempts';quarantine.mkdir(exist_ok=True);orphan.rename(quarantine/(str(time.time_ns())+'-'+orphan.name))
        from .reference_archive import archive_orphans,archive_traces
        archive_orphans(run)
        if state['traces']:archive_traces(run,state['traces'])
    else:
        if run.exists() and any(run.iterdir()):raise ValueError('existing NLI run needs resume')
        run.mkdir(parents=True);initialize_checkpoint(run,path,contract,save)
    cadence=Cadence(getattr(args,'execution_policy',None),cursor=state['cursor'])
    pause=[False];handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for s in handlers:signal.signal(s,lambda *_:pause.__setitem__(0,True))
    started=time.monotonic();start_cursor=state['cursor']
    try:
        while state['cursor']<len(plan):
            poll_archives()
            if pause[0] or args.pause_after is not None and state['cursor']>=args.pause_after or args.diagnostic and state['cursor']>=args.diagnostic_updates:
                if state['cursor']!=cadence.cursor:save();cadence.saved(state['cursor'])
                atomic_json(run/'RUN_PAUSED.json',{'cursor':state['cursor'],'phase':'NLI_pairs'});return 75
            epoch,ids=plan[state['cursor']];adapter.model.train();opt.zero_grad(set_to_none=True);pending=[];batch_loss=0.
            reset_update_peak(args.device)
            for start in range(0,len(ids),args.micro_batch):
                part=[pairs[i] for i in ids[start:start+args.micro_batch]];batch=adapter.batch(part)
                rows=[i for i,f in enumerate(part) if f['sample_id']==selected_id and f['candidate_relation'] not in state['captured_candidates']]
                cap=Capture(adapter.model,rows,len(part)) if rows else None
                before={name:p.detach().cpu().clone() for name,p in adapter.model.named_parameters() if name.startswith('classifier.')} if rows else None
                with cap if cap else __import__('contextlib').nullcontext():
                    loss,logits,_=adapter.forward(batch);scaled=loss*len(part)/len(ids)
                    if not torch.isfinite(scaled):raise ValueError('nonfinite NLI pair loss')
                scaled.backward();batch_loss+=float(scaled.detach())
                if cap:pending.append((batch,part,cap,loss.detach(),logits.detach(),before))
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(),r['max_grad_norm']);opt.step();sched.step()
            for batch,part,cap,loss,logits,before in pending:
                candidates=[part[i]['candidate_relation'] for i in cap.rows]
                state['traces'].append(emit(adapter,batch,[data['train'][selected]],cap,phase='train',step=state['cursor']+1,event=len(state['traces']),
                    directory=run/'case_study',loss=loss,logits=logits,parameter_before=before,
                    extra={'selected_hypotheses':[groups['train'][selected]['group']['candidates'][part[i]['candidate_index']] for i in cap.rows],
                           'candidate_relations':candidates,'nli_order':list(adapter.nli_order),'training_unit':'expanded_pair'}))
                state['captured_candidates']+=candidates
            state['cursor']+=1
            update_memory=memory_metadata(args.device)
            end_epoch=state['cursor']==len(plan) or plan[state['cursor']][0]!=epoch
            if end_epoch:
                dev,_=evaluate(adapter,groups['valid'],data['valid'],args.micro_batch)
                if dev['micro_f1']>state['best_dev']:
                    state['best_dev']=dev['micro_f1'];state['best_cursor']=state['cursor'];state['best_file']=f"best-u{state['cursor']}.pt"
                    save_torch(run/'checkpoints'/state['best_file'],adapter.model.state_dict());state['best_sha256']=digest(run/'checkpoints'/state['best_file'])
                atomic_json(run/f'dev_epoch{epoch}.json',dev)
            if cadence.due() or pending or end_epoch:save();cadence.saved(state['cursor'])
            if pending:
                from .reference_archive import archive_traces
                archive_traces(run,state['traces'])
            for old in (run/'checkpoints').glob('best-u*.pt'):
                if old.name!=state['best_file'] and old.name[6:-3].isdigit() and not old.is_symlink():old.unlink()
            speed=(state['cursor']-start_cursor)/max(time.monotonic()-started,1e-9)
            heartbeat={'phase':'NLI_pairs','batch':state['cursor'],'batches':len(plan),'loss':batch_loss,
                       'batches_per_second':speed,'eta_seconds':(len(plan)-state['cursor'])/max(speed,1e-9),
                       **metadata(adapter,state['cursor'],cadence.cursor)}
            heartbeat.update(update_memory)
            atomic_json(run/'heartbeat.json',heartbeat)
            print(f"[MetaEntail] {state['cursor']}/{len(plan)} pair updates | loss {batch_loss:.5f}",flush=True)
        if state['cursor']!=cadence.cursor:save();cadence.saved(state['cursor'])
        if set(state['captured_candidates'])!=required:raise ValueError('selected training sample did not record all feasible hypotheses')
        adapter.model.load_state_dict(torch.load(run/'checkpoints'/state['best_file'],map_location='cpu',weights_only=True,mmap=True),strict=True)
        splits={};traces=state['traces']
        for s in ('train','valid','test'):
            # Inference groups on training are newly encoded without NLI target;
            # no training target enters the producer or grouped decision.
            gs=groups[s] if s!='train' else [adapter.encode({k:v for k,v in row.items() if k!='relation'},feasibility,split='test',max_length=512) for row in data[s]]
            selected_test=next((i for i,g in enumerate(gs) if g['features']),None) if s=='test' else None
            if s=='test' and selected_test is None:raise ValueError('no inference group has any feasible hypothesis; trace requires an observed forward')
            report,events=evaluate(adapter,gs,data[s],args.micro_batch,case_index=selected_test,
                run=run,step=state['cursor'],event_start=len(traces))
            splits[s]=report;traces+=events
        from .reference_execution import drain_archives
        drain_archives()
        validate(run,traces)
        from .reference_complete import finish_summary
        finish_summary(run,args.model,args.seed,r,splits,state['cursor'],traces,args.data_dir,assets,diagnostic=args.diagnostic)
        return 0
    finally:
        for s,h in handlers.items():signal.signal(s,h)

def evaluate(adapter,groups,records,micro,*,case_index=None,run=None,step=0,event_start=0):
    import torch
    adapter.model.eval();gold=[];pred=[];traces=[];loss_sum=0.;pair_count=0
    with torch.no_grad():
        for index,(encoded,record) in enumerate(zip(groups,records)):
            probabilities=[]
            for start in range(0,len(encoded['features']),micro):
                part=encoded['features'][start:start+micro];batch=adapter.batch(part)
                cap=Capture(adapter.model,list(range(len(part))),len(part)) if index==case_index else None
                with cap if cap else __import__('contextlib').nullcontext():_,logits,_=adapter.forward(batch)
                probs=logits.softmax(-1);probabilities+=probs.cpu().tolist()
                if cap:traces.append(emit(adapter,batch,[record],cap,phase='test',step=step,event=event_start+len(traces),
                    directory=run/'case_study',loss=None,logits=logits,
                    extra={'hypotheses':encoded['group']['candidates'][start:start+micro],'nli_order':list(adapter.nli_order)}))
            decision=adapter.decide(encoded,probabilities);gold.append(record['relation']);pred.append(decision['relation'])
            # Post-prediction metric only. No gold is given to encode/forward/decide.
            for candidate,p in zip(encoded['group']['candidates'],probabilities):
                target=adapter.nli_order.index(nli_target(record['relation'],candidate['relation']))
                loss_sum-=math.log(max(p[target],1e-30));pair_count+=1
            if index==case_index:
                for event in traces:
                    path=run/'case_study'/event['path'];trace=json.loads(path.read_text())
                    trace['grouped_decision']=decision
                    sample=trace['samples'][0];sample['roles']['prediction']=decision['relation']
                    sample['stages'].append({'stage_id':'group_relation_decision','canonical_stage':'selection','input_refs':['output.logits_or_scores'],
                        'output_refs':['final_relation'],'outputs':decision,'purpose':'汇总该句全部候选的三类NLI分数，再执行蕴涵优先和置信度选择，或明确弃答。'})
                    atomic_json(path,trace);event['sha256']=digest(path)
                atomic_json(run/'case_study'/'grouped-decision.json',{'phase':'test','raw_input':record,'decision':decision,'group':encoded['group'],
                    'trace_events':traces,'empty_group':not encoded['features'],'status':'complete','numerics_required_for_nonempty_group':True})
    result=classification_with_abstention(gold,pred,LABELS)
    result['loss']=loss_sum/pair_count if pair_count else None
    result['loss_scope']='post_prediction_NLI_pair_cross_entropy_not_relation_NLL'
    result['loss_items']=pair_count
    result['loss_status']='observed_pair_metric' if pair_count else 'no_feasible_pairs'
    return result,traces
