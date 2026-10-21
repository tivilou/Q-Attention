"""Frozen CPU Q-CFRA subset screen; never a complete-data formal runner."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from q_attention.adapters.q_cfra import CFRAModel,objective
from q_attention.analysis.cfra_trace import Trace,values,vector,state_sha,validate_trace,sha
from q_attention.analysis.ecpa_trace import atomic_json
from q_attention.experiments.ecpa_protocol import metrics,exclusive,save_state
from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig
from q_attention.tasks import cfra_data as d

SOURCE_FILES = ('src/q_attention/plugins/q_cfra.py','src/q_attention/adapters/q_cfra.py',
    'src/q_attention/tasks/cfra_data.py','src/q_attention/analysis/cfra_trace.py','experiments/run_cfra_toy.py',
    'src/q_attention/models/relation_transformer.py','src/q_attention/tasks/relation.py',
    'src/q_attention/analysis/ecpa_contract_validator.py','src/q_attention/analysis/ecpa_trace.py',
    'src/q_attention/experiments/ecpa_protocol.py','src/q_attention/metrics.py',
    'tests/test_cfra.py','configs/q_cfra_toy.json','configs/q_cfra_trace.json')


def source_identity():
    return d.digest({name:sha(ROOT/name) for name in SOURCE_FILES})


def inputs(batch,views=True):
    names = ('input_ids','attention_mask','subject_mask','object_mask')+(('neutral_ids',) if views else ())
    return {name:batch[name] for name in names}


def indexes(count,seed,epoch):
    order = torch.randperm(count,generator=torch.Generator().manual_seed(seed+epoch*10007)).tolist()
    # Fixed selected train case is captured in the first actual update.
    if epoch == 0:
        order.remove(0); order.insert(0,0)
    return order


def evaluate(model,split,data,datasets,vocab,labels,batch_size=2,views=True):
    model.eval()
    predicted,gold,loss = [],[],0.
    with torch.no_grad():
        for start in range(0,len(data[split]),batch_size):
            ids = list(range(start,min(start+batch_size,len(data[split]))))
            batch = d.batch(data[split],datasets[split],vocab,ids)
            logits = model(**inputs(batch,views))
            if views:
                logits = logits[:,0]
            predicted.extend(logits.argmax(-1).tolist()); gold.extend(batch['labels'].tolist())
            loss += float(torch.nn.functional.cross_entropy(logits,batch['labels'],reduction='sum'))
    return metrics(predicted,gold,labels,loss)


def logical_backward(model,batch,loss_config,micro_batch,*,capture=False):
    """Same logical objective; bound statevector autograd memory per microbatch."""
    count=len(batch['labels'])
    query_count=batch['attention_mask'].sum()
    components=None
    observed=None
    for offset in range(0,count,micro_batch):
        micro={name:value[offset:offset+micro_batch] for name,value in batch.items()}
        keep=capture and offset==0
        logits=model(**inputs(micro),capture=keep)
        _,parts=objective(logits,model.attention,micro['attention_mask'],micro['labels'],**loss_config)
        sample_weight=len(micro['labels'])/count
        query_weight=micro['attention_mask'].sum()/query_count
        weighted=torch.stack((parts[0]*sample_weight,parts[1]*sample_weight,parts[2]*query_weight))
        total=weighted[0]+loss_config['prediction_js']*weighted[1]+loss_config['attention_js']*weighted[2]
        if not torch.isfinite(total):
            raise ValueError('nonfinite training objective')
        total.backward()
        detached=torch.cat((weighted.detach(),total.detach().reshape(1)))
        components=detached if components is None else components+detached
        if keep:
            observed={name:value.detach().clone() for name,value in model.observed.items()}
    if capture:
        model.observed=observed
    return components


def warmup(root,seed,config,data,datasets,vocab,labels,contract):
    path = root/'warmup.pt'
    phase_identity = dict(phase='warmup',seed=seed,mode='shared')
    torch.manual_seed(seed)
    model = RelationExtractionModel(RelationTransformerConfig(vocab_size=len(vocab),num_labels=len(labels),
        max_length=config['max_length'],**config['model']))
    optim = torch.optim.Adam(model.parameters(),lr=config['warmup']['lr'])
    start,best_metric,best = 0,-math.inf,None
    if path.exists():
        state = torch.load(path,map_location='cpu',weights_only=False)
        if state['contract'] != contract or state.get('phase_identity') != phase_identity:
            raise ValueError('warmup identity mismatch')
        model.load_state_dict(state['model']); optim.load_state_dict(state['optimizer'])
        start,best_metric,best = state['next_epoch'],state['best_metric'],state['best']
        torch.set_rng_state(state['rng'])
    cfg = config['warmup']
    for epoch in range(start,cfg['epochs']):
        model.train()
        order = indexes(len(data['train']),seed,epoch)
        for offset in range(0,len(order),cfg['batch_size']):
            batch = d.batch(data['train'],datasets['train'],vocab,order[offset:offset+cfg['batch_size']])
            optim.zero_grad(set_to_none=True)
            logits = model(**inputs(batch,False))
            torch.nn.functional.cross_entropy(logits,batch['labels']).backward()
            optim.step()
        report = evaluate(model,'valid',data,datasets,vocab,labels,batch_size=config['runtime']['evaluation_batch'],views=False)
        score = report['metrics']['macro_f1']
        if score > best_metric:
            best_metric,best = score,copy.deepcopy(model.state_dict())
        save_state(path,dict(contract=contract,phase_identity=phase_identity,next_epoch=epoch+1,model=model.state_dict(),optimizer=optim.state_dict(),
                             best_metric=best_metric,best=best,rng=torch.get_rng_state()))
        print(json.dumps(dict(seed=seed,mode='warmup',epoch=epoch+1,valid_macro_f1=score)),flush=True)
    model.load_state_dict(best)
    d.initialize_neutral_embeddings(model,data['train'],vocab)
    return model


def replay(model,writer,data,datasets,vocab,labels,config,checkpoint,updates):
    for split in ('train','valid','test'):
        batch = d.batch(data[split],datasets[split],vocab,[0])
        model.eval(); model.zero_grad(set_to_none=True)
        before = vector(model)
        identity = state_sha(model)
        logits = model(**inputs(batch),capture=True)
        loss,parts = objective(logits,model.attention,batch['attention_mask'],batch['labels'],**config['loss'])
        loss.backward()
        writer.add(data[split][0],split,checkpoint,'replay',
                   values(model,batch['labels'],parts,before,vector(model,True),before),updates,identity)
        model.zero_grad(set_to_none=True)


def run_mode(root,mode,seed,base,config,data,datasets,vocab,labels,contract,*,stop_after_epoch=None):
    directory = root/mode
    directory.mkdir(parents=True,exist_ok=True)
    marker = directory/'RUN_COMPLETE.json'
    phase_identity = dict(phase='screen',seed=seed,mode=mode,base_state_sha256=state_sha(base))
    if marker.exists():
        complete = json.loads(marker.read_text())
        if complete['contract']!=contract or complete.get('phase_identity')!=phase_identity or complete['summary_sha256']!=sha(directory/'summary.json') or complete['trace_sha256']!=sha(directory/'sample_trace.json'):
            raise ValueError('completed job identity mismatch')
        validate_trace(directory/'sample_trace.json')
        summary = json.loads((directory/'summary.json').read_text())
        if summary.get('phase_identity') != phase_identity:
            raise ValueError('completed summary phase mismatch')
        return summary
    torch.manual_seed(seed+1000)
    model = CFRAModel(copy.deepcopy(base),mode,config['channel'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params,lr=config['screen']['lr'])
    names = [dict(name=name,shape=list(p.shape),numel=p.numel()) for name,p in model.named_parameters() if p.requires_grad]
    writer = Trace(directory,mode,dict(run_id=root.parent.name,seed=seed,model_identity=mode,
        config_sha256=contract['config_sha256'],dataset='bounded Re-TACRED subset',
        code_revision=contract['source_sha256'],dataset_identity=contract['subset_sha256']),labels,names)
    latest = directory/'latest.pt'
    start,updates,best_metric,best,best_epoch,best_updates = 0,0,-math.inf,None,-1,0
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['contract'] != contract or state.get('phase_identity') != phase_identity:
            raise ValueError('source/config/subset or phase identity mismatch; refuse resume')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start,updates,best_metric,best,best_epoch,best_updates = (state[k] for k in ('next_epoch','updates','best_metric','best','best_epoch','best_updates'))
        torch.set_rng_state(state['rng'])
    else:
        replay(model,writer,data,datasets,vocab,labels,config,'initial',0)
        save_state(latest,dict(contract=contract,phase_identity=phase_identity,next_epoch=0,updates=0,best_metric=best_metric,best=best,best_epoch=best_epoch,
            best_updates=best_updates,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
    cfg = config['screen']
    started = time.monotonic()
    for epoch in range(start,cfg['epochs']):
        model.train()
        order = indexes(len(data['train']),seed,epoch)
        for offset in range(0,len(order),cfg['batch_size']):
            batch = d.batch(data['train'],datasets['train'],vocab,order[offset:offset+cfg['batch_size']])
            optimizer.zero_grad(set_to_none=True)
            capture = updates == 0
            before = vector(model) if capture else None
            identity = state_sha(model) if capture else None
            parts=logical_backward(model,batch,config['loss'],config['runtime']['micro_batch'],capture=capture)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise ValueError('nonfinite training gradient')
            gradient = vector(model,True) if capture else None
            optimizer.step()
            if capture:
                writer.add(data['train'][0],'train','update0','training',
                    values(model,batch['labels'],parts,before,gradient,vector(model)),0,identity)
            updates += 1
        report = evaluate(model,'valid',data,datasets,vocab,labels,batch_size=config['runtime']['evaluation_batch'])
        metric = report['metrics']['macro_f1']
        if metric > best_metric:
            best_metric,best,best_epoch,best_updates = metric,copy.deepcopy(model.state_dict()),epoch+1,updates
        save_state(latest,dict(contract=contract,phase_identity=phase_identity,next_epoch=epoch+1,updates=updates,best_metric=best_metric,best=best,
            best_epoch=best_epoch,best_updates=best_updates,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
        atomic_json(directory/'heartbeat.json',dict(seed=seed,mode=mode,epoch=epoch+1,epochs=cfg['epochs'],updates=updates,valid_macro_f1=metric))
        print(json.dumps(dict(seed=seed,mode=mode,epoch=epoch+1,valid_macro_f1=metric,elapsed=round(time.monotonic()-started,2))),flush=True)
        if stop_after_epoch is not None and epoch+1>=stop_after_epoch and epoch+1<cfg['epochs']:
            return None
    final = copy.deepcopy(model.state_dict())
    replay(model,writer,data,datasets,vocab,labels,config,'final',updates)
    model.load_state_dict(best)
    replay(model,writer,data,datasets,vocab,labels,config,'best',best_updates)
    test = evaluate(model,'test',data,datasets,vocab,labels,batch_size=config['runtime']['evaluation_batch'])
    valid = evaluate(model,'valid',data,datasets,vocab,labels,batch_size=config['runtime']['evaluation_batch'])
    writer.payload['trace_completeness']='complete'; writer.save()
    readiness = validate_trace(directory/'sample_trace.json')
    summary = dict(seed=seed,mode=mode,contract=contract,phase_identity=phase_identity,best_epoch=best_epoch,updates=updates,
        trainable_parameters=sum(p.numel() for p in params),plugin_parameters=sum(p.numel() for p in model.kernel.parameters()) if model.kernel else 0,
        valid=valid,test=test,trace_readiness=readiness,elapsed_seconds=time.monotonic()-started)
    atomic_json(directory/'summary.json',summary)
    atomic_json(marker,dict(contract=contract,phase_identity=phase_identity,summary_sha256=sha(directory/'summary.json'),trace_sha256=sha(directory/'sample_trace.json')))
    model.load_state_dict(final)
    return summary


def aggregate(summaries,config):
    primary = {s['seed']:s['test']['metrics']['macro_f1'] for s in summaries if s['mode']=='quantum'}
    base = {s['seed']:s['test']['metrics']['macro_f1'] for s in summaries if s['mode']=='disabled'}
    if set(primary)!=set(config['seeds']) or set(base)!=set(primary):
        raise ValueError('incomplete paired seed population')
    delta = [primary[s]-base[s] for s in config['seeds']]
    sd = statistics.stdev(delta) if len(delta)>1 else 0.
    base_sd = statistics.stdev(base.values()) if len(delta)>1 else 0.
    mean,positive = statistics.mean(delta),sum(x>0 for x in delta)
    gates = config['gates']
    utility = mean>=gates['mean_delta'] and positive>=gates['positive_seeds'] and (sd<=gates['delta_sd'] or sd<=base_sd)
    shuffle = [s['test']['metrics']['macro_f1'] for s in summaries if s['mode']=='shuffled']
    parity = mean>=gates['mean_delta'] and statistics.mean(shuffle)>=statistics.mean(primary.values())-gates['shuffled_parity_tolerance']
    collapsed = False
    mechanism_diagnostics=[]
    # Actual first update carries the plugin gradients and attention movement.
    for s in summaries:
        if s['mode']=='quantum':
            trace_path = Path(s['_trace_path'])
            payload = json.loads(trace_path.read_text())
            selected = next(x for x in payload['samples'] if x['lane']=='training')
            grad = read_tensor(trace_path,selected,'parameter_gradient')
            groups={'projection':[],'channel':[],'readout':[]}
            cursor=0
            for item in payload['parameter_order']:
                part=grad[cursor:cursor+item['numel']]; cursor+=item['numel']
                for group in groups:
                    if item['name'].startswith('kernel.'+group):
                        groups[group].extend(part.tolist())
            norms={key:np_norm(value) for key,value in groups.items()}
            attention=read_tensor(trace_path,selected,'attention')
            raw=read_tensor(trace_path,selected,'base_scores')
            mask=read_tensor(trace_path,selected,'attention_mask').astype(bool)
            raw[...,~mask]=-1e30
            baseline_attention=torch.from_numpy(raw).softmax(-1).numpy()
            shift=float(abs(attention-baseline_attention)[...,mask,:].sum(-1).max())
            if min(norms.values())<=1e-12 or shift<=1e-8:
                collapsed=True
            mechanism_diagnostics.append(dict(seed=s['seed'],plugin_group_gradient_norms=norms,max_valid_attention_l1_shift=shift))
    kill = sum(x<=0 for x in delta)>=4 or parity or collapsed
    return dict(stage='bounded real-data toy only',seeds=config['seeds'],paired_deltas=delta,mean_delta=mean,
        delta_sample_sd=sd,disabled_sample_sd=base_sd,positive_seeds=positive,shuffled_parity=parity,
        collapsed_plugin_or_intervention=collapsed,mechanism_diagnostics=mechanism_diagnostics,
        classification='kill' if kill else 'promote_to_formal_preparation' if utility else 'inconclusive',
        quantum_attribution='not established; compare all controls separately',formal_execution='collaborator only')


def read_tensor(trace_path,sample,name):
    import numpy as np
    rep = next(x for x in sample['representations'] if x['id']==name)
    return np.load(trace_path.parent/rep['artifact_refs'][0]['path'],allow_pickle=False)


def np_norm(array):
    import numpy as np
    return float(np.linalg.norm(array))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--canary',action='store_true',help='Tiny diagnostic budget; never a toy utility result.')
    parser.add_argument('--stop-after-epoch',type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    config = json.loads((ROOT/'configs/q_cfra_toy.json').read_text())
    if args.canary:
        config['seeds']=[13]; config['per_class']=1
        config['warmup']['epochs']=1; config['screen']['epochs']=2
        config['model'].update(dim=8,num_heads=2,ff_dim=16)
    root = Path(args.output_dir).resolve()
    if root.exists() and not args.resume:
        raise ValueError('fresh run directory already exists; use --resume only with matching identities')
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'.run.lock'):
        data,datasets,vocab,labels,subset = d.prepare(args.data_dir,config)
        contract = dict(source_sha256=source_identity(),config_sha256=d.digest(config),subset_sha256=d.digest(subset))
        path = root/'run_manifest.json'
        if path.exists() and json.loads(path.read_text())['contract']!=contract:
            raise ValueError('source/config/subset changed; refuse resume')
        atomic_json(path,dict(contract=contract,config=config,subset=subset,data_dir=str(Path(args.data_dir).resolve()),
                             git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        results = []
        for seed in config['seeds']:
            directory = root/f'seed{seed}'
            directory.mkdir(exist_ok=True)
            base = warmup(directory,seed,config,data,datasets,vocab,labels,contract)
            for mode in config['modes']:
                summary = run_mode(directory,mode,seed,base,config,data,datasets,vocab,labels,contract,stop_after_epoch=args.stop_after_epoch)
                if summary is None:
                    return 0
                summary['_trace_path']=str(directory/mode/'sample_trace.json')
                results.append(summary)
        counts = {s['plugin_parameters'] for s in results if s['mode']!='disabled'}
        updates = {s['updates'] for s in results}
        if len(counts)!=1 or len(updates)!=1:
            raise ValueError('unmatched plugin parameter or update budget')
        decision = dict(stage='numerical canary only',classification='not a utility screen') if args.canary else aggregate(results,config)
        summary = dict(contract=contract,decision=decision,conditions=[{k:v for k,v in s.items() if k!='_trace_path'} for s in results])
        atomic_json(root/'summary.json',summary)
        atomic_json(root/'RUN_COMPLETE.json',dict(contract=contract,summary_sha256=sha(root/'summary.json')))
        print(json.dumps(decision),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
