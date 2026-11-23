"""Scientific callbacks; the accepted training/worker/COS core stays shared."""
from pathlib import Path
from . import rmea_relative_return_protocol as protocol
from . import rmea_task_runtime as task
from .rmea_task_runtime import backward, score_batch

def require(ok,message):
    if not ok:raise ValueError(message)

def make_head(kind):
    from .rmea_relative_return import RelativeReturnHead
    return RelativeReturnHead(kind)

def launch_spec(output,kind,device,*,resume):
    return task.launch_spec(protocol,'experiments/q_rmea_relative_return_train.py',
                            output,kind,device,resume=resume)

def prepare(recovery,labels,data_dir):
    return task.prepare(protocol,recovery,labels,data_dir)

def capture(output,head,kind,split,batch,metadata):
    # Rebuild ordinary case studies from final checkpoint + pinned features.
    # No duplicate intermediate tensors and no materialized trace claim.
    (Path(output)/protocol.ANALYSIS_DIRECTORY).mkdir(exist_ok=True)
    return []

def worker_receipt(output,condition):
    return dict(analysis='final_checkpoint_replayable',trace_materialized=False)

def validate_completed(output,kind,row,receipt):
    require(row['samples']==[] and row.get('diagnostics') is None
        and receipt.get('analysis')=='final_checkpoint_replayable'
        and receipt.get('trace_materialized') is False,'Wrong scientific completion scope')

def validate_training_cache(output,cache):
    for name,pin in cache.items():
        f=Path(output)/'cache'/name
        require(f.is_file() and not f.is_symlink() and f.stat().st_size==pin['bytes']
                and protocol.digest(f)==pin['sha256'],'Training input bytes differ: '+name)

def finish(cfg,output,parts,*,data_dir=None,assets=None):
    from .rmea_token_transport_recovery import sync_directory
    p=protocol;output=Path(output);results={};initials=set();orders=set()
    for row in parts:
        require(not set(results)&set(row['results']) and row['samples']==[]
                and row.get('diagnostics') is None,'Duplicate or unexpected result')
        results.update(row['results']);initials.add(row['initial_sha']);orders.add(row['order_sha'])
    require(set(results)==set(p.MODES) and initials=={p.BASE_INITIAL} and len(orders)==1,
            'Condition/init/order coverage mismatch')
    m=p.read(output/'manifest.json')
    require(cfg==m['config']==p.config(cfg['profile']),'Frozen config differs')
    if cfg['profile']=='discovery':
        require(next(iter(orders))==m['reference']['order_sha256'],'Historical order mismatch')
    reference=m['reference']['result'];contrasts={}
    candidate=results['return_coherent']['test']
    for name,other in [('native_reference',reference['test'])]+[(k,results[k]['test']) for k in p.MODES[1:]]:
        delta=candidate['positive39_macro_f1']-other['positive39_macro_f1'];base=abs(other['positive39_macro_f1'])
        contrasts[name]=dict(absolute_macro_delta=delta,relative_macro_delta=delta/base if base else None,
                             ce_delta=candidate['ce']-other['ce'])
    full=cfg['profile']=='discovery'
    utility=contrasts['native_reference']['absolute_macro_delta']>0 if full else None
    attribution=all(contrasts[k]['absolute_macro_delta']>0 for k in p.MODES[1:]) if full else None
    summary=dict(schema=p.SCHEMA_PREFIX+'.result.v1',results=results,reference=reference,contrasts=contrasts,
        ordinary_utility_gate=utility,matched_control_gate=attribution,promotion_gate=bool(utility and attribution) if full else None,
        seed=cfg['seed'],full_data_training=cfg['profile']=='discovery',common_initial_state_sha256=p.BASE_INITIAL,
        common_epoch_order_sha256=next(iter(orders)),source_sha256=p.identity(p.sources()),
        config_sha256=p.identity(cfg),selection=cfg['selection'],L5='NOT_ACCEPTED',
        limitations='Exposed-cohort single-seed discovery only; utility and attribution do not establish novelty or independent confirmation')
    p.atomic_json(output/'summary.json',summary)
    readiness=dict(schema=p.SCHEMA_PREFIX+'.analysis-readiness.v1',trace_materialized=False,
        status='Final checkpoints retained; case study rebuildable, not materialized',
        checkpoints={k:dict(path='checkpoints/'+k+'-final.pt',sha256=results[k]['checkpoint_sha256']) for k in p.MODES})
    for row in readiness['checkpoints'].values():
        require(p.digest(output/row['path'])==row['sha256'],'Final checkpoint differs')
    p.atomic_json(output/'analysis-readiness.json',readiness)
    sync_directory(output)
    p.atomic_json(output/'RUN_COMPLETE',dict(artifacts={n:dict(sha256=p.digest(output/n),bytes=(output/n).stat().st_size)
        for n in ('manifest.json','summary.json','analysis-readiness.json')}))
    sync_directory(output)

def closed_analysis(output):
    p=protocol;output=Path(output);m=p.read(output/'manifest.json');ready=p.read(output/'analysis-readiness.json')
    require(m['source_hashes']==p.sources() and m['config']==p.config(m['config']['profile']),'Completed provenance differs')
    require(ready.get('trace_materialized') is False and set(ready['checkpoints'])==set(p.MODES),'Analysis coverage differs')
    for kind,row in ready['checkpoints'].items():
        require(row['path']=='checkpoints/'+kind+'-final.pt' and p.digest(output/row['path'])==row['sha256'],
                'Final checkpoint differs')
    return ready

def export(output,target,*,inject_failure=False):
    return task.export(protocol,closed_analysis,output,target,inject_failure=inject_failure)

def replay(output,condition,split):
    raise ValueError('Use final flat-state head with pinned features; sample-trace capture is not materialized')

def unit(out):
    raise ValueError('Use the already executed GPU feasibility record; no lifecycle matrix per plugin')
