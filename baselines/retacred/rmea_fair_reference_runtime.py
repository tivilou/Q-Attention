"""Dense FP32 callbacks for the original-budget symmetric comparison."""
from pathlib import Path
from . import rmea_fair_reference_protocol as protocol
from . import rmea_task_runtime as task


def require(ok,message):
    if not ok:raise ValueError(message)


def initialize_head(kind,initial,cfg):
    from .rmea_matched_routers import initialize_head as initialize
    return initialize(kind,initial,cfg)


def validate_initialized_head(head,kind,initial,cfg):
    from .rmea_matched_routers import validate_initialized_head as validate
    return validate(head,kind,initial,cfg)


def score_batch(head,inputs):
    return head(*inputs)


def backward(head,inputs,y):
    from torch.nn import functional as F
    value=F.cross_entropy(head(*inputs),y)
    require(bool(value.isfinite()),'Nonfinite dense loss')
    value.backward()
    return value.detach()


def launch_spec(output,kind,device,*,resume):
    return task.launch_spec(protocol,'experiments/q_rmea_fair_reference_train.py',
                            output,kind,device,resume=resume)


def prepare(recovery,labels,data_dir):
    return task.prepare(protocol,recovery,labels,data_dir)


def capture_selected(output,head,kind,split,batch,metadata,evaluation_state):
    # Finalization authenticates the committed selected checkpoint, not final.
    (Path(output)/protocol.ANALYSIS_DIRECTORY).mkdir(exist_ok=True)
    return []


def worker_receipt(output,condition):
    return dict(analysis='selected_checkpoint_replayable',trace_materialized=False)


def validate_completed(output,kind,row,receipt):
    require(row['samples']==[] and row.get('diagnostics') is None
        and receipt.get('analysis')=='selected_checkpoint_replayable'
        and receipt.get('trace_materialized') is False,'Completion scope differs')


def validate_training_cache(output,cache):
    for name,pin in cache.items():
        path=Path(output)/'cache'/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==pin['bytes']
                and protocol.digest(path)==pin['sha256'],'Input bytes differ: '+name)


def selected_inventory(output,results,cfg):
    import torch
    from .rmea_token_transport_study import state_identity
    p=protocol;output=Path(output);inventory={}
    from .rmea_training_options import expected_training_identities,per_epoch
    initial,order=expected_training_identities(cfg)
    for kind,row in results.items():
        chosen=row['evaluation_checkpoint'];path=output/chosen['path']
        require(chosen['path']=='checkpoints/'+kind+'-selected.pt' and path.is_file() and not path.is_symlink()
            and chosen['sha256']==p.digest(path),'Selected checkpoint differs')
        state=torch.load(path,map_location='cpu',weights_only=True)
        require(state_identity(state)==chosen['state_sha256'] and row['selected_epoch']==chosen['epoch']
            and 1<=chosen['epoch']<=cfg['epochs'],'Selected state/epoch differs')
        final=output/'checkpoints'/(kind+'-final.pt')
        require(final.is_file() and not final.is_symlink() and p.digest(final)==row['checkpoint_sha256']
            and state_identity(torch.load(final,map_location='cpu',weights_only=True))==row['final_state_sha256'],
            'Final checkpoint identity differs')
        require(row['updates']==cfg['epochs']*per_epoch(cfg) and row['initial_state_sha256']==initial
            and row['parameters']==(68200 if kind=='uniform' else 81028),'Budget or initialization differs')
        inventory[kind]=dict(chosen,final_path='checkpoints/'+final.name,final_sha256=row['checkpoint_sha256'])
    return inventory,initial,order


def finish(cfg,output,parts,*,data_dir=None,assets=None):
    from .rmea_token_transport_recovery import sync_directory
    p=protocol;output=Path(output);results={};initials=set();orders=set()
    for row in parts:
        require(not set(results)&set(row['results']) and row['samples']==[] and row.get('diagnostics') is None,
                'Duplicate or unexpected condition result')
        results.update(row['results']);initials.add(row['initial_sha']);orders.add(row['order_sha'])
    require(set(results)==set(p.MODES),'Condition coverage differs')
    manifest=p.read(output/'manifest.json')
    require(cfg==manifest['config']==p.config(cfg['profile']) and manifest['source_hashes']==p.sources(),
            'Frozen config/source differs')
    inventory,initial,order=selected_inventory(output,results,cfg)
    require(initials=={initial} and orders=={order},'Shared initial state/order differs')
    contrasts={kind:dict(test_all40_macro_delta=results['quantum']['test']['macro_f1']-results[kind]['test']['macro_f1'],
        test_positive39_macro_delta=results['quantum']['test']['positive39_macro_f1']-results[kind]['test']['positive39_macro_f1'])
        for kind in p.MODES if kind!='quantum'}
    qualification=all(r['test_all40_macro_delta']>0 for r in contrasts.values()) if cfg['profile']=='discovery' else None
    summary=dict(schema=p.SCHEMA_PREFIX+'.result.v1',results=results,contrasts=contrasts,
        qualification_gate=qualification,reference=dict(role='provenance_only',used_as_comparator=False,
            source_input_manifest=p.BASE_MANIFEST,historical_summary=p.BASE_SUMMARY),
        seed=cfg['seed'],full_data_training=cfg['profile']=='discovery',common_initial_state_sha256=initial,
        common_epoch_order_sha256=order,source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg),
        selection=cfg['selection'],L3='NOT_ACCEPTED',L4='NOT_ACCEPTED',L5='NOT_ACCEPTED',
        limitations=['Single seed, previously exposed benchmark; no significance or independent confirmation.',
            'Equal registered counts do not establish equal effective capacity.',
            'No metric/checkpoint/seed/budget rescue; all five endpoints retained.'])
    p.atomic_json(output/'summary.json',summary)
    p.atomic_json(output/'analysis-readiness.json',dict(schema=p.SCHEMA_PREFIX+'.analysis-readiness.v1',
        trace_materialized=False,status='Selected-checkpoint replayable; no materialized case-study claim',checkpoints=inventory))
    sync_directory(output)
    p.atomic_json(output/'RUN_COMPLETE',dict(artifacts={n:dict(sha256=p.digest(output/n),bytes=(output/n).stat().st_size)
        for n in ('manifest.json','summary.json','analysis-readiness.json')}))
    sync_directory(output)


def closed_analysis(output):
    p=protocol;output=Path(output);m=p.read(output/'manifest.json');s=p.read(output/'summary.json')
    ready=p.read(output/'analysis-readiness.json')
    require(m['source_hashes']==p.sources() and m['config']==p.config(m['config']['profile'])
        and set(s['results'])==set(p.MODES),'Completed provenance/coverage differs')
    inventory,initial,order=selected_inventory(output,s['results'],m['config'])
    require(ready['checkpoints']==inventory and ready['trace_materialized'] is False
        and s['common_initial_state_sha256']==initial and s['common_epoch_order_sha256']==order,
        'Selected-checkpoint analysis differs')
    return ready


def export(output,target,*,inject_failure=False):
    return task.export(protocol,closed_analysis,output,target,inject_failure=inject_failure)


def verify_admission(args,environment):
    import os
    p=protocol
    require(os.environ.get('QATTN_COLLABORATOR_FORMAL')=='1','Collaborator formal execution flag required')
    def regular(path):
        path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<2*1024**2,
                              'Regular compact acceptance required');return path
    a=p.read(regular(args.admission))
    require(a.get('schema')==p.SCOPED_ADMISSION_SCHEMA and a.get('gate') is True
        and args.admission_sha256==p.digest(args.admission),'Admission identity differs')
    require(a['source_hashes']==p.sources() and a['source_sha256']==p.identity(p.sources())
        and a['contract_sha256']==p.identity(p.contract()) and a['config_sha256']==p.identity(p.config('discovery'))
        and a['environment']==environment,'Admission source/config/environment differs')
    require(a['formal_executor']=='collaborator' and a['historical_results_unchanged'] is True,
            'Formal execution boundary differs')
    folder=Path(args.admission).parent
    for name,base in [('unit-validation.json',Path(args.unit_output)),('canary-validation.json',folder),
                      ('dependency-verification.json',folder)]:
        path=regular(base/name);require(a['evidence_sha256'][name]==p.digest(path),'Acceptance evidence differs')
        record=p.read(path)
        success_field='verified' if name=='dependency-verification.json' else 'gate'
        require(record.get(success_field) is True and all(record[k] is True
            for k in ('gate','verified') if k in record),'Acceptance failed or contradictory')
        if name!='dependency-verification.json':
            require(record['source_sha256']==p.identity(p.sources()) and record['contract_sha256']==p.identity(p.contract()),
                    'Acceptance scope differs')
        seal=regular(base/'RUN_COMPLETE')
        if name!='dependency-verification.json':
            require(p.read(seal)==dict(artifacts={name:p.digest(path)}),'Acceptance seal differs')
    c=p.read(folder/'canary-validation.json')
    require(c['actual_parent_child_launch'] is True and c['export_injected_failure_clean_retry'] is True
            and c['official_test_rows']==0 and c['official_valid_rows']==0,'Focused lifecycle evidence differs')
    d=p.read(folder/'dependency-verification.json')
    require(d['request_sha256']==p.INPUT_REQUEST_SHA and d['acknowledgement_sha256']==p.INPUT_ACK_ID
        and d['objects']==13 and d['versioned_head_checks']==13 and d['total_pinned_bytes']==6947440640
        and d['payload_bytes_read']==0,'Pinned input dependency differs')


def validate_worker_admission(output,environment):
    from types import SimpleNamespace
    p=protocol;output=Path(output);cfg=p.read(output/'manifest.json')['config']
    if cfg['profile']=='canary':return
    require(cfg==p.config('discovery'),'Worker discovery configuration differs')
    root=output.parent;folder=root/'recovery-inputs/admission'
    require(p.read(root.parent/'run-identity.json')==dict(run_id=root.name,profile='discovery',
        source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg)),
        'Worker lacks the admitted outer run identity')
    admission=folder/'canary/admission.json'
    verify_admission(SimpleNamespace(admission=admission,admission_sha256=p.digest(admission),
                                    unit_output=folder/'unit'),environment)


def unit(output):
    raise ValueError('Use retained numerical/factory evidence plus the scoped adapter canary; no lifecycle matrix')


def replay(output,condition,split):
    raise ValueError('Selected checkpoint and pinned features are retained; sample traces are not materialized')
