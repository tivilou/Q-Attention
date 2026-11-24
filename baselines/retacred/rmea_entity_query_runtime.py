"""Only scientific factory, scoped admission and result contrast change."""
from pathlib import Path
from types import SimpleNamespace
from . import rmea_entity_query_protocol as protocol
from . import rmea_fair_reference_runtime as base
from . import rmea_task_runtime as task
from .rmea_entity_query import initialize_head,validate_initialized_head

require=base.require
score_batch=base.score_batch
backward=base.backward
capture_selected=base.capture_selected
worker_receipt=base.worker_receipt
validate_completed=base.validate_completed
validate_training_cache=base.validate_training_cache
selected_inventory=base.selected_inventory


def launch_spec(output,kind,device,*,resume):
    return task.launch_spec(protocol,'experiments/q_rmea_entity_query_train.py',output,kind,device,resume=resume)


def prepare(recovery,labels,data_dir):
    return task.prepare(protocol,recovery,labels,data_dir)


def finish(cfg,output,parts,*,data_dir=None,assets=None):
    from .rmea_token_transport_recovery import sync_directory
    p=protocol;output=Path(output);results={};initials=set();orders=set()
    for row in parts:
        require(not set(results)&set(row['results']) and row['samples']==[] and row.get('diagnostics') is None,'Unexpected result')
        results.update(row['results']);initials.add(row['initial_sha']);orders.add(row['order_sha'])
    require(set(results)==set(p.MODES),'Five new conditions required')
    manifest=p.read(output/'manifest.json')
    require(cfg==manifest['config']==p.config('discovery') and manifest['source_hashes']==p.sources(),'Frozen configuration differs')
    inventory,initial,order=selected_inventory(output,results,cfg)
    native=p.read(p.ROOT/(p.EVIDENCE+'native-summary.json'))
    old=p.read(p.ROOT/(p.EVIDENCE+'native-manifest.json'))
    keys=('seed','counts','epochs','batch_size','eval_batch_size','training_dtype','order_generator','optimizer',
        'lr','weight_decay','betas','optimizer_eps','clip_norm','validation_metric','reported_checkpoint','selection')
    require(all(cfg[k]==old['config'][k] for k in keys),'Native comparator training protocol differs')
    require(old['cache']==manifest['cache'] and old['source_input_manifest']==manifest['source_input_manifest'],
            'Native comparator inputs differ')
    require(initials=={initial}=={native['common_initial_state_sha256']} and orders=={order}=={native['common_epoch_order_sha256']},
            'Common initialization/order differs')
    require(set(native['results'])==set(base.protocol.MODES) and all(r['updates']==9140 for r in native['results'].values()),
            'Completed native controls differ')
    controls={k:v for k,v in results.items() if k!='quantum_context'}
    controls.update({'native/'+k:v for k,v in native['results'].items()})
    q=results['quantum_context']['test']['macro_f1']
    contrasts={k:dict(test_all40_macro_delta=q-v['test']['macro_f1']) for k,v in controls.items()}
    p.atomic_json(output/'summary.json',dict(schema=p.SCHEMA_PREFIX+'.result.v1',results=results,
        reused_native_results=native['results'],contrasts=contrasts,qualification_gate=all(v['test_all40_macro_delta']>0 for v in contrasts.values()),
        seed=cfg['seed'],full_data_training=True,common_initial_state_sha256=initial,common_epoch_order_sha256=order,
        source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg),selection=cfg['selection'],
        reference=dict(role='completed full native controls',used_as_comparator=True,**cfg['reference']),
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',limitations=['Single previously exposed seed13 qualification; no confirmation.',
        'Conditioned queries and Born measurements are known tools; theory novelty not established.',
        'Registered parameters equal; effective capacity not claimed equal; real circuit uses more CNOTs.']))
    p.atomic_json(output/'analysis-readiness.json',dict(trace_materialized=False,status='Selected checkpoint replayable',checkpoints=inventory))
    sync_directory(output)
    p.atomic_json(output/'RUN_COMPLETE',dict(artifacts={n:dict(sha256=p.digest(output/n),bytes=(output/n).stat().st_size)
        for n in ('manifest.json','summary.json','analysis-readiness.json')}));sync_directory(output)


def closed_analysis(output):
    p=protocol;output=Path(output);m=p.read(output/'manifest.json');s=p.read(output/'summary.json')
    require(m['config']==p.config('discovery') and m['source_hashes']==p.sources() and set(s['results'])==set(p.MODES),'Completed provenance differs')
    inventory,_,_=selected_inventory(output,s['results'],m['config'])
    ready=p.read(output/'analysis-readiness.json');require(ready['checkpoints']==inventory,'Selected checkpoint differs')
    return ready


def export(output,target,*,inject_failure=False):
    return task.export(protocol,closed_analysis,output,target,inject_failure=inject_failure)


def verify_admission(args,environment):
    p=protocol;a=p.read(args.admission)
    require(a.get('schema')==p.SCOPED_ADMISSION_SCHEMA and a.get('gate') is True
        and p.digest(args.admission)==args.admission_sha256,'Changed-plugin admission differs')
    require(a['source_sha256']==p.identity(p.sources()) and a['contract_sha256']==p.identity(p.contract())
        and a['environment']==environment and a['formal_executor']=='owner','Admission scope differs')
    unit=Path(args.unit_output)/'unit-validation.json';u=p.read(unit)
    require(p.digest(unit)==a['unit_sha256'] and u['gate'] is True and u['source_sha256']==a['source_sha256']
        and u['environment']==environment and u['input_manifest_sha256']==p.BASE_MANIFEST
        and u['contract_sha256']==a['contract_sha256'] and u['config_sha256']==p.identity(p.config('discovery')),
            'Changed-model technical check differs')
    require(p.read(Path(args.unit_output)/'RUN_COMPLETE')=={'artifacts':{'unit-validation.json':p.digest(unit)}},'Technical seal differs')
    # Actual old accepted evidence is carried unchanged, not reissued for this source.
    require(a['reused_capability_files']==p.CORE_EVIDENCE,'Complete accepted core capability inventory required')
    for name,h in a['reused_capability_files'].items():
        require(p.digest(Path(args.admission).parent/name)==h,'Retained core capability evidence differs')
    require(u['test_model_rows']==0 and u['test_target_reads']==0 and u['real_train_updates_per_condition']==1,
            'Technical scope differs')


def validate_worker_admission(output,environment):
    root=Path(output).parent;folder=root/'recovery-inputs/admission'
    a=folder/'canary/admission.json'
    verify_admission(SimpleNamespace(admission=a,admission_sha256=protocol.digest(a),unit_output=folder/'unit'),environment)


def unit(output):raise ValueError('Use changed-model feasibility; reuse stable-core capability evidence')
def replay(output,condition,split):raise ValueError('Use selected checkpoint with pinned features after completion')
