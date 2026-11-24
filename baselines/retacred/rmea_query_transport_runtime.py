"""Scientific factory, exact technical-evidence reuse and the twelve comparisons."""
from pathlib import Path
import os
from . import rmea_query_transport_protocol as protocol
from . import rmea_fair_reference_runtime as base
from . import rmea_task_runtime as task
from .rmea_query_transport import initialize_head,validate_initialized_head

require=base.require
score_batch=base.score_batch
backward=base.backward
capture_selected=base.capture_selected
worker_receipt=base.worker_receipt
validate_completed=base.validate_completed
validate_training_cache=base.validate_training_cache
selected_inventory=base.selected_inventory

def launch_spec(output,kind,device,*,resume):
    return task.launch_spec(protocol,'experiments/q_rmea_query_transport_train.py',output,kind,device,resume=resume)
def prepare(recovery,labels,data_dir):return task.prepare(protocol,recovery,labels,data_dir)

def verify_admission(args,environment):
    p=protocol;a=p.read(args.admission);u=p.read(p.ROOT/p.UNIT);d=p.read(p.ROOT/p.UNIT_DESIGN)
    require(a.get('schema')==p.SCOPED_ADMISSION_SCHEMA and a.get('gate') is True and p.digest(args.admission)==args.admission_sha256,'Admission differs')
    require(a['source_sha256']==p.identity(p.sources()) and a['contract_sha256']==p.identity(p.contract())
        and a['environment']==environment and a['formal_executor']=='owner','Current adapter/environment differs')
    # Keep actual prior evidence unchanged; its parent-source identity is NOT this full source.
    require(p.digest(p.ROOT/p.UNIT)==a['prior_unit_sha256'] and u['gate'] is True
        and u['source_sha256']==d['input_hashes']['source'] and u['plugin_sha256']==p.digest(p.ROOT/'baselines/retacred/rmea_query_transport.py')
        and u['code_sha256']==d['code_sha256']==p.digest(p.ROOT/(p.EVIDENCE+'unit-replay.py')),'Exact prior plugin execution differs')
    require(set(u['conditions'])==set(p.MODES) and u['test_model_rows']==u['test_target_reads']==u['valid_model_rows']==0
        and all(v['real_train_updates']==1 and v['finite_gradients'] and v['parameters']==81028 for v in u['conditions'].values()),'Prior technical scope differs')
    require(a['unit_execution_exit_code']==0 and a['unit_execution_code_sha256']==u['code_sha256'],'Actual unit execution not bound')
    supplied=Path(args.unit_output)
    require(p.digest(supplied/'unit-validation.json')==a['prior_unit_sha256']
        and p.read(supplied/'RUN_COMPLETE')=={'artifacts':{'unit-validation.json':a['prior_unit_sha256']}},'Supplied technical evidence differs')
    for name,h in d['source_hashes'].items():require(p.digest(p.ROOT/name)==h,'Changed primitive since technical check')
    require(a['reused_capability_files']==p.CORE_EVIDENCE,'Core capability inventory differs')
    for name,h in a['reused_capability_files'].items():require(p.digest(Path(args.admission).parent/name)==h,'Actual core evidence differs')
    # The accepted queue inherits this parent's environment for each worker.
    os.environ['QATTN_QUERY_TRANSPORT_ADMISSION_SHA256']=args.admission_sha256

def validate_worker_admission(output,environment):
    from types import SimpleNamespace
    folder=Path(output).parent/'recovery-inputs/admission';a=folder/'canary/admission.json'
    pin=os.environ.get('QATTN_QUERY_TRANSPORT_ADMISSION_SHA256','')
    require(len(pin)==64 and all(c in '0123456789abcdef' for c in pin),'Parent admission pin missing')
    verify_admission(SimpleNamespace(admission=a,admission_sha256=pin,unit_output=folder/'unit'),environment)

def finish(cfg,output,parts,*,data_dir=None,assets=None):
    from .rmea_token_transport_recovery import sync_directory
    p=protocol;output=Path(output);results={};initials=set();orders=set()
    for row in parts:
        require(not set(results)&set(row['results']) and row['samples']==[] and row.get('diagnostics') is None,'Unexpected result')
        results.update(row['results']);initials.add(row['initial_sha']);orders.add(row['order_sha'])
    require(set(results)==set(p.MODES),'Three new conditions required')
    manifest=p.read(output/'manifest.json')
    require(cfg==manifest['config']==p.config('discovery') and manifest['source_hashes']==p.sources(),'Frozen configuration differs')
    inventory,initial,order=selected_inventory(output,results,cfg)
    keys=('seed','counts','epochs','batch_size','eval_batch_size','training_dtype','order_generator','optimizer',
        'lr','weight_decay','betas','optimizer_eps','clip_norm','validation_metric','reported_checkpoint','selection')
    cohorts={}
    for label,sp,mp,expected in [('native',p.NATIVE_SUMMARY,p.NATIVE_MANIFEST,set(base.protocol.MODES)),
        ('context',p.CONTEXT_SUMMARY,p.CONTEXT_MANIFEST,{'quantum_context','dephased_context','real12_context','softmax12_context','real_circuit_context'})]:
        old=p.read(p.ROOT/mp);summary=p.read(p.ROOT/sp)
        require(all(cfg[k]==old['config'][k] for k in keys),'Comparator protocol differs: '+label)
        require(old['cache']==manifest['cache'] and old['source_input_manifest']==manifest['source_input_manifest'],'Comparator inputs differ: '+label)
        require(initials=={initial}=={summary['common_initial_state_sha256']} and orders=={order}=={summary['common_epoch_order_sha256']},'Initialization/order differs: '+label)
        require(set(summary['results'])==expected and all(v['updates']==9140 for v in summary['results'].values()),'Complete comparison differs: '+label)
        cohorts[label]=summary['results']
    controls={k:v for k,v in results.items() if k!='quantum_transport'}
    controls.update({label+'/'+k:v for label,cohort in cohorts.items() for k,v in cohort.items()})
    require(len(controls)==12,'Exactly12 declared controls required')
    q=results['quantum_transport']['test']['macro_f1']
    contrasts={k:dict(test_all40_macro_delta=q-v['test']['macro_f1']) for k,v in controls.items()}
    p.atomic_json(output/'summary.json',dict(schema=p.SCHEMA_PREFIX+'.result.v1',results=results,reused_results=cohorts,
        contrasts=contrasts,qualification_gate=all(v['test_all40_macro_delta']>0 for v in contrasts.values()),seed=cfg['seed'],full_data_training=True,
        common_initial_state_sha256=initial,common_epoch_order_sha256=order,source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg),
        selection=cfg['selection'],reference=cfg['reference'],L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',
        limitations=['Single exposed seed13 qualification; independent confirmation not performed.',
            'Shared-unitary Gram invariance is known and not preserved by token normalization; no utility or novelty theorem.',
            '81028 registered parameters do not establish equal effective capacity; real control has more gates.']))
    p.atomic_json(output/'analysis-readiness.json',dict(trace_materialized=False,status='Selected checkpoint replayable',checkpoints=inventory))
    sync_directory(output);p.atomic_json(output/'RUN_COMPLETE',dict(artifacts={n:dict(sha256=p.digest(output/n),bytes=(output/n).stat().st_size)
        for n in ('manifest.json','summary.json','analysis-readiness.json')}));sync_directory(output)

def closed_analysis(output):
    p=protocol;output=Path(output);m=p.read(output/'manifest.json');s=p.read(output/'summary.json')
    require(m['config']==p.config('discovery') and m['source_hashes']==p.sources() and set(s['results'])==set(p.MODES),'Completed provenance differs')
    inventory,_,_=selected_inventory(output,s['results'],m['config']);ready=p.read(output/'analysis-readiness.json')
    require(ready['checkpoints']==inventory,'Selected checkpoint differs');return ready
def export(output,target,*,inject_failure=False):return task.export(protocol,closed_analysis,output,target,inject_failure=inject_failure)
def unit(output):raise ValueError('Reuse exact-plugin technical results')
def replay(output,condition,split):raise ValueError('Use a separately declared selected checkpoint diagnostic')
