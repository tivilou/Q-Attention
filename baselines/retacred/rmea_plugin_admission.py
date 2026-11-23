"""Prospective plugin acceptance: new science unit plus reused unchanged core."""
import math
from pathlib import Path

SCHEMA='q-rmea.plugin-training-admission.v3'
PRIOR='docs/evidence/rmea/20261122-scoped-training-admission/scoped-admission.json'
PRIOR_SHA='79d32ab799c041d30952caf8169eeda47a0c76c9eccc6f2d8d90ccda61174916'
CORE=('rmea_token_transport_study','rmea_token_transport_parallel',
      'rmea_token_transport_recovery','rmea_task_runtime','rmea_scoped_admission')

def verify(p,args,environment):
    def require(ok,message):
        if not ok:raise ValueError(message)
    def regular(path):
        f=Path(path)
        require(f.is_file() and not f.is_symlink() and f.stat().st_size<2*1024**2,'Compact regular admission required')
        return f
    a=p.read(regular(args.admission));cfg=p.config('discovery');source=p.sources()
    require(getattr(p,'SCOPED_ADMISSION_SCHEMA',None)==SCHEMA and a.get('schema')==SCHEMA
            and a.get('gate') is True and p.digest(args.admission)==args.admission_sha256,'Plugin admission differs')
    require(a['source_hashes']==source and a['source_sha256']==p.identity(source)
            and a['contract_sha256']==p.identity(p.contract())
            and a['config_sha256']==p.identity(cfg) and a['environment']==environment,'Current provenance/environment differs')
    require(a['execution']==dict(gpus=cfg['allocated_gpus'],seed=13,conditions=list(p.MODES),
            epochs=4,batch_size=32,baseline_retrained=False),'Execution assumptions differ')
    prior_file=regular(p.ROOT/PRIOR)
    require(p.digest(prior_file)==PRIOR_SHA,'Accepted capability record differs')
    prior=p.read(prior_file)
    require(prior['gate'] is True and prior['historical_complete_acceptance'] is False
            and prior['historical_canary_status']=='PARTIAL_RUNTIME_PASS_RNG_ACCEPTANCE_FAILED','Historical scope changed')
    require(a['prior_admission_sha256']==PRIOR_SHA,'Prior capability reference differs')
    for name in CORE:
        key='baselines/retacred/'+name+'.py'
        require(source[key]==prior['source_hashes'][key],'Core capability changed: '+name)
    folder=Path(args.admission).parent
    for name,field in (('canary-validation.json','prior_canary_validation_sha256'),
                       ('RUN_COMPLETE','prior_canary_seal_sha256'),
                       ('dependency-verification.json','dependency_verification_sha256')):
        require(p.digest(regular(folder/name))==prior[field],'Historical evidence bytes differ')
    canary=p.read(folder/'canary-validation.json')
    require(canary['gate'] is True and all(canary.get(n) is True
        for n in ('serial_parallel_resume_equal','export_retry','cos_live_pinned_restore')),'Required capability missing')
    unit_file=regular(Path(args.unit_output)/'unit-validation.json');u=p.read(unit_file)
    require(p.digest(unit_file)==a['unit_sha256']
        and p.read(regular(unit_file.parent/'RUN_COMPLETE'))['summary_sha256']==p.digest(unit_file)
        and u['status']=='PASS' and u['initial_sha256']==p.BASE_INITIAL
        and set(u['results'])==set(p.MODES),'Current plugin feasibility differs')
    design_file=regular(unit_file.parent/'design.json');design=p.read(design_file)
    required={'baselines/retacred/rmea.py','baselines/retacred/frozen_feature_head.py',
              'experiments/diagnostics/rmea_relative_return/relative_return.py',
              'baselines/retacred/rmea_task_runtime.py','baselines/retacred/rmea_token_transport_study.py'}
    require(required <= set(design['source_hashes']), 'Scientific unit dependency coverage missing')
    require(p.digest(design_file)==u['design_sha256']==a['unit_design_sha256']
        and design['code_hashes']['rmea_relative_return.py']==source['baselines/retacred/rmea_relative_return.py']
        and all(source[name]==pin for name,pin in design['source_hashes'].items()),
        'Scientific unit dependency changed')
    for row in u['results'].values():
        require(row['parameters']==81028 and row['flat_initial_state_sha256']==p.BASE_INITIAL
            and row['optimizer_updates']==1 and row['worst4_optimizer_updates']==0
            and row['checkpoint_saved'] is False and row['parity_error']<1e-10,
            'Plugin unit coverage differs')
        for field in ('gradient_norms','worst4_gradient_norms'):
            require(len(row[field])==7 and all(math.isfinite(x) and x>0 for x in row[field].values()),
                    'Plugin finite gradient coverage differs')
    focused_file=regular(folder/'focused-integration.json')
    focused=p.read(focused_file)
    require(p.digest(focused_file)==a['focused_integration_sha256']
        and focused['status']=='PASS' and focused['gate'] is True
        and focused['source_hashes']==source and focused['source_sha256']==p.identity(source)
        and focused['contract_sha256']==p.identity(p.contract())
        and focused['config_sha256']==p.identity(cfg)
        and focused['unchanged_checkpoint_semantics'] is True
        and focused['actual_allocated_gpus']==cfg['allocated_gpus']
        and focused['before_core_sha256']==prior['source_hashes']['baselines/retacred/rmea_frozen_training.py']
        and focused['cli']['exit_code']==0
        and len(focused['synthetic_report_cases'])==2
        and all(x['report_members']==7 and x['retry_identical'] is True for x in focused['synthetic_report_cases']),
        'Completed focused integration evidence missing')
    return a
