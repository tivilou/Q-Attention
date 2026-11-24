"""Prospective entity-conditioned-query full-data qualification."""
from . import rmea_fair_reference_protocol as base
from .rmea_fair_reference_protocol import (BASE_MANIFEST,BASE_SUMMARY,BASE_INITIAL,LABELS_SHA,
    INPUT_REQUEST_SHA,INPUT_ACK_ID,REPORT_FILES,CANARY_MANIFEST,ROOT,EXPECTED,identity,read,git,atomic_json,digest)

EXPERIMENT='entity-query'
SCHEMA_PREFIX='q-rmea.entity-query'
SCOPED_ADMISSION_SCHEMA=SCHEMA_PREFIX+'.changed-plugin-admission.v1'
REFERENCE_METHOD=base.REFERENCE_METHOD
REFERENCE_DESCRIPTION=base.REFERENCE_DESCRIPTION
ANALYSIS_DIRECTORY='analysis'
MODES=('quantum_context','dephased_context','real12_context','softmax12_context','real_circuit_context')
EVIDENCE='docs/evidence/rmea/20261124-entity-query/'
REVIEW=EVIDENCE+'review.json'
CORE_EVIDENCE={'dependency-verification.json':'923ec6836d0c32747327f521dffcfde39a83ed84ebca31fea88122df15234530',
    'canary-validation.json':'782fa6b3410c6d8ab5de0849826961eba88b8ccab05c6d67c1af13aa3ec32867',
    'RUN_COMPLETE':'9ed903c679d8d778550b47df1758bd01251c83cfb4dafb84ad62d7a90eb4b5a9'}
EXTRA=('baselines/retacred/rmea_entity_query.py','baselines/retacred/rmea_entity_query_protocol.py',
       'baselines/retacred/rmea_entity_query_runtime.py','experiments/q_rmea_entity_query_train.py',
       'experiments/check_rmea_entity_query.py',
       EVIDENCE+'protocol.md',EVIDENCE+'native-summary.json',EVIDENCE+'native-manifest.json')


def sources():
    rows=base.sources()
    for n in EXTRA:
        p=ROOT/n
        if not p.is_file() or p.is_symlink():raise ValueError('Missing entity-query source: '+n)
        rows[n]=digest(p)
    return rows


def config(profile):
    if profile!='discovery':raise ValueError('Reuse core capabilities; no repeated canary profile')
    cfg=base.config(profile)
    native=base.parameter_schemas()
    mapping={'quantum_context':'quantum','dephased_context':'dephased','real12_context':'real12',
        'softmax12_context':'softmax12','real_circuit_context':'quantum'}
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1',conditions=list(MODES),
        parameter_schemas={k:native[v] for k,v in mapping.items()},
        primary='Selected TEST all40Macro: quantum_context strictly greater than all nine new and reused controls',
        qualification='All nine differences positive; otherwise close this exact profile; no maturity inherited',
        reference=dict(native_manifest=digest(ROOT/(EVIDENCE+'native-manifest.json')),
            native_summary=digest(ROOT/(EVIDENCE+'native-summary.json')),retrain_all_conditions=False),
        replication='Seed13 qualification only; independent multi-seed confirmation must be frozen prospectively if it passes',
        input_condition='Average of subject and object masked means of concatenated h/roles; tied input projection before query nonlinearity',
        classical_control='Identical anchor; parameter-tied query conditioning; four-layer real circuit has12CNOT versus native6')
    return cfg


def contract():
    return dict(profiles={'discovery':config('discovery')},formal_owner='owner via existing allocation',
        launch='Unchanged frozen-feature core and dynamic multi-GPU queue; five new conditions only',
        feasibility='Input entity flags on3splits withoutTESTlabels/inference; changed model checks and oneTRAINupdate',
        recovery='Unchanged local best/latest and full-state10percent aggregate COS barriers; original pinned inputs reused',
        controls='Five completed full native controls authenticated by original summary/manifest; no retraining',
        scope='No automatic L3/L4-T; exact failed profiles stay closed')


def review_gate(*,action):
    row=read(ROOT/REVIEW)
    if row.get('verdict')!='PASS' or row.get('source_sha256')!=identity(sources()) or row.get('contract_sha256')!=identity(contract()):
        raise ValueError('Changed-plugin review identity differs')
    if action not in ('run','worker','export'):raise ValueError('No repeated lifecycle matrix')
    return row
