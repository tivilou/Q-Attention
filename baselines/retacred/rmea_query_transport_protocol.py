"""Three new shared-unitary conditions; ten sealed full-data controls reused."""
from . import rmea_fair_reference_protocol as base
from . import rmea_entity_query_protocol as previous
from .rmea_fair_reference_protocol import (BASE_MANIFEST,BASE_SUMMARY,BASE_INITIAL,LABELS_SHA,
    INPUT_REQUEST_SHA,INPUT_ACK_ID,REPORT_FILES,CANARY_MANIFEST,ROOT,EXPECTED,identity,read,git,atomic_json,digest)

EXPERIMENT='query-transport'
SCHEMA_PREFIX='q-rmea.query-transport'
SCOPED_ADMISSION_SCHEMA=SCHEMA_PREFIX+'.changed-adapter-admission.v1'
REFERENCE_METHOD=base.REFERENCE_METHOD
REFERENCE_DESCRIPTION=base.REFERENCE_DESCRIPTION
ANALYSIS_DIRECTORY='analysis'
MODES=('quantum_transport','dephased_transport','real_circuit_transport')
EVIDENCE='docs/evidence/rmea/20261124-query-transport/'
REVIEW=EVIDENCE+'adapter-review.json'
CORE_EVIDENCE=previous.CORE_EVIDENCE
NATIVE_SUMMARY=previous.EVIDENCE+'native-summary.json'
NATIVE_MANIFEST=previous.EVIDENCE+'native-manifest.json'
CONTEXT_SUMMARY=EVIDENCE+'context-summary.json'
CONTEXT_MANIFEST=EVIDENCE+'context-manifest.json'
UNIT=EVIDENCE+'unit-validation.json'
UNIT_DESIGN=EVIDENCE+'unit-design.json'
EXTRA=('baselines/retacred/rmea_query_transport.py','baselines/retacred/rmea_query_transport_protocol.py',
    'baselines/retacred/rmea_query_transport_runtime.py','baselines/retacred/rmea_entity_query.py',
    'baselines/retacred/rmea_entity_query_protocol.py','experiments/q_rmea_query_transport_train.py',
    EVIDENCE+'protocol.md',UNIT,UNIT_DESIGN,EVIDENCE+'unit-replay.py',EVIDENCE+'review.json',
    NATIVE_SUMMARY,NATIVE_MANIFEST,CONTEXT_SUMMARY,CONTEXT_MANIFEST)

def sources():
    rows=base.sources()
    for n in EXTRA:
        p=ROOT/n
        if not p.is_file() or p.is_symlink():raise ValueError('Missing query-transport source: '+n)
        rows[n]=digest(p)
    return rows

def config(profile):
    if profile!='discovery':raise ValueError('Reuse accepted core capabilities')
    cfg=base.config(profile);schemas=base.parameter_schemas()
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1',conditions=list(MODES),
        parameter_schemas={k:schemas['dephased' if k=='dephased_transport' else 'quantum'] for k in MODES},
        primary='Selected TEST all40Macro: quantum_transport strictly greater than all12 new/reused controls',
        qualification='All12 deltas positive; otherwise close exact profile without seed/budget/metric/control rescue',
        reference=dict(native_summary=digest(ROOT/NATIVE_SUMMARY),native_manifest=digest(ROOT/NATIVE_MANIFEST),
            context_summary=digest(ROOT/CONTEXT_SUMMARY),context_manifest=digest(ROOT/CONTEXT_MANIFEST),retrain_all_conditions=False),
        replication='Seed13 exposed qualification only; independent confirmation must be frozen prospectively after all12 positive',
        input_condition='Previous subject/object mean of concatenated h/roles; biased tied prepare(c) controls a common unitary AFTER native query preparation',
        classical_control='Real4RYlayers per circuit; query has base+context24CNOT versus complex12; native and previous context width12 controls reused')
    return cfg

def contract():
    return dict(profiles={'discovery':config('discovery')},formal_owner='owner via existing allocation',
        launch='Unchanged frozen-feature core and dynamic dualGPU queue; only three new conditions,27420 updates',
        feasibility='Reused exact-plugin source review and actual first8TRAIN technical updates; no repeated fullsplit scan/lifecycle matrix',
        recovery='Unchanged local best/latest and full-state10percent aggregate COS barriers, immutable original input dependency',
        controls='Native5 and previous entity-context5 sealed full results, protocol/cache/initial/order authentication, no retraining',
        scope='Gram invariance is known; no automatic L3/T and no reopening exact failed profiles')

def review_gate(*,action):
    row=read(ROOT/REVIEW)
    if row.get('verdict')!='PASS' or row.get('source_sha256')!=identity(sources()) or row.get('contract_sha256')!=identity(contract()):
        raise ValueError('Changed-adapter review identity differs')
    if action not in ('run','worker','export'):raise ValueError('No repeated core lifecycle matrix')
    return row
