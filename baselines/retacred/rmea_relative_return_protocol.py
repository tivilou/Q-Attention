"""Prospective fair-budget three-readout study; completed native is reused."""
from . import rmea_token_transport_protocol as base
from .rmea_entity_scattering_protocol import (BASE_MANIFEST, BASE_SUMMARY, BASE_INITIAL,
    CANARY_MANIFEST, LABELS_SHA, INPUT_REQUEST_SHA, INPUT_ACK_ID, REPORT_FILES)
from .rmea_protocol import ROOT, EXPECTED, identity, read, git
from .common import atomic_json, digest
from .rmea_relative_return import MODES

EXPERIMENT = 'relative-return'
SCHEMA_PREFIX = 'q-rmea.relative-return'
REFERENCE_METHOD = 'native'
REFERENCE_DESCRIPTION = 'Pinned full-data seed13 native, identical fresh initial state/order/budget; not retrained'
ANALYSIS_DIRECTORY = 'analysis'
SCOPED_ADMISSION_SCHEMA = 'q-rmea.plugin-training-admission.v3'
OWNER_HASHES = {'allocation_publisher.py': '56c567973d095d8587a899f4109fd031a797f83b2539cd32b815c8a72a61ed80', 'full_host_run.py': 'a577c5f84dd749f19aab437d63f9f0be454ad2c0a593fe6f154498e5798e478b', 'launch_full.py': 'b6aa516374ed1dcf58defd265754aad2b7a2dcecf90cc0c6550132188d17b647', 'accepted_publisher_backend.py': 'e2b611543836d288daed2bebb95a0c8e68c6b795df7173ef1e3359740aa9e971'}
EXTRA_RECOVERY_INPUTS = ('recovery-inputs/admission/unit/design.json',
                         'recovery-inputs/admission/canary/focused-integration.json')
EXTRA = (
    'baselines/retacred/rmea_relative_return.py',
    'baselines/retacred/rmea_relative_return_protocol.py',
    'baselines/retacred/rmea_relative_return_runtime.py',
    'baselines/retacred/rmea_frozen_training.py',
    'baselines/retacred/rmea_plugin_admission.py',
    'baselines/retacred/rmea_scoped_admission.py',
    'baselines/retacred/rmea_task_runtime.py',
    'baselines/retacred/rmea_experiments.py',
    'baselines/retacred/rmea_entity_scattering_protocol.py',
    'experiments/q_rmea_train.py',
    'experiments/q_rmea_relative_return_train.py',
    'experiments/diagnostics/rmea_relative_return/relative_return.py',
    'baselines/retacred/orchestrate.py',
    'docs/evidence/rmea/20261122-scoped-training-admission/scoped-admission.json',
)

def sources():
    result = base.sources()
    for name in EXTRA:
        path = ROOT/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or symlinked source: '+name)
        result[name] = digest(path)
    return result

def config(profile):
    cfg = base.config(profile)
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1',conditions=list(MODES),
        allocated_gpus=2,micro_batch_size=4,diagnostic_samples=0,
        trace_samples_per_split=0,
        mechanism='W=V^dagger U; coherent |(W^2)00|^2 versus computational-basis intermediate dephasing',
        control='Native p^2 before the same EPS/normalizer; same81028 parameters/features/value/decoder',
        reference=dict(manifest=BASE_MANIFEST,summary=BASE_SUMMARY,initial=BASE_INITIAL,
                       method='native',selection='fixed epoch4',retrain=False),
        primary='Full test positive39Macro of return_coherent minus completed native',
        ordinary_utility_gate='Strictly positive candidate-minus-native Macro',
        attribution_gate='Separately coherent Macro strictly exceeds dephased and scalar-native-square',
        selection='Fixed final epoch only; valid retention descriptive; no early stopping or epoch selection',
        promotion='Utility and matched controls reported separately; stop exact recipe on failed conjunction, no rescue seed/epoch/readout',
        analysis='Final flat-state checkpoints retained; ordinary case study may be rebuilt; no materialized trace claim',
        claim_scope='One seed on repeatedly exposed official cohort; encoding-unitary access, not state-only; no inherited L4/L5, independent confirmation or quantum exclusivity')
    return cfg

def contract():
    return dict(profiles={x:config(x) for x in ('canary','discovery')},
        launch='One atomic allocation of2GPU; unchanged internal ready queue executes3independent conditions, one per card',
        recovery='Unchanged full-state local latest every update, validation-best, quiescent10percent COS and explicit resume; immutable original input versions reused',
        output='Three final checkpoints, checkpoint-replayable analysis inventory and exact seven compact report files',
        validation='Actual new-plugin GPU shape/gradient/update/parity; accepted unchanged core reuse, focused allocated-card count delta; historical complete canary failure retained')

def review_gate(*,action):
    row=read(ROOT/'docs/evidence/rmea/20261122-relative-return-training/formal-review.json')
    if (row.get('verdict')!='PASS' or row.get('review_scope')!='both' or row.get('must_fix')
            or row.get('source_sha256')!=identity(sources())
            or row.get('contract_sha256')!=identity(contract())
            or row.get('owner_hashes')!=OWNER_HASHES):
        raise ValueError('Current plugin/adapter/admission review required')
    return row
