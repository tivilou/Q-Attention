"""Fixed three-fit discovery contract; historical native is not retrained."""
import json
from . import rmea_token_transport_protocol as base
from .rmea_entity_scattering_protocol import (
    BASE_MANIFEST, BASE_SUMMARY, BASE_INITIAL, CANARY_MANIFEST, LABELS_SHA,
    INPUT_REQUEST_SHA, INPUT_ACK_ID, REPORT_FILES)
from .rmea_protocol import ROOT, EXPECTED, identity, read, git, inventory, seal
from .common import atomic_json, digest

EXPERIMENT = 'protected-context'
SCOPED_ADMISSION_SCHEMA = 'q-rmea.scoped-training-admission.v2'
SCHEMA_PREFIX = 'q-rmea.protected-context'
REFERENCE_METHOD = 'native'
REFERENCE_DESCRIPTION = 'Pinned completed full-data native baseline; not retrained'
ANALYSIS_DIRECTORY = 'trace'
EXTRA_RECOVERY_INPUTS = ('recovery-inputs/native-final.pt',)
MODES = ('protected_context', 'context_scrambled', 'unprotected_context')
TRACE_MODES = ('native', *MODES)
NATIVE_CHECKPOINT = '2da52b1ce826046a6f443269f47f49e4e9fa63e10f22f9ebdcde6f890e12ce51'
EXTRA = ('baselines/retacred/rmea_frozen_training.py',
         'baselines/retacred/rmea_protected_context.py',
         'baselines/retacred/rmea_protected_context_protocol.py',
         'baselines/retacred/rmea_protected_context_runtime.py',
         'baselines/retacred/rmea_protected_context_trace.py',
         'baselines/retacred/rmea_entity_scattering_runtime.py',
         'baselines/retacred/rmea_entity_scattering_protocol.py',
         'experiments/q_rmea_protected_context_train.py',
         'experiments/q_rmea_protected_context_unit.py',
         'scripts/view_q_rmea_protected_context_trace.py',
         'experiments/q_rmea_query_alignment.py',
         'baselines/retacred/orchestrate.py')
EXTRA = (*EXTRA, 'baselines/retacred/rmea_task_runtime.py',
         'baselines/retacred/rmea_experiments.py', 'experiments/q_rmea_train.py',
         'baselines/retacred/rmea_scoped_admission.py')


def sources():
    result = base.sources()
    for name in EXTRA:
        if (ROOT/name).is_symlink():
            raise ValueError('Symlinked source')
        result[name] = digest(ROOT/name)
    return result


def config(profile):
    cfg = base.config(profile)
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1', conditions=list(MODES),
        trace_conditions=['native', *MODES], epochs=4 if profile == 'discovery' else 3,
        micro_batch_size=4, diagnostic_samples=0,
        mechanism=dict(ridge=1/8, projector_relative_cutoff=1e-10, source_probability_floor=1e-6,
            derivative='Actual retained-span first derivative on fixed-rank positive-gap strata; no detach/surrogate'),
        control='Fixed same-example DFT8 covariance conjugation; unprotected Pi=I; unchanged parameter count and value readout',
        reference=dict(manifest=BASE_MANIFEST, summary=BASE_SUMMARY, initial=BASE_INITIAL,
            method=REFERENCE_METHOD, selection='fixed epoch4', retrain=False),
        primary='Protected-context test positive39Macro minus completed native; signed absolute and relative delta',
        ordinary_utility_gate='Strictly positive primary delta after shared integrity/leakage gates; no hidden practical threshold',
        attribution_gate='Separately protected-context Macro exceeds both DFT-scrambled and unprotected; CE is descriptive, no automatic significance/replication',
        promotion='Ordinary utility and matched-control comparisons are reported separately; failed exact recipe stops, no rescue seed/epoch/ridge/checkpoint',
        trace_checkpoints=['final'], trace_samples_per_split=2,
        analysis='Post-training complete final-head sample-trace.v2 on first2 fixed cache units/split, including unchanged old native; private coordinates only',
        claim_scope='One training seed on repeatedly exposed official cohort; no inherited L4/L5, independent confirmation, encoder trajectory or quantum-necessity claim')
    return cfg


def contract():
    return dict(profiles={x: config(x) for x in ('canary', 'discovery')},
        launch='One scheduler3GPUgroup, guarded existing internal auto runner, one independent fit/card, seed13 only; explicit shared lifecycle/runtime callbacks',
        output='Per-condition atomic full-state latest every update and validation-best; fixed final models, private complete v2 trace and exact seven-file compact report',
        recovery='Existing quiescent global10percent COS seals, fixed source/config/input/model/AdamW/RNG/cursor; original13 immutable inputs reused. Explicit resume, no automatic science retry.',
        input='Pinned discovery/canary manifest and old native model; no baseline retrain or data/weights download to Windows',
        review='Exact current both-scope review; prospective scoped v2 admission may reuse exercised primary capabilities with bound source equivalence and focused RNG repair; historical failed full seal remains unchanged')


def review_gate(*, action):
    row = read(ROOT/'docs/evidence/rmea/20261122-protected-context-training/training-review.json')
    scoped = row.get('verdict') == 'PASS_WITH_SCOPE' and row.get('admission') == 'scheduler_cpu_training_adapter_unit_only'
    if (not (row.get('verdict') == 'PASS' or scoped and action == 'unit')
            or row.get('review_scope') != 'both' or row.get('must_fix') != []
            or row.get('source_sha256') != identity(sources())
            or row.get('contract_sha256') != identity(contract())
            or not row.get('owner_hashes')
            or row.get('owner_lifecycle_sha256') != identity(row['owner_hashes'])):
        raise ValueError('Exact source/contract/owner lifecycle both-scope review required')
    return row
