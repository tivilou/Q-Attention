"""Prospective common-cache qualification; historical tables remain unchanged."""
from . import rmea_token_transport_protocol as base
from .rmea_entity_scattering_protocol import (BASE_MANIFEST, BASE_SUMMARY, BASE_INITIAL,
    LABELS_SHA, INPUT_REQUEST_SHA, INPUT_ACK_ID, REPORT_FILES)
from .rmea_protocol import ROOT, EXPECTED, identity, read, git
from .common import atomic_json, digest

CANARY_MANIFEST = 'c5b119938e0d0e34a7f87ba3deeeeefe2e1ac916bbb27dc3e1b949026684315c'
EXPERIMENT = 'fair-reference'
SCHEMA_PREFIX = 'q-rmea.fair-reference'
MODES = ('uniform', 'quantum', 'dephased', 'softmax12', 'real12')
REFERENCE_METHOD = 'native'
REFERENCE_DESCRIPTION = 'Cache provenance only; older four-epoch metrics are NOT a comparator'
ANALYSIS_DIRECTORY = 'analysis'
SCOPED_ADMISSION_SCHEMA = SCHEMA_PREFIX+'.training-admission.v1'
REVIEW = 'docs/evidence/rmea/20261124-fair-reference/review.json'
EXTRA = (
    'baselines/retacred/rmea_fair_reference_protocol.py',
    'baselines/retacred/rmea_fair_reference_runtime.py',
    'baselines/retacred/rmea_frozen_training.py',
    'baselines/retacred/rmea_matched_routers.py',
    'baselines/retacred/rmea_training_options.py',
    'baselines/retacred/rmea_token_transport_parallel.py',
    'baselines/retacred/rmea_token_transport_recovery.py',
    'baselines/retacred/rmea_task_runtime.py',
    'baselines/retacred/rmea_experiments.py',
    'baselines/retacred/rmea_entity_scattering_protocol.py',
    'experiments/q_rmea_fair_reference_train.py',
    'experiments/q_rmea_train.py',
)


def sources():
    rows = base.sources()
    for name in EXTRA:
        path = ROOT/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or symlinked source: '+name)
        rows[name] = digest(path)
    return rows


def parameter_schemas():
    # Standard-library protocol import keeps the parent independent of Torch.
    shared = {'head.value.weight':[64,1024], 'head.value.bias':[64],
              'head.decoder':[40,64], 'head.bias':[40]}
    return {m:dict(shared, **({} if m=='uniform' else
        {'head.prepare.weight':[12,1028], 'head.prepare.bias':[12], 'head.effect':[40,2,3,2]}
        if m in ('quantum','dephased') else
        {'head.key.weight':[12,1028], 'head.key.bias':[12], 'head.query':[40,12]})) for m in MODES}


def config(profile):
    cfg = base.config(profile)
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1',conditions=list(MODES),
        epochs=20 if profile=='discovery' else 1,batch_size=128,eval_batch_size=128,
        training_dtype='float32',dtype='float32/complex64',allocated_gpus='auto',
        order_generator='torch_epoch_seed',order='Torch randperm(seed+zero_based_epoch)',
        validation_metric='macro_f1',reported_checkpoint='validation_best',
        selection='strictly greatest valid all40 macro_f1; earliest tie; test only after selection',
        parameter_schemas=parameter_schemas(),parameters=None,diagnostic_samples=0,
        trace_samples_per_split=0,trace_checkpoints=['selected'],
        primary='TEST all40 Macro-F1: quantum minus EACH of uniform/dephased/softmax12/real12',
        qualification='All four differences strictly positive; otherwise close this fixed profile. No automatic rating.',
        reference=dict(manifest=BASE_MANIFEST,summary=BASE_SUMMARY,retrain_all_conditions=True),
        replication='Single fixed seed13 qualification. No extra seeds, epochs, learning rates or metric rescue.',
        claim_scope='Same rebuilt cache only; repeated development exposure; no independent confirmation, significance, quantum exclusivity or inherited maturity',
        analysis='Selected checkpoints replayable with pinned features; traces not materialized')
    return cfg


def contract():
    return dict(profiles={p:config(p) for p in ('canary','discovery')},
        launch='Existing shared runner; one task per scheduler-assigned GPU, dynamic five-condition queue; no fleet access',
        recovery='Existing full-state latest and best locally; minimal COS preparation plus ten-percent aggregate barriers; explicit resume',
        output='Exact seven-file compact report; selected AND final checkpoint identities; selected-state metrics',
        scientific_scope='New symmetric common-cache table. Old four-epoch/native and fresh5 metrics never used as controls.',
        formal_owner='Collaborator only; owner may execute a tiny TRAIN-derived lifecycle canary')


def review_gate(*,action):
    row = read(ROOT/REVIEW)
    if (row.get('verdict')!='PASS' or row.get('review_scope')!='both' or row.get('must_fix')
            or row.get('source_sha256')!=identity(sources())
            or row.get('contract_sha256')!=identity(contract())):
        raise ValueError('Exact fair-reference review required')
    if action not in row.get('allowed_actions',[]):
        raise ValueError('Review does not admit this action')
    return row
