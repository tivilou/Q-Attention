"""Prospective single-seed utility test; historical baselines/runs remain pinned."""
import json
from . import rmea_token_transport_protocol as base
from .rmea_protocol import ROOT, EXPECTED, identity, read, git, inventory, seal
from .common import atomic_json, digest

EXPERIMENT='entity-scattering'
SCHEMA_PREFIX='q-rmea.scattering'
REFERENCE_METHOD='linear_coherent'
REFERENCE_DESCRIPTION='Pinned completed full-data linear baseline; not retrained'
ANALYSIS_DIRECTORY='analysis'

MODES = ('complex','real_magnitude','constant_onsite')
BASE_MANIFEST = 'dfb347aef35657c5463486e1043aca0617ccc11d3aca1eea851a59af15ec5f49'
BASE_SUMMARY = '053b8aef68dfec5ecb6c3d5706365f898814243137fa826af2bd3c3b4e77633a'
BASE_INITIAL = '037232c406d5cb1b23cea0ee289b7ab2d82366a2febfcedfdeb6fc69ec8da369'
CANARY_MANIFEST = 'd1e17dfd0f8f75d2d90ab1b9c5365b7a752b7e44a70c950c3c2b94ad28d207ea'
LABELS_SHA = '6e822509a07c5c865130ebe315122a9d83e97b1b11496b9e86c6b2a600f8a429'
INPUT_REQUEST_SHA = 'a2903267be99693d2a5e248eff4fed899d31d0c3615fd09a0cd7bc1911b996cf'
INPUT_ACK_ID = 'e6c468164fb9358b0f262c9a60aeed370f60a182f6e3a8cdbaa1a8465fd0e4e0'
EXTRA = ('baselines/retacred/rmea_frozen_training.py','baselines/retacred/rmea_entity_scattering.py',
         'baselines/retacred/rmea_entity_scattering_protocol.py',
         'baselines/retacred/rmea_entity_scattering_runtime.py',
         'experiments/q_rmea_entity_scattering_train.py',
         'experiments/q_rmea_entity_scattering.py',
         'experiments/q_rmea_query_alignment.py',
         'baselines/retacred/orchestrate.py')
EXTRA = (*EXTRA, 'baselines/retacred/rmea_task_runtime.py',
         'baselines/retacred/rmea_experiments.py', 'experiments/q_rmea_train.py')
REPORT_FILES = {'summary.json','manifest.json','analysis-readiness.json','receipt.json',
                'run_summary.json','data.sha256','data_counts.txt'}


def sources():
    result = base.sources()
    for n in EXTRA:
        if (ROOT/n).is_symlink():
            raise ValueError('Symlinked source')
        result[n] = digest(ROOT/n)
    return result


def config(profile):
    cfg = base.config(profile)
    cfg.update(schema='q-rmea.entity-scattering.training.v1', conditions=list(MODES),
        epochs=4 if profile=='discovery' else 3, micro_batch_size=4,
        mechanism=dict(hopping=1.,energy=0.,damping=.1,terminal_lead=1.,epsilon=1e-12),
        control='constant_onsite uses the unweighted mean of tanh utility over valid words at every word; all parameters remain trainable',
        reference=dict(manifest=BASE_MANIFEST,summary=BASE_SUMMARY,initial=BASE_INITIAL,
            method='linear_coherent',selection='fixed epoch4',retrain=False,
            comparison='Same seed13/initialization/cache/order/AdamW/batch32/4epochs/parametercount; new fits use microbatch4 accumulation, not bitwise-identical kernels'),
        primary='Complex test positive39Macro minus pinned linear reference, signed absolute/relative deltas; CE reported',
        promotion='Complex Macro exceeds linear, real_magnitude and constant_onsite, and complex CE<=linear; no automatic replication',
        trace_checkpoints=[],trace_samples_per_split=0,
        analysis='Freeze final-epoch replay interface and first2 immutable units/split; compact private replay anchors, NOT sample-trace.v2 or materialized Case Study',
        claim_scope='Single seed on repeatedly exposed official cohort; no inherited L4/L5, fresh-cohort or quantum-exclusive claim')
    return cfg


def contract():
    return dict(profiles={x:config(x) for x in ('canary','discovery')},
        launch='One shared scheduler3GPUgroup, independent conditions via existing guarded runner; explicit runtime callbacks, no monkeypatch; seed13 only',
        output='Three local full-state latest and validation-best states; final-epoch evaluation weights and compact replay anchors; exact seven compact public files',
        recovery='Existing atomic full-state and quiescent10-percent global barriers; exact protocol source/config/cache/AdamW/RNG/cursor; minimal preparation refers to already verified original COS inputs instead of duplicating cache. No automatic retry/migration.',
        input='Pinned original discovery manifest/summary/initialization and cache; existing version-pinned preparation dependency; newcanary uses protected16/8/8 fixture3epochs',
        review='Exact both-scope review required before tests; real parent/child pause-resume, pinned live COS restore and immutable export failure/retry before formal admission')


def review_gate(*, action=None):
    r = read(ROOT/'docs/evidence/rmea/20261122-entity-scattering/training-review.json')
    if (r.get('verdict')!='PASS' or r.get('review_scope')!='both' or r.get('must_fix')!=[]
            or r.get('source_sha256')!=identity(sources()) or r.get('contract_sha256')!=identity(contract())):
        raise ValueError('Exact training mechanism/lifecycle review required')
    if not r.get('owner_hashes') or r.get('owner_lifecycle_sha256')!=identity(r['owner_hashes']):raise ValueError('Approved owner lifecycle identity required')
    return r
