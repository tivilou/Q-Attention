"""Frozen Q-RMEA study contract and standard-library artifact integrity."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .common import atomic_json,digest
from scripts.audit_rmea_inputs import EXPECTED

ROOT=Path(__file__).resolve().parents[2]
MODES=('static','ordered','no_content','symmetric','classical_pair')
FILES=('baselines/retacred/rmea_entity.py','baselines/retacred/rmea.py',
 'baselines/retacred/frozen_feature_head.py','baselines/retacred/common.py',
 'baselines/retacred/assets.lock.json','baselines/retacred/rmea_protocol.py',
 'baselines/retacred/rmea_run.py','baselines/retacred/rmea_entity_trace.py',
 'baselines/retacred/rmea_entity_study_protocol.py',
 'baselines/retacred/rmea_entity_study.py','baselines/retacred/rmea_entity_study_trace.py',
 'baselines/retacred/rmea_entity_handoff.py','scripts/audit_rmea_inputs.py',
 'scripts/audit_rmea_tokenization.py','scripts/validate_sample_trace_v2.py',
 'docs/contracts/sample-trace.v2.schema.json','docs/contracts/q_rmea_entity_labels.json','experiments/q_rmea_entity_handoff.py','experiments/q_rmea_entity_dev.py',
 'experiments/check_rmea_entity_handoff.py','experiments/run_rmea_entity_pilot.py',
 'scripts/run_q_rmea_entity_handoff.sh','scripts/check_q_rmea_entity_handoff.sh',
 'scripts/export_q_rmea_entity_handoff.sh','scripts/view_rmea_entity_trace.py',
 'tests/test_rmea_entity.py','tests/test_rmea_entity_cache.py','tests/test_rmea_entity_handoff.py')

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def sources():
    if any((ROOT/name).is_symlink() for name in FILES):raise ValueError('Symlinked source')
    return {name:digest(ROOT/name) for name in FILES}

def config(profile):
    if profile not in ('canary','discovery','pilot'):raise ValueError('Unknown frozen profile')
    counts=({'train':2048,'valid':1024} if profile=='pilot' else
        {s:(EXPECTED[s][0] if profile=='discovery' else n) for s,n in [('train',16),('valid',8),('test',8)]})
    return dict(schema='q-rmea.entity.protocol.v1',profile=profile,seed=307 if profile=='pilot' else 13,
        conditions=list(MODES),counts=counts,epochs=4 if profile=='pilot' else (20 if profile=='discovery' else 1),
        batch_size=32 if profile=='pilot' else 128,optimizer='AdamW',lr=.001,weight_decay=.01,clip_norm=1.,
        encoder_revision='722cf37b1afa9454edce342e7895e588b6ff1d59',max_tokens=512,
        hidden_size=1024,value_size=64,classes=40,qubits=3,depth=2,eps=1e-6,
        selection='strictly greatest valid macro_f1; earliest tie; test metrics only after selection',
        sample_rule='SHA256(rmea-sample-v1:split:original_id), ascending; pilot reuses pinned readiness-v1 train/valid selection',
        trace_samples_per_split=2,trace_checkpoints=['initial','selected'],
        primary_utility='ordered minus static full40 test macro_f1; positive is provisional single-seed gate only',
        primary_mechanism='ordered selected-checkpoint CE(branch train-mean)-CE(actual), CE(branch role-swap)-CE(actual), CE(branch whole-split deterministic mismatch)-CE(actual); all samples retained',
        controls='Retrained no_content and symmetric; classical_pair equal active parameter count, not proven capacity/optimization equivalence',
        replication='Never automatic. Owner audits single-seed report before any multi-seed release.',
        private_policy='Report-only Git return; no exchange authorization required; private evidence stays collaborator-side',
        claim_scope='Frozen encoder output head; no L3, novelty, hardware, speedup, quantum exclusivity or statistical claim from a single seed')

def contract():
    return dict(profiles={s:config(s) for s in ('canary','discovery','pilot')},
        trace='sample-trace.v2; original tokens->tokenization->saved encoder output->original-word mean->entities/values/weights->prediction->posthoc target; 2 fixed samples per split and initial/valid-selected checkpoint',
        lifecycle='Portable parent; isolated preflight; live GPU canary before collaborator-only discovery; sealed inventory; exact seven-file atomic report; no raw transfer; single ordinary report commit/push; export retry without training; no partial training resume')

def review_gate(formal=False):
    record=read(ROOT/'docs/validation/q_rmea_entity_implementation_review.json')
    if (record.get('review_scope')!='both' or record.get('must_fix') or
        record.get('source_sha256')!=identity(sources()) or record.get('contract_sha256')!=identity(contract())):
        raise ValueError('Exact-source scope-both review required')
    allowed=('PASS',) if formal else ('PASS','PASS_WITH_RISK')
    if record.get('verdict') not in allowed:raise ValueError('Review blocks this scope')
    return record

@contextlib.contextmanager
def lock(path):
    path=Path(path);fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);yield
    finally:path.unlink()

def inventory(root):
    root=Path(root)
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Symlinked private artifact')
        if path.is_file() and path.name not in ('RUN_COMPLETE','.run.lock'):
            result[path.relative_to(root).as_posix()]=dict(sha256=digest(path),bytes=path.stat().st_size)
    return result

def seal(root):
    atomic_json(Path(root)/'RUN_COMPLETE',dict(schema='q-rmea.closed.v1',artifacts=inventory(root)))

def audit(root):
    root=Path(root)
    if root.is_symlink() or not (root/'RUN_COMPLETE').is_file():raise ValueError('Run incomplete')
    expected=read(root/'RUN_COMPLETE')['artifacts']
    if inventory(root)!=expected:raise ValueError('Closed artifact inventory differs')
    manifest=read(root/'manifest.json')
    if manifest['source_sha256']!=identity(sources()) or manifest['config']!=config(manifest['config']['profile']):raise ValueError('Wrong completed source/config')
    summary=read(root/'summary.json')
    if set(summary['results'])!=set(MODES) or summary['common_initialization_equal'] is not True:raise ValueError('Incomplete conditions')
    return manifest,summary
