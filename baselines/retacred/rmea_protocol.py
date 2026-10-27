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
MODES=('uniform','relation_softmax','real_overlap','quantum','dephased')
FILES=('baselines/retacred/rmea.py','baselines/retacred/rmea_protocol.py',
       'baselines/retacred/rmea_trace.py','baselines/retacred/rmea_run.py',
       'baselines/retacred/rmea_handoff.py','baselines/retacred/rmea_exchange.py','baselines/retacred/frozen_feature_head.py',
       'baselines/retacred/common.py','baselines/retacred/assets.lock.json',
       'scripts/audit_rmea_inputs.py','scripts/audit_rmea_tokenization.py',
       'scripts/validate_sample_trace.py','scripts/check_qattn_exchange.py','scripts/collab/lib/exchange_upload.py','scripts/collab/certs/qattn-exchange-ca.crt','experiments/q_rmea_handoff.py',
       'scripts/run_q_rmea_handoff.sh','scripts/check_q_rmea_handoff.sh',
       'scripts/export_q_rmea_handoff.sh','tests/test_rmea.py','tests/test_rmea_lifecycle.py','tests/test_rmea_exchange.py')

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def sources():
    if any((ROOT/name).is_symlink() for name in FILES):raise ValueError('Symlinked source')
    return {name:digest(ROOT/name) for name in FILES}

def config(profile):
    if profile not in ('canary','discovery'):raise ValueError('Unknown frozen profile')
    return dict(schema='q-rmea.protocol.v1',profile=profile,seed=13,
        conditions=list(MODES),counts={s:(EXPECTED[s][0] if profile=='discovery' else n) for s,n in [('train',16),('valid',8),('test',8)]},
        epochs=20 if profile=='discovery' else 1,batch_size=128,optimizer='AdamW',lr=.001,weight_decay=.01,clip_norm=1.,
        encoder_revision='722cf37b1afa9454edce342e7895e588b6ff1d59',max_tokens=512,
        hidden_size=1024,value_size=64,classes=40,qubits=3,depth=2,eps=1e-6,
        selection='strictly greatest valid macro_f1; earliest tie; test only after selection',
        sample_rule='SHA256(rmea-sample-v1:split:original_id), ascending; no label/outcome access',
        trace_samples_per_split=2,trace_checkpoints=['initial','selected'],
        primary_utility='quantum minus uniform full40 macro_f1',
        primary_mechanism='mean test CE(shuffled_context_weights)-CE(actual); all samples, including short context',
        replication='Not automatic. Audit discovery first; fresh29/53/71/97/113 only after explicit release.',
        claim_scope='frozen encoder output head; not Transformer internal attention, SOTA, quantum speedup or exclusivity')

def review_gate():
    record=read(ROOT/'docs/validation/q_rmea_implementation_review.json')
    if record.get('review_scope')!='both' or record.get('must_fix') or record.get('source_sha256')!=identity(sources()):raise ValueError('Exact-source scope-both review required')
    if record.get('verdict') not in ('PASS','PASS_WITH_RISK'):raise ValueError('Review blocked')
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
