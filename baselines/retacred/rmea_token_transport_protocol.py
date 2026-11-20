"""Prospective cubic transport protocol; legacy studies remain immutable."""
import json
from pathlib import Path
from .common import atomic_json, digest
from .rmea_protocol import ROOT, EXPECTED, identity, read, git, lock, inventory, seal

MODES = ('native', 'cubic_coherent', 'cubic_initial_dephased', 'linear_coherent')
REPORT_FILES = {'summary.json', 'manifest.json', 'trace_validation.json', 'receipt.json',
                'run_summary.json', 'data.sha256', 'data_counts.txt'}
FILES = (
    'baselines/retacred/rmea_token_transport.py',
    'baselines/retacred/rmea_token_transport_diagnostics.py',
    'baselines/retacred/rmea_token_transport_protocol.py',
    'baselines/retacred/rmea_token_transport_study.py',
    'baselines/retacred/rmea_token_transport_trace.py',
    'baselines/retacred/rmea_token_transport_handoff.py',
    'baselines/retacred/rmea.py', 'baselines/retacred/frozen_feature_head.py',
    'baselines/retacred/common.py', 'baselines/retacred/rmea_run.py',
    'baselines/retacred/rmea_protocol.py', 'baselines/retacred/assets.lock.json',
    'baselines/retacred/rmea_entity_study.py',
    'baselines/retacred/rmea_entity_study_protocol.py',
    'docs/contracts/q_rmea_entity_labels.json',
    'scripts/audit_rmea_inputs.py', 'scripts/audit_rmea_tokenization.py',
    'scripts/validate_sample_trace_v2.py', 'docs/contracts/sample-trace.v2.schema.json',
    'experiments/q_rmea_token_transport_handoff.py',
    'experiments/q_rmea_token_transport_canary.py',
    'experiments/check_rmea_token_transport_delivery.py',
    'scripts/run_q_rmea_token_transport_handoff.sh',
    'scripts/check_q_rmea_token_transport_handoff.sh',
    'scripts/export_q_rmea_token_transport_handoff.sh',
    'scripts/view_rmea_token_transport_trace.py',
    'tests/test_rmea_token_transport.py',
    'docs/evidence/rmea/20261121-token-transport-formal/design.json',
)


def sources():
    if any((ROOT / name).is_symlink() for name in FILES):
        raise ValueError('Symlinked source')
    return {name: digest(ROOT / name) for name in FILES}


def config(profile):
    if profile not in ('canary', 'discovery'):
        raise ValueError('Unknown fixed profile')
    return dict(schema='q-rmea.token-transport.protocol.v1', profile=profile, seed=13,
        conditions=list(MODES), counts={s: EXPECTED[s][0] if profile == 'discovery' else n
                                       for s, n in [('train', 16), ('valid', 8), ('test', 8)]},
        epochs=4 if profile == 'discovery' else 1, batch_size=32,
        optimizer='AdamW', lr=.001, weight_decay=.01, betas=[.9, .999], optimizer_eps=1e-8,
        clip_norm=1., dtype='float64/complex128', parameters=81028, tau=.25, born_eps=1e-6,
        encoder_revision='722cf37b1afa9454edce342e7895e588b6ff1d59',
        order='NumPy default_rng(seed); one permutation per epoch; identical all four conditions',
        selection='Fixed final epoch only; valid descriptive; no early stopping or epoch selection',
        sample_rule='SHA256(rmea-sample-v1:split:original_id), ascending, no label access',
        primary='positive39Macro fixed denominator39; cubic>each control and CE<=native',
        trace_checkpoints=['final'], trace_samples_per_split=2,
        diagnostic_samples=32, diagnostic_tolerance=1e-8,
        replication='No automatic replication or rescue seeds/epochs/tau',
        claim_scope='Single seed on previously exposed official splits; no inherited L4/L5/novelty or mediation')


def contract():
    return dict(profiles={s: config(s) for s in ('canary', 'discovery')},
        launch='Portable parent; isolated probe; tiny canary then collaborator-only full discovery; one GPU; serial heads',
        output='Private cache, four final checkpoints, complete final sample-trace.v2; exact seven-file report only',
        recovery='Export-only retry from sealed immutable run; no automatic retrain or partial training resume; no COS claim',
        diagnostic='First32 hash-ordered testunits; fixed cubic state and four swaps; reviewed coordinate and phase-gauge bounds; separate task gate')


def review_gate(formal=False):
    record = read(ROOT / 'docs/validation/q_rmea_token_transport_implementation_review.json')
    if (record.get('review_scope') != 'both' or record.get('must_fix')
            or record.get('source_sha256') != identity(sources())
            or record.get('contract_sha256') != identity(contract())):
        raise ValueError('Exact source/launch/output/recovery scope-both review required')
    allowed = ('PASS',) if formal else ('PASS', 'PASS_WITH_RISK')
    if record.get('verdict') not in allowed:
        raise ValueError('Review blocks requested execution scope')
    return record


def verify_loader_directory(assets):
    """For this new study, Transformers may see only the five locked inputs."""
    assets=Path(assets)
    rows=read(ROOT/'baselines/retacred/assets.lock.json')['files']
    if assets.is_symlink() or {x.name for x in assets.iterdir()}!={r['name'] for r in rows}:
        raise ValueError('Loader directory must contain exactly the pinned asset files')
    for row in rows:
        path=assets/row['name']
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['size_bytes'] or digest(path)!=row['sha256']:
            raise ValueError('Pinned loader input differs')


def audit(root):
    root = Path(root)
    if root.is_symlink() or not (root / 'RUN_COMPLETE').is_file():
        raise ValueError('Run incomplete')
    if read(root / 'RUN_COMPLETE')['artifacts'] != inventory(root):
        raise ValueError('Sealed run changed')
    m, s, t = (read(root / name) for name in ('manifest.json', 'summary.json', 'trace_validation.json'))
    cfg = config(m['config']['profile'])
    if m['config'] != cfg or m['source_sha256'] != identity(sources()):
        raise ValueError('Completed source/config differs')
    if set(s['results']) != set(MODES) or not s['common_initialization_equal']:
        raise ValueError('Missing condition or unmatched initialization')
    expected = cfg['epochs'] * ((cfg['counts']['train'] + 31) // 32)
    for result in s['results'].values():
        if result['updates'] != expected or result['parameters'] != 81028 or result['selected_epoch'] != cfg['epochs']:
            raise ValueError('Budget or final-state policy differs')
    if any(t.get(k) != 'pass' for k in ('structural', 'semantic', 'replay', 'coordinate_reader')):
        raise ValueError('Trace incomplete')
    if t['samples'] != 24 or s['trace'] != t:
        raise ValueError('Trace coverage differs')
    for split, n in cfg['counts'].items():
        if m['data'][split] != dict(source_sha256=EXPECTED[split][1], source_count=EXPECTED[split][0], selected_count=n):
            raise ValueError('Data contract differs')
    if cfg['profile'] == 'discovery' and m['environment'].get('fixture_sha256'):
        raise ValueError('Fixture cannot certify formal data')
    if not m['environment'].get('fixture_sha256') and t.get('tokenization_validation')!='live_pinned_source_replay':
        raise ValueError('Live tokenization readiness missing')
    return m, s
