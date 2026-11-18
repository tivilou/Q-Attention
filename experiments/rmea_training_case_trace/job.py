"""Unadmitted draft: paired training-prefix traces from four frozen final heads.

Requires a new exact BOTH review, private execution contract and shared queue.
No optimizer, new weights, cloud operation, formal experiment or promotion.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rmea_original_scores.common import guard, read, sha
from experiments.rmea_training_case_trace.trace import AXES, build, ref, save, value_cli

DESIGN = ROOT / 'docs/evidence/rmea/20261119-l5-round88/training-case-design.json'
PATH_ARGUMENTS = ['assets', 'receipt', 'cohort', 'donors', 'checkpoints', 'original_report',
                  'execution_contract', 'output']


def arguments():
    p = argparse.ArgumentParser()
    for name in PATH_ARGUMENTS:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    return p.parse_args()


def checked(a):
    guard()
    d, c = read(DESIGN), read(a.execution_contract)
    assert d['schema'] == 'qattn.training-paired-case.design.v1'
    assert c['schema'] == 'qattn.training-paired-case.execution.v1'
    assert c['job_sha256'] == sha(Path(__file__)) and c['design_sha256'] == sha(DESIGN)
    assert c['budget'] == d['proposed_budget'] and c['budget']['optimizer_steps'] == 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert {n: importlib.metadata.version(n) for n in d['environment']} == d['environment']
    assert c['implementation_review']['scope'] == 'both' and c['implementation_review']['must_fix'] == []
    assert c['implementation_review']['verdict'] in ('PASS', 'PASS_WITH_RISK')
    for n, digest in {**d['source_files'], **d['implementation_files']}.items():
        assert sha(ROOT / n) == digest
    for name in PATH_ARGUMENTS:
        p = getattr(a, name).absolute()
        assert all(not q.is_symlink() for q in [p, *p.parents])
    assert a.assets.is_dir() and a.checkpoints.is_dir()
    assert not a.output.exists() or not list(a.output.iterdir())
    assert sha(a.original_report) == d['original_report_sha256']
    for row in d['checkpoints'].values():
        p = a.checkpoints / row['filename']
        assert not p.is_symlink() and p.is_file() and sha(p) == row['sha256']
    return d, c


def child(a, design):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
    from experiments.rmea_ecqp_screen import metrics, runner
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261118)
    identity = design['data']
    data = runner.read_training_prefix(a.assets, a.receipt, a.cohort, identity['cohort_sha256'],
        a.donors, identity['donors_sha256'], expected_receipt_sha256=identity['prefix_receipt_sha256'],
        expected_plan_sha256=identity['plan_sha256'])
    ids = data['splits']['eval']
    assert len(ids) == 972 and len({data['cohort'][int(i)]['group'] for i in ids}) == 866
    # Freeze input/label-stratified selection before loading/forwarding the heads.
    selected = {}
    for i in ids:
        label = int(data['targets'][i])
        key = data['metadata'][i]['id_hash']
        if label not in selected or key < data['metadata'][selected[label]]['id_hash']:
            selected[label] = int(i)
    assert len(selected) == design['selection']['expected_base_cases'] == 34
    assert sorted(set(range(40)) - set(selected)) == [17, 23, 24, 32, 36, 37]
    private = a.output.absolute().parent / 'scientific-work/private-trace'
    private.mkdir(mode=0o700)
    selection = dict(rule=design['selection']['rule'], model_outputs_read_for_selection=False,
        rows=[dict(source_index=i, id_hash=data['metadata'][i]['id_hash'], label=label)
              for label, i in sorted(selected.items())])
    save(private / 'selection.json', selection)
    save(private / 'case-inputs.json', dict(rows=[dict(source_index=i,
        **{k: data['metadata'][i][k] for k in ('id_hash', 'tokens', 'subject', 'object')},
        label=int(data['targets'][i])) for i in selected.values()]))
    original = read(a.original_report)
    assert original['status'] == 'TRAIN_ONLY_EXPLORATORY_SCREEN' and original['seed'] == 20261118
    conditions = {r['id']: r for r in read(ROOT / 'docs/evidence/rmea/20261118-l5-round72/screen-design.json')['conditions']}
    native = RMEAHead('quantum')
    models = {}
    for name, row in design['checkpoints'].items():
        assert not conditions[name]['donor']
        model = runner.TrainingModel(conditions[name], native)
        model.load_state_dict(torch.load(a.checkpoints / row['filename'], map_location='cpu', weights_only=True))
        model.eval().requires_grad_(False)
        assert sum(p.numel() for p in model.parameters()) == row['parameters']
        assert original['conditions'][name]['checkpoint_sha256'] == row['sha256']
        models[name] = model
    predictions = {n: [] for n in models}
    losses = {n: 0.0 for n in models}
    records = []
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            batch_ids = ids[start:start + 32]
            h, roles, mask, y, donor = runner.batch(data, batch_ids, 'eval', False)
            assert donor is None
            details = {n: m(h, roles, mask, details=True) for n, m in models.items()}
            for name, d in details.items():
                assert torch.isfinite(d['logits']).all()
                predictions[name].extend(d['logits'].argmax(-1).tolist())
                losses[name] += float(F.cross_entropy(d['logits'], y, reduction='sum'))
            for j, i in enumerate(map(int, batch_ids)):
                if i not in selected.values():
                    continue
                sid = data['metadata'][i]['id_hash']
                filename = sid + '-inputs.npz'
                np.savez_compressed(private / filename, h=h[j:j+1].numpy().copy(),
                    roles=roles[j:j+1].numpy().copy(), mask=mask[j:j+1].numpy().copy(),
                    features=torch.cat((h[j:j+1], roles[j:j+1]), -1).numpy().copy())
                for name, d in details.items():
                    values = {}
                    for key, value in dict(d, decoder=models[name].head.decoder, bias=models[name].head.bias).items():
                        if not torch.is_tensor(value):
                            continue
                        assert key in AXES, key
                        if AXES[key][0] == 'sample':
                            value = value[j:j+1]
                        values[key] = value.detach().cpu().numpy().copy()
                    # Verify detached full padded-token values reproduce the actual logits.
                    reconstructed = (torch.from_numpy(values['alpha']) * torch.from_numpy(values['utilities'])).sum(-1) + torch.from_numpy(values['bias'])
                    assert torch.equal(reconstructed, torch.from_numpy(values['logits']))
                    headfile = sid + '-' + name + '.npz'
                    np.savez_compressed(private / headfile, **values)
                    records.append(dict(sample_id=sid + ':' + name, id_hash=sid, condition=name,
                        prediction=int(values['logits'].argmax()), input_file=filename, head_file=headfile))
    assert len(records) == 136 and len({r['sample_id'] for r in records}) == 136
    gold = np.asarray(data['targets'][ids], dtype=np.int64)
    measured = {}
    for name, pred in predictions.items():
        row = metrics.from_confusion(metrics.confusion(gold, np.asarray(pred, dtype=np.int64)))
        row['ce'] = losses[name] / len(ids)
        assert row == original['metrics'][name], 'Exact original aggregate parity failed'
        measured[name] = row
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip()
    validation = build(private, records, design, sha(DESIGN), revision)
    manifest = {p.name: ref(private, p) for p in sorted(private.iterdir())}
    assert all(p.is_file() and not p.is_symlink() for p in private.iterdir())
    assert sum(r['bytes'] for r in manifest.values()) <= design['proposed_budget']['private_trace_bytes_max']
    save(private / 'RUN_COMPLETE', dict(schema='qattn.training-case-trace.closed.v1',
        files={n: r['sha256'] for n, r in manifest.items()}, source_revision=revision, training=False))
    # Public result whitelist contains counts/identities, never per-example data or predictions.
    print(json.dumps(dict(status='TRAIN_ONLY_PAIRED_CASE_TRACE_COMPLETE', design_sha256=sha(DESIGN),
        job_sha256=sha(Path(__file__)), base_cases=34, condition_cases=136, evaluation_examples=972,
        selection_sha256=sha(private / 'selection.json'), trace_sha256=sha(private / 'sample-trace.json'),
        private_trace_bytes=sum(r['bytes'] for r in manifest.values()),
        private_closed_sha256=sha(private / 'RUN_COMPLETE'),
        validation=validation, exact_original_aggregate_parity=list(measured),
        historical_per_example_logits_available=False, semantic_causality_tested=False,
        environment=design['environment'], elapsed_seconds=time.monotonic() - started,
        optimizer_steps=0, task_training=False, COS_requests=0, cloud_writes=0,
        new_checkpoints=0, promotion=False, formal_handoff=False, L5='NOT_ACCEPTED'), allow_nan=False), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'value':
        value_cli(sys.argv[2:])
        return
    a = arguments()
    design, c = checked(a)
    if a.child:
        child(a, design)
        return
    assert 'torch' not in sys.modules
    work = a.output.absolute().parent / 'scientific-work'
    assert not work.exists(), 'Existing work retained; no implicit replay'
    work.mkdir()
    command = [sys.executable, '-B', str(Path(__file__).resolve())]
    for name in PATH_ARGUMENTS:
        command += ['--' + name.replace('_', '-'), str(getattr(a, name).absolute())]
    command += ['--child']
    stdout, stderr = work / 'child-stdout.json', work / 'child-stderr.log'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
            timeout=c['budget']['timeout_seconds'] - 60)
    assert result.returncode == 0, 'Child failed; private logs retained, no public completion marker'
    report = read(stdout)
    assert report['status'] == 'TRAIN_ONLY_PAIRED_CASE_TRACE_COMPLETE' and not report['task_training']
    assert report['design_sha256'] == sha(DESIGN) and report['job_sha256'] == sha(Path(__file__))
    assert sha(work / 'private-trace/RUN_COMPLETE') == report['private_closed_sha256']
    report['parent_child'] = dict(command=command, returncode=0, synchronous=True,
        stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), parent_imported_torch=False)
    output = a.output.absolute()
    output.mkdir(exist_ok=True)
    assert not list(output.iterdir())
    save(output / 'compact-report.json', report)
    save(output / 'RUN_COMPLETE', dict(schema='qattn.training-case-public.closed.v1',
        files={'compact-report.json': sha(output / 'compact-report.json')}, training=False))
    print(json.dumps(dict(status=report['status'], task_training=False, optimizer_steps=0, L5='NOT_ACCEPTED')))


if __name__ == '__main__':
    main()
