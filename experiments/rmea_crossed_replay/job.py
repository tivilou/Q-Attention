"""Draft: frozen routing/readout interchange; exact review and admission required.

No optimizer, fitted adapter, new checkpoints or cloud access. A future reviewed
execution contract must bind this exact file and the frozen public design.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rmea_original_scores.common import guard, read, save, sha

DESIGN = ROOT / 'docs/evidence/rmea/20261118-l5-round82/crossed-replay-design.json'
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
    design = read(DESIGN)
    c = read(a.execution_contract)
    assert design['schema'] == 'qattn.routing-readout-crossed-replay.design.v1'
    assert c['schema'] == 'qattn.routing-readout-crossed-replay.execution.v1'
    assert c['job_sha256'] == sha(Path(__file__)) and c['design_sha256'] == sha(DESIGN)
    assert c['budget'] == design['proposed_budget'] and c['budget']['optimizer_steps'] == 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert {n: importlib.metadata.version(n) for n in design['environment']} == design['environment']
    for n, digest in design['source_files'].items():
        assert sha(ROOT / n) == digest
    for name in PATH_ARGUMENTS:
        p = getattr(a, name).absolute()
        assert all(not q.is_symlink() for q in [p, *p.parents])
    assert a.assets.is_dir() and a.checkpoints.is_dir()
    assert not a.output.exists() or not list(a.output.iterdir())
    assert sha(a.original_report) == design['original_report_sha256']
    for n, row in design['checkpoints'].items():
        path = a.checkpoints / row['filename']
        assert path.is_file() and not path.is_symlink() and sha(path) == row['sha256']
    return design, c


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
    original = read(a.original_report)
    assert original['status'] == 'TRAIN_ONLY_EXPLORATORY_SCREEN'
    assert original['input']['evaluation'] == 972 and original['seed'] == 20261118
    training_design = read(ROOT / 'docs/evidence/rmea/20261118-l5-round72/screen-design.json')
    conditions = {row['id']: row for row in training_design['conditions']}
    native = RMEAHead('quantum')
    models = {}
    for name, row in design['checkpoints'].items():
        assert not conditions[name]['donor']
        model = runner.TrainingModel(conditions[name], native)
        model.load_state_dict(torch.load(a.checkpoints / row['filename'], map_location='cpu', weights_only=True))
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == row['parameters']
        assert original['conditions'][name]['checkpoint_sha256'] == row['sha256']
        models[name] = model
    ids = data['splits']['eval']
    assert len(ids) == identity['evaluation_examples'] == 972
    gold = np.asarray(data['targets'][ids], dtype=np.int64)
    groups = [data['cohort'][int(i)]['group'] for i in ids]
    assert len(set(groups)) == identity['groups'] == 866
    cells = {(row['selector'], row['readout']): row['feature'] for row in design['cells']}
    assert len(cells) == len(design['cells']) == 18
    predictions = {key: [] for key in cells}
    loss = {key: 0.0 for key in cells}
    diagonal_tensor_checks = 0
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start + 32]
            h, roles, mask, y, donor = runner.batch(data, selected, 'eval', False)
            assert donor is None
            details = {n: model(h, roles, mask, details=True) for n, model in models.items()}
            for selector, readout in cells:
                out = (details[selector]['alpha'] * details[readout]['utilities']).sum(-1) + models[readout].head.bias
                assert out.shape == (len(selected), 40) and torch.isfinite(out).all()
                if selector == readout:
                    assert torch.equal(out, details[selector]['logits'])
                    diagonal_tensor_checks += 1
                key = selector, readout
                predictions[key].extend(out.argmax(-1).tolist())
                loss[key] += float(F.cross_entropy(out, y, reduction='sum'))
    measured = {}
    for key, pred in predictions.items():
        row = metrics.from_confusion(metrics.confusion(gold, np.asarray(pred, dtype=np.int64)))
        row['ce'] = loss[key] / len(ids)
        measured[key] = row
    for name in models:
        # Same source, CPU, batching and arithmetic: no tolerance relaxation.
        assert measured[name, name] == original['metrics'][name], 'Original diagonal parity failed'

    comparisons = []
    for pair in design['endpoints']['comparisons']:
        q, c = pair['quantum'], pair['classical']
        values = {}
        for endpoint in ['macro_f1', 'ce']:
            sign = 1 if endpoint == 'macro_f1' else -1
            qq, cq, qc, cc = [sign * measured[k][endpoint] for k in [(q, q), (c, q), (q, c), (c, c)]]
            values[endpoint] = dict(higher_is_better=True, total=cc - qq,
                routing_at_quantum_readout=cq - qq, routing_at_classical_readout=cc - qc,
                readout_at_quantum_selector=qc - qq, readout_at_classical_selector=cc - cq,
                interaction=cc - cq - qc + qq,
                convention='S=Macro-F1 or -CE; descriptive fixed-model cells only.')
        comparisons.append(dict(**pair, contrasts=values))
    report = dict(status='COMPLETED_DESCRIPTIVE_CROSSED_REPLAY',
        design_sha256=sha(DESIGN), job_sha256=sha(Path(__file__)),
        original_report_sha256=sha(a.original_report), checkpoints=design['checkpoints'],
        source_files=design['source_files'], input=identity,
        cells=[dict(feature=feature, selector=s, readout=v, metrics=measured[s, v])
            for (s, v), feature in cells.items()], comparisons=comparisons,
        diagonal_parity='EXACT_ALL_SIX_AGGREGATES_AND_BATCH_LOGITS',
        diagonal_tensor_checks=diagonal_tensor_checks,
        missing_positive_classes=[i for i, count in enumerate(measured[next(iter(cells))]['support']) if i and not count],
        environment=design['environment'], elapsed_seconds=time.monotonic() - started,
        optimizer_steps=0, task_training=False, COS_requests=0, cloud_writes=0,
        new_checkpoints=0, fitted_adapter=False, promotion=False, formal_handoff=False, L5='NOT_ACCEPTED')
    print(json.dumps(report, allow_nan=False), flush=True)


def main():
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
    stdout = work / 'child-stdout.json'
    stderr = work / 'child-stderr.log'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
            timeout=c['budget']['timeout_seconds'] - 60)
    assert result.returncode == 0, 'Child failed; private logs retained, no completion marker'
    report = read(stdout)
    assert report['status'] == 'COMPLETED_DESCRIPTIVE_CROSSED_REPLAY' and not report['task_training']
    assert report['design_sha256'] == sha(DESIGN) and report['job_sha256'] == sha(Path(__file__))
    report['parent_child'] = dict(command=command, returncode=0, synchronous=True,
        stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), parent_imported_torch=False)
    output = a.output.absolute()
    output.mkdir(exist_ok=True)
    assert not list(output.iterdir())
    save(output / 'compact-report.json', report)
    save(output / 'RUN_COMPLETE', dict(schema='qattn.crossed-replay.closed.v1',
        files={'compact-report.json': sha(output / 'compact-report.json')}, training=False))
    print(json.dumps(dict(status=report['status'], task_training=False, optimizer_steps=0, L5='NOT_ACCEPTED')))


if __name__ == '__main__':
    main()
