"""Frozen synthetic entry; reuses the existing Torch-free argument interface."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rmea_original_scores.common import guard, read, save, sha
from experiments.rmea_utility_measurement.class_centered_job import arguments

DESIGN = ROOT/'docs/evidence/rmea/20261120-l5-token-graph/canary-design.json'


def main():
    a = arguments()
    guard()
    d, c = read(DESIGN), read(a.execution_contract)
    assert d['schema'] == 'qattn.token-graph.canary.design.v1'
    assert c['schema'] == 'qattn.token-graph.canary.execution.v1'
    assert c['job_sha256'] == sha(Path(__file__)) and c['design_sha256'] == sha(DESIGN)
    assert c['budget'] == d['budget'] and c['review_scope'] == 'both'
    assert c['budget']['optimizer_steps'] == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert {n: importlib.metadata.version(n) for n in d['environment']} == d['environment']
    for n, h in d['source_files'].items():
        assert sha(ROOT/n) == h
    for p in (a.output.absolute(), a.execution_contract.absolute()):
        assert all(not q.is_symlink() for q in (p, *p.parents))
    assert a.output.is_dir() and not list(a.output.iterdir())
    if a.child:
        from experiments.rmea_utility_measurement.token_transport_canary import run
        r = run()
        r.update(job_sha256=sha(Path(__file__)), design_sha256=sha(DESIGN), environment=d['environment'])
        print(json.dumps(r, allow_nan=False), flush=True)
        return
    assert 'torch' not in sys.modules
    work = a.output.parent/'scientific-work'
    assert not work.exists()
    work.mkdir()
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--execution-contract',
            str(a.execution_contract.absolute()), '--output', str(a.output.absolute()), '--child']
    save(work/'child-command.json', dict(argv=argv))
    stdout, stderr = work/'child-stdout.json', work/'child-stderr.log'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        done = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=err,
            timeout=c['budget']['timeout_seconds']-30,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    assert done.returncode == 0, 'Canary failed; preserve partials, no retry or completion marker'
    r = read(stdout)
    assert r['status'] == 'TOKEN_GRAPH_NUMERICAL_CANARY_COMPLETE'
    assert r['check_count'] == len(r['checks']) == d['checks'] == 8
    assert len({v['name'] for v in r['checks']}) == 8
    assert {v['name'] for v in r['checks']} == set(d['check_names'])
    assert all(v['status'] == 'PASS' for v in r['checks'])
    assert r['job_sha256'] == sha(Path(__file__)) and r['design_sha256'] == sha(DESIGN)
    assert all(r[k] == 0 for k in ('optimizer_steps', 'real_dataset_reads', 'new_checkpoints', 'COS_requests', 'cloud_writes'))
    assert all(r[k] is False for k in ('task_training', 'independent_confirmation', 'formal_handoff'))
    assert r['L5'] == 'NOT_ACCEPTED'
    r['parent_child'] = dict(command_sha256=sha(work/'child-command.json'), returncode=0,
        synchronous=True, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), parent_imported_torch=False)
    assert not list(a.output.iterdir())
    save(a.output/'compact-report.json', r)
    save(a.output/'RUN_COMPLETE', dict(schema='qattn.token-graph.canary.closed.v1',
        files={'compact-report.json': sha(a.output/'compact-report.json')}, training=False))
    print(json.dumps(dict(status=r['status'], task_training=False, L5='NOT_ACCEPTED')))


if __name__ == '__main__':
    main()
