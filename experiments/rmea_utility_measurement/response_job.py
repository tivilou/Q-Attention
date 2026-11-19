"""One reviewed CPU child analyzing frozen models, with compact publication."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rmea_original_scores.common import guard, read, save, sha

DESIGN = ROOT/'docs/evidence/rmea/20261119-l5-round98/response-design.json'
PATH_ARGUMENTS = ['assets', 'receipt', 'cohort', 'donors', 'checkpoints', 'execution_contract', 'output']


def arguments():
    parser = argparse.ArgumentParser()
    for name in PATH_ARGUMENTS:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def checked_checkpoints(a, d):
    assert a.checkpoints.is_dir()
    assert set(p.name for p in a.checkpoints.iterdir()) == set(d['checkpoints']) | {'RUN_COMPLETE'}
    for name, value in d['checkpoints'].items():
        f = a.checkpoints/name
        assert not f.is_symlink() and f.stat().st_size == value['bytes'] and sha(f) == value['sha256']
    assert sha(a.checkpoints/'RUN_COMPLETE') == d['checkpoint_marker_sha256']
    assert read(a.checkpoints/'RUN_COMPLETE') == dict(schema='qattn.utility-training.checkpoints.v1',
        files={n: v['sha256'] for n, v in d['checkpoints'].items()}, training=True)


def checked(a):
    guard()
    d, c = read(DESIGN), read(a.execution_contract)
    assert d['schema'] == 'qattn.utility-response.design.v1' and c['schema'] == 'qattn.utility-response.execution.v1'
    assert c['job_sha256'] == sha(Path(__file__)) and c['design_sha256'] == sha(DESIGN)
    assert c['budget'] == d['budget'] and c['review_scope'] == 'both'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert {n: importlib.metadata.version(n) for n in d['environment']} == d['environment']
    for n, h in d['source_files'].items():
        assert sha(ROOT/n) == h
    assert sha(ROOT/d['reported_result_path']) == d['reported_result_sha256']
    for name in PATH_ARGUMENTS:
        path = getattr(a, name).absolute()
        assert all(not p.is_symlink() for p in [path, *path.parents])
    assert a.output.is_dir() and not list(a.output.iterdir())
    checked_checkpoints(a, d)
    return d, c


def main():
    a = arguments()
    d, c = checked(a)
    if a.child:
        from experiments.rmea_utility_measurement.response import run
        r = run(a, d)
        checked_checkpoints(a, d)
        r.update(job_sha256=sha(Path(__file__)), design_sha256=sha(DESIGN), environment=d['environment'])
        print(json.dumps(r, allow_nan=False), flush=True)
        return
    assert 'torch' not in sys.modules
    work = a.output.parent/'scientific-work'
    assert not work.exists()
    work.mkdir()
    command = [sys.executable, '-B', str(Path(__file__).resolve())]
    for name in PATH_ARGUMENTS:
        command += ['--' + name.replace('_', '-'), str(getattr(a, name).absolute())]
    command += ['--child']
    save(work/'child-command.json', dict(argv=command))
    stdout, stderr = work/'child-stdout.json', work/'child-stderr.log'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        done = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err,
            timeout=c['budget']['timeout_seconds'] - 60,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    assert done.returncode == 0, 'Analysis failed; retain private evidence, no public marker'
    checked_checkpoints(a, d)
    r = read(stdout)
    assert r['status'] == 'UTILITY_LOCAL_RESPONSE_DIAGNOSTIC_COMPLETE'
    assert r['job_sha256'] == sha(Path(__file__)) and r['design_sha256'] == sha(DESIGN)
    assert r['canary']['status'] == 'PASS' and r['conditions'] == d['conditions']
    assert r['evaluation_examples'] == 972 and r['groups'] == 866
    assert all(r[k] == 0 for k in ['optimizer_steps', 'new_checkpoints', 'COS_requests', 'cloud_writes'])
    assert not r['task_training'] and not r['formal_handoff'] and r['L5'] == 'NOT_ACCEPTED'
    assert set(r['models']) == set(d['conditions'])
    assert all(v['checkpoint_unchanged'] for v in r['models'].values())
    r['parent_child'] = dict(command_sha256=sha(work/'child-command.json'), returncode=0,
        synchronous=True, stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), parent_imported_torch=False)
    assert not list(a.output.iterdir())
    save(a.output/'compact-report.json', r)
    save(a.output/'RUN_COMPLETE', dict(schema='qattn.utility-response.closed.v1',
        files={'compact-report.json': sha(a.output/'compact-report.json')}, training=False))
    print(json.dumps(dict(status=r['status'], decision=r['decision'], L5='NOT_ACCEPTED')))


if __name__ == '__main__':
    main()
