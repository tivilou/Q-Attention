"""Focused algebra, real CLI launch and failure cleanup; no model/data access."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.q_rmea_query_alignment import generator, distribution, derangement, command, execute
from baselines.retacred.rmea_token_transport import route
from baselines.retacred.rmea_token_transport_protocol import atomic_json


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261121)
    lengths = [1, 3, 9]
    mask = torch.arange(9)[None, :] < torch.tensor(lengths)[:, None]
    psi = torch.randn(3, 9, 8, dtype=torch.complex128)
    psi = psi/psi.norm(dim=-1, keepdim=True)
    phi = torch.randn(40, 8, dtype=torch.complex128)
    phi = phi/phi.norm(dim=-1, keepdim=True)
    g = generator(psi, mask)
    alpha, rotation = distribution(psi, phi, mask, g)
    expected = route(psi, phi, mask, kind='linear_coherent')['alpha']
    assert torch.allclose(alpha, expected, atol=2e-12, rtol=2e-12)
    changed = psi.clone()
    changed[~mask] = 7+11j
    assert torch.equal(g, generator(changed, mask))
    assert torch.equal(alpha, distribution(changed, phi, mask, g)[0])
    assert torch.allclose(rotation@rotation.mH, torch.eye(8, dtype=rotation.dtype), atol=2e-12, rtol=0)
    for shifted in (g[[1,2,0]], g.mean(0).expand_as(g)):
        a, _ = distribution(psi, phi, mask, shifted)
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=2e-12, rtol=0)
        assert not a[~mask[:, None, :].expand_as(a)].any()
    for n in (2, 7, 19584):
        perm = derangement(n)
        assert perm == derangement(n) and sorted(perm) == list(range(n))
        assert all(i != j for i, j in enumerate(perm))
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        previous = os.environ.get('RESEARCH_OUTPUT_DIR')
        os.environ['RESEARCH_OUTPUT_DIR'] = str(out)
        try:
            args = SimpleNamespace(run=out/'unused', output=out)
            jobs = [(a, command(args, 'worker', a)+['--help'], 'cpu-launch-fixture-'+str(i))
                    for i,a in enumerate(('aligned','deranged','training_mean'))]
            execute(jobs, out, 'launch-fixture')
            rows = json.loads((out/'launch-fixture-commands.json').read_text())['commands']
            assert len({r['pid'] for r in rows}) == 3
            assert all(r['argv'][2] == str(Path(__file__).with_name('q_rmea_query_alignment.py')) for r in rows)
            # Real synthetic child handles test the finally path after a launch failure.
            fail = command(args, 'worker', 'aligned')+['--invalid-argument']
            execute_jobs = [('bad', fail, 'cpu-fixture'), ('sleeper',
                [sys.executable, '-B', '-c', 'import time;time.sleep(60)'], 'cpu-fixture')]
            try:
                execute(execute_jobs, out, 'failure-fixture')
                raise AssertionError('Expected child failure')
            except RuntimeError:
                pass
            rows = json.loads((out/'failure-fixture-commands.json').read_text())['commands']
            for row in rows:
                try:
                    os.kill(row['pid'], 0)
                except ProcessLookupError:
                    continue
                raise AssertionError('Owned child remains after cleanup')
            real_popen = subprocess.Popen
            handles = []
            def signal_after_creation(*a, **kw):
                proc = real_popen(*a, **kw)
                handles.append(proc)
                # The exact Popen-to-registration window; repeated signals only set a flag.
                os.kill(os.getpid(), signal.SIGTERM)
                os.kill(os.getpid(), signal.SIGTERM)
                return proc
            with patch('subprocess.Popen', signal_after_creation):
                try:
                    execute([('signal-window', execute_jobs[1][1], 'cpu-fixture')], out, 'signal-window')
                    raise AssertionError('Expected cancellation')
                except InterruptedError:
                    pass
            assert handles and all(p.poll() is not None for p in handles)
            real_wait = subprocess.Popen.wait
            signaled = []
            def signal_after_wait(proc, *a, **kw):
                rc = real_wait(proc, *a, **kw)
                if not signaled:
                    signaled.append(True)
                    os.kill(os.getpid(), signal.SIGTERM)
                return rc
            with patch('subprocess.Popen.wait', signal_after_wait):
                try:
                    execute([jobs[0]], out, 'cleanup-window')
                    raise AssertionError('Expected cleanup cancellation')
                except InterruptedError:
                    pass
            assert signaled
        finally:
            if previous is None:
                os.environ.pop('RESEARCH_OUTPUT_DIR', None)
            else:
                os.environ['RESEARCH_OUTPUT_DIR'] = previous
    if os.environ.get('RESEARCH_OUTPUT_DIR'):
        out = Path(os.environ['RESEARCH_OUTPUT_DIR'])
        atomic_json(out/'unit-validation.json', dict(accepted=True, checks=[
            '8D identity versus original token route', 'singleton/padding', 'control normalization',
            'label-blind bijection', 'three exact CLI targets', 'failed child/sibling quiescence',
            'signal during Popen-to-registration plus repeated signal']))
        atomic_json(out/'RUN_COMPLETE', dict(accepted=True))
    print('PASS: query identity, controls, masking, exact three-child launch and failure cleanup')


if __name__ == '__main__':
    main()
