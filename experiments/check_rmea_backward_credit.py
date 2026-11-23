"""Focused graph parity and real CLI-target checks, no formal data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from types import SimpleNamespace
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred import rmea_token_transport_protocol as p
    from experiments import q_rmea_backward_credit as b
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261121)
    head = RMEAHead('quantum').double()
    h = torch.randn(2,5,1024,dtype=torch.float64)*.1
    roles = torch.randn(2,5,4,dtype=torch.float64)*.1
    mask = torch.tensor([[True]*5,[True]*3+[False]*2])
    inputs = h,roles,mask
    row = b.probe(head,inputs,torch.tensor([1,2]))
    if row['detach_difference']<=1e-10:
        raise AssertionError('Fixture does not discriminate the detached graphs')
    for arm in b.ARMS:
        d = head(*inputs,details=True)
        a = b.alpha(d['psi'],d['phi'],mask,arm)
        g = torch.autograd.grad(a.sum(),head.prepare.weight)[0]
        if g.abs().max()>1e-10:
            raise AssertionError('Normalized distribution has a nonzero mass Jacobian')
        if a[1,:,3:].abs().max()!=0:
            raise AssertionError('Padding gained mass')
    with tempfile.TemporaryDirectory() as temp:
        args = SimpleNamespace(run=Path(temp)/'input',output=Path(temp)/'output')
        args.output.mkdir()
        jobs = []
        for action,condition in [('prepare',None),('worker','full')]:
            argv = b.command(args,action,condition)+['--help']
            if Path(argv[2]).resolve()!=Path(b.__file__).resolve():
                raise AssertionError('Command target differs')
            jobs.append((action,argv,''))
        events = []
        b.qa.execute(jobs,args.output,'cli-target',report=lambda *v:events.append(v[2]))
        if not events:
            raise AssertionError('Real parent did not invoke the progress reporter')
        commands = p.read(args.output/'cli-target-commands.json')['commands']
        if len(commands)!=2 or any(c['cwd']!=str(ROOT) for c in commands):
            raise AssertionError('Real parent CLI command capture differs')
    result = dict(status='PASS',probe=row,normalized_mass_gradient=True,padding_zero=True,
                  real_parent_target='prepare/worker exact argv + --help',callback_exercised=True)
    output = Path(os.environ.get('RESEARCH_OUTPUT_DIR','.'))
    p.atomic_json(output/'unit-validation.json',result)
    p.atomic_json(output/'RUN_COMPLETE',dict(validation_sha256=p.digest(output/'unit-validation.json')))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
