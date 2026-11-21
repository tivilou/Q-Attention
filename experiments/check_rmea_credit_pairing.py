"""Focused cotangent, matching, optimizer and real launch checks."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred import rmea_token_transport_protocol as p
    from baselines.retacred.rmea_token_transport_recovery import cpu_tree
    from experiments import q_rmea_credit_pairing as c
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261121)
    head=RMEAHead('quantum').double()
    inputs=(torch.randn(3,5,1024,dtype=torch.float64)*.1,
            torch.randn(3,5,4,dtype=torch.float64)*.1,
            torch.tensor([[True]*5,[True]*4+[False],[True]*3+[False]*2]))
    y=torch.tensor([1,2,3])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);F.cross_entropy(head(*inputs)[0],y).backward();optimizer.step()
    state=cpu_tree(optimizer.state_dict())
    before={n:v.detach().clone() for n,v in head.named_parameters()}
    gradients,meta,trace=c.gradient_packet(head,inputs,y)
    perm=c.permutation(3)
    assert sorted(perm)==list(range(3)) and all(i!=v for i,v in enumerate(perm))
    r,v=trace['rotation'],trace['cotangent']
    own,shuffle,k,_=c.transported_cotangents(r,v,perm)
    phase=torch.exp(1j*torch.tensor([.7,-1.1,2.3],dtype=torch.float64))[:,None,None]
    a,b,k2,_=c.transported_cotangents(phase*r,phase*v,perm)
    assert torch.allclose(k,k2,atol=1e-12,rtol=1e-10)
    assert torch.allclose(a,phase*own,atol=1e-12,rtol=1e-10)
    assert torch.allclose(b,phase*shuffle,atol=1e-12,rtol=1e-10)
    steps={a:c.proposed_step(head,state,g)[0] for a,g in gradients.items()}
    assert all(torch.equal(before[n],t) for n,t in head.named_parameters())
    assert all(torch.equal(state['state'][k]['exp_avg'],optimizer.state_dict()['state'][k]['exp_avg']) for k in state['state'])
    # Compare the proposed full step to a separate actual AdamW clone.
    clone=copy.deepcopy(head);actual=torch.optim.AdamW(clone.parameters(),lr=.001,weight_decay=.01);actual.load_state_dict(copy.deepcopy(state))
    for n,t in clone.named_parameters():t.grad=gradients['full'][n].clone()
    torch.nn.utils.clip_grad_norm_(clone.parameters(),1.,error_if_nonfinite=True);actual.step()
    assert all(torch.equal(t.detach()-before[n],steps['full'][n]) for n,t in clone.named_parameters())
    matched,matching=c.match(steps['full'],steps['direct'],steps['shuffle'])
    contrast={n:steps['full'][n]-matched[n] for n in matched}
    rotation=torch.eye(8,dtype=torch.complex128)
    finite=c.finite_check(head,inputs,y,rotation,contrast)
    for f,d,q in [(torch.tensor([1.,0.]),torch.tensor([0.,0.]),torch.tensor([0.,1.])),
                  (torch.tensor([1.,0.]),torch.tensor([1.,0.]),torch.tensor([0.,1.])),
                  (torch.tensor([0.,1.]),torch.tensor([1.,0.]),torch.tensor([1.,0.]))]:
        try:c.match({'x':f},{'x':d},{'x':q})
        except c.Uninformative:pass
        else:raise AssertionError('Degenerate matching admitted')
    with tempfile.TemporaryDirectory() as tmp:
        args=SimpleNamespace(run=Path(tmp)/'run',output=Path(tmp)/'out',anchor=Path(tmp)/'anchor');args.output.mkdir()
        jobs=[('prepare',c.command(args,'prepare')+['--help'],''),('worker',c.command(args,'worker',0)+['--help'],'')]
        events=[];c.b.qa.execute(jobs,args.output,'cli-target',report=lambda *v:events.append(v[2]))
        assert events
        commands=p.read(args.output/'cli-target-commands.json')['commands']
        assert all(Path(x['argv'][2]).resolve()==Path(c.__file__).resolve() for x in commands)
    result=dict(status='PASS',VJP=meta,matching=matching,finite_difference=finite,
        global_phase_covariance=True,original_model_optimizer_unchanged=True,actual_AdamW_step_parity=True,
        declared_degeneracy_rejected=True,real_parent_CLI_callback=True)
    out=Path(os.environ.get('RESEARCH_OUTPUT_DIR','.'));p.atomic_json(out/'unit-validation.json',result)
    p.atomic_json(out/'RUN_COMPLETE',dict(validation_sha256=p.digest(out/'unit-validation.json')))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
