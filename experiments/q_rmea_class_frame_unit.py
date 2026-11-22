"""Bounded source-level fixtures; no real dataset or optimizer step."""
import argparse,hashlib,json,math
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.retacred.rmea import EPS,RMEAHead
from baselines.retacred.rmea_class_frame import ClassFrameHead,inverse_sqrt,route,hermitian

def main():
    if not __debug__:raise RuntimeError('Unit validation requires assertions')
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'unit-validation.json').exists(),'Fresh unit output required'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(13)
    eye=torch.eye(8,dtype=torch.complex128);checks={}
    # Exact repeated eigenvalues: no eigengap division in the derivative.
    x=(torch.eye(3,dtype=torch.complex128)*2).requires_grad_()
    assert torch.autograd.gradcheck(lambda t:inverse_sqrt(hermitian(t)),(x,),eps=1e-6,atol=1e-6,rtol=1e-4)
    checks['repeated_eigenvalue_gradcheck']=True
    x=(torch.diag(torch.tensor([.8,1.1,2.3],dtype=torch.float64)).to(torch.complex128)).requires_grad_()
    assert torch.autograd.gradcheck(lambda t:inverse_sqrt(hermitian(t)),(x,),eps=1e-6,atol=1e-6,rtol=1e-4)
    checks['distinct_eigenvalue_gradcheck']=True
    phi=eye.repeat(5,1);psi=eye[:3][None];mask=torch.ones(1,3,dtype=torch.bool)
    native=route(psi,phi,mask,kind='native');full=route(psi,phi,mask,kind='class_frame')
    assert torch.allclose(native['alpha'],full['alpha'],atol=1e-12,rtol=0)
    checks['tight_frame_native_equivalence']=True
    phi=eye[0:1].repeat(40,1);full=route(psi,phi,mask,kind='class_frame')
    assert torch.allclose(full['alpha'],torch.full((1,40,3),1/3,dtype=torch.float64),atol=1e-10,rtol=0)
    assert torch.allclose(full['raw'].sum(1),torch.ones(1,3,dtype=torch.float64),atol=1e-9,rtol=0)
    checks['identical_query_uniform_limit']=True
    q=torch.randn(40,8,dtype=torch.complex128);q=q/q.norm(dim=-1,keepdim=True)
    s=torch.randn(2,4,8,dtype=torch.complex128);s=s/s.norm(dim=-1,keepdim=True);m=torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool)
    f=route(s,q,m,kind='class_frame');assert torch.allclose(f['raw'].sum(1),m.double(),atol=1e-11,rtol=0)
    assert torch.allclose(f['alpha'].sum(-1),torch.ones(2,40,dtype=torch.float64),atol=1e-12,rtol=0)
    assert torch.equal(f['alpha'][~m[:,None,:].expand_as(f['alpha'])],torch.zeros_like(f['alpha'][~m[:,None,:].expand_as(f['alpha'])]))
    # Explicit dense effects and exact classical transformed-key expression agree.
    w=f['inverse_sqrt'];P=q[:,:,None]*q[:,None,:].conj()+EPS*eye[None];E=w[None]@P@w[None]
    dense=torch.einsum('btd,rde,bte->brt',s.conj(),E,s).real*m[:,None,:]
    assert torch.allclose(dense,f['raw'],atol=1e-12,rtol=0)
    assert torch.allclose(E.sum(0),eye,atol=1e-11,rtol=0)
    checks['dense_classical_probability_completeness_padding']=True
    # Global token phases must not alter any condition.
    for kind in ('native','class_frame','class_frame_diagonal'):
        t=route(s,q,m,kind=kind);s2=s*torch.exp(1j*torch.arange(8,dtype=torch.float64).reshape(2,4))[...,None]
        assert torch.allclose(t['alpha'],route(s2,q,m,kind=kind)['alpha'],atol=1e-11,rtol=0)
    checks['token_gauge_invariance']=True
    torch.manual_seed(13);base=RMEAHead('quantum').double();initial={k:v.clone() for k,v in base.state_dict().items()}
    h=torch.randn(2,4,1024,dtype=torch.float64)*.03;roles=torch.zeros(2,4,4,dtype=torch.float64);roles[:,0,2]=1;roles[:,1,3]=1
    ref=base(h,roles,m,details=True);rows={};differences={};model=None
    for kind in ('native','class_frame','class_frame_diagonal'):
        model=ClassFrameHead(kind).double();model.load_state_dict(initial,strict=True)
        assert sum(p.numel() for p in model.parameters())==81028
        d=model(h,roles,m,details=True);assert torch.isfinite(d['logits']).all()
        if kind=='native':assert torch.equal(d['logits'],ref['logits']) and torch.equal(d['alpha'],ref['alpha'])
        loss=F.cross_entropy(d['logits'],torch.tensor([1,2]));loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        rows[kind]=dict(loss=float(loss.detach()),parameter_count=81028,gradient_norm=float(torch.sqrt(sum((p.grad*p.grad).sum() for p in model.parameters()))))
        if kind=='class_frame':
            analytic=float(model.effect.grad[1,0,0,0]);step=1e-5
            with torch.no_grad():
                model.effect[1,0,0,0]+=step
                plus=float(F.cross_entropy(model(h,roles,m)[0],torch.tensor([1,2])))
                model.effect[1,0,0,0]-=2*step
                minus=float(F.cross_entropy(model(h,roles,m)[0],torch.tensor([1,2])))
                model.effect[1,0,0,0]+=step
            finite=(plus-minus)/(2*step)
            assert abs(analytic-finite)<=2e-6+1e-4*abs(finite)
            checks['actual_head_effect_directional_derivative']=dict(analytic=analytic,finite=finite)
        before=d['alpha'].detach().clone()
        with torch.no_grad():model.effect[1,0,0,0]+=.2
        after=model(h,roles,m,details=True)['alpha'];delta=float((after[:,0]-before[:,0]).abs().max())
        if kind=='native':assert delta==0
        elif kind=='class_frame':assert delta>1e-10
        differences[kind]=delta
    checks['actual_head_native_parity_gradient_and_competitor_dependency']=True
    # Inputs and selected intermediates for exact private unit replay, no gold inference inputs.
    torch.manual_seed(13);model=ClassFrameHead('class_frame').double();model.load_state_dict(initial,strict=True)
    with torch.no_grad():d=model(h,roles,m,details=True)
    payload=dict(schema='q-rmea.class-frame.source-fixture.v1',stage='initial_source_unit',selection='fixed seed13 first synthetic2x4 fixture; no outcome-based selection',scope='Synthetic actual-head tensors only; NOT sample-trace.v2 or formal Case Study',inputs=dict(h=h,roles=roles,mask=m),initial=initial,states={k:d[k].detach().cpu() for k in ('psi','phi','alpha','native_alpha','utilities','values','logits')},measurement={k:v.detach().cpu() for k,v in d['measurement'].items() if isinstance(v,torch.Tensor)})
    torch.save(payload,out/'source-fixture.pt')
    loaded=torch.load(out/'source-fixture.pt',map_location='cpu',weights_only=True)
    replay=ClassFrameHead('class_frame').double();replay.load_state_dict(loaded['initial'],strict=True)
    with torch.no_grad():again=replay(loaded['inputs']['h'],loaded['inputs']['roles'],loaded['inputs']['mask'],details=True)
    replay_error=max(float((again[k]-v).abs().max()) for k,v in loaded['states'].items())
    assert replay_error<=1e-12
    checks['private_fixture_reopen_replay']=dict(max_error=replay_error)
    source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['baselines/retacred/rmea_class_frame.py','experiments/q_rmea_class_frame_unit.py','baselines/retacred/rmea.py','baselines/retacred/frozen_feature_head.py']}
    result=dict(source_hashes=source_hashes,schema='q-rmea.class-frame.unit.v1',status='pass',checks=checks,head_results=rows,competitor_class0_alpha_deltas=differences,fixture_sha256=hashlib.sha256((out/'source-fixture.pt').read_bytes()).hexdigest(),formal_ready=False,optimizer_steps=0,model_parameters=81028,L5='NOT_ACCEPTED')
    data=(json.dumps(result,indent=2)+'\n').encode();(out/'unit-validation.json').write_bytes(data);(out/'RUN_COMPLETE').write_text(hashlib.sha256(data).hexdigest()+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
