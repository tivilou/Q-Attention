"""Eight fixed forward falsifiers; numerical execution belongs to shared scheduler."""
import math
import torch
from experiments.rmea_context_geometry.legacy.rmea import EPS,RMEAHead
from experiments.rmea_utility_measurement.token_transport import KINDS,TAU,TokenTransportHead,transport_attention
CHECK_NAMES=[
 'native_tau0_mask_padding_and_normalization',
 'invalid_interfaces_and_parameter_isolation',
 'gauge_query_phase_and_token_permutation',
 'single_zero_graph_and_degenerate_spectrum',
 'finite_native_source_decision_and_harmful_relabel_witness',
 'signed_pair_current_and_finite_difference',
 'astra_absorption_and_exact_contextual_key_counterpart',
 'deterministic_replay_and_complete_synthetic_fixture',
]

def encoded(t):
    t=t.detach().cpu()
    r=dict(shape=list(t.shape),dtype=str(t.dtype))
    if t.is_complex():r.update(real=t.real.tolist(),imag=t.imag.tolist())
    else:r['values']=t.tolist()
    return r

@torch.no_grad()
def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261120)
    checks=[]
    def record(**v):checks.append(dict(name=CHECK_NAMES[len(checks)],status='PASS',**v))
    def close(a,b,tol=2e-9):
        assert torch.allclose(a,b,atol=tol,rtol=0),float((a-b).abs().max())
    def states(shape):
        a=torch.randn(shape,dtype=torch.float64)+1j*torch.randn(shape,dtype=torch.float64)
        return a/a.norm(dim=-1,keepdim=True)
    psi=states((2,5,8));phi=states((40,8));mask=torch.tensor([[True,False,True,True,False],[True,True,False,True,True]])
    dirty=psi.clone();dirty[~mask]=complex(float('nan'),0)
    p=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square();native=(p+EPS)*mask[:,None];native/=native.sum(-1,keepdim=True)
    for kind in KINDS:
        zero=transport_attention(dirty,phi,mask,kind=kind,_tau=0.)['alpha'];close(zero,native)
        a=transport_attention(dirty,phi,mask,kind=kind)['alpha'];close(a.sum(-1),torch.ones((2,40),dtype=torch.float64));assert a.min()>=0
        assert torch.equal(a.masked_select(~mask[:,None]),torch.zeros(120,dtype=torch.float64))
    record(tau0_native=True,padding_nan_ignored=True,all40=True)

    bad=psi.clone();bad[0,0,0]=complex(float('nan'),0)
    invalid=[(psi,phi[1:],mask),(psi.to(torch.complex64),phi,mask),(psi,phi,mask.float()),(psi,phi,torch.zeros_like(mask)),(psi*2,phi,mask),(bad,phi,mask)]
    for args in invalid:
        try:transport_attention(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid transport interface accepted')
    original=RMEAHead('quantum').double();initial={k:v.clone() for k,v in original.state_dict().items()}
    for kind in KINDS:
        h=TokenTransportHead(kind).double();h.load_state_dict(initial);assert set(h.state_dict())==set(initial) and sum(p.numel() for p in h.parameters())==81028
    try:TokenTransportHead('dephased')
    except ValueError:pass
    else:raise AssertionError('Physical-register dephasing conflated with token control')
    record(rejected=len(invalid)+1,stored_parameters=81028,forward_only=True)

    phases=torch.exp(1j*torch.arange(10,dtype=torch.float64).reshape(2,5)*.31)
    queryphases=torch.exp(1j*torch.arange(40,dtype=torch.float64)*.17)
    permutation=torch.tensor([2,4,0,3,1])
    for kind in KINDS:
        d=transport_attention(psi,phi,mask,kind=kind)
        shifted=transport_attention(psi*phases[...,None],phi*queryphases[:,None],mask,kind=kind)
        close(d['alpha'],shifted['alpha'])
        reordered=transport_attention(psi[:,permutation],phi,mask[:,permutation],kind=kind)
        close(reordered['alpha'],d['alpha'][:,:,permutation])
        for b in range(2):
            q=torch.diag(phases[b,mask[b]]);close(shifted['samples'][b]['U'],q.mH@d['samples'][b]['U']@q)
    record(token_and_query_gauge=True,permutation=True)

    one=psi[:1,:1];mm=torch.ones((1,1),dtype=torch.bool)
    orth=torch.zeros((1,2,8),dtype=torch.complex128);orth[0,0,0]=1;orth[0,1,1]=1
    same=torch.zeros((1,4,8),dtype=torch.complex128);same[:,:,0]=1
    for s in [one,orth,same]:
        m=torch.ones(s.shape[:2],dtype=torch.bool)
        for kind in KINDS:
            d=transport_attention(s,phi,m,kind=kind);close(d['alpha'],d['samples'][0]['native_alpha'][None])
            close(d['samples'][0]['U'].mH@d['samples'][0]['U'],torch.eye(s.shape[1],dtype=torch.complex128))
    record(single_token=True,zero_graph=True,repeated_eigenvalues_forward_finite=True,training_gradients_not_qualified=True)

    witness=RMEAHead('quantum').double()
    witness.prepare.weight.zero_();witness.prepare.bias.zero_();witness.effect.zero_()
    witness.prepare.weight[6,1]=math.atanh(.5)
    witness.effect[1,1,0,0]=math.atanh(.5);witness.effect[1,1,0,1]=math.atanh(.5)
    witness.value.weight.zero_();witness.value.bias.zero_();witness.value.weight[0,0]=1;witness.value.weight[0,1]=-1
    witness.decoder.zero_();witness.decoder[1,0]=1;witness.bias.fill_(-2);witness.bias[0]=.25;witness.bias[1]=0
    inputs=torch.zeros((1,2,1024),dtype=torch.float64);inputs[0,0,0]=1;inputs[0,1,1]=1
    roles=torch.zeros((1,2,4),dtype=torch.float64);valid=torch.ones((1,2),dtype=torch.bool)
    fixed={k:v.clone() for k,v in witness.state_dict().items()};details={};predictions={}
    for kind in KINDS:
        h=TokenTransportHead(kind).double();h.load_state_dict(fixed);d=h(inputs,roles,valid,details=True);details[kind]=d;predictions[kind]=int(d['logits'].argmax(-1))
        assert all(torch.equal(v,h.state_dict()[k]) for k,v in fixed.items())
    assert predictions==dict(native=0,coherent=1,token_dephased=0)
    shift=math.sin(2*TAU)/(2*math.sqrt(2)*(1+2*EPS))
    expected=torch.tensor([.5+shift,.5-shift],dtype=torch.float64)
    close(details['coherent']['alpha'][0,1],expected);close(details['native']['alpha'][0,1],torch.full((2,),.5,dtype=torch.float64))
    close(details['token_dephased']['alpha'][0,1],details['native']['alpha'][0,1])
    source=witness(inputs,roles,valid,details=True);close(source['alpha'],details['native']['alpha']);close(source['logits'],details['native']['logits'])
    record(predictions=predictions,coherent_class1_logit=float(details['coherent']['logits'][0,1]),null_logit=.25,harmful_if_true_null=True,scope='Analytic source-parameter fixture; no text benefit')

    s=psi[:1,:4];m=torch.ones((1,4),dtype=torch.bool);u=torch.randn((40,4),dtype=torch.float64);step=1e-6
    d=transport_attention(s,phi,m,kind='coherent',_tau=0.)['samples'][0];a=d['a'];H=d['H'];D=d['D']
    theory=(2*(a.conj()*(a@H.T)).imag*u/D[:,None]).sum(-1)
    plus=transport_attention(s,phi,m,kind='coherent',_tau=step)['alpha'][0]
    minus=transport_attention(s,phi,m,kind='coherent',_tau=-step)['alpha'][0]
    fd=((plus-minus)*u).sum(-1)/(2*step);close(fd,theory,2e-7)
    pairs=torch.zeros(40,dtype=torch.float64)
    for i in range(4):
        for j in range(i+1,4):pairs+=2*(u[:,i]-u[:,j])*(a[:,i].conj()*H[i,j]*a[:,j]).imag/D
    close(pairs,theory);assert theory.min()<0 and theory.max()>0
    record(max_fd_error=float((fd-theory).abs().max()),signed_response=True,pair_current_identity=True)

    three=torch.zeros((1,3,8),dtype=torch.complex128);three[0,0,0]=1;three[0,1,0]=math.sqrt(3)/2;three[0,1,1]=.5;three[0,2,1]=1
    d=transport_attention(three,phi,torch.ones((1,3),dtype=torch.bool))['samples'][0]
    G=d['G'];U=d['U'];K=d['K'];P=G@torch.linalg.pinv(G,hermitian=True)
    residual=(torch.eye(3,dtype=torch.complex128)-P)@U@three[0].conj()
    assert residual.norm()>1e-5 and (K@P-P@K).norm()>1e-5
    transformed=three[0].T@U.mH
    encoded_amplitude=phi@transformed.conj();walk_amplitude=d['a']@U.T;close(encoded_amplitude,walk_amplitude)
    close((encoded_amplitude.abs().square()+EPS)/d['D'][:,None],d['alpha'])
    record(nonabsorbed_amplitude_residual=float(residual.norm()),exact_contextual_keys=True,fixture_scope='Abstract normalized states; no source-circuit or decision separation asserted')

    for kind in KINDS:
        h=TokenTransportHead(kind).double();h.load_state_dict(fixed);again=h(inputs,roles,valid,details=True)
        close(again['alpha'],details[kind]['alpha'],0.);close(again['logits'],details[kind]['logits'],0.)
    t=details['coherent']['transport']['samples'][0]
    fixture=dict(schema='qattn.token-graph.synthetic-full-values.v1',selection='One predeclared source-parameter witness,synthetic h0=e0/h1=e1',inputs={k:encoded(v) for k,v in dict(h=inputs,roles=roles,mask=valid,psi=source['psi'],phi=source['phi'],utilities=source['utilities']).items()},transport={k:encoded(t[k]) for k in ['a','D','G','K','H','U']},outputs={kind:{k:encoded(details[kind][k]) for k in ['alpha','logits']} for kind in KINDS},semantic_checks=dict(native_source_parity=True,complete_coordinate_values=True,unchanged_values_and_bias=True,not_sample_trace_v2_certification=True))
    record(all_conditions_replayed=True,full_fixture_fields=list(fixture['inputs'])+list(fixture['transport']),no_weights_or_dataset_saved=True)
    assert len(checks)==8
    return dict(status='TOKEN_GRAPH_NUMERICAL_CANARY_COMPLETE',checks=checks,check_count=8,seed=20261120,synthetic_source_fixture=fixture,task_training=False,optimizer_steps=0,real_dataset_reads=0,new_checkpoints=0,COS_requests=0,cloud_writes=0,independent_confirmation=False,formal_handoff=False,L5='NOT_ACCEPTED',qualification='Forward synthetic qualification only; no training-gradient,CaseStudy-readiness,novelty or task gain')
