"""Four prospective source/proof correspondence checks; execute only via scheduler."""
import cmath
import hashlib
import math
import torch
from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead
from experiments.rmea_utility_measurement.token_transport import KINDS, TAU, TokenTransportHead, transport_attention

CHECKS=('AC-1-finite-native-source-separation','AC-2-balanced-ceiling-null',
        'AC-3-complex-linear-Gram-region-null','AC-4-tau0-and-dephased-domain-boundary')

def encoded(t):
    t=t.detach().cpu()
    out=dict(shape=list(t.shape),dtype=str(t.dtype))
    if t.is_complex():out.update(real=t.real.tolist(),imag=t.imag.tolist())
    else:out['values']=t.tolist()
    return out

def parameter_hash(head):
    h=hashlib.sha256()
    for name,t in sorted(head.state_dict().items()):
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def run():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'One scheduler-allocated CUDA device required'
    device=torch.device('cuda:0');torch.set_num_threads(1)
    errors=[];records=[]
    def close(a,b):
        if not isinstance(b,torch.Tensor):b=torch.as_tensor(b,dtype=a.dtype,device=a.device)
        error=float((a-b).abs().max());assert math.isfinite(error) and error<1e-9,error
        errors.append(error);return error
    def projectors(s):return s[..., :,None]*s.conj()[...,None,:]
    eye=torch.eye(8,dtype=torch.complex128,device=device)
    def ceiling(effect,M):
        B=M+3*EPS*eye;vals,vecs=torch.linalg.eigh(B);assert vals.min()>0
        inverse=(vecs*vals.rsqrt()[None,:])@vecs.mH
        whitened=inverse@(effect+EPS*eye)@inverse
        return torch.linalg.eigvalsh((whitened+whitened.mH)/2).max()
    b0=eye[0];b3=eye[3];e=(b0-b3)/math.sqrt(2);f=(b0+b3)/math.sqrt(2)
    a=math.sqrt(3)/2;b=.5;r=3*math.sqrt(3)/(2*math.sqrt(7));s=1/(2*math.sqrt(7));h=5/(2*math.sqrt(7))
    q=math.sin(TAU)**2;ell=.5+3*q/56;beta=.25+9*q/112
    L=(1+EPS)/(2+3*EPS);P=(ell*ell+EPS*beta)/(ell+3*EPS*beta);margin=(P-L)/2
    assert margin>0
    ze=.5*math.cos(TAU)*a+1j*math.sin(TAU)*(r-h*a/2)
    zf=.5*math.cos(TAU)*b+1j*math.sin(TAU)*(s-h*b/2)
    A=(ze+zf)/math.sqrt(2*beta);B=(-ze+zf)/math.sqrt(2*beta)
    y=2*math.atan2(abs(B),abs(A));zeta=math.remainder(cmath.phase(B)-cmath.phase(A),2*math.pi)
    assert 0<y<math.pi and -math.pi<zeta<0
    source=RMEAHead('quantum').double().to(device)
    for parameter in source.parameters():parameter.zero_()
    source.prepare.weight[6,0]=math.atanh(-.5);source.prepare.weight[6,1]=math.atanh(-1/6);source.prepare.weight[6,2]=math.atanh(.5)
    source.effect[1,1,0,0]=math.atanh(y/math.pi);source.effect[1,1,0,1]=math.atanh(zeta/math.pi)
    source.value.weight[0,1]=1;source.decoder[1,0]=1;source.bias[0]=(P+L)/2;source.bias[2:]=-2
    assert all(torch.isfinite(t).all() for t in source.state_dict().values())
    features=torch.zeros((1,3,1024),dtype=torch.float64,device=device)
    features[0,torch.arange(3,device=device),torch.arange(3,device=device)]=1
    roles=torch.zeros((1,3,4),dtype=torch.float64,device=device);mask=torch.ones((1,3),dtype=torch.bool,device=device)
    baseline_hash=parameter_hash(source);native=source(features,roles,mask,details=True)
    expected_keys=torch.stack((e,a*e+b*f,f));expected_query=A*b0+B*b3
    close(projectors(native['psi'][0]),projectors(expected_keys));close(projectors(native['phi'][1]),projectors(expected_query))
    assert native['psi'].dtype==native['phi'].dtype==torch.complex128 and features.dtype==torch.float64
    outputs={};predictions={}
    for kind in KINDS:
        head=TokenTransportHead(kind).double().to(device);head.load_state_dict(source.state_dict());assert parameter_hash(head)==baseline_hash
        d=head(features,roles,mask,details=True);outputs[kind]=d;predictions[kind]=int(d['logits'].argmax(-1))
        close(d['alpha'].sum(-1),torch.ones((1,40),dtype=torch.float64,device=device));assert torch.isfinite(d['logits']).all()
        close(d['utilities'],native['utilities']);assert parameter_hash(head)==baseline_hash
    close(outputs['native']['alpha'],native['alpha']);close(outputs['native']['logits'],native['logits'])
    graph=outputs['coherent']['transport']['samples'][0];psi=native['psi'][0];M=psi.T@psi.conj()
    close(graph['G'],expected_keys.conj()@expected_keys.T)
    expected_H=torch.tensor([[0,r,0],[r,0,s],[0,s,0]],dtype=torch.complex128,device=device);close(graph['H'],expected_H)
    native_max=ceiling(projectors(psi[1]),M)
    weights=graph['U'][1].abs().square();inc_effect=(projectors(psi)*weights[:,None,None]).sum(0);inc_max=ceiling(inc_effect,M)
    close(native_max,L);assert float(inc_max)<=L+1e-9
    close(outputs['coherent']['alpha'][0,1,1],P)
    assert predictions==dict(native=0,coherent=1,token_dephased=0)
    assert float(outputs['coherent']['logits'][0,1]-source.bias[0])>margin*.999
    for kind in ('native','token_dephased'):assert float(source.bias[0]-outputs[kind]['logits'][0,1])>=margin*.999
    assert parameter_hash(source)==baseline_hash
    records.append(dict(name=CHECKS[0],status='PASS',native_ceiling=float(native_max),initial_dephased_ceiling=float(inc_max),coherent_middle=float(outputs['coherent']['alpha'][0,1,1]),analytic_L=L,analytic_P=P,decision_margin=margin,predictions=predictions,parameter_sha256=baseline_hash,finite_source_parameters=True))

    balanced=torch.stack((e,(e+f)/math.sqrt(2),f))[None]
    balanced_query=balanced[0,1].expand(40,-1).clone()
    d=transport_attention(balanced,balanced_query,mask)['samples'][0];bp=balanced[0];bm=bp.T@bp.conj();w=bp.T@d['U'].mH[:,1]
    close(ceiling(projectors(w),bm),L);close(d['alpha'][:,1],L)
    records.append(dict(name=CHECKS[1],status='PASS',coherent_middle_ceiling=float(ceiling(projectors(w),bm)),analytic_ceiling=L))

    keys=torch.stack((b0,(b0+1j*b3)/math.sqrt(2),b3,a*b0+cmath.exp(.7j)*b*b3))
    G=keys.conj()@keys.T;K=G-torch.eye(4,dtype=G.dtype,device=device);scale=torch.linalg.eigvalsh(K).abs().max()
    U=torch.matrix_exp(-1j*TAU*K/scale);physical_M=keys.T@keys.conj();W=torch.matrix_exp(-1j*TAU*(physical_M-eye)/scale)
    theta=torch.arange(40,dtype=torch.float64,device=device)/41+.1
    query=torch.cos(theta)[:,None]*b0+torch.sin(theta)[:,None]*torch.exp(1j*(theta+.2))[:,None]*b3+.1j*eye[4]
    query=query/query.norm(dim=-1,keepdim=True)
    amp=query@keys.conj().T;den=amp.abs().square().sum(-1)+4*EPS
    linear=((amp@U.T).abs().square()+EPS)/den[:,None]
    transformed=query@W.T;m4=torch.ones((1,4),dtype=torch.bool,device=device)
    counterpart=transport_attention(keys[None],transformed,m4,kind='native')['alpha'][0]
    close(transformed.norm(dim=-1),query.norm(dim=-1));close((transformed@keys.conj().T).abs().square().sum(-1)+4*EPS,den)
    residual=close(linear,counterpart);close(U@keys.conj(),keys.conj()@W)
    records.append(dict(name=CHECKS[2],status='PASS',max_vector_residual=residual,complex_rank_deficient_keys=True,unrestricted_query_counterpart=True))

    for kind in KINDS:close(transport_attention(native['psi'],native['phi'],mask,kind=kind,_tau=0.)['alpha'],native['alpha'])
    boundary=[]
    gamma=(1+2*EPS)/(2+3*EPS);cutoff=70*gamma/61
    for qq in (.5,.75):
        d=transport_attention(native['psi'],native['phi'],mask,kind='token_dephased',_tau=math.asin(math.sqrt(qq)))['samples'][0]
        effect=(projectors(psi)*d['U'][1].abs().square()[:,None,None]).sum(0);maximum=float(ceiling(effect,M))
        determinant=qq*(210*gamma-183*qq)/784
        if qq<cutoff:assert determinant>0 and maximum<=L+1e-9
        else:assert determinant<0 and maximum>L+1e-6
        boundary.append(dict(q=qq,determinant=determinant,dephased_ceiling=maximum,native_ceiling=L))
    records.append(dict(name=CHECKS[3],status='PASS',exact_q_cutoff=cutoff,stress_points=boundary,tau0_identity=True))
    assert max(errors)<margin*.001
    fixture=dict(schema='qattn.attention-ceiling.synthetic-source-trace.v1',sample_selection='Predeclared finite three-token source witness; no text or outcome selection',features=dict(shape=[1,3,1024],dtype='float64',nonzero=[[0,0,0,1],[0,1,1,1],[0,2,2,1]],default=0),roles=dict(shape=[1,3,4],default=0),mask=encoded(mask),psi=encoded(native['psi']),query_class1=encoded(native['phi'][1]),utilities=encoded(native['utilities']),G=encoded(graph['G']),K=encoded(graph['K']),H=encoded(graph['H']),U=encoded(graph['U']),amplitudes=encoded(graph['a']),normalizer=encoded(graph['D']),outputs={k:{n:encoded(outputs[k][n]) for n in ('alpha','logits')} for k in KINDS},sample_trace_v2_certified=False)
    return dict(status='ATTENTION_CEILING_CUDA_CHECK_COMPLETE',checks=records,check_count=4,max_absolute_residual=max(errors),fixture=fixture,device=dict(type='cuda',name=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda),optimizer_steps=0,real_dataset_reads=0,new_checkpoints=0,COS_requests=0,cloud_writes=0,task_training=False,independent_task_confirmation=False,formal_handoff=False,L5='NOT_ACCEPTED')
