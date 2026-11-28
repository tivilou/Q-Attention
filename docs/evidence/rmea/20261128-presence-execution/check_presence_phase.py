"""Synthetic check of parity-separated Q-RMEA with exact pinned native circuit."""
from pathlib import Path
import ast,hashlib,json,math,sys
import numpy as np
import torch

EPS=1e-6
PIN='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
torch.set_num_threads(1)

def sig(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def original(p):
    p=Path(p);assert sig(p)==PIN
    nodes=ast.parse(p.read_text(encoding='utf-8')).body
    c=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='circuit'];assert len(c)==1
    e={'torch':torch,'QUBITS':3,'DEPTH':2};exec(compile(ast.Module(body=c,type_ignores=[]),str(p),'exec'),e)
    return e['circuit']

def permutation(i):
    for control,target in ((0,1),(1,2),(2,0)):
        if i&(1<<(2-control)):i^=1<<(2-target)
    return i

PART=np.zeros(8,dtype=np.int64)
for k in range(8):PART[permutation(k)]=k//4

def pool(m,v,mask):
    raw=(m+EPS)*mask[:,None];return raw.T@v/raw.sum(0)[:,None]

def head(psi,chi,q,v,mask,w,b,d):
    # All effects are evaluated on the same token state family, not two encoders.
    masses=[(psi*(torch.as_tensor(PART)==j))@q.T.conj() for j in (0,1)]
    ident=masses[0].abs().square()+masses[1].abs().square()
    m=(.25+.5*(psi@chi.conj()).abs().square())
    up=pool(torch.stack((m,1-m),axis=1),v,mask)
    ur=pool(ident,v,mask);u=(ur*w).sum(-1)+b
    g=(d*(up[0]-up[1])).sum();l=torch.cat((torch.zeros_like(g)[None],g+torch.log_softmax(u,dim=0)))
    return ident,m,up,ur,u,g,l

def dense_check(psi,chi,q,v,mask,w,b,d):
    p=psi.detach().numpy(); c=chi.numpy();q=q.numpy();v=v.numpy();w=w.numpy();b=b.numpy();dg=d.numpy();ms=mask.numpy()
    P=[np.diag(PART==j) for j in (0,1)]
    ech=np.eye(8)/4+np.outer(c,c.conj())/2
    effects=[sum(Z@np.outer(qr,qr.conj())@Z for Z in P) for qr in q]
    densities=[np.outer(pt,pt.conj()) for pt in p]
    m=np.array([np.trace(ech@r).real for r in densities]);a=np.array([[np.trace(e@r).real for e in effects] for r in densities])
    def independent(z):
        out=[]
        for j in range(z.shape[1]):
            wt=np.array([z[t,j]+EPS for t in range(len(ms)) if ms[t]])
            out.append(np.average(v[ms],axis=0,weights=wt))
        return np.array(out)
    vp=independent(np.stack([m,1-m],axis=1)); vr=independent(a)
    s=np.sum(vr*w,axis=1)+b;g=np.dot(vp[0]-vp[1],dg)
    l=np.r_[0.,g+s-np.logaddexp.reduce(s)]
    return a,m,vp,vr,s,g,l

def maxerr(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

def test(source):
    ci=original(source);rng=np.random.default_rng(20261128);td=torch.float64
    g0=lambda shape:torch.tensor(rng.normal(size=shape),dtype=td)
    a=g0((7,2,3,2));ch=ci(g0((2,3,2)));q=ci(g0((39,2,3,2)))
    v=g0((7,64));b=g0(39);w=g0((39,64));dg=g0(64);ms=torch.tensor([1,1,0,1,1,0,1],dtype=torch.bool)
    ps=ci(a);r=head(ps,ch,q,v,ms,w,b,dg);r2=dense_check(ps,ch,q,v,ms,w,b,dg)
    names=('identity_mass','presence_mass','presence_full64','identity_full64','positive_scores','presence_logodds','joint40_logits')
    error={name:maxerr(x.detach(),y) for name,x,y in zip(names,r,r2)}
    pp=r[-1].softmax(0);error['hierarchical_probability']=maxerr(pp[1:],r[-2].sigmoid()*r[-3].softmax(0))
    error['no_relation_probability']=maxerr(pp[0],(-r[-2]).sigmoid())
    m0=r[1][ms];v0=v[ms];mean=m0.mean();cov=((m0-mean)[:,None]*(v0-v0.mean(0))).mean(0)
    error['presence_full64_covariance_identity']=maxerr(r[2][0]-r[2][1],cov*(1+2*EPS)/((mean+EPS)*(1-mean+EPS)))
    # Native angle perturbation: arbitrary RZ offset in final layer on logical wire 0.
    # After final CNOT permutation, this is a sector-relative phase (global phase irrelevant).
    delta=torch.tensor(.37,dtype=td,requires_grad=True)
    e=torch.zeros_like(a);e[0,1,0,1]=1.;one=head(ci(a+delta*e),ch,q,v,ms,w,b,dg)
    dpos=torch.autograd.grad(one[-3].square().sum(),delta,retain_graph=True)[0].item()
    dp=torch.autograd.grad(one[-2],delta)[0].item()
    step=1e-5
    ga=lambda d:head(ci(a+d*e),ch,q,v,ms,w,b,dg)[-2].item()
    error['presence_derivative']=abs(dp-(ga(.37+step)-ga(.37-step))/(2*step))
    error['protected_identity_derivative']=abs(dpos)
    assert abs(dp)>1e-6 and max(error.values())<1e-8
    bad=v.clone();bad[~ms]=1e5
    assert torch.equal(r[-1],head(ps,ch,q,bad,ms,w,b,dg)[-1])

    # Both states are native circuit outputs with finite in-range angles.
    a=torch.zeros((2,2,3,2),dtype=td);a[:,1,0,0]=math.pi/2;a[0,1,0,1]=math.pi/2;a[1,1,0,1]=-math.pi/2
    ps=ci(a);c=ci(a[0]);v0=torch.zeros((2,64),dtype=td);v0[0,0]=1.;v0[1,0]=-1.
    reader=torch.zeros(64,dtype=td);reader[0]=1.;mask=torch.ones(2,dtype=torch.bool)
    A=head(ps,c,q,v0,mask,w,b,reader);B=head(ps.flip(0),c,q,v0,mask,w,b,reader)
    assert maxerr(abs(ps[0])**2,abs(ps[1])**2)<1e-14 and maxerr(A[-3],B[-3])<1e-14
    assert maxerr(A[1].detach(),[.75,.25])<1e-14 and maxerr(B[1].detach(),[.25,.75])<1e-14
    gap=abs(A[-2].item()-B[-2].item());assert abs(gap-2/(1+2*EPS))<1e-13
    # Same frozen measurement, erase only inter-sector coherence at input.
    deph_m=.25+.5*sum((((torch.as_tensor(PART)==j)*ps)@c.conj()).abs().square() for j in (0,1))
    assert maxerr(deph_m,[.5,.5])<1e-14
    assert (pool(torch.stack([deph_m,1-deph_m],dim=1),v0,mask).diff(dim=0)==0).all()

    # Product-CE factorization alone is an identity, while the forward separation is testable.
    for y in (0,1,39):
        ce=-r[-1].log_softmax(0)[y];main=torch.nn.functional.softplus((-1 if y else 1)*r[-2])
        if y:main=main-r[-3].log_softmax(0)[y-1]
        assert maxerr(ce,main)<1e-14
    return dict(status='PASS_NATIVE_CIRCUIT_PHASE_SEPARATION_SYNTHETIC',code_sha256=sig(__file__),native_source_sha256=sig(source),
      columns=40,signed_value=64,witness_presence_logits=[A[-2].item(),B[-2].item()],witness_gap=gap,
      witness_identity_maxdiff=maxerr(A[-3],B[-3]),block_dephase_presence_logits=[0.,0.],
      learned_D_note='Matched block-D keeps trained within-sector overlaps; .50 is witness-specific.',
      maximum_errors=error,presence_phase_derivative=dp,task_rows=0,task_forwards=0,updates=0,gpu_allocations=0,
      torch_version=torch.__version__,scope='CPU synthetic; native circuit AST verbatim, not old checkpoint, no natural task representation or utility claim')

if __name__=='__main__':print(json.dumps(test(sys.argv[1]),indent=2,allow_nan=False))
