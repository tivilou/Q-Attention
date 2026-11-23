"""Exact free-state algebra fixtures only. No model/data/optimizer or GPU imports."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
EPS=1e-6

def laws(g,q,u):
 n=len(q);ts=list(itertools.combinations(range(n),3));k=(np.eye(n)+g)/2
 b=[];c=[];d=[];w=[];v=[]
 for t in ts:
  i,j,l=t
  b.append(1-(abs(g[i,j])**2+abs(g[j,l])**2+abs(g[l,i])**2)/4)
  c.append(np.real(g[i,j]*g[j,l]*g[l,i])/4)
  z=np.linalg.det(k[np.ix_(t,t)])
  assert abs(z.imag)<2e-12
  d.append(z.real);w.append(np.prod(q[list(t)]));v.append(np.mean(u[list(t)]))
 b,c,d,w,v=map(np.array,(b,c,d,w,v))
 assert np.max(abs(b+c-d))<3e-12 and np.min(b)>=.25-1e-12 and np.min(d)>=.5-1e-12
 pf=w*d/np.dot(w,d);pb=w*b/np.dot(w,b)
 rho=max(abs(g[i,j]) for i in range(n) for j in range(i))
 dp=rho**3/(4-3*rho*rho);dm=dp if rho<=.5 else 1/26
 assert min(c/b)>=-dm-1e-12 and max(c/b)<=dp+1e-12
 tv=.5*np.sum(abs(pf-pb));tvbound=(np.sqrt(1+dp)-np.sqrt(1-dm))/(np.sqrt(1+dp)+np.sqrt(1-dm))
 assert tv<=tvbound+1e-12
 delta=abs(np.dot(pf-pb,v));assert delta<=2*np.max(abs(u))*tvbound+2e-12
 alpha=np.zeros(n)
 for t,p in zip(ts,pf):alpha[list(t)]+=p/3
 assert alpha.max()<=1/3+1e-12 and abs(alpha.sum()-1)<1e-12
 assert abs(alpha@u-pf@v)<1e-12
 # Known 3-DPP marginal polynomial cross-check, not an implementation contribution.
 l=np.sqrt(q[:,None]*q[None,:])*k;e1=np.trace(l).real;e2=(e1**2-np.trace(l@l).real)/2
 e3=(e1**3-3*e1*np.trace(l@l).real+2*np.trace(l@l@l).real)/6
 inc=np.diag(l@l@l-e1*l@l+e2*l).real/e3
 assert np.max(abs(inc/3-alpha))<2e-10
 return dict(rho=float(rho),tv=float(tv),tv_bound=float(tvbound),score=float(pf@v),pair_score=float(pb@v),cap=float(alpha.max()))

def witness(rho,theta):
 g=np.eye(4,dtype=complex);z=rho*np.exp(1j*theta/3)
 for i,j in ((0,1),(1,2),(2,0)):g[i,j]=z;g[j,i]=z.conjugate()
 c=np.sqrt((1-2*rho)/8);joint=np.zeros((5,5),complex);joint[:4,:4]=g;joint[:4,4]=c;joint[4,:4]=c;joint[4,4]=1
 ev,evec=np.linalg.eigh(joint);assert ev.min()>0
 f=np.sqrt(ev)[:,None]*evec.conj().T;states=f[:,:4].T;phi=f[:,4]
 assert np.max(abs(states.conj()@states.T-g))<2e-12
 a=phi@states.conj().T;den=np.sum(abs(a)**2)+4*EPS
 native=(abs(a)**2+EPS)/den;graph=g-np.eye(4);e,v=np.linalg.eigh(graph);scale=max(abs(e))
 unitary=(v*np.exp(-1j*.25*e/scale))@v.conj().T
 coherent=(abs(a@unitary.T)**2+EPS)/den
 assert max(abs(native-.25))<2e-12 and max(abs(coherent-.25))<2e-12
 out=laws(g,coherent,np.array([0.,0.,0.,1.]))
 d=1-rho*rho/4;t=1-3*rho*rho/4+rho**3*np.cos(theta)/4
 assert abs(out['score']-d/(t+3*d))<2e-12
 out['minimum_joint_eigenvalue']=float(ev.min());out['theta']=float(theta)
 return out

def main():
 rng=np.random.default_rng(20261122);cases=[]
 for n in (3,4,6,9):
  for _ in range(40):
   x=rng.normal(size=(n,8))+1j*rng.normal(size=(n,8));x/=np.linalg.norm(x,axis=1,keepdims=True)
   q=np.exp(rng.uniform(-4,4,n));q/=q.sum();u=rng.uniform(-1,1,n)
   cases.append(laws(x.conj()@x.T,q,u))
 witnesses=[]
 for rho in (.02,.05,.1,.2,.4):
  a=witness(rho,0);b=witness(rho,np.pi);d=1-rho*rho/4
  formula=d*rho**3/2/((4-1.5*rho*rho)**2-rho**6/16)
  assert abs(abs(a['score']-b['score'])-formula)<2e-12
  assert abs(a['pair_score']-b['pair_score'])<2e-12
  witnesses.append(dict(rho=rho,positive=a,negative=b,phase_delta=formula,cubic_ratio=formula/rho**3))
 g=np.full((3,3),-.5,dtype=complex);np.fill_diagonal(g,1);sharp=laws(g,np.ones(3)/3,np.array([1.,0.,0.]))
 assert abs(sharp['score']-1/3)<1e-12
 for rho in (0.1,0.5,1.):
  g=np.full((3,3),rho,dtype=complex);np.fill_diagonal(g,1);laws(g,np.ones(3)/3,np.array([0.,1.,0.]))
 output=dict(schema='q-rmea.three-token-math-fixture.v1',status='pass',scope='mechanism_algebra_only',random_cases=len(cases),witness_pairs=witnesses,negative_ratio_endpoint=-1/26,maximum_observed_tv=max(x['tv'] for x in cases),no_model_training=True,no_task_utility_claim=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'three-token-fixture-result.json').write_bytes((json.dumps(output,indent=2)+'\n').encode())
 print(json.dumps({k:v for k,v in output.items() if k!='witness_pairs'}))
if __name__=='__main__':main()
