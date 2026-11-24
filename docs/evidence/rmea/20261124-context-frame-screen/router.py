"""Known square-root word-frame measurement with matched classical controls."""
import math
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import circuit,EPS
from baselines.retacred.rmea_matched_routers import Head,initialize_head

KINDS=dict(quantum='quantum',quantum_frame='quantum',dephased='dephased',
 dephased_frame='dephased',real12='real12',real12_frame='real12',softmax12='softmax12',uniform='uniform')

class InvSqrt(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):
  eigen,v=torch.linalg.eigh((x+x.mH)/2)
  if not torch.all(eigen>0):raise ValueError('Frame must be positive definite')
  s=eigen.sqrt();ctx.save_for_backward(s,v)
  return (v/s.unsqueeze(-2))@v.mH
 @staticmethod
 def backward(ctx,gradient):
  s,v=ctx.saved_tensors;g=(gradient+gradient.mH)/2
  divided=-1/(s.unsqueeze(-1)*s.unsqueeze(-2)*(s.unsqueeze(-1)+s.unsqueeze(-2)))
  result=v@((v.mH@g@v)*divided)@v.mH
  return (result+result.mH)/2

def frame_attention(key,query,mask,diagonal=False):
 n=mask.sum(-1).to(key.real.dtype)
 if diagonal:
  p=key.abs().square();q=query.abs().square()
  component=(p*mask[...,None]).sum(1)+n[:,None]*EPS
  raw=torch.einsum('rd,btd->brt',q,(p+EPS)/component[:,None,:])
 else:
  gram=torch.einsum('bti,btj->bij',key*mask[...,None],key.conj())
  gram=gram+n[:,None,None]*EPS*torch.eye(key.shape[-1],device=key.device,dtype=key.dtype)
  inverse=InvSqrt.apply(gram)
  whitened=torch.einsum('bij,btj->bti',inverse,key)
  projected=torch.einsum('bij,rj->bri',inverse,query)
  raw=torch.einsum('rd,btd->brt',query.conj(),whitened).abs().square()
  raw=raw+EPS*projected.abs().square().sum(-1)[...,None]
 raw=raw*mask[:,None,:];normalizer=raw.sum(-1,keepdim=True)
 # In exact arithmetic normalizer=1 for unit queries; normalize roundoff only.
 return raw/normalizer,normalizer

class FrameHead(Head):
 def __init__(self,name):
  super().__init__(KINDS[name]);self.frame=name.endswith('_frame')
 def forward(self,h,roles,mask):
  if self.mode=='uniform':
   weights=mask.to(h.dtype);weights=weights/weights.sum(-1,keepdim=True)
   pooled=torch.einsum('bt,btd->bd',weights,self.head.value(h))
   return pooled@self.head.decoder.T+self.head.bias
  if not self.frame:return super().forward(h,roles,mask)
  a=self.head
  if self.mode in ('quantum','dephased'):
   key=circuit((math.pi*torch.tanh(a.prepare(torch.cat((h,roles),-1)))).reshape(h.shape[:2]+(2,3,2)))
   query=circuit(math.pi*torch.tanh(a.effect))
  else:
   key=F.normalize(a.key(torch.cat((h,roles),-1)),dim=-1);query=F.normalize(a.query,dim=-1)
  alpha,_=frame_attention(key,query,mask,self.mode=='dephased')
  return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize(initial,seed):
 out={}
 for name,kind in KINDS.items():
  native=initialize_head(kind,initial,dict(seed=seed,training_dtype='float64'))
  model=FrameHead(name).double();model.load_state_dict(native.state_dict());out[name]=model
 return out

def make_head(name):return FrameHead(name)

def fixtures(models,inputs):
 inputs=tuple(x.cpu() for x in inputs);h,roles,mask=inputs;errors=[]
 uniform=models['uniform'];z=uniform(*inputs)
 expected=(uniform.head.value(h)*mask[...,None]).sum(1)/mask.sum(1)[:,None]
 uniform_error=float((z-(expected@uniform.head.decoder.T+uniform.head.bias)).abs().max())
 assert uniform_error<1e-12
 for name in ('quantum','dephased','real12'):
  assert all(torch.equal(x,y) for x,y in zip(models[name].state_dict().values(),models[name+'_frame'].state_dict().values()))
 for name in ('quantum_frame','dephased_frame','real12_frame'):
  m=models[name]
  with torch.no_grad():
   z=m(*inputs)
   padded=(F.pad(h,(0,0,0,3)),F.pad(roles,(0,0,0,3)),F.pad(mask,(0,3)))
   errors.append(float((z-m(*padded)).abs().max()))
   duplicate=tuple(torch.cat((v,v),dim=1) for v in inputs)
   errors.append(float((z-m(*duplicate)).abs().max()))
 # Repeated eigenvalues are intentional: differentiating eigenvectors directly
 # is singular here. The Loewner derivative has a finite repeated-eigen limit.
 x=(2*torch.eye(3,dtype=torch.complex128)).requires_grad_()
 k=torch.tensor([[1,1j,0],[-1j,2,1],[0,1,3]],dtype=x.dtype)
 value=(InvSqrt.apply(x)*k.conj()).sum().real
 grad=torch.autograd.grad(value,x)[0]
 derivative_error=float((grad+k/(2*2**1.5)).abs().max());assert derivative_error<1e-12
 direction=k/4;delta=1e-6
 fd=float((((InvSqrt.apply(x.detach()+delta*direction)-InvSqrt.apply(x.detach()-delta*direction))*k.conj()).sum().real)/(2*delta))
 analytic=float((grad.conj()*direction).sum().real);fd_error=abs(fd-analytic)
 assert max(errors)<1e-8 and fd_error<1e-7
 return dict(status='PASS',uniform_double_reference_error=uniform_error,padding_and_duplicate_max_error=max(errors),repeated_eigen_derivative_error=derivative_error,
             complex_finite_difference_error=fd_error,scope='Only new word-frame forward/mask/Fréchet derivative; real updates verified during fixed training. No lifecycle matrix.')

def summarize(rows,logits,labels,cfg):
 q=rows['quantum_frame'];deltas={k:q['all40_macro_f1']-v['all40_macro_f1'] for k,v in rows.items() if k!='quantum_frame'}
 return dict(results=rows,all40_macro_deltas=deltas,gates={k:v>0 for k,v in deltas.items()},
  gate='PASS_FOR_NEW_FULL_PROTOCOL_DESIGN_ONLY' if all(v>0 for v in deltas.values()) else 'NO_PROMOTION_EXACT_SCREEN',
  L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',
  scope='Known PGM/word-frame transformation, no task-optimality or novelty theorem. Eight fixed matched CE arms; exposed TRAIN feasibility only. No coefficient/seed/budget/control/metric rescue.')
