"""Label-free, query-free pairing statistic on all valid unordered word pairs."""
NAMES=('S','sameQ_dephased_minus_G','length','purity_pair_mean','AQ','AG','AD')

def compute(h,mask,psi,scaled_gaussian_key):
 import torch
 x=h.double();p=psi.to(torch.complex128);g=scaled_gaussian_key.double()
 n=mask.sum(-1);norm=x.norm(dim=-1)
 # Padded and zero-norm vectors use a computational denominator1; real zero-norm
 # inputs are flagged undefined below, never silently admitted to the statistic.
 u=x/torch.where(norm>0,norm,torch.ones_like(norm))[...,None]
 d=1-torch.bmm(u,u.transpose(1,2))
 size=h.shape[1];ids=torch.arange(size,device=h.device)
 valid=mask[:,:,None]&mask[:,None,:]&(ids[:,None]<ids[None,:])[None]
 count=valid.sum((1,2));den=count.clamp_min(1)
 ordinary=(d*valid).sum((1,2))/den
 K=torch.bmm(p.conj(),p.transpose(1,2)).abs().square()
 pop=p.abs().square();KD=torch.bmm(pop,pop.transpose(1,2))
 v=torch.cat((p.real,p.imag),-1);jv=torch.cat((-p.imag,p.real),-1)
 K16=torch.bmm(v,v.transpose(1,2)).square()+torch.bmm(v,jv.transpose(1,2)).square()
 distance=torch.zeros_like(K)
 for axis in range(g.shape[-1]):distance+=(g[:,:,axis,None]-g[:,None,:,axis]).square()
 KG=torch.exp(-distance)
 undefined=(n<2)|(((norm==0)&mask).any(-1))
 def assoc(kernel):
  mass=(kernel*valid).sum((1,2))
  result=(kernel*d*valid).sum((1,2))/torch.where(mass>0,mass,torch.ones_like(mass))-ordinary
  return result.masked_fill(undefined|(mass<=0),torch.nan),mass
 Aq,mq=assoc(K);Ad,md=assoc(KD);Ag,mg=assoc(KG)
 purity=(K*valid).sum((1,2))/den
 scores=torch.stack((Aq-Ag,Ad-Ag,n.double(),purity,Aq,Ag,Ad),-1)
 equality=(K-K16).abs().masked_fill(~valid,0).amax()
 info=dict(distance=d,KQ=K,KD=KD,KG=KG,valid=valid,pair_count=count,
  masses=torch.stack((mq,md,mg),-1),real16_error=equality,
  zero_norm_rows=((norm==0)&mask).any(-1),short_rows=n<2)
 return scores,info
