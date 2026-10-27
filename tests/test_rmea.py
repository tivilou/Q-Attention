import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import math
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import RMEAHead,circuit,replay,MODES,EPS
from baselines.retacred.frozen_feature_head import FrozenFeatureHead


class RMEATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(13)
        self.device='cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        self.h=torch.randn(2,7,1024,device=self.device)*.2
        self.roles=torch.zeros(2,7,4,device=self.device)
        self.roles[:,0,2]=1;self.roles[:,2,3]=1
        self.mask=torch.ones(2,7,dtype=torch.bool,device=self.device);self.mask[0,6]=False

    def test_circuit_against_independent_dense_reference(self):
        rng=np.random.default_rng(9);a=rng.normal(size=(2,3,2))
        state=np.eye(8,dtype=complex)[0]
        for layer in range(2):
            for q in range(3):
                y,z=a[layer,q];ry=np.array([[np.cos(y/2),-np.sin(y/2)],[np.sin(y/2),np.cos(y/2)]])
                rz=np.diag([np.exp(-.5j*z),np.exp(.5j*z)]);g=np.ones((1,1),complex)
                for wire in range(3):g=np.kron(g,rz@ry if wire==q else np.eye(2))
                state=g@state
            for c,t in ((0,1),(1,2),(2,0)):
                matrix=np.zeros((8,8))
                for i in range(8):matrix[i ^ ((1<<(2-t)) if i & (1<<(2-c)) else 0),i]=1
                state=matrix@state
        actual=circuit(torch.tensor(a,device=self.device)).cpu().numpy()
        np.testing.assert_allclose(actual,state,atol=1e-12,rtol=1e-12)

    def test_complex_state_norm_and_measurement_bounds(self):
        head=RMEAHead('quantum').to(self.device);d=head(self.h,self.roles,self.mask,True)
        torch.testing.assert_close(d['psi'].abs().square().sum(-1),torch.ones_like(self.mask,dtype=torch.float))
        self.assertTrue(bool((d['probability']>=0).all() & (d['probability']<=1+1e-6).all()))
        self.assertGreater(float(d['psi'].imag.abs().max()),1e-4)
        for mode in MODES:
            h=RMEAHead(mode).to(self.device);_,a=h(self.h,self.roles,self.mask)
            torch.testing.assert_close(a.sum(-1),torch.ones(2,40,device=self.device))
            self.assertEqual(float(a[0,:,-1].abs().sum()),0.)

    def test_exact_classical_compatibility_and_common_init(self):
        shared=[]
        for mode in MODES:
            torch.manual_seed(19);head=RMEAHead(mode).to(self.device)
            shared.append([p.detach().clone() for p in (head.value.weight,head.value.bias,head.decoder,head.bias)])
            if mode in MODES[:3]:
                torch.manual_seed(19);old=FrozenFeatureHead(mode).to(self.device)
                for a,b in zip(head(self.h,self.roles,self.mask),old(self.h,self.roles,self.mask)):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
        for row in shared[1:]:
            for a,b in zip(shared[0],row):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_ce_reaches_both_angle_families(self):
        for mode in MODES[3:]:
            torch.manual_seed(13);head=RMEAHead(mode).to(self.device)
            d=head(self.h,self.roles,self.mask,True)
            F.cross_entropy(d['logits'],torch.tensor([1,12],device=self.device)).backward()
            for p in (head.prepare.weight,head.effect):
                self.assertTrue(bool(torch.isfinite(p.grad).all()));self.assertGreater(float(p.grad.norm()),1e-7)

    def test_dephasing_is_diagonal_density_measurement(self):
        head=RMEAHead('dephased').to(self.device);d=head(self.h,self.roles,self.mask,True)
        rho=torch.diag_embed(d['psi'].abs().square()).to(d['psi'].dtype)
        e=d['phi'][:,:,None]*d['phi'][:,None,:].conj()
        expected=torch.einsum('rij,btji->brt',e,rho).real
        torch.testing.assert_close(d['probability'],expected)

    def test_sensitivity_matches_autograd_and_zero_utility_case(self):
        head=RMEAHead('quantum').to(self.device);d=head(self.h,self.roles,self.mask,True)
        p=d['probability'].detach().clone().requires_grad_();m=self.mask[:,None,:]
        raw=(p+EPS)*m;a=raw/raw.sum(-1,keepdim=True)
        logits=(a*d['utilities'].detach()).sum(-1)+head.bias.detach()
        g=torch.autograd.grad(logits.sum(),p,retain_graph=True)[0]
        torch.testing.assert_close(g,d['sensitivity'].detach(),atol=3e-5,rtol=3e-5)
        u=torch.ones_like(p)*2;s=(a*u).sum(-1)
        torch.testing.assert_close(torch.autograd.grad(s.sum(),p)[0],torch.zeros_like(p),atol=1e-5,rtol=0)

    def test_replay_mass_entity_mask_and_identity(self):
        permutations=[]
        for mode in MODES:
            head=RMEAHead(mode).to(self.device);d=head(self.h,self.roles,self.mask,True)
            r=replay(head,d,self.mask,self.roles,['a','b']);permutations.append(r['permutation'])
            torch.testing.assert_close(r['shuffled_alpha'].sum(-1),d['alpha'].sum(-1))
            fixed=(~r['eligible_context_mask'])[:,None,:].expand_as(d['alpha'])
            torch.testing.assert_close(r['shuffled_alpha'][fixed],d['alpha'][fixed],rtol=0,atol=0)
            torch.testing.assert_close(r['identity_residual'],torch.zeros_like(d['logits']),atol=2e-5,rtol=0)
            if mode=='uniform':torch.testing.assert_close(r['shuffled_logits'],d['logits'],atol=2e-5,rtol=0)
        for a in permutations[1:]:self.assertTrue(torch.equal(a,permutations[0]))

    def test_padding_inert_and_short_context_unchanged(self):
        head=RMEAHead('quantum').to(self.device)
        x=self.h.clone();x[0,6]=100
        torch.testing.assert_close(head(x,self.roles,self.mask)[0],head(self.h,self.roles,self.mask)[0])
        mask=self.mask.clone();mask[:]=False;mask[:,[0,2,3]]=True
        d=head(self.h,self.roles,mask,True);r=replay(head,d,mask,self.roles,['a','b'])
        torch.testing.assert_close(r['shuffled_alpha'],d['alpha'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
