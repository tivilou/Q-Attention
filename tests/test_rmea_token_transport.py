"""Mechanism fixtures for the scheduler-assigned GPU environment only."""
import math
import unittest
import torch
from baselines.retacred.rmea import RMEAHead, EPS
from baselines.retacred.rmea_token_transport import TrainableTokenTransportHead, route, KINDS
from baselines.retacred.rmea_token_transport_diagnostics import inspect_unit, gauge_gap
from baselines.retacred import rmea_token_transport_protocol as p


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(701); torch.set_num_threads(1)
        self.device='cuda'

    def states(self, count):
        z=torch.randn(count,8,dtype=torch.complex128,device=self.device)
        return z/z.norm(dim=-1,keepdim=True)

    def test_native_matches_canonical_head_and_has_same_parameters(self):
        torch.manual_seed(13); base=RMEAHead('quantum').double().to(self.device)
        head=TrainableTokenTransportHead('native').double().to(self.device);head.load_state_dict(base.state_dict())
        h=torch.randn(2,7,1024,dtype=torch.float64,device=self.device);r=torch.randn(2,7,4,dtype=torch.float64,device=self.device)
        m=torch.tensor([[1]*7,[1]*5+[0]*2],dtype=torch.bool,device=self.device)
        for a,b in zip(base(h,r,m),head(h,r,m)):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        self.assertEqual(sum(v.numel() for v in head.parameters()),81028)

    def test_phase_permutation_and_padding_invariance(self):
        psi=self.states(6);phi=self.states(40);m=torch.ones(1,6,dtype=torch.bool,device=self.device)
        phase=torch.exp(1j*torch.arange(6,dtype=torch.float64,device=self.device));perm=torch.tensor([3,1,5,0,4,2],device=self.device)
        padded=torch.cat((psi,self.states(3)),0)[None];pm=torch.tensor([[1]*6+[0]*3],dtype=torch.bool,device=self.device)
        for kind in KINDS:
            a=route(psi[None],phi,m,kind=kind)['alpha']
            b=route((psi*phase[:,None])[None],phi,m,kind=kind)['alpha']
            c=route(psi[perm][None],phi,m,kind=kind)['alpha']
            d=route(padded,phi,pm,kind=kind)['alpha']
            torch.testing.assert_close(a,b,atol=1e-11,rtol=0)
            torch.testing.assert_close(a[...,perm],c,atol=1e-11,rtol=0)
            torch.testing.assert_close(a,d[...,:6],atol=1e-11,rtol=0)
            self.assertEqual(float(d[...,6:].abs().max()),0.)

    def test_two_token_cubic_has_physical_query_absorption(self):
        psi=self.states(2);phi=self.states(40);m=torch.ones(1,2,dtype=torch.bool,device=self.device)
        d=route(psi[None],phi,m,kind='cubic_coherent')
        factor=(psi[0].conj()@psi[1]).abs().square()
        physical=psi.T@psi.conj()
        v=torch.matrix_exp(-.25j*factor*(physical-torch.eye(8,device=self.device))/d['spectral_scale'][0])
        transformed=(v@phi.T).T
        native=route(psi[None],transformed,m,kind='native')['alpha']
        torch.testing.assert_close(d['alpha'],native,atol=1e-11,rtol=0)
        self.assertLess(float(gauge_gap(psi,d['unitary'][0])),1e-11)

    def test_asymmetric_three_key_theorem_with_smoothing(self):
        e=torch.zeros(8,dtype=torch.complex128,device=self.device);e[0]=1
        f=e.roll(1);v=math.sqrt(3)/2*e+.5*f;psi=torch.stack((e,v,f))
        m=psi.T@psi.conj();graph=(psi.conj()@psi.T);graph=graph*graph.abs().square();graph.fill_diagonal_(0)
        u=torch.matrix_exp(-.25j*graph/torch.linalg.eigvalsh(graph).abs().max())
        # The middle transported effective key supplies the established witness.
        w=(u.conj()@psi)[1];z=torch.linalg.solve(m[:2,:2],w[:2]);phi=e.clone();phi[:2]=z;phi[2:]=0;phi/=phi.norm()
        detail=inspect_unit(psi,phi[None].expand(40,-1),torch.zeros(40,3,dtype=torch.float64,device=self.device),
                            torch.zeros(40,dtype=torch.float64,device=self.device),1)
        ceiling=(1+EPS)/(2+3*EPS)
        alpha=route(psi[None],phi[None].expand(40,-1),torch.ones(1,3,dtype=torch.bool,device=self.device),kind='cubic_coherent')['alpha']
        self.assertGreater(float(alpha[0,0,1]),ceiling+1e-8)
        self.assertTrue(detail['simultaneous_ceiling_witnesses'])
        # Both real three-node path generators are bipartite. Evolution leaves
        # diag(U G U†)=1 here, so the sufficient row-norm certificate is silent
        # even though the coordinate probability ceiling is strictly exceeded.
        self.assertFalse(detail['routing']['cubic_coherent']['phase_gauge_obstruction'])
        self.assertFalse(detail['routing']['linear_coherent']['phase_gauge_obstruction'])

    def test_full_head_gradients_and_one_finite_difference(self):
        head=TrainableTokenTransportHead('cubic_coherent').double().to(self.device)
        h=torch.randn(2,5,1024,dtype=torch.float64,device=self.device);r=torch.zeros(2,5,4,dtype=torch.float64,device=self.device)
        mask=torch.ones(2,5,dtype=torch.bool,device=self.device);y=torch.tensor([1,5],device=self.device)
        loss=torch.nn.functional.cross_entropy(head(h,r,mask)[0],y);loss.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters()))
        index=(0,0);actual=float(head.prepare.weight.grad[index]);epsilon=1e-5
        with torch.no_grad():
            initial=float(head.prepare.weight[index]);values=[]
            for sign in (1,-1):
                head.prepare.weight[index]=initial+sign*epsilon
                values.append(float(torch.nn.functional.cross_entropy(head(h,r,mask)[0],y)))
            head.prepare.weight[index]=initial
        self.assertAlmostEqual(actual,(values[0]-values[1])/(2*epsilon),delta=2e-7)

    def test_finite_depth_two_head_realizes_designated_middle_witness(self):
        head=TrainableTokenTransportHead('cubic_coherent').double().to(self.device)
        h=torch.zeros(1,3,1024,dtype=torch.float64,device=self.device)
        h[0,torch.arange(3),torch.arange(3)]=1
        roles=torch.zeros(1,3,4,dtype=torch.float64,device=self.device)
        mask=torch.ones(1,3,dtype=torch.bool,device=self.device)
        b0=torch.zeros(8,dtype=torch.complex128,device=self.device);b0[0]=1
        b3=torch.zeros_like(b0);b3[3]=1
        e=(b0-b3)/math.sqrt(2);f=(b0+b3)/math.sqrt(2)
        expected=torch.stack((e,math.sqrt(3)/2*e+.5*f,f))
        gram=expected.conj()@expected.T;graph=gram*gram.abs().square();graph.fill_diagonal_(0)
        unitary=torch.matrix_exp(-.25j*graph/torch.linalg.eigvalsh(graph).abs().max())
        physical=expected.T@expected.conj();w=(unitary.conj()@expected)[1]
        ix=torch.tensor([0,3],device=self.device)
        solved=torch.linalg.solve(physical[ix][:,ix],w[ix]);solved/=solved.norm()
        query=torch.zeros_like(b0);query[ix]=solved
        y=2*math.atan2(float(solved[1].abs()),float(solved[0].abs()))
        zeta=(float(solved[1].angle()-solved[0].angle())+math.pi)%(2*math.pi)-math.pi
        self.assertLess(abs(y),math.pi);self.assertLess(abs(zeta),math.pi)
        with torch.no_grad():
            head.prepare.weight.zero_();head.prepare.bias.zero_();head.effect.zero_()
            head.prepare.weight[6,0]=math.atanh(-.5)
            head.prepare.weight[6,1]=math.atanh(-1/6)
            head.prepare.weight[6,2]=math.atanh(.5)
            head.effect[1,1,0,0]=math.atanh(y/math.pi)
            head.effect[1,1,0,1]=math.atanh(zeta/math.pi)
            d=head(h,roles,mask,details=True)
        torch.testing.assert_close((d['psi'][0].conj()*expected).sum(-1).abs(),
                                  torch.ones(3,dtype=torch.float64,device=self.device),atol=1e-12,rtol=0)
        self.assertAlmostEqual(float((d['phi'][1].conj()*query).sum().abs()),1.,delta=1e-12)
        measured=inspect_unit(d['psi'][0],d['phi'],d['utilities'][0],head.bias,1)
        self.assertTrue(any(v['relation']==1 and v['token']==1 for v in measured['simultaneous_ceiling_witnesses']))
        self.assertGreater(float(d['alpha'][0,1,1]),(1+EPS)/(2+3*EPS)+1e-8)

    def test_budget_and_final_state_contract(self):
        cfg=p.config('discovery')
        self.assertEqual(cfg['counts'],{'train':58465,'valid':19584,'test':13418})
        self.assertEqual(cfg['epochs']*((cfg['counts']['train']+31)//32),7312)
        self.assertEqual(cfg['trace_checkpoints'],['final'])
        self.assertEqual(cfg['conditions'],list(KINDS))


if __name__=='__main__':
    unittest.main()
