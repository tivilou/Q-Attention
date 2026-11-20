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

    @torch.no_grad()
    def test_full_rank_overcomplete_separation_survives_key_perturbations(self):
        """Exact generalized-eigenvalue competitors, not sampled-query maxima."""
        eye=torch.eye(8,dtype=torch.complex128,device=self.device)
        v=math.sqrt(3)/2*eye[0]+.5*eye[1]
        base=torch.cat((torch.stack((eye[0],v,eye[1])),eye[2:]),0)
        mask=torch.ones(1,9,dtype=torch.bool,device=self.device)
        frame=base.T@base.conj();gram=base.conj()@base.T
        graph=gram*gram.abs().square();graph.fill_diagonal_(0)
        unitary=torch.matrix_exp(-.25j*graph/torch.linalg.eigvalsh(graph).abs().max())
        w=(unitary.conj()@base)[1];query=torch.linalg.solve(frame,w);query/=query.norm()
        phi=query[None].expand(40,-1)
        q=math.sin(.25)**2;ell=.5+3*q/56;beta=.25+9*q/112
        predicted=(ell*ell+EPS*beta)/(ell+9*EPS*beta)
        ceiling=(1+EPS)/(2+9*EPS)
        self.assertGreater(predicted-ceiling,.003)

        def bound(effect,denominator):
            inverse=torch.linalg.inv(torch.linalg.cholesky(denominator))
            whitened=inverse@effect@inverse.mH
            return float(torch.linalg.eigvalsh((whitened+whitened.mH)/2)[-1])

        generator=torch.Generator(device=self.device);generator.manual_seed(20261121)
        angle=2*math.asin(9.99e-6/2);smallest_margin=1.;largest_linear_error=0.
        for case in range(65):
            psi=base
            if case:
                tangent=torch.randn(base.shape,dtype=base.dtype,device=self.device,generator=generator)
                tangent-=base*(base.conj()*tangent).sum(-1,keepdim=True)
                tangent/=tangent.norm(dim=-1,keepdim=True)
                psi=math.cos(angle)*base+math.sin(angle)*tangent
            self.assertLessEqual(float((psi-base).norm(dim=-1).max()),1e-5)
            physical=psi.T@psi.conj()
            self.assertGreater(float(torch.linalg.eigvalsh(physical)[0]),.9998)
            result=route(psi[None],phi,mask,kind='cubic_coherent')
            projectors=psi[:,:,None]*psi.conj()[:,None,:]
            denominator=physical+9*EPS*eye
            native=bound(projectors[1]+EPS*eye,denominator)
            weights=result['unitary'][0,1].abs().square()
            dephased=bound((weights[:,None,None]*projectors).sum(0)+EPS*eye,denominator)
            actual=float(result['alpha'][0,0,1]);margin=actual-max(native,dephased)
            self.assertGreater(margin,.0009);smallest_margin=min(smallest_margin,margin)
            if not case:
                self.assertAlmostEqual(actual,predicted,delta=1e-12)
                self.assertAlmostEqual(native,ceiling,delta=1e-12)
                self.assertLessEqual(dephased,ceiling+1e-12)
            linear=route(psi[None],phi,mask,kind='linear_coherent')
            rotation=torch.matrix_exp(-.25j*(physical-eye)/linear['spectral_scale'][0])
            reference=route(psi[None],(rotation@phi.T).T,mask,kind='native')['alpha']
            error=float((linear['alpha']-reference).abs().max())
            self.assertLess(error,1e-11);largest_linear_error=max(largest_linear_error,error)
        print(p.json.dumps(dict(status='FULL_RANK_STABLE_SEPARATION_PASS',cases=65,dimension=8,tokens=9,
            base_coordinate_gain=predicted-ceiling,minimum_coordinate_margin=smallest_margin,
            maximum_linear_query_absorption_error=largest_linear_error,task_utility=False)),flush=True)

    @torch.no_grad()
    def test_finite_actual_pqc_realizes_the_stable_full_rank_witness(self):
        """Use one finite shared prepare layer and the actual source head."""
        eye=torch.eye(8,dtype=torch.complex128,device=self.device)
        v=math.sqrt(3)/2*eye[0]+.5*eye[1]
        base=torch.cat((torch.stack((eye[0],v,eye[1])),eye[2:]),0)
        delta=1e-5;eta=delta/2
        angles=torch.zeros(9,2,3,2,dtype=torch.float64,device=self.device)
        for j in range(8):
            x,y,z=(j>>2)&1,(j>>1)&1,j&1
            row=0 if j==0 else j+1
            angles[row,0,:,0]=torch.tensor([x^y,z,x],dtype=angles.dtype,device=self.device)*(math.pi-eta)
        angles[1,0,1,0]=math.pi/3
        raw=torch.atanh(angles/math.pi).reshape(9,12)
        head=TrainableTokenTransportHead('cubic_coherent').double().to(self.device)
        head.prepare.weight.zero_();head.prepare.bias.zero_()
        head.prepare.weight[:,:9].copy_(raw.T)
        u=torch.stack((math.cos(.25)*math.sqrt(3)/4*eye[0]+1j*math.sin(.25)*math.sqrt(21)/8*eye[0],
                       math.cos(.25)/4*eye[1]-1j*math.sin(.25)/(8*math.sqrt(7))*eye[1])).sum(0)
        query=u/u.norm()
        theta=2*math.atan2(float(query[1].abs()),float(query[0].abs()))
        phase=float(torch.angle(query[1])-torch.angle(query[0]))
        self.assertTrue(0<theta<math.pi and -math.pi<phase<0)
        head.effect.zero_();head.effect[:,0,1,0]=math.atanh(theta/math.pi)
        head.effect[:,0,1,1]=math.atanh(phase/math.pi)
        h=torch.zeros(1,9,1024,dtype=torch.float64,device=self.device)
        h[0,:,:9]=torch.eye(9,dtype=torch.float64,device=self.device)
        roles=torch.zeros(1,9,4,dtype=h.dtype,device=self.device)
        mask=torch.ones(1,9,dtype=torch.bool,device=self.device)
        self.assertTrue(all(torch.isfinite(t).all() for t in head.parameters()))
        detail=head(h,roles,mask,details=True);psi=detail['psi'][0];phi=detail['phi'][0]
        key_error=float((psi-base).norm(dim=-1).max());self.assertLess(key_error,delta)
        aligned=phi*torch.vdot(query,phi).conj()/torch.vdot(query,phi).abs()
        query_error=float((aligned-query).norm());self.assertLess(query_error,1e-12)
        frame=psi.T@psi.conj();minimum_eigenvalue=float(torch.linalg.eigvalsh(frame)[0])
        self.assertGreater(minimum_eigenvalue,.9998)
        projectors=psi[:,:,None]*psi.conj()[:,None,:];denominator=frame+9*EPS*eye
        inverse=torch.linalg.inv(torch.linalg.cholesky(denominator))
        weights=detail['transport']['unitary'][0,1].abs().square()
        bounds=[]
        for effect in (projectors[1]+EPS*eye,(weights[:,None,None]*projectors).sum(0)+EPS*eye):
            normalized=inverse@effect@inverse.mH
            bounds.append(float(torch.linalg.eigvalsh((normalized+normalized.mH)/2)[-1]))
        actual=float(detail['alpha'][0,0,1]);margin=actual-max(bounds)
        self.assertGreater(margin,.0009)
        print(p.json.dumps(dict(status='FINITE_PQC_STABLE_WITNESS_PASS',keys=9,dimension=8,
            maximum_key_error=key_error,query_phase_aligned_error=query_error,
            minimum_frame_eigenvalue=minimum_eigenvalue,maximum_prepare_raw=float(raw.abs().max()),
            native_ceiling=bounds[0],dephased_ceiling=bounds[1],coherent_probability=actual,
            simultaneous_margin=margin,feature_scope='Synthetic nine independent feature rows; no data or training',
            task_utility=False)),flush=True)


    @torch.no_grad()
    def test_task_aligned_loss_bound_and_dark_token_dephasing_null(self):
        """Known readout tools must agree with the actual fixed route."""
        eye=torch.eye(8,dtype=torch.complex128,device=self.device)
        v=math.sqrt(3)/2*eye[0]+.5*eye[1]
        psi=torch.cat((torch.stack((eye[0],v,eye[1])),eye[2:]),0)
        mask=torch.ones(1,9,dtype=torch.bool,device=self.device)
        initial=route(psi[None],eye[0,None].expand(40,-1),mask,kind='cubic_coherent')
        frame=psi.T@psi.conj();w=(initial['unitary'][0].conj()@psi)[1]
        query=torch.linalg.solve(frame,w);query/=query.norm()
        phi=query[None].expand(40,-1).clone()
        routes={k:route(psi[None],phi,mask,kind=k) for k in KINDS}
        projectors=psi[:,:,None]*psi.conj()[:,None,:]
        denominator=frame+9*EPS*eye;chol=torch.linalg.cholesky(denominator)
        inverse=torch.linalg.inv(chol)
        utilities=torch.zeros(40,9,dtype=torch.float64,device=self.device);utilities[1,1]=1
        native_ceiling=(1+EPS)/(2+9*EPS)
        coherent_probability=float(routes['cubic_coherent']['alpha'][0,1,1])
        bias=torch.full((40,),-2.,dtype=torch.float64,device=self.device)
        bias[0]=(coherent_probability+native_ceiling)/2;bias[1]=0
        target=torch.tensor([1],device=self.device)
        losses={};ceilings={};attainment_errors={}
        for kind in ('native','cubic_initial_dephased'):
            effects=projectors+EPS*eye
            if kind=='cubic_initial_dephased':
                weights=routes[kind]['unitary'][0].abs().square()
                effects=torch.einsum('ij,jab->iab',weights.to(psi.dtype),effects)
            readout=torch.einsum('rt,tab->rab',utilities.to(psi.dtype),effects)
            normalized=inverse@readout@inverse.mH
            values,vectors=torch.linalg.eigh((normalized+normalized.mH)/2)
            lower=bias+values[:,0];upper=bias+values[:,-1]
            best=lower.clone();best[1]=upper[1]
            losses[kind]=float(torch.nn.functional.cross_entropy(best[None],target))
            ceilings[kind]=float(upper[1])
            # All rival utilities are zero; only the true-class extremizer matters.
            optimum=torch.linalg.solve(chol.mH,vectors[1,:,-1]);optimum/=optimum.norm()
            extremizers=phi.clone();extremizers[1]=optimum
            attained=(route(psi[None],extremizers,mask,kind=kind)['alpha'][0]*utilities).sum(-1)+bias
            attainment_errors[kind]=float((attained-best).abs().max())
            self.assertLess(attainment_errors[kind],1e-12)
        actual=(routes['cubic_coherent']['alpha'][0]*utilities).sum(-1)+bias
        ce=float(torch.nn.functional.cross_entropy(actual[None],target))
        self.assertGreater(min(losses.values())-ce,1e-4)
        self.assertEqual(int(actual.argmax()),1)
        self.assertLess(ceilings['native'],float(bias[0]))
        self.assertLess(ceilings['cubic_initial_dephased'],float(bias[0]))

        keys=psi[:3];m=torch.ones(1,3,dtype=torch.bool,device=self.device)
        fixed=eye[0,None].expand(40,-1)
        coherent=route(keys[None],fixed,m,kind='cubic_coherent')['alpha'][0,0,2]
        dephased=route(keys[None],fixed,m,kind='cubic_initial_dephased')['alpha'][0,0,2]
        r=3*math.sqrt(3)/(2*math.sqrt(7));s=1/(2*math.sqrt(7))
        floor=EPS/(1.75+3*EPS)
        excess=s*s*((1-math.cos(.25))**2*r*r+math.sin(.25)**2*.75)/(1.75+3*EPS)
        self.assertAlmostEqual(float(coherent),floor+excess,delta=1e-12)
        self.assertAlmostEqual(float(coherent),float(dephased),delta=1e-12)
        isolated=keys.clone();isolated[2]=eye[2]
        isolated_probability=float(route(isolated[None],fixed,m,kind='cubic_coherent')['alpha'][0,0,2])
        self.assertAlmostEqual(isolated_probability,floor,delta=1e-12)
        print(p.json.dumps(dict(status='TASK_ALIGNED_LIMITS_PASS',coherent_ce=ce,
            query_relaxed_minimum_ce=losses,minimum_ce_separation=min(losses.values())-ce,
            maximum_extremizer_attainment_error=max(attainment_errors.values()),
            dark_token_coherent_probability=float(coherent),dark_token_dephased_probability=float(dephased),
            isolated_token_probability=isolated_probability,dark_token_epsilon_floor=floor,
            scope='Synthetic fixed keys/utilities; arbitrary physical-query relaxation; no text or training',
            task_utility=False)),flush=True)


if __name__=='__main__':
    unittest.main()
