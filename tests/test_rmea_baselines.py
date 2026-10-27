"""Scientific contract checks for common-feature heads, on CPU and assigned GPU."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import unittest
import torch
from torch.nn import functional as F
from baselines.retacred.frozen_feature_head import FrozenFeatureHead,MODES
from baselines.retacred.common import metrics
from scripts.audit_rmea_tokenization import marked_words


class CommonHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(1)
        self.device='cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        self.h=torch.randn(2,5,1024,device=self.device)
        self.roles=torch.randn(2,5,4,device=self.device)
        self.mask=torch.tensor([[True,True,True,False,False],[True]*5],device=self.device)

    def test_padding_is_inert(self):
        changed=self.h.clone();changed[0,3:]=123
        for mode in MODES:
            head=FrozenFeatureHead(mode).to(self.device)
            a,weights=head(self.h,self.roles,self.mask);b,_=head(changed,self.roles,self.mask)
            torch.testing.assert_close(a,b)
            self.assertEqual(float(weights[0,:,3:].abs().sum()),0.)
            torch.testing.assert_close(weights.sum(-1),torch.ones(2,40,device=self.device))

    def test_uniform_pooling_matches_arithmetic_mean(self):
        head=FrozenFeatureHead('uniform').to(self.device)
        logits,_=head(self.h,self.roles,self.mask)
        for i,length in enumerate((3,5)):
            mean=head.value(self.h[i,:length]).mean(0)
            expected=(mean[None,:]*head.decoder).sum(-1)+head.bias
            torch.testing.assert_close(logits[i],expected)

    def test_attention_parameters_receive_ce_gradients(self):
        for mode in MODES[1:]:
            head=FrozenFeatureHead(mode).to(self.device)
            logits,_=head(self.h,self.roles,self.mask)
            F.cross_entropy(logits,torch.tensor([1,12],device=self.device)).backward()
            for parameter in (head.query,head.key.weight):
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.norm()),0.)

    def test_common_initialization(self):
        states=[]
        for mode in MODES:
            torch.manual_seed(13);head=FrozenFeatureHead(mode)
            states.append([p.detach().clone() for p in (head.value.weight,head.value.bias,head.decoder,head.bias)])
        for state in states[1:]:
            for a,b in zip(states[0],state):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_empty_mask_rejected(self):
        head=FrozenFeatureHead('uniform').to(self.device)
        with self.assertRaises(ValueError):head(self.h,self.roles,torch.zeros_like(self.mask))

    def test_label_blind_entity_construction(self):
        row=dict(tokens=['one','two','three','four'],subject=[0,1],object=[1,3],label='a')
        a=marked_words(row);row.update(label='b',metadata={'id':'different'})
        self.assertEqual(a,marked_words(row))
        self.assertEqual([i for i in a[1] if i is not None],list(range(4)))
        self.assertEqual(len(a[0]),8)

    def test_macro_keeps_zero_support_class(self):
        labels=['no_relation']+[str(i) for i in range(39)]
        report=metrics([0,0],[0,0],labels,None)
        self.assertAlmostEqual(report['macro_f1'],1/40)
        self.assertEqual(report['micro_f1'],0.)
        self.assertEqual(report['all_label_micro_f1'],1.)


if __name__=='__main__':unittest.main()
