"""Label-free mechanism acceptance. No dataset, training loop or model selection."""
import math
import unittest

import torch

from baselines.retacred.rmea import EPS, RMEAHead
from baselines.retacred.rmea_entity import entity_inputs
from baselines.retacred.rmea_joint import (CROSS_EDGES, LOCAL_EDGES, VARIANTS,
    make_head, normalized_attention, register_state, separable_floor)


def dense_oracle(angles, edges):
    """Independent scalar circuit construction with full Kronecker operators."""
    n = len(angles)
    state = torch.zeros(2**n, dtype=torch.complex128)
    state[0] = 1
    eye = torch.eye(2, dtype=torch.complex128)
    for wire, (y, z) in enumerate(angles):
        ry = torch.stack((torch.stack((torch.cos(y/2), -torch.sin(y/2))),
                          torch.stack((torch.sin(y/2), torch.cos(y/2))))).to(torch.complex128)
        rz = torch.diag(torch.stack((torch.exp(-.5j*z), torch.exp(.5j*z))))
        op = torch.ones((1, 1), dtype=torch.complex128)
        for w in range(n):
            op = torch.kron(op, (rz @ ry) if wire == w else eye)
        state = op @ state
    for control, target in edges:
        matrix = torch.zeros((2**n, 2**n), dtype=torch.complex128)
        for j in range(2**n):
            bits = list(f'{j:0{n}b}')
            if bits[control] == '1':
                bits[target] = '0' if bits[target] == '1' else '1'
            matrix[int(''.join(bits), 2), j] = 1
        state = matrix @ state
    return state


def fixture(dtype=torch.float64):
    torch.manual_seed(307)
    h = torch.randn(3, 5, 1024, dtype=dtype) * .2
    roles = torch.zeros(3, 5, 4, dtype=dtype)
    roles[..., :2] = torch.randn(3, 5, 2, dtype=dtype) * .3
    roles[:, 0, 2] = 1
    roles[:, 1, 3] = 1
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    return h, roles, mask


class JointMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.inputs = fixture()

    def test_dense_gate_oracle_and_angle_gradients(self):
        for n, edges in ((2, ((0, 1),)), (6, LOCAL_EDGES+CROSS_EDGES)):
            angles = torch.randn(n, 2, dtype=torch.float64, requires_grad=True) * .7
            value = register_state(angles, edges)
            oracle = dense_oracle(angles, edges)
            torch.testing.assert_close(value, oracle, rtol=1e-12, atol=1e-12)
            target = torch.randn(2**n, dtype=torch.complex128)
            grad = torch.autograd.grad((value*target).sum().abs().square(), angles, retain_graph=True)[0]
            expected = torch.autograd.grad((oracle*target).sum().abs().square(), angles)[0]
            torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-12)

    def test_static_is_exact_legacy(self):
        torch.manual_seed(19)
        legacy = RMEAHead('quantum').double()
        torch.manual_seed(19)
        adapter = make_head('static').double()
        self.assertEqual(set(adapter.state_dict()), set(legacy.state_dict()))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, adapter.state_dict()[name]))
        for a, b in zip(legacy(*self.inputs), adapter(*self.inputs)):
            self.assertTrue(torch.equal(a, b))

    def test_registry_counts_shared_initialization_and_masks(self):
        reference = None
        for variant in VARIANTS:
            torch.manual_seed(23)
            head = make_head(variant).double()
            expected = 81508 if variant == 'classical_cp' else 81028
            self.assertEqual(sum(p.numel() for p in head.parameters()), expected)
            if reference is None:
                reference = head
            for name in ('decoder', 'bias'):
                self.assertTrue(torch.equal(getattr(head, name), getattr(reference, name)))
            self.assertTrue(torch.equal(head.value.weight, reference.value.weight))
            if variant == 'no_content':
                head.fit_role_means([self.inputs], split='train')
            logits, alpha = head(*self.inputs)
            self.assertTrue(torch.isfinite(logits).all())
            torch.testing.assert_close(alpha.sum(-1), torch.ones(3, 40, dtype=alpha.dtype))
            self.assertEqual(alpha.masked_select(~self.inputs[2][:, None, :]).count_nonzero().item(), 0)
            h, roles, mask = self.inputs
            changed_h, changed_roles = h.clone(), roles.clone()
            changed_h[~mask] = 999
            changed_roles[..., :2][~mask] = -999
            # Memberships stay binary even on padding: the model validates its
            # input schema before masking. Flip legal padding flags instead.
            changed_roles[..., 2:4][~mask] = 1-changed_roles[..., 2:4][~mask]
            torch.testing.assert_close(head(changed_h, changed_roles, mask)[0], logits, rtol=0, atol=0)

    def test_direct_born_tensor_and_all_parameter_gradient_parity(self):
        head = make_head('joint').double()
        d = head(*self.inputs, details=True)
        z = torch.einsum('bti,bj,bk->btijk', d['psi'], d['subject'], d['object']).reshape(3, 5, 64)
        direct = torch.einsum('rd,btd->brt', d['phi'].conj(), z).abs().square()
        torch.testing.assert_close(direct, d['probability'], rtol=1e-11, atol=1e-14)
        alpha, _ = normalized_attention(direct, self.inputs[2])
        logits = (alpha*d['utilities']).sum(-1)+head.bias
        torch.testing.assert_close(logits, d['logits'], rtol=1e-11, atol=1e-13)
        weight = torch.linspace(.1, .9, 120, dtype=torch.float64).reshape(3, 40)
        params = list(head.parameters())
        a = torch.autograd.grad((d['logits']*weight).sum(), params, retain_graph=True)
        b = torch.autograd.grad((logits*weight).sum(), params)
        for grad, expected in zip(a, b):
            self.assertTrue(torch.isfinite(grad).all())
            torch.testing.assert_close(grad, expected, rtol=1e-9, atol=1e-11)
            self.assertGreater(grad.abs().max().item(), 1e-10)
        # Honest structural accounting: opposite-role channels are always zero
        # for disjoint entities. These eight weights are NOT active capacity.
        self.assertEqual(a[head_parameter_index(head, 'prepare_subject.weight')][:, -1].count_nonzero(), 0)
        self.assertEqual(a[head_parameter_index(head, 'prepare_object.weight')][:, -2].count_nonzero(), 0)

    def test_product_effect_factorization_and_floor_null(self):
        head = make_head('separable').double()
        d = head(*self.inputs, details=True)
        null = d['separable_null']
        torch.testing.assert_close(d['probability'], null['q']*null['c'], rtol=1e-11, atol=1e-14)
        torch.testing.assert_close(d['alpha'], null['alpha'], rtol=1e-11, atol=1e-13)
        params = list(head.parameters())
        grad = torch.autograd.grad(d['alpha'].square().sum(), params, allow_unused=True, retain_graph=True)
        expected = torch.autograd.grad(null['alpha'].square().sum(), params, allow_unused=True)
        for a, b in zip(grad, expected):
            if a is None:
                self.assertIsNone(b)
            else:
                torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-10)
        q, c = torch.zeros(1, 2, 3, dtype=torch.float64), torch.zeros(1, 2, 1, dtype=torch.float64)
        c.requires_grad_()
        edge = separable_floor(q, c, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(edge['alpha'], torch.tensor([[[.5, .5, 0], [.5, .5, 0]]], dtype=torch.float64))
        self.assertTrue(torch.isfinite(torch.autograd.grad(edge['alpha'].sum(), c)[0]).all())

    def test_entities_change_joint_token_direction_not_only_floor(self):
        head = make_head('joint').double()
        pair = entity_inputs(*self.inputs)
        original = head(*self.inputs, details=True)
        swapped = head(*self.inputs, details=True, entity_override=pair.flip(1))
        mismatched = head(*self.inputs, details=True, entity_override=pair.roll(1, 0))
        mean = head(*self.inputs, details=True, entity_override=pair.mean(0, keepdim=True).expand_as(pair))
        def direction(d):
            p = d['probability'] * self.inputs[2][:, None, :]
            return p / p.sum(-1, keepdim=True)
        for intervention in (swapped, mismatched, mean):
            # Without the additive floor, product effects cannot pass this.
            self.assertGreater((direction(original)-direction(intervention)).abs().max().item(), 1e-5)
            self.assertTrue(torch.equal(original['values'], intervention['values']))
            self.assertTrue(torch.equal(original['utilities'], intervention['utilities']))
        product = make_head('separable').double()
        p1 = product(*self.inputs, details=True)
        p2 = product(*self.inputs, details=True, entity_override=pair.flip(1))
        torch.testing.assert_close(direction(p1), direction(p2), rtol=1e-10, atol=1e-12)

    def test_train_mean_calibration_is_equal_example_train_only_and_serialized(self):
        head = make_head('no_content').double()
        with self.assertRaises(ValueError):
            head(*self.inputs)
        with self.assertRaises(ValueError):
            head.fit_role_means([self.inputs], split='valid')
        head.fit_role_means([self.inputs], split='train')
        torch.testing.assert_close(head.role_means, entity_inputs(*self.inputs).mean(0))
        clone = make_head('no_content').double()
        clone.load_state_dict(head.state_dict())
        torch.testing.assert_close(clone(*self.inputs)[0], head(*self.inputs)[0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            clone.fit_role_means([self.inputs], split='train')

    def test_classical_cp_is_trilinear_not_a_classical_output_bypass(self):
        head = make_head('classical_cp').double()
        d = head(*self.inputs, details=True)
        oracle = []
        for r in range(40):
            terms = []
            for k in range(2):
                terms.append((d['word_angles']@head.cp[r, k, 0])
                             * (d['subject_angles']@head.cp[r, k, 1])[:, None]
                             * (d['object_angles']@head.cp[r, k, 2])[:, None])
            oracle.append(sum(terms))
        torch.testing.assert_close(d['amplitude'], torch.stack(oracle, 1))
        alpha, _ = normalized_attention(torch.stack(oracle, 1).square(), self.inputs[2])
        torch.testing.assert_close(d['logits'], (alpha*d['utilities']).sum(-1)+head.bias)
        d['logits'].square().sum().backward()
        self.assertTrue(torch.isfinite(head.cp.grad).all())
        self.assertGreater(head.cp.grad.abs().min().item(), 0)

    def test_float32_and_invalid_inputs(self):
        for variant in ('joint', 'separable', 'classical_cp'):
            torch.manual_seed(73)
            single = make_head(variant)
            double = make_head(variant).double()
            double.load_state_dict(single.state_dict())
            x = fixture(torch.float32)
            precise = (x[0].double(), x[1].double(), x[2])
            torch.testing.assert_close(single(*x)[0].double(), double(*precise)[0], rtol=2e-4, atol=2e-5)
        h, roles, mask = self.inputs
        bad = mask.clone(); bad[0] = False
        with self.assertRaises(ValueError):
            make_head('joint').double()(h, roles, bad)
        bad_roles = roles.clone(); bad_roles[:, :, 2] = 0
        with self.assertRaises(ValueError):
            make_head('joint').double()(h, bad_roles, mask)
        with self.assertRaises(ValueError):
            make_head('unknown')


def head_parameter_index(head, name):
    return [n for n, _ in head.named_parameters()].index(name)


if __name__ == '__main__':
    unittest.main()
