import inspect
import unittest

import torch

from baselines.retacred.rmea import RMEAHead, circuit
from baselines.retacred.rmea_entity import (
    VARIANTS, EntityRMEAHead, apply_circuit, entity_inputs, make_head, mismatched_pairs,
)


def fixture(n=3, dtype=torch.float64):
    g = torch.Generator().manual_seed(701)
    h = torch.randn(n, 6, 1024, generator=g, dtype=dtype)*.15
    roles = torch.zeros(n, 6, 4, dtype=dtype)
    roles[:, :, :2] = torch.randn(n, 6, 2, generator=g, dtype=dtype)*.1
    roles[:, 0:2, 2] = 1
    roles[:, 3, 3] = 1
    mask = torch.ones(n, 6, dtype=torch.bool)
    mask[:, -1] = False
    return h, roles, mask


class EntityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(307)
        torch.set_num_threads(1)

    def head(self, variant, data=None):
        head = make_head(variant).double()
        if variant == 'no_content':
            head.fit_role_means([fixture() if data is None else data], split='train')
        return head

    def test_static_exact_legacy_outputs_and_gradients(self):
        h, r, m = fixture()
        old = RMEAHead('quantum').double()
        new = self.head('static')
        state = new.state_dict()
        state.update(old.state_dict())
        new.load_state_dict(state)
        for a, b in zip(old(h, r, m), new(h, r, m)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        old(h, r, m)[0].square().sum().backward()
        new(h, r, m)[0].square().sum().backward()
        for (name, a), (name2, b) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, name2)
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_unitary_inverse_and_legacy_preparation(self):
        angles = torch.randn(2, 2, 3, 2, dtype=torch.float64, requires_grad=True)
        zero = torch.zeros(2, 8, dtype=torch.complex128)
        zero[:, 0] = 1
        prepared = apply_circuit(angles, zero)
        torch.testing.assert_close(prepared, circuit(angles), atol=1e-14, rtol=1e-14)
        arbitrary = torch.randn(2, 8, dtype=torch.complex128)
        torch.testing.assert_close(apply_circuit(angles, apply_circuit(angles, arbitrary), True), arbitrary,
                                   atol=1e-14, rtol=1e-14)

    def test_effective_measurement_equals_direct_with_gradients(self):
        head = self.head('ordered')
        data = fixture()
        d = head(*data, details=True)
        evolved = apply_circuit(d['object_angles'][:, None],
                                apply_circuit(d['subject_angles'][:, None], d['psi']))
        direct = torch.einsum('rd,btd->brt', d['phi'].conj(), evolved).abs().square()
        torch.testing.assert_close(direct, d['probability'], atol=2e-14, rtol=2e-13)
        params = tuple(head.parameters())
        a = torch.autograd.grad(direct.square().sum(), params, retain_graph=True, allow_unused=True)
        b = torch.autograd.grad(d['probability'].square().sum(), params, allow_unused=True)
        for x, y in zip(a, b):
            self.assertEqual(x is None, y is None)
            if x is not None:
                torch.testing.assert_close(x, y, atol=2e-11, rtol=2e-11)

    def test_branch_gradcheck_and_nonzero_entity_path(self):
        head = self.head('ordered')
        h, r, m = fixture(1)
        pair = entity_inputs(h, r, m).detach().requires_grad_()
        # Numerical differences on three selected semantic coordinates.
        direction = torch.zeros_like(pair)
        direction[0, 0, 3] = 1
        direction[0, 1, 8] = -.5
        scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
        fn = lambda x: head(h, r, m, entity_override=pair.detach()+x*direction)[0]
        self.assertTrue(torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=2e-6, rtol=2e-4))
        d = head(h, r, m, True, entity_override=pair)
        gradient = torch.autograd.grad(d['logits'].square().sum(), pair)[0]
        self.assertGreater(float(gradient.norm()), 1e-9)

    def test_all_active_counts_common_initialization_and_finite_gradients(self):
        common = []
        for variant in VARIANTS:
            torch.manual_seed(1137)
            head = self.head(variant)
            self.assertEqual(sum(p.numel() for p in head.parameters()), 81028)
            common.append(tuple(x.detach().clone() for x in (head.value.weight, head.value.bias, head.decoder, head.bias)))
            loss = torch.nn.functional.cross_entropy(head(*fixture())[0], torch.tensor([1, 4, 9]))
            loss.backward()
            for name, p in head.named_parameters():
                self.assertIsNotNone(p.grad, (variant, name))
                self.assertTrue(torch.isfinite(p.grad).all(), (variant, name))
                self.assertGreater(float(p.grad.norm()), 0, (variant, name))
        for row in common[1:]:
            for a, b in zip(row, common[0]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_word_permutation_padding_and_batch_isolation(self):
        h, r, m = fixture()
        perm = torch.tensor([3, 0, 4, 2, 1, 5])
        for variant in VARIANTS:
            head = self.head(variant)
            base, alpha = head(h, r, m)
            permuted, weights = head(h[:, perm], r[:, perm], m[:, perm])
            torch.testing.assert_close(base, permuted, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(alpha[:, :, perm], weights, atol=1e-12, rtol=1e-12)
            hp = torch.cat((h, torch.randn(3, 2, 1024, dtype=h.dtype)), 1)
            rp = torch.cat((r, torch.zeros(3, 2, 4, dtype=r.dtype)), 1)
            mp = torch.cat((m, torch.zeros(3, 2, dtype=torch.bool)), 1)
            torch.testing.assert_close(base, head(hp, rp, mp)[0], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(base[:1], head(h[:1], r[:1], m[:1])[0], atol=1e-12, rtol=1e-12)
            self.assertTrue((alpha[:, :, -1] == 0).all())
            torch.testing.assert_close(alpha.sum(-1), torch.ones_like(alpha.sum(-1)))

    def test_no_content_train_only_frozen_and_reload(self):
        data = fixture()
        head = make_head('no_content').double()
        with self.assertRaises(ValueError):
            head(*data)
        with self.assertRaises(ValueError):
            head.fit_role_means([data], split='valid')
        head.fit_role_means([data], split='train')
        torch.testing.assert_close(head.role_means, entity_inputs(*data).mean(0))
        with self.assertRaises(ValueError):
            head.fit_role_means([data], split='train')
        with self.assertRaises(ValueError):
            head(*data, entity_override=entity_inputs(*data))
        other = make_head('no_content').double()
        other.load_state_dict(head.state_dict())
        torch.testing.assert_close(head(*data)[0], other(*data)[0], atol=0, rtol=0)

    def test_order_sensitivity_symmetric_invariance_and_valid_probability(self):
        h, r, m = fixture()
        pair = entity_inputs(h, r, m)
        for variant in ('ordered', 'symmetric'):
            head = self.head(variant)
            d = head(h, r, m, True, entity_override=pair)
            swap = head(h, r, m, True, entity_override=pair.flip(1))
            self.assertTrue((d['probability'] >= 0).all() and (d['probability'] <= 1+1e-12).all())
            if variant == 'symmetric':
                torch.testing.assert_close(d['logits'], swap['logits'], atol=1e-12, rtol=1e-12)
            else:
                self.assertGreater(float((d['alpha']-swap['alpha']).abs().max()), 1e-7)

    def test_invalid_inputs_and_explicit_registry(self):
        with self.assertRaises(ValueError):
            make_head('auto_discover')
        h, r, m = fixture()
        r[:, :, 2] = 0
        with self.assertRaises(ValueError):
            self.head('ordered')(h, r, m)
        h, r, m = fixture()
        h[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.head('ordered')(h, r, m)
        self.assertNotIn('labels', inspect.signature(EntityRMEAHead.forward).parameters)
        self.assertNotIn('target', inspect.signature(EntityRMEAHead.forward).parameters)

    def test_mismatch_ids_not_labels(self):
        pair = entity_inputs(*fixture())
        ids = ['a', 'b', 'c']
        a = mismatched_pairs(pair, ids)
        b = mismatched_pairs(pair, ids)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertTrue(((a-pair).abs().sum((1, 2)) > 0).all())
        with self.assertRaises(ValueError):
            mismatched_pairs(pair[:1], ids[:1])


if __name__ == '__main__':
    unittest.main()
