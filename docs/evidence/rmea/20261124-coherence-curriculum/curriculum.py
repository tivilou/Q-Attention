"""Fixed dephasing-to-coherence training path; unchanged pure endpoints."""
from pathlib import Path
import json
import math
import torch
from torch.nn import functional as F
import router as reference
from baselines.retacred.rmea import EPS

MODES = ('quantum_curriculum', 'real_curriculum')
BASE = {'quantum_curriculum': 'quantum', 'real_curriculum': 'real12'}


def baseline():
    return json.loads(Path(__file__).with_name('reference-summary.json').read_bytes())


class Head(reference.Head):
    def __init__(self, mode):
        super().__init__(BASE[mode])
        self.arm = mode
        self.eta = 1.0

    def set_training_progress(self, step, total):
        if not 1 <= step <= total or total < 2:
            raise ValueError('Invalid logical-batch progress')
        self.eta = (step - 1) / (total - 1)

    def forward(self, h, roles, mask):
        eta = self.eta if self.training else 1.0
        if eta == 1.0:
            return super().forward(h, roles, mask)
        a = self.head
        if self.mode == 'quantum':
            d = a(h, roles, mask, details=True)
            coherent = d['probability']
            diagonal = torch.einsum('rd,btd->brt', d['phi'].abs().square(), d['psi'].abs().square())
        else:
            key = F.normalize(a.key(torch.cat((h, roles), -1)), dim=-1)
            query = F.normalize(a.query, dim=-1)
            coherent = torch.einsum('btd,rd->brt', key, query).square()
            diagonal = torch.einsum('btd,rd->brt', key.square(), query.square())
        probability = (1 - eta) * diagonal + eta * coherent
        raw = (probability + EPS) * mask[:, None]
        alpha = raw / raw.sum(-1, keepdim=True)
        utilities = torch.einsum('btd,rd->brt', a.value(h), a.decoder)
        return (alpha * utilities).sum(-1) + a.bias


def make_head(mode):
    return Head(mode)


def initialize(initial, seed):
    from baselines.retacred.rmea_token_transport_study import state_identity
    refs = reference.initialize(initial, seed)
    old = baseline()
    models = {}
    for mode in MODES:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = Head(mode).double()
        model.load_state_dict(refs[BASE[mode]].state_dict())
        assert state_identity(model.state_dict()) == old['results'][BASE[mode]]['initial_sha256']
        assert sum(p.numel() for p in model.parameters()) == 81028
        models[mode] = model
    return models


def fixtures(models, inputs):
    errors = {}
    for mode, original in models.items():
        model = Head(mode).double().to(inputs[0].device)
        model.load_state_dict(original.state_dict())
        endpoint = reference.Head(BASE[mode]).double().to(inputs[0].device)
        endpoint.load_state_dict(original.state_dict())
        model.train()
        model.set_training_progress(1024, 1024)
        errors[mode+'_pure_endpoint'] = float((model(*inputs)-endpoint(*inputs)).abs().max())
        model.set_training_progress(1, 1024)
        assert model.eta == 0
        h, roles, mask = inputs
        a = model.head
        if mode == 'quantum_curriculum':
            from baselines.retacred.rmea import RMEAHead
            diagonal = RMEAHead('dephased').double().to(h.device)
            diagonal.load_state_dict(a.state_dict())
            expected = diagonal(*inputs)[0]
        else:
            k = F.normalize(a.key(torch.cat((h, roles), -1)), dim=-1)
            q = F.normalize(a.query, dim=-1)
            p = (k[:, None].square() * q[None, :, None].square()).sum(-1)
            w = (p+EPS)*mask[:, None]
            w = w/w.sum(-1, keepdim=True)
            expected = (w * (a.value(h)[:, None]*a.decoder[None, :, None]).sum(-1)).sum(-1)+a.bias
        errors[mode+'_diagonal_endpoint'] = float((model(*inputs)-expected).abs().max())
        model.set_training_progress(512, 1024)
        assert model.eta == 511/1023
        got = model(*inputs)
        if mode == 'quantum_curriculum':
            d = a(*inputs, details=True)
            state, effect = d['psi'], d['phi']
        else:
            state = F.normalize(a.key(torch.cat((h, roles), -1)), dim=-1)
            effect = F.normalize(a.query, dim=-1)
        density = model.eta * state[..., :, None] * state.conj()[..., None, :]
        density = density + (1-model.eta) * torch.diag_embed(state.abs().square())
        probability = torch.einsum('ri,btij,rj->brt', effect.conj(), density, effect).real
        weight = (probability+EPS)*mask[:, None]
        weight = weight/weight.sum(-1, keepdim=True)
        utility = (a.value(h)[:, None]*a.decoder[None, :, None]).sum(-1)
        direct = (weight*utility).sum(-1)+a.bias
        errors[mode+'_mixed_density_formula'] = float((got-direct).abs().max())
        grads = torch.autograd.grad(got.square().mean(), tuple(model.parameters()), retain_graph=True)
        other = torch.autograd.grad(direct.square().mean(), tuple(model.parameters()))
        errors[mode+'_mixed_density_gradient'] = max(float((u-v).abs().max()) for u,v in zip(grads,other))
        assert all(torch.isfinite(g).all() for g in grads)
        padded = (F.pad(h,(0,0,0,2),value=.123),F.pad(roles,(0,0,0,2),value=1.),F.pad(mask,(0,2),value=False))
        errors[mode+'_padding'] = float((model(*padded)-got).abs().max())
        # Evaluation is always the original pure endpoint, including after a load.
        model.eval()
        errors[mode+'_eval_pure_endpoint'] = float((model(*inputs)-endpoint(*inputs)).abs().max())
    assert all(math.isfinite(v) and v < 1e-8 for v in errors.values())
    return dict(status='PASS', errors=errors, scope='Only new channel endpoints, logical-step schedule, finite gradient and padding; existing runtime/COS capabilities reused.')


def summarize(rows, logits, labels, cfg):
    old = baseline()
    assert old['source_hashes'] == cfg['source_hashes']
    assert old['train_items'] == 4192 and old['assessment_items'] == 4000
    assert all(x['updates']==1024 and x['parameters']==81028 for x in rows.values())
    assert all(x['order_sha256']==old['results'][BASE[m]]['order_sha256'] for m,x in rows.items())
    assert all(x['initial_sha256']==old['results'][BASE[m]]['initial_sha256'] for m,x in rows.items())
    all_rows = dict(old['results'], **rows)
    q = rows['quantum_curriculum']
    comparisons = {k:dict(macro_delta=q['positive39_macro_f1']-v['positive39_macro_f1'],
                          ce_comparator_minus_quantum=v['ce']-q['ce'])
                   for k,v in all_rows.items() if k != 'quantum_curriculum'}
    gates = {k:v['macro_delta']>0 for k,v in comparisons.items()}
    return dict(results=rows, reused_reference_results=old['results'], comparisons=comparisons,
        gates=gates, gate='PASS_FOR_FORMAL_STUDY_DESIGN_ONLY' if all(gates.values()) else 'NO_PROMOTION_FIXED_LINEAR_CURRICULUM',
        schedule='eta=(logical_step-1)/(1024-1); evaluation eta=1; same endpoint function/state/parameter count.',
        interpretation='Exposed TRAIN one-seed planning only; no independent utility/novelty/statistical confirmation. No curve, endpoint, seed, split or update-budget rescue.',
        L3='NOT_ACCEPTED', L4='NOT_ACCEPTED', L5='NOT_ACCEPTED')
