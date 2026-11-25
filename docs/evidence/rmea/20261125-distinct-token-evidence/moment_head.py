"""Fixed shared relation queries plus the previously specified value moments."""
import math
import torch
from torch import nn
from baselines.retacred.rmea import EPS, circuit
from shared_query import SharedQueryHead, initialize as initialize_shared

MODES = ('quantum_shared_covariance', 'dephased_shared_covariance',
         'gaussian_shared_covariance', 'quantum_shared_mean_product', 'quantum_permuted_covariance')


def shared_kind(kind):
    if kind not in MODES:
        raise ValueError(kind)
    return ('quantum_permuted' if kind.startswith('quantum_permuted') else
            'gaussian_shared' if kind.startswith('gaussian') else
            'dephased_shared' if kind.startswith('dephased') else 'quantum_shared')


class SharedMomentHead(SharedQueryHead):
    def __init__(self, kind):
        super().__init__(shared_kind(kind))
        self.joint_kind = kind
        self.moment_decoder = nn.Parameter(torch.zeros(40, 32))

    def components(self, h, roles, mask):
        if (h.ndim != 3 or h.shape[-1] != 1024 or roles.shape != h.shape[:2]+(4,)
                or mask.shape != h.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Invalid frozen-word inputs')
        a = self.head; query = self.mapped_query()
        if self.native_mode == 'real12':
            key = a.key(torch.cat((h, roles), -1))
            scale = (key.new_tensor([3-math.sqrt(2),3+math.sqrt(2)]*6)/24).sqrt()
            k,q = key*scale,query*scale
            probability = torch.exp(-(k[:,None]-q[None,:,None]).square().sum(-1))
        else:
            angles = math.pi*torch.tanh(a.prepare(torch.cat((h, roles), -1)))
            psi = circuit(angles.reshape(h.shape[:2]+(2,3,2)))
            phi = circuit(math.pi*torch.tanh(query.reshape(40,2,3,2)))
            if self.native_mode == 'quantum':
                probability = torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
            else:
                probability = torch.einsum('rd,btd->brt',phi.abs().square(),psi.abs().square())
        raw = (probability+EPS)*mask[:,None];alpha = raw/raw.sum(-1,keepdim=True)
        values = a.value(h);mean = torch.einsum('brt,btd->brd',alpha,values)
        utilities = torch.einsum('btd,rd->brt',values,a.decoder)
        base = (alpha*utilities).sum(-1)+a.bias
        return alpha,values,mean,base

    def statistic(self, alpha, values, mean):
        product = mean[...,0::2]*mean[...,1::2]
        if self.joint_kind.endswith('mean_product'):
            return product
        return torch.einsum('brt,btj->brj',alpha,values[...,0::2]*values[...,1::2])-product

    def forward(self, h, roles, mask):
        alpha,values,mean,base = self.components(h,roles,mask)
        return base+(self.statistic(alpha,values,mean)*self.moment_decoder).sum(-1)


def make_head(kind):
    return SharedMomentHead(kind)


def initialize(initial, seed):
    bases = initialize_shared(initial,seed);models = {}
    for kind in MODES:
        m = make_head(kind)
        out = m.load_state_dict(bases[shared_kind(kind)].state_dict(),strict=False)
        assert out.missing_keys == ['moment_decoder'] and not out.unexpected_keys
        assert torch.count_nonzero(m.moment_decoder)==0
        models[kind] = m
    return models
