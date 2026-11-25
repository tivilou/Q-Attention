"""Remove same-position products, with a collision-mass-only control."""
import torch
from moment_head import SharedMomentHead, initialize as initialize_moments

MODES=('quantum_distinct','dephased_distinct','gaussian_distinct',
       'quantum_permuted_distinct','quantum_collision_mass')
BASE=dict(quantum_distinct='quantum_shared_covariance',
          dephased_distinct='dephased_shared_covariance',
          gaussian_distinct='gaussian_shared_covariance',
          quantum_permuted_distinct='quantum_permuted_covariance',
          quantum_collision_mass='quantum_shared_mean_product')


class DistinctHead(SharedMomentHead):
    def __init__(self,kind):
        super().__init__(BASE[kind])
        self.distinct_kind=kind

    def statistic(self,alpha,values,mean):
        product=mean[...,0::2]*mean[...,1::2]
        if self.distinct_kind=='quantum_collision_mass':
            return (1-alpha.square().sum(-1,keepdim=True))*product
        self_product=torch.einsum('brt,btj->brj',alpha.square(),values[...,0::2]*values[...,1::2])
        return product-self_product


def make_head(kind):
    return DistinctHead(kind)


def initialize(initial,seed):
    bases=initialize_moments(initial,seed);models={}
    for kind in MODES:
        m=make_head(kind)
        m.load_state_dict(bases[BASE[kind]].state_dict(),strict=True)
        assert torch.count_nonzero(m.moment_decoder)==0
        models[kind]=m
    return models
