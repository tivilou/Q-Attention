"""Explicit four-arm factory for the phase-presence screen."""
import torch

from baselines.retacred.rmea import circuit
from baselines.retacred.rmea_matched_routers import initialize_head
from presence_phase import COMMON, MODES, make_models as make_presence_models
from rmea_history_posterior import GaussianCovarianceHead


def make_models(seed=13, initial=None):
    if initial is None:
        raise ValueError('A pinned native initial state is required')
    models = make_presence_models(circuit, initial, seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = GaussianCovarianceHead()
        state = reference.state_dict()
        legacy = initialize_head('real12', initial, dict(seed=seed, training_dtype='float32')).head.state_dict()
        if set(state) != set(legacy) | {'moment_decoder'}:
            raise ValueError('Unexpected Gcov state keys')
        state.update({key:value.detach().clone() for key,value in legacy.items()})
        reference.load_state_dict(state, strict=True)
    models['gaussian_covariance'] = reference
    return models


def real_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
