"""Exercise the real dev coefficient fitter on discriminating branch scores."""
import torch
from baselines.retacred.reference_models import LABELS
from baselines.retacred.reference_worker import tune


class BranchModel:
    def eval(self):
        return self


class BranchAdapter:
    def __init__(self, model):
        self.kind = model
        self.model = BranchModel()

    def batch(self, features):
        return {'features': features,
                'candidate_mask': torch.ones(len(features), len(LABELS), dtype=torch.bool)}

    def forward(self, batch):
        rows = batch['features']
        factual = torch.zeros(len(rows), len(LABELS))
        nuisance = torch.zeros_like(factual)
        for index, row in enumerate(rows):
            true = row['labels']
            wrong = 2 if true == 1 else 1
            factual[index, true] = .4
            factual[index, wrong] = .6
            nuisance[index, true] = .1
            nuisance[index, wrong] = .9
        if self.kind == 'luke_core':
            self.model.last = {'factual_probability': factual,
                               'entity_only_probability': nuisance,
                               'blank_probability': torch.zeros_like(factual)}
        else:
            self.model.last = {'factual_logits': factual,
                               'counterfactual_logits': nuisance}


def test_core_dev_grid_selects_correcting_coefficients():
    adapter = BranchAdapter('luke_core')
    result = tune(adapter, [{'labels': 1}, {'labels': 2}], None,
                  {'model': 'luke_core', 'core_grid': [0., 1.]}, 1)
    assert result['grid_size'] == 4
    assert result['micro_f1'] == 1.
    assert result['coefficients'] == (1., 0.)
    assert adapter.coefficients == (1., 0.)
    assert result['fit_split'] == 'valid' and result['test_access'] is False


def test_cfie_dev_grid_applies_chosen_alpha():
    adapter = BranchAdapter('luke_cfie')
    result = tune(adapter, [{'labels': 1}, {'labels': 2}], None,
                  {'model': 'luke_cfie', 'alpha_grid': [0., 1.]}, 1)
    assert result['grid_size'] == 2
    assert result['micro_f1'] == 1.
    assert result['coefficients'] == 1.
    assert adapter.coefficients == adapter.model.alpha == 1.
