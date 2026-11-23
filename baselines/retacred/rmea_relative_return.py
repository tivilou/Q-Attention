"""Relative-unitary return readouts with the native flat parameter/checkpoint schema."""
from .rmea import RMEAHead, circuit, EPS
from experiments.diagnostics.rmea_relative_return.relative_return import return_scores

MODES = ('return_coherent', 'return_dephased', 'native_square')


class RelativeReturnHead(RMEAHead):
    """No new parameters, features, classical residual, optimizer or persistence."""
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError('Unknown relative-return mode')
        super().__init__('quantum')
        self.readout_mode = mode

    def forward(self, h, roles, mask, details=False):
        native = super().forward(h, roles, mask, details=True)
        if self.readout_mode == 'native_square':
            probability = native['probability'].square()
        else:
            mode = {'return_coherent': 'coherent', 'return_dephased': 'dephased'}[self.readout_mode]
            probability = return_scores(circuit, native['angles'], native['effect_angles'])[mode]
        raw = (probability + EPS) * mask[:, None]
        normalizer = raw.sum(-1, keepdim=True)
        alpha = raw / normalizer
        logits = (alpha * native['utilities']).sum(-1) + self.bias
        if details:
            return dict(logits=logits, alpha=alpha, probability=probability,
                        normalizer=normalizer, native=native)
        return logits
