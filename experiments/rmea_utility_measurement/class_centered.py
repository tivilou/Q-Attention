"""R102 utility-interface experiment; float64 CPU, no new learned tensors.

The original head and measured_attention remain unchanged. Stored parameter
counts agree; centered readout removes 64 effective decoder directions.
"""
import torch

from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
from experiments.rmea_utility_measurement.head import measured_attention

KINDS = ('native', 'full', 'centered_native', 'centered_full', 'construction_only')


def class_centered_utilities(utilities, mask):
    """Subtract the all-40-class mean independently at every valid token."""
    if (utilities.ndim != 3 or utilities.shape[1] != 40
            or utilities.dtype != torch.float64 or utilities.device.type != 'cpu'):
        raise ValueError('Expected float64 CPU utilities[B,40,T], including null')
    if (mask.shape != (utilities.shape[0], utilities.shape[2])
            or mask.dtype != torch.bool or mask.device != utilities.device
            or not mask.any(-1).all() or utilities.shape[0] == 0):
        raise ValueError('Nonempty boolean valid-token mask required')
    if not torch.isfinite(utilities.masked_select(mask[:, None])).all():
        raise ValueError('Nonfinite valid utility')
    raw = torch.where(mask[:, None], utilities, torch.zeros_like(utilities))
    common = raw.mean(1, keepdim=True)
    return raw - common, common


def class_centered_measurement(psi, phi, utilities, mask, *, construction_only=False):
    """Use centered construction; optionally retain original recipient readout."""
    centered, common = class_centered_utilities(utilities, mask)
    raw = torch.where(mask[:, None], utilities, torch.zeros_like(utilities))
    readout = raw if construction_only else centered
    update = measured_attention(psi, phi, readout, mask,
                               construction_utilities=centered)
    return dict(update=update, raw_utilities=raw, class_mean=common,
                centered_utilities=centered, construction_utilities=centered,
                readout_utilities=readout, alpha=update['alpha'],
                aggregate=update['aggregate'])


class ClassCenteredUtilityHead(RMEAHead):
    """One explicit adapter; native/full controls preserve their original maps."""
    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError('Unknown class-centering condition')
        super().__init__('quantum')
        self.kind = kind

    def forward(self, h, roles, mask, details=False):
        if h.dtype != torch.float64 or h.device.type != 'cpu':
            raise ValueError('This adapter is qualified only for float64 CPU')
        original = super().forward(h, roles, mask, details=True)
        raw = original['utilities']
        centered, common = class_centered_utilities(raw, mask)
        update = None
        if self.kind == 'native':
            logits, alpha, readout = original['logits'], original['alpha'], raw
        elif self.kind == 'centered_native':
            alpha, readout = original['alpha'], centered
            logits = (alpha * readout).sum(-1) + self.bias
        elif self.kind == 'full':
            update = measured_attention(original['psi'], original['phi'], raw, mask)
            alpha, readout = update['alpha'], raw
            logits = update['aggregate'] + self.bias
        else:
            result = class_centered_measurement(original['psi'], original['phi'], raw,
                mask, construction_only=self.kind == 'construction_only')
            update, alpha, readout = result['update'], result['alpha'], result['readout_utilities']
            logits = result['aggregate'] + self.bias
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite class-centered output; no repair')
        if not details:
            return logits, alpha
        construction = (None if self.kind in ('native', 'centered_native') else
                        centered if self.kind in ('centered_full', 'construction_only') else raw)
        return dict(logits=logits, alpha=alpha, raw_utilities=raw,
                    class_mean=common, centered_utilities=centered,
                    construction_utilities=construction, readout_utilities=readout,
                    psi=original['psi'], phi=original['phi'], update=update)
