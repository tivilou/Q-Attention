"""Optional class competition; original preparation/value/training remain unchanged."""
from baselines.retacred.rmea import RMEAHead, EPS


def attention(q, mask, coupled=True):
    """q[B,R,T] is strictly positive; padding has no probability mass."""
    if q.ndim != 3 or mask.shape != (q.shape[0], q.shape[2]):
        raise ValueError('Bad score/mask shape')
    if not mask.any(-1).all() or not q.isfinite().all() or not (q > 0).all():
        raise ValueError('Nonempty mask and finite positive scores required')
    raw = q / q.sum(1, keepdim=True) if coupled else q
    raw = raw * mask[:, None, :]
    return raw / raw.sum(-1, keepdim=True)


class CompetitiveHead(RMEAHead):
    def __init__(self, variant):
        if variant not in ('native', 'competitive', 'population_competitive'):
            raise ValueError('Unknown scientific variant')
        super().__init__('dephased' if variant == 'population_competitive' else 'quantum')
        self.variant = variant

    def forward(self, h, roles, mask, details=False):
        if self.variant == 'native':
            return super().forward(h, roles, mask, details=details)
        d = super().forward(h, roles, mask, details=True)
        q = d['probability'] + EPS
        alpha = attention(q, mask)
        logits = (alpha * d['utilities']).sum(-1) + self.bias
        if not details:
            return logits, alpha
        # The old row-local sensitivity is invalid after coupling relation rows.
        d.pop('sensitivity', None)
        d.update(alpha=alpha, logits=logits, class_mass=q.sum(1, keepdim=True),
                 normalizer=((q/q.sum(1, keepdim=True))*mask[:, None, :]).sum(-1, keepdim=True))
        return d
