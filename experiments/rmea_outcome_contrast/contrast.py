"""Both outcomes of the SAME binary Born measurement; no labels or fitting."""
import numpy as np

EPS = 1e-6
VARIANTS = ('static', 'outcome_contrast', 'dephased_contrast', 'centered_positive')


def branch_weights(probability, mask):
    p = np.asarray(probability, np.float64)
    m = np.asarray(mask, bool)[:, None, :]
    if p.ndim != 3 or p.shape[0] != m.shape[0] or p.shape[2] != m.shape[2]:
        raise ValueError('Bad probability/mask shape')
    if not m.any(-1).all() or not np.isfinite(p).all():
        raise ValueError('Invalid mask or nonfinite probability')
    # Only floating-point excursions of normalized complex64 states are clipped.
    if p.min() < -1e-5 or p.max() > 1 + 1e-5:
        raise ValueError('Not a binary probability')
    p = np.clip(p, 0, 1)
    positive = (p + EPS) * m
    negative = (1 - p + EPS) * m
    return positive / positive.sum(-1, keepdims=True), negative / negative.sum(-1, keepdims=True)


def intervention(saved):
    # The two effects are P_r and I-P_r. No second circuit column or pair term.
    psi = np.asarray(saved['psi'], np.complex128)
    phi = np.asarray(saved['phi'], np.complex128)
    p = np.abs(np.einsum('rd,btd->brt', phi.conj(), psi)) ** 2
    pd = np.einsum('rd,btd->brt', np.abs(phi) ** 2, np.abs(psi) ** 2)
    mask = np.asarray(saved['mask'], bool)
    ap, am = branch_weights(p, mask)
    dp, dm = branch_weights(pd, mask)
    uniform = mask[:, None, :] / mask.sum(-1)[:, None, None]
    weights = dict(static=ap, outcome_contrast=ap-am,
                   dephased_contrast=dp-dm, centered_positive=ap-uniform)
    utility = np.asarray(saved['utilities'], np.float64)
    bias = np.asarray(saved['bias'], np.float64)
    logits = {k: (w*utility).sum(-1)+bias for k,w in weights.items()}
    np.testing.assert_allclose(ap, saved['alpha'], atol=5e-6, rtol=3e-4)
    np.testing.assert_allclose(logits['static'], saved['logits'], atol=5e-5, rtol=3e-4)
    for name in VARIANTS:
        assert np.isfinite(logits[name]).all()
        np.testing.assert_allclose(weights[name].sum(-1), 1 if name=='static' else 0, atol=1e-12)
        assert not weights[name][np.broadcast_to(~mask[:,None,:], weights[name].shape)].any()
    t = mask.sum(-1)[:,None]
    q = (np.clip(p,0,1)+EPS)/(1+2*EPS)
    mean = (q*mask[:,None,:]).sum(-1)/t
    umean = (utility*mask[:,None,:]).sum(-1)/t
    cov = ((q-mean[...,None])*(utility-umean[...,None])*mask[:,None,:]).sum(-1)/t
    np.testing.assert_allclose(logits['outcome_contrast']-bias, cov/(mean*(1-mean)), atol=2e-7, rtol=3e-7)
    np.testing.assert_allclose(weights['outcome_contrast'], weights['centered_positive']/(1-mean[...,None]), atol=1e-10, rtol=1e-8)
    return weights, logits, dict(mean_probability=float(mean.mean()),
        contrast_l1_mean=float(np.abs(weights['outcome_contrast']).sum(-1).mean()),
        contrast_logit_linf=float(np.abs(logits['outcome_contrast']-logits['static']).max()))
