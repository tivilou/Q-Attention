"""Fixed-readout identities only: no task rows, training, network or COS."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parent
rng = np.random.default_rng(88)
eps = 1e-6
residuals = []
slacks = []
for trial in range(256):
    relations, tokens = 5, 17
    mask = rng.random(tokens) > .2
    mask[0] = True
    base = np.exp(rng.uniform(-20, 0, (relations, tokens)))
    gate = rng.uniform(0, 1, tokens)
    gate[trial % tokens] = 0  # EPS-dominated entries are deliberately included.
    gate[-1] = 1.00001  # Do not require numerically exact g<=1.
    value = rng.normal(size=(relations, tokens))
    bias = rng.normal(size=relations)
    raw0 = (base + eps) * mask
    raw1 = (base * gate + eps) * mask
    alpha0 = raw0 / raw0.sum(-1, keepdims=True)
    alpha1 = raw1 / raw1.sum(-1, keepdims=True)
    c = (base * gate + eps) / (base + eps)
    mean_c = (alpha0 * c).sum(-1)
    mean_u = (alpha0 * value).sum(-1)
    covariance = (alpha0 * c * value).sum(-1) - mean_c * mean_u
    z0 = mean_u + bias
    z1 = (alpha1 * value).sum(-1) + bias
    delta = covariance / mean_c
    residuals.append(float(np.max(np.abs(z1 - z0 - delta))))
    assert np.allclose(z1 - z0, delta, atol=2e-14, rtol=0)
    cmax = c[:, mask].max(-1)
    lam = 1 - mean_c / cmax
    tv = .5 * np.abs(alpha1 - alpha0).sum(-1)
    span = value[:, mask].max(-1) - value[:, mask].min(-1)
    bound = lam * span
    assert np.all(tv <= lam + 2e-14)
    assert np.all(np.abs(z1 - z0) <= bound + 2e-14)
    slacks.extend((bound - np.abs(z1 - z0)).tolist())
    for gold in range(relations):
        rival = (gold + 1) % relations
        actual = (z1[gold] - z1[rival]) - (z0[gold] - z0[rival])
        assert abs(actual - (delta[gold] - delta[rival])) < 2e-14

# Raising the gold logit can still turn a correct argmax into an error.
# This is a readout-port fixture, not a native joint-reachability assertion.
g = np.array([1., .1, 1.])
w = g + eps
a1 = w / w.sum()
u = np.array([[1., 0., 0.], [2., 0., 0.]])
bias = np.array([.4, 0.])
before = u.mean(-1) + bias
after = u @ a1 + bias
assert before.argmax() == 0 and after.argmax() == 1 and after[0] > before[0]
record = dict(
    status='PASS', trials=256, maximum_identity_error=max(residuals),
    minimum_score_bound_slack=min(slacks), epsilon=eps,
    includes_zero_gate=True, includes_gate_above_one=True,
    counterexample=dict(before=before.tolist(), after=after.tolist(),
                        gold_logit_increased=True, gold_prediction_lost=True),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    optimizer_updates=0, task_rows=0, cos_operations=0,
    native_joint_reachability_tested=False, new_theory_certified=False)
(root / 'readout-verification.json').write_text(
    json.dumps(record, indent=2) + '\n', encoding='utf8')
print(json.dumps(record))
