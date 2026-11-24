"""Offline algebra for the two fixed mixed controls; no data/model loading."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'modified/baselines/retacred/rmea_number_optical.py'
EPS = 1e-6
C = 2 ** 6


def weights(p, epsilon):
    return (p + epsilon) / (p.sum(-1, keepdims=True) + p.shape[-1] * epsilon)


def main():
    rng = np.random.default_rng(20261125)
    errors = []
    for n in (1, 2, 7, 37):
        p = np.exp(rng.uniform(-50, -1, (128, n)))
        p[0] = 0
        total = p.sum(-1, keepdims=True)
        beta = np.divide(p, total, out=np.full_like(p, 1 / n), where=total != 0)
        lam = n * EPS / (total + n * EPS)
        alpha = weights(p, EPS)
        scaled = weights(C * p, EPS)
        errors.append(float(np.abs(alpha - ((1 - lam) * beta + lam / n)).max()))
        errors.append(float(np.abs(scaled - weights(p, EPS / C)).max()))
        values = rng.normal(size=p.shape)
        lam_c = n * EPS / (C * total + n * EPS)
        predicted = ((lam_c - lam) *
                     (values.mean(-1, keepdims=True) - (beta * values).sum(-1, keepdims=True)))
        actual = ((scaled - alpha) * values).sum(-1, keepdims=True)
        errors.append(float(np.abs(predicted - actual).max()))
    # t = mean(raw six-mode probability)/EPS, not a fitted task quantity.
    t = np.logspace(-8, 8, 100001)
    gap = 1 / (1 + t) - 1 / (1 + C * t)
    t_star = 1 / np.sqrt(C)
    exact_max = (np.sqrt(C) - 1) / (np.sqrt(C) + 1)
    at_star = 1 / (1 + t_star) - 1 / (1 + C * t_star)
    assert gap.max() <= exact_max + 1e-14
    assert abs(at_star - 7 / 9) < 1e-14
    p = np.array([[EPS / 4, 0.]])
    raw = weights(p, EPS)
    scaled = weights(C * p, EPS)
    assert np.allclose(raw, [[5 / 9, 4 / 9]], rtol=0, atol=1e-15)
    assert np.allclose(scaled, [[17 / 18, 1 / 18]], rtol=0, atol=1e-15)
    v = np.array([[1., -1.]])
    score_raw = float((raw * v).sum())
    score_scaled = float((scaled * v).sum())
    assert score_raw - .5 < 0 < score_scaled - .5
    assert -score_raw + .5 > 0 > -score_scaled + .5
    assert max(errors) < 2e-14
    result = dict(
        scope='fixed-parameter normalized routing algebra; not task evidence or novel theory',
        implementation_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        six_mode_scale=C, epsilon=EPS, max_identity_error=max(errors),
        exact_max_uniform_mixture_gap=exact_max,
        mean_raw_probability_at_max=EPS * t_star,
        witness=dict(raw_weights=raw.tolist(), scaled_weights=scaled.tolist(),
                     raw_score=score_raw, scaled_score=score_scaled,
                     both_margin_directions_demonstrated=True),
        task_examples_read=0, model_or_feature_loads=0, training_updates=0,
        scientific_protocol_changed=False, maturity_changed=False,
    )
    (ROOT / 'floor-interpretation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
