"""Reanalyse two authenticated stored outputs; no model, labels, or new inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EXPECTED = [
    '25827062f443b8394263955f0facbf8de20ffb76e5dfded9e883e279e8782ad6',
    'ce400bc5a2ff7e016fb790095a20b68c4bbdfcc8816847a0dba4cb756b8acd1f',
]
EPS = 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace0', required=True, type=Path)
    parser.add_argument('--trace1', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for index, path in enumerate([args.trace0, args.trace1]):
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED[index]
        with np.load(path, allow_pickle=False) as saved:
            p = saved['probability'].astype(np.float64)
            alpha = saved['alpha'].astype(np.float64)
            mask = saved['mask'].astype(bool)
        assert p.ndim == 3 and p.shape[:2] == (1, 40) and mask.shape == (1, p.shape[-1])
        p, alpha = p[0, :, mask[0]], alpha[0, :, mask[0]]
        # Boolean advanced indexing moves the selected token axis to the front.
        assert p.shape[1] == 40
        p, alpha = p.T, alpha.T
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 + 2e-5
        means = p.mean(-1, keepdims=True)
        assert (means > 0).all()
        conditional = p / p.sum(-1, keepdims=True)
        weight = means / (means + EPS)
        mixture = weight * conditional + (1 - weight) / p.shape[-1]
        direct = (p + EPS) / (p + EPS).sum(-1, keepdims=True)
        assert np.max(abs(mixture - direct)) < 1e-14
        residual = np.max(abs(alpha - mixture))
        assert residual < 2e-6
        # The deterministic bound uses the reconstructed real-valued probabilities.
        tv = abs(mixture - conditional).sum(-1) / 2
        floor = (1 - weight).ravel()
        assert (tv <= floor + 1e-14).all()
        rows.append(dict(case_index=index, authenticated_trace_sha256=EXPECTED[index], labels=40,
                         valid_tokens=p.shape[-1], mean_born_min=float(means.min()),
                         mean_born_max=float(means.max()), uniform_floor_weight_min=float(floor.min()),
                         uniform_floor_weight_max=float(floor.max()),
                         native_alpha_vs_ideal_mixture_max_abs=float(residual),
                         epsilon_smoothing_tv_max=float(tv.max()),
                         algebraic_identity_max_abs=float(np.max(abs(mixture-direct)))))
    result = dict(schema='qattn.normalizer-observable-audit.v1', status='PASS_SCOPED_STORED_ARRAY_AUDIT',
                  epsilon=EPS, numpy=np.__version__, cases=rows, stored_attention_rows=80,
                  selection='Original fixed first2 stable-ID-hash selected validation traces, seed29 quantum selected checkpoint.',
                  measurements='Derived from original stored probability/alpha/mask arrays; no model execution.',
                  limits=['Two fixed cases, not benchmark or training-cohort distributions.',
                          'Tiny floor weights on these cases do not establish a population bound or irrelevance.',
                          'No utility, calibration, semantic-null, information-sufficiency or L5 conclusion.',
                          'Representation loss from alpha alone is not loss of all head information; values also depend on inputs.'],
                  model_jobs=0, training_jobs=0, cos_requests=0, labels_or_outcomes_read=False, L5='NOT_ACCEPTED')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(dict(status=result['status'], stored_attention_rows=80,
                          native_residual_max=max(row['native_alpha_vs_ideal_mixture_max_abs'] for row in rows),
                          uniform_floor_weight_max=max(row['uniform_floor_weight_max'] for row in rows),
                          model_jobs=0, L5='NOT_ACCEPTED')))


if __name__ == '__main__':
    main()
