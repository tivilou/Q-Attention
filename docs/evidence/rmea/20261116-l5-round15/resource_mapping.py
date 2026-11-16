"""Algebra-only sanity fixtures. No quantum circuit, learned model or data access."""
import hashlib
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parent


def main():
    eps = 1e-6
    p0 = eps / (1 + eps)
    cases = [([0., 0., 0.], [-2., 0., 2.]),
             ([.8, .2], [2., -1.]),
             ([0., 1e-9, .5, 1.], [-3., 2., 1., -2.])]
    rows = []
    for q, u in cases:
        n = len(q)
        eta = eps / (1 + eps)
        a = [eta + (1 - eta) * x for x in q]
        alpha = [(x + eps) / (sum(q) + n * eps) for x in q]
        conditioned = [x / sum(a) for x in a]
        upper = max(abs(x) for x in u)
        w = [(x + upper) / (2 * upper) for x in u]
        prob = sum(a) / n
        moment = sum(x * y for x, y in zip(a, w)) / n
        direct = sum(x * y for x, y in zip(alpha, u))
        ratio = 2 * upper * moment / prob - upper
        err = max(abs(x - y) for x, y in zip(alpha, conditioned))
        assert err < 1e-14 and abs(direct - ratio) < 1e-13
        assert prob >= p0
        rows.append(dict(q=q, signed_u=u, p=prob, m=moment,
                         weight_error=err, direct_logit=direct, ratio_logit=ratio))

    # This fixture checks the error-accounting inequalities, not AE execution.
    worst_ratio = 0.
    corners = 0
    for error in [1., .1, .01]:
        iterations = math.ceil(16 * math.pi / (error * math.sqrt(p0)))
        for prob in [p0, 1e-5, .001, .01, .1, 1.]:
            e_p = 2 * math.pi * math.sqrt(prob * (1 - prob)) / iterations + math.pi**2 / iterations**2
            assert e_p <= error * prob / 4
            for conditional in [0., .1, .5, .9, 1.]:
                moment = prob * conditional
                e_m = 2 * math.pi * math.sqrt(moment * (1 - moment)) / iterations + math.pi**2 / iterations**2
                assert e_m <= error * prob / 4
                for sign_p in [-1, 1]:
                    for sign_m in [-1, 1]:
                        estimate = min(1., max(0., (moment + sign_m * e_m) / (prob + sign_p * e_p)))
                        actual = abs(estimate - conditional)
                        assert actual <= error + 1e-14
                        worst_ratio = max(worst_ratio, actual / error)
                        corners += 1

    raw = dict(schema='qattn.conditional-resource-algebra.v1', status='PASS',
               epsilon=eps, lower_bound_p=p0, inverse_sqrt_lower_bound=1 / math.sqrt(p0),
               conditioning_fixtures=rows, ratio_error_corners=corners,
               largest_relative_to_declared_error=worst_ratio,
               theorem_iteration_formula='ceil(16*pi/(e*sqrt(p0)))',
               proof_scope='Ideal coherent query access, 0<e<=1; known AE bound plus elementary ratio perturbation.',
               known_source='Montanaro arXiv:1504.06987v3, p8 Theorem2 and pp8-9 Algorithm1/Theorem3.',
               limits=['No quantum algorithm executed or finite-precision gate count certified.',
                       'No trained model, dataset, checkpoint or prediction array loaded.',
                       'No end-to-end speedup, task-benefit, optimality or novelty claim.',
                       'Exact constant-d classical Born evaluation is an available baseline.'],
               training_runs=0, model_forwards=0, l5_accepted=False,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (P / 'resource-mapping-audit.json').write_bytes((json.dumps(raw, indent=2) + '\n').encode())
    print(json.dumps(dict(status='PASS', conditioning_fixtures=len(rows), ratio_error_corners=corners,
                         inverse_sqrt_p0=raw['inverse_sqrt_lower_bound'], model_forwards=0, l5_accepted=False)))


if __name__ == '__main__':
    main()
