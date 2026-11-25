"""Exact rational 2x2 checks, independent of the NumPy dense implementation."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

r = Path(__file__).resolve().parent
scope = json.loads((r / 'scope.json').read_bytes())
primary = json.loads((r / 'results.json').read_bytes())
eps = F(1, 1000000)
assert F(str(scope['EPS'])) == eps


def mat(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(2)), F(0)) for j in range(2)] for i in range(2)]


def trace(a):
    return a[0][0] + a[1][1]


def scale(a, s):
    return [[x * s for x in row] for row in a]


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(2)] for i in range(2)]


zero = [[F(1), F(0)], [F(0), F(0)]]
one = [[F(0), F(0)], [F(0), F(1)]]
plus = [[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]]
minus = [[F(1, 2), F(-1, 2)], [F(-1, 2), F(1, 2)]]
maxmix = [[F(1, 2), F(0)], [F(0), F(1, 2)]]
X = [[F(0), F(1)], [F(1), F(0)]]
Z = [[F(1), F(0)], [F(0), F(-1)]]
checked = skipped_zero_events = 0
for E in (zero, one, plus, minus):
    for rho in (zero, one, plus, minus, maxmix):
        p = trace(mat(E, rho))
        for V in (X, Z, scale(add(X, Z), F(1, 2))):
            if not p:
                skipped_zero_events += 1
                continue
            post = scale(mat(mat(E, rho), E), 1 / p)
            assert post == E
            assert trace(mat(V, post)) == trace(mat(V, E))
            checked += 1


def dephase(a):
    return [[a[0][0], F(0)], [F(0), a[1][1]]]


def pool(E, V, states, rd=False, vd=False):
    p = [trace(mat(dephase(E), dephase(q))) if rd else trace(mat(E, q)) for q in states]
    u = [trace(mat(dephase(V), dephase(q))) if vd else trace(mat(V, q)) for q in states]
    return sum(((pr + eps) * val for pr, val in zip(p, u)), F(0)) / sum((pr + eps for pr in p), F(0))


assert add(plus, minus) == add(zero, one)
exact_contrast = {}
for name, states in [('X_pair', [plus, minus]), ('Z_pair', [zero, one])]:
    exact_contrast[name] = {}
    for rr in ('Q', 'D'):
        for vv in ('Q', 'D'):
            label = 'route_' + rr + '_value_' + vv
            value = pool(plus, X, states, rr == 'D', vv == 'D')
            expected = 1 / (1 + 2 * eps) if (name, rr, vv) == ('X_pair', 'Q', 'Q') else F(0)
            assert value == expected
            assert abs(float(value) - primary['same_density_contrast'][name][label]) < 1e-14
            exact_contrast[name][label] = str(value)
    p = [trace(mat(plus, q)) for q in states]
    n = [trace(mat(mat(plus, X), q)) for q in states]
    regularized = sum(n, F(0)) / (sum(p, F(0)) + 2 * eps)
    assert regularized == 1 / (1 + 2 * eps)

weak = []
for power in scope['edge_checks']['weak_value_rational_parameter_powers']:
    t = F(1, 10 ** power)
    a, b = 2 * t / (1 + t * t), (1 - t * t) / (1 + t * t)
    rho = [[a * a, a * b], [a * b, b * b]]
    assert trace(rho) == 1 and mat(rho, rho) == rho
    p, n = trace(mat(zero, rho)), trace(mat(mat(zero, X), rho))
    w = n / p
    assert w == (1 - t * t) / (2 * t)
    value = (p + eps) * w / (1 + p + 2 * eps)
    old = next(x for x in primary['weak_value_pole_table'] if x['t_power'] == power)
    assert abs(float(value) - old['native_EPS_pool']) <= max(1e-12, 1e-12 * abs(float(value)))
    weak.append({'t_power': power, 'exact_p': str(p), 'exact_n': str(n),
                 'exact_weak_value': str(w), 'exact_native_EPS_pool': str(value), 'native_EPS_pool_float': float(value)})
assert F(weak[-1]['exact_native_EPS_pool']) > 100000
assert primary['native_head_forwards'] == primary['native_circuit_executions'] == primary['task_rows'] == 0
out = {'status': 'PASS_INDEPENDENT_EXACT_RATIONAL_CHECKS', 'rank_one_positive_events_checked': checked,
       'zero_events_left_undefined': skipped_zero_events, 'exact_same_density_contrast': exact_contrast,
       'exact_weak_value_table': weak, 'comparison_to_primary': 'PASS',
       'scope_sha256': hashlib.sha256((r / 'scope.json').read_bytes()).hexdigest(),
       'primary_results_sha256': hashlib.sha256((r / 'results.json').read_bytes()).hexdigest(),
       'not_external_research_review': True, 'not_task_confirmation': True}
(r / 'independent-results.json').write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + '\n').encode())
print(json.dumps({'status': out['status'], 'positive_event_checks': checked, 'zero_events_left_undefined': skipped_zero_events,
                  'same_density_counterexample': 'EXACT', 'weak_value_table': 'EXACT', 'new_theory': False}))
