"""Fixed small dense-operator audit; no native head, datasets or training."""
from pathlib import Path
import json
import time
import numpy as np

r = Path(__file__).resolve().parent
cfg = json.loads((r / 'scope.json').read_bytes())
rng = np.random.default_rng(cfg['random_seed'])
eps, d = cfg['EPS'], cfg['dimension']
errors = dict(native_ratio=0., initial_C_expectation=0., initial_derivative_relative=0.,
              realified_score=0., realified_state=0., attention_sum=0., scalar_tilt_variance=0.)
baseline_min, convex_violation = 0., 0.


def unit(shape):
    v = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def score(a, b, q):
    return float(np.vdot(q, a @ q).real / np.vdot(q, b @ q).real)


def evolve(c, phi, s):
    lam, vec = np.linalg.eigh(c)
    shift = lam.max() if s >= 0 else lam.min()
    w = vec @ (np.exp(s * (lam - shift)) * (vec.conj().T @ phi))
    return w / np.linalg.norm(w)


def realify(m):
    return np.block([[m.real, -m.imag], [m.imag, m.real]])


start = time.monotonic()
for case in range(cfg['random_contexts']):
    t = 3 + case % 6
    psi, phis = unit((t, d)), unit((cfg['classes'], d))
    rho = np.einsum('ti,tj->tij', psi, psi.conj())
    bt = rho + eps * np.eye(d)
    b = bt.sum(0)
    utility = rng.normal(size=(cfg['classes'], t))
    for phi, u in zip(phis, utility):
        a = np.einsum('t,tij->ij', u, bt)
        p = abs(psi.conj() @ phi)**2 + eps
        alpha = p / p.sum()
        z0 = score(a, b, phi)
        errors['native_ratio'] = max(errors['native_ratio'], abs(z0 - alpha @ u))
        c = a - z0 * b
        assert np.max(abs(c - c.conj().T)) < cfg['tolerances']['identity_absolute']
        errors['initial_C_expectation'] = max(errors['initial_C_expectation'], abs(np.vdot(phi, c @ phi)))
        analytic = 2 * np.vdot(c @ phi, c @ phi).real / np.vdot(phi, b @ phi).real
        h = cfg['derivative_difference_step']
        numeric = (score(a, b, evolve(c, phi, h)) - score(a, b, evolve(c, phi, -h))) / (2*h)
        errors['initial_derivative_relative'] = max(errors['initial_derivative_relative'], abs(numeric-analytic)/(1+abs(analytic)))
        assert analytic >= 0
        # An exact representation control, not an independently trained real model.
        cr, ar, br = realify(c), realify(a), realify(b)
        pr = np.r_[phi.real, phi.imag]
        for s in cfg['representative_algebra_s']:
            q = evolve(c, phi, s)
            qr = evolve(cr, pr, s)
            errors['realified_state'] = max(errors['realified_state'], float(np.max(abs(qr-np.r_[q.real,q.imag]))))
            errors['realified_score'] = max(errors['realified_score'], abs(score(ar,br,qr)-score(a,b,q)))
            z = score(a,b,q)
            baseline_min = min(baseline_min, z-z0)
            p = abs(psi.conj() @ q)**2 + eps
            alpha_s = p/p.sum()
            errors['attention_sum'] = max(errors['attention_sum'], abs(alpha_s.sum()-1))
            assert np.min(alpha_s) > 0
            convex_violation = max(convex_violation, min(u)-z, z-max(u))
        def tilted(tau):
            weights = alpha*np.exp(tau*u - np.max(tau*u))
            return weights/weights.sum()
        tau = .5  # Fixed derivative fixture, no task setting selected.
        w = tilted(tau)
        variance = w @ (u-(w@u))**2
        numeric = ((tilted(tau+h) @ u)-(tilted(tau-h) @ u))/(2*h)
        errors['scalar_tilt_variance'] = max(errors['scalar_tilt_variance'], abs(variance-numeric))

e = np.eye(d)
minus, plus = (e[0]-e[1])/np.sqrt(2), (e[0]+e[1])/np.sqrt(2)
states = np.stack([minus,minus,minus,plus,e[0],e[1]])
bt = np.einsum('ti,tj->tij',states,states) + eps*np.eye(d)
b, a = bt.sum(0), np.einsum('t,tij->ij',np.array([0,0,0,0,-1,1]),bt)
assert abs(score(a,b,plus)) < 1e-14
nonmonotone = []
for x in (6,100):
    q = e[0]+x*e[1]
    actual = score(a,b,q)
    reference = (x*x-1)/((3+6*eps)*(x*x+1)-2*x)
    assert abs(actual-reference) < 1e-14
    nonmonotone.append(actual)
assert nonmonotone[0] > nonmonotone[1] > 0

states = e[:2]
bt = np.einsum('ti,tj->tij',states,states) + eps*np.eye(d)
b = bt.sum(0)
utilities = np.full((40,2),-100.)
utilities[0], utilities[1] = [1,1], [-2,2]
before, after = [], []
for u in utilities:
    a = np.einsum('t,tij->ij',u,bt)
    z = score(a,b,plus)
    before.append(z)
    after.append(score(a,b,evolve(a-z*b,plus,1.)))
before, after = np.array(before), np.array(after)
assert int(before.argmax()) == 0 and int(after.argmax()) == 1
assert np.min(after-before) > -1e-12


def ce(z):
    return float(np.log(np.exp(z-z.max()).sum())+z.max()-z[0])


assert ce(after) > ce(before)
assert errors['initial_derivative_relative'] < cfg['tolerances']['derivative_relative']
assert errors['scalar_tilt_variance'] < cfg['tolerances']['derivative_relative']
assert all(v < cfg['tolerances']['identity_absolute'] for k,v in errors.items() if k not in ('initial_derivative_relative','scalar_tilt_variance'))
assert baseline_min > -cfg['tolerances']['baseline_lower_slack'] and convex_violation < 1e-11
out = dict(status='PASS_BOUNDED_EFFECT_ALGEBRA_CHECKS', random_contexts=cfg['random_contexts'], classes=cfg['classes'],
           random_rows=cfg['random_contexts']*cfg['classes'], algebra_evaluations=cfg['random_contexts']*cfg['classes']*len(cfg['representative_algebra_s']),
           max_errors={k:float(v) for k,v in errors.items()}, baseline_min=float(baseline_min), convex_hull_violation=float(convex_violation),
           nonmonotone_strength=dict(x=[6,100],scores=nonmonotone),
           forty_class_counterexample=dict(original_prediction=int(before.argmax()),updated_prediction=int(after.argmax()),
                                          gold=0, original_first2=before[:2].tolist(),updated_first2=after[:2].tolist(),
                                          original_CE=ce(before),updated_CE=ce(after),all_scores_baseline_nondecreasing=True),
           seconds=time.monotonic()-start, native_ansatz_reachability=False, selected_task_method=False,
           new_theory=False, external_research_review=False, goal='ACTIVE', newL3='NOT_ACCEPTED', newL4T='NOT_ACCEPTED')
(r/'results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
