"""Small dense equation check; no model fitting, data, GPU, or runtime mutation."""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
EPS = 1e-6

def ry(x):
    return np.array([[math.cos(x/2), -math.sin(x/2)],
                     [math.sin(x/2), math.cos(x/2)]], dtype=complex)

def rz(x):
    return np.diag(np.exp(np.array([-0.5j*x, 0.5j*x])))

def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)

def cnot(control, target):
    out = np.zeros((8, 8), dtype=complex)
    for x in range(8):
        y = x ^ (1 << (2-target)) if (x >> (2-control)) & 1 else x
        out[y, x] = 1
    return out

F = cnot(2, 0) @ cnot(1, 2) @ cnot(0, 1)

def dense(angles):
    out = np.eye(8, dtype=complex)
    for layer in angles:
        gates = [rz(z) @ ry(y) for y, z in layer]
        out = F @ kron3(*gates) @ out
    return out

def measurements(w):
    paths = w[0, :] * w[:, 0]
    p1 = float(abs(w[0, 0])**2)
    return dict(native=p1, native_square=p1**2,
                dephased=float(np.sum(abs(paths)**2)),
                coherent=float(abs(paths.sum())**2),
                process=float(abs(np.trace(w))**2/64),
                process_times_native=float(abs(np.trace(w))**2*p1/64))

def ce(logits, label):
    a = float(np.max(logits))
    return a + float(np.log(np.exp(logits-a).sum())) - float(logits[label])

rows = []
for phi in [math.pi/6, math.pi/3, 2*math.pi/5]:
    psi = math.pi-phi
    relation = np.zeros((2, 3, 2)); low = np.zeros_like(relation)
    relation[0, 0, 1] = -phi/2; relation[0, 1, 1] = -psi/2
    low[0, 0] = [math.pi/2, phi/2]; low[0, 1, 1] = psi/2
    high = low.copy(); high[0, :2, 0] *= -1; high[0, :2, 1] -= math.pi
    assert max(np.max(abs(x)) for x in [relation, low, high]) < math.pi
    v, u0, u1 = map(dense, [relation, low, high])
    w0, w1 = v.conj().T @ u0, v.conj().T @ u1
    expected = kron3(rz(phi)@ry(math.pi/2), rz(psi), np.eye(2))
    state0, state1 = u0[:, 0], u1[:, 0]
    density_error = float(np.max(abs(np.outer(state0, state0.conj())-np.outer(state1, state1.conj()))))
    a, b = measurements(w0), measurements(w1)
    errors = [density_error, float(np.max(abs(w0-expected))),
              abs(a['coherent']-(1-math.cos(phi))/2),
              abs(b['coherent']-(1+math.cos(phi))/2)]
    for k in ['native', 'native_square', 'dephased', 'process', 'process_times_native']:
        errors.append(abs(a[k]-b[k])); assert min(a[k], b[k]) > 0
    errors.append(abs(a['process_times_native']-math.sin(phi)**2/16))
    # One common Linear(1028,12) realizes all four input/value assignments.
    prep = np.zeros((12, 1028)); prep[:, 0] = np.arctanh(low.reshape(-1)/math.pi)
    prep[:, 1] = np.arctanh(high.reshape(-1)/math.pi)
    for branch, target in [(0, low), (1, high)]:
        for sign in [-1, 1]:
            x = np.zeros(1028); x[branch] = 1; x[2] = sign; x[1027] = 1
            errors.append(float(np.max(abs(math.pi*np.tanh(prep@x).reshape(2,3,2)-target))))
    assert max(errors) < 2e-12
    branches = []
    for s in range(8):
        z = np.diag([(-1)**((s & j).bit_count()) for j in range(8)])
        pair = [measurements(w@z)['coherent'] for w in [w0, w1]]
        direction = -1 if s & 4 else 1
        assert abs(pair[1]-pair[0]-direction*math.cos(phi)) < 2e-12
        branches.append(pair)
    rows.append(dict(phi=phi, low=a, high=b, density_error=density_error,
                     max_equation_error=max(errors), shared_linear_mapper=True,
                     common_branch_coherent_pairs=branches))

phi = math.pi/3
loss_rows = []
for classes, other_bias in [(2, 0.), (40, -4.)]:
    for m in [.5, 4.]:
        q = .6; value_scale = m*(1+2*EPS)/math.cos(phi)
        losses = []
        for s in range(8):
            total = 0.
            low_probability, high_probability = rows[1]['common_branch_coherent_pairs'][s]
            weight_difference = (high_probability-low_probability)/(high_probability+low_probability+2*EPS)
            for sign in [-1, 1]:
                for y in [-1, 1]:
                    weight = .5*(q if sign == y else 1-q)
                    logits = np.full(classes, other_bias)
                    logit = sign*value_scale*weight_difference/2
                    logits[:2] = [logit, -logit]
                    total += weight*ce(logits, 0 if y == 1 else 1)
            losses.append(total)
        base = math.log(2+(classes-2)*math.exp(other_bias))
        predicted = math.log(2*math.cosh(m/2)+(classes-2)*math.exp(other_bias))-(q-.5)*m
        delta = float(np.mean(losses[1:])-losses[0])
        target_delta = 4/7*(2*q-1)*m
        assert abs(losses[0]-predicted) < 2e-12 and abs(delta-target_delta) < 2e-12
        assert delta > 0 and ((losses[0] < base) if m == .5 else (losses[0] > base))
        loss_rows.append(dict(classes=classes, other_bias=other_bias, m=m, q=q,
                              value_scale=value_scale, baseline_ce=base, coherent_ce=losses[0],
                              utility_delta=base-losses[0], seven_branch_delta=delta,
                              formula_error=max(abs(losses[0]-predicted),abs(delta-target_delta))))

out = dict(status='DENSE_MATHEMATICAL_IDENTITY_PASS', source_order='RY then RZ; CNOT01,12,20; two layers; wire0 MSB',
           circuit_rows=rows, task_rows=loss_rows,
           native_source_read='rmea.py9f4bfc57 / frozen_feature_head.pyf4e63a43',
           scope='Constructed input/angle ports and conditional2/40class readouts; not native Torch execution, natural text, training, novelty, or L5 acceptance',
           training_changed=False, gpu_used=False, exit_code=0)
(ROOT/'balanced-completion-check.json').write_text(json.dumps(out,indent=2)+'\n', encoding='utf8')
print(json.dumps(out))
