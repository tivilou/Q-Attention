"""Recompute a scoped exact local-rank result from one authenticated stored trace.
No circuit/model evaluation, labels or predictions. Does not settle global fibers.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

TRACE_SHA = '25827062f443b8394263955f0facbf8de20ffb76e5dfded9e883e279e8782ad6'
EPS = F(1, 1000000)
ZERO = (F(0), F(0))
ONE = (F(1), F(0))

def add(a, b):
    return a[0] + b[0], a[1] + b[1]

def neg(a):
    return -a[0], -a[1]

def sub(a, b):
    return add(a, neg(b))

def mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]

def scale(a, f):
    return a[0]*f, a[1]*f

def conj(a):
    return a[0], -a[1]

def total(items):
    out = ZERO
    for item in items:
        out = add(out, item)
    return out

def from_complex(z):
    return F(float(z.real)), F(float(z.imag))

def squared(z):
    return z[0]**2 + z[1]**2

def rational_string(f):
    return str(f.numerator) + '/' + str(f.denominator)

def projector(v):
    norm = sum(squared(z) for z in v)
    assert norm > 0
    return [[scale(mul(v[i], conj(v[j])), 1/norm) for j in range(8)] for i in range(8)]

def matvec(matrix, vector):
    return [total(mul(matrix[i][j], vector[j]) for j in range(8)) for i in range(8)]

def numeric_column_pivots(matrix, count):
    remaining = list(range(matrix.shape[1]))
    residual = matrix.copy()
    selected = []
    for _ in range(count):
        j = max(remaining, key=lambda k: np.linalg.norm(residual[:, k]))
        v = residual[:, j]
        assert np.linalg.norm(v) > 1e-14
        v = v / np.linalg.norm(v)
        residual -= v[:, None] @ (v @ residual)[None, :]
        selected.append(j)
        remaining.remove(j)
    return selected

def div(a, b):
    return scale(mul(a, conj(b)), 1/squared(b))

def determinant(matrix):
    a = [row[:] for row in matrix]
    value = F(1)
    for j in range(len(a)):
        pivot = max(range(j, len(a)), key=lambda i: abs(a[i][j]))
        if a[pivot][j] == 0:
            return F(0)
        if pivot != j:
            a[j], a[pivot] = a[pivot], a[j]
            value = -value
        p = a[j][j]
        value *= p
        for i in range(j+1, len(a)):
            factor = a[i][j]/p
            for k in range(j, len(a)):
                a[i][k] -= factor*a[j][k]
    return value

def main():
    # Exact minors can exceed Python's default decimal-output length; only this
    # bounded worker's private proof serialization uses the increased limit.
    sys.set_int_max_str_digits(100000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.trace.read_bytes()).hexdigest() == TRACE_SHA
    assert not args.trace.is_symlink()
    with np.load(args.trace, allow_pickle=False) as saved:
        words = saved['psi'][0][saved['mask'][0].astype(bool)].astype(np.complex128)
        query = saved['phi'][0].astype(np.complex128)
    pivot = int(np.argmax(abs(query)))
    coordinates = [j for j in range(8) if j != pivot]
    w = query/query[pivot]
    psi = words/np.linalg.norm(words, axis=1)[:, None]
    c_numeric = np.einsum('ti,tj->tij', psi, psi.conj())+float(EPS)*np.eye(8)[None]
    cw = c_numeric @ w
    n = (w.conj()[None, :]*cw).sum(-1).real
    dn = np.concatenate((2*cw[:, coordinates].real, 2*cw[:, coordinates].imag), axis=1)
    denominator = n.sum()
    j_numeric = (dn*denominator-n[:, None]*dn.sum(0)[None])/denominator**2
    singular = np.linalg.svd(j_numeric, compute_uv=False)
    assert singular[-1] > 1e-12
    rows = numeric_column_pivots(j_numeric.T, 14)
    q = [from_complex(z) for z in query]
    w_exact = [div(z, q[pivot]) for z in q]
    effects = []
    for word in words:
        a = projector([from_complex(z) for z in word])
        effects.append([[add(a[i][j], (EPS, F(0))) if i == j else a[i][j] for j in range(8)] for i in range(8)])
    cw_exact = [matvec(c, w_exact) for c in effects]
    n_exact = [total(mul(conj(w_exact[k]), v[k]) for k in range(8)) for v in cw_exact]
    assert all(z[1] == 0 and z[0] > 0 for z in n_exact)
    n_exact = [z[0] for z in n_exact]
    dn_exact = [[2*v[k][0] for k in coordinates]+[2*v[k][1] for k in coordinates] for v in cw_exact]
    den_exact = sum(n_exact)
    den_derivative = [sum(v[k] for v in dn_exact) for k in range(14)]
    jacobian = [[(dn_exact[t][k]*den_exact-n_exact[t]*den_derivative[k])/den_exact**2 for k in range(14)] for t in rows]
    det = determinant(jacobian)
    assert det != 0
    private = dict(trace_sha256=TRACE_SHA, projective_chart_pivot=pivot, selected_alpha_rows=rows,
                   exact_minor_determinant=rational_string(det))
    raw = (json.dumps(private, indent=2)+'\n').encode()
    args.output.with_name('local-rank-certificate-private.json').write_bytes(raw)
    result = dict(schema='qattn.local-observability.audit.v1', status='EXACT_RATIONAL_LOCAL_RANK14_PASS',
        input_trace_sha256=TRACE_SHA, trace_index=0, label_row=0, valid_tokens=40,
        rank=14, exact_minor_nonzero=True, numeric_jacobian_singular_min=float(singular[-1]),
        numeric_jacobian_singular_max=float(singular[0]),
        certificate_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Local pure-state identifiability in a projective chart of the unit-normalized rational stored-state frame.',
        implies='Inverse function theorem gives some neighborhood with a unique query ray and hence locally determined mean Born response.',
        not_proven=['Global identifiability or absence of distant native collisions.',
                    'Exact circuit vectors, a quantified neighborhood radius, or numerically stable reconstruction.',
                    'Predictive relevance, a useful gate, new architecture, formal utility or L5.'],
        model_jobs=0, circuit_evaluations=0, cos_requests=0, labels_or_outcomes_read=False, candidate='NONE', L5='NOT_ACCEPTED')
    args.output.write_bytes((json.dumps(result, indent=2)+'\n').encode())
    print(json.dumps({k:result[k] for k in ['status','rank','numeric_jacobian_singular_min','model_jobs','L5']}))

if __name__ == '__main__':
    main()
