"""R0 dense identities/counterexamples; no model, dataset or training execution."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
EPS = 1e-6
rng = np.random.default_rng(20261123)


def unitary(d):
    z = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * np.exp(1j*np.angle(np.diag(r)))[None, :]


def ry(a):
    return np.array([[np.cos(a/2), -np.sin(a/2)],
                     [np.sin(a/2), np.cos(a/2)]], complex)


def rz(a):
    return np.diag(np.exp(np.array([-.5j, .5j])*a))


def native(a):
    u = np.eye(8, dtype=complex)
    for layer in range(2):
        for wire in range(3):
            g = rz(a[layer, wire, 1]) @ ry(a[layer, wire, 0])
            g = np.kron(np.kron(np.eye(2**wire), g), np.eye(2**(2-wire)))
            u = g @ u
        for c, t in ((0, 1), (1, 2), (2, 0)):
            ids = [i ^ ((1 << (2-t)) if (i >> (2-c)) & 1 else 0) for i in range(8)]
            u = u[ids]
    return u


def alpha(p, eps=EPS):
    return (p+eps)/(p.sum()+len(p)*eps)


def correlator(u, v):
    e = np.eye(len(u), dtype=complex)[:, 0]
    return np.vdot(u@v@e, v@u@e)


def main():
    errors = {k: 0. for k in ('proportional_score', 'normalization_mixture',
        'eps0_cancellation', 'finite_logit', 'generic_effect',
        'commutator_otoc', 'commutator_symmetry', 'quadrature_orientation')}
    x = np.kron(np.array([[0, 1], [1, 0]]), np.eye(4))
    z = np.kron(np.diag([1, -1]), np.eye(4))
    for _ in range(100):
        state = unitary(8); phi = unitary(8)[:, 0]
        a, b = x, z
        theta = float(rng.uniform(-np.pi, np.pi))
        c = abs(-1+np.exp(1j*theta))**2/4
        q = abs(phi.conj() @ b @ a @ state)**2
        p = abs(phi.conj() @ ((a@b+np.exp(1j*theta)*b@a)/2) @ state)**2
        lam = c*q.sum()/(c*q.sum()+len(q)*EPS)
        mixed = lam*q/q.sum()+(1-lam)/len(q)
        errors['proportional_score'] = max(errors['proportional_score'], float(abs(p-c*q).max()))
        errors['normalization_mixture'] = max(errors['normalization_mixture'], float(abs(alpha(p)-mixed).max()))
        errors['eps0_cancellation'] = max(errors['eps0_cancellation'], float(abs(alpha(p,0)-alpha(q,0)).max()))
        values = rng.normal(size=len(q))
        lam1 = q.sum()/(q.sum()+len(q)*EPS)
        exact = (lam-lam1)*(q@values/q.sum()-values.mean())
        errors['finite_logit'] = max(errors['finite_logit'], float(abs((alpha(p)-alpha(q))@values-exact)))
        assert abs(exact) <= np.ptp(values)*abs(lam-lam1)+1e-12
        a, b = unitary(8), unitary(8)
        k = (a@b+np.exp(1j*theta)*b@a)/2
        chi = k.conj().T@phi
        errors['generic_effect'] = max(errors['generic_effect'], float(abs(abs(phi.conj()@k@state)**2-abs(chi.conj()@state)**2).max()))
        assert np.linalg.norm(chi) <= 1+1e-12
        e = np.eye(8)[:, 0]; c_ab = correlator(a,b); c_ba = correlator(b,a)
        minus = np.linalg.norm((a@b-b@a)@e)**2/4
        reverse = np.linalg.norm((b@a-a@b)@e)**2/4
        errors['commutator_otoc'] = max(errors['commutator_otoc'], abs(minus-(1-c_ab.real)/2))
        errors['commutator_symmetry'] = max(errors['commutator_symmetry'], abs(minus-reverse))
        errors['quadrature_orientation'] = max(errors['quadrature_orientation'], abs((1+c_ab.imag)/2+(1+c_ba.imag)/2-1))
    zero = np.zeros((2,3,2)); word = zero.copy(); effect = zero.copy()
    word[0,0,1] = np.pi/2; effect[0,0,0] = np.pi/2
    c0 = native(zero); center = c0.conj().T
    u = [center@native(zero), center@native(word)]; v = center@native(effect)
    e = np.eye(8)[:,0]
    fidelities = np.array([abs(np.vdot(v@e,a@e))**2 for a in u])
    comm = np.array([np.linalg.norm((a@v-v@a)@e)**2/4 for a in u])
    quad = np.array([(1+correlator(a,v).imag)/2 for a in u])
    rho = [(a@e)[:,None]*(a@e).conj()[None,:] for a in u]
    density_error = float(abs(rho[0]-rho[1]).max())
    assert density_error < 1e-12 and np.allclose(fidelities,[.5,.5])
    assert np.allclose(comm,[0,.25])
    assert max(errors.values()) < 1e-12
    out = dict(status='PASS', profile='R0 mathematical identities only', random_cases=100,
        errors=errors, native_completion=dict(density_max_error=density_error,
        native_scores=fidelities.tolist(), commutator_scores=comm.tolist(),
        quadrature_scores=quad.tolist(), native_alpha=alpha(fidelities).tolist(),
        commutator_alpha=alpha(comm).tolist(), quadrature_alpha=alpha(quad).tolist(),
        raw_finite_angle_parameter=float(np.arctanh(.5))),
        scope='Known-tool/source-correspondence fixtures; no R2 plugin execution, task/model/dataset, novelty, training or L5 claim',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'algebra-check.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
