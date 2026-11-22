"""Explicit fixed-work PSD moment-slice centre and finite nullspace motion."""
import numpy as np

def construct(fs, rhs, c, floor=1e-3, iterations=64):
    d = fs.shape[-1]
    gram = np.einsum('aij,bij->ab', fs.conj(), fs).real
    gram = (gram + gram.T) / 2
    eig = np.linalg.eigvalsh(gram)
    out = dict(admitted=False, constraint_ratio=float(eig[0]/eig[-1]))
    if eig[0] <= 1e-10*eig[-1]:
        out['reason']='ill_conditioned_moment_rows'
        return out
    def moments(z):
        return np.einsum('aij,ij->a', fs.conj(), z).real
    def component(values):
        return np.einsum('a,aij->ij', np.linalg.solve(gram, values), fs)
    def affine(z):
        z=(z+z.conj().T)/2
        return z + component(rhs-moments(z))
    z=np.eye(d,dtype=np.complex128)/d
    for unused in range(iterations):
        z=affine(z)
        values,vectors=np.linalg.eigh(z)
        z=(vectors*np.maximum(values,floor))@vectors.conj().T
    centre=affine(z)
    sigma=float(np.linalg.eigvalsh(centre).min())
    error=float(np.max(np.abs(moments(centre)-rhs)))
    out.update(centre_min_eigenvalue=sigma,centre_equality_error=error)
    if error>1e-12 or sigma<floor/2:
        out['reason']='fixed_work_centre_not_resolved'
        return out
    direction=(c+c.conj().T)/2
    direction=direction-component(moments(direction))
    norm=float(np.linalg.norm(direction))
    if norm<=1e-12:
        out['reason']='utility_orthogonal_to_moment_nullspace'
        return out
    step=sigma/(2*norm)
    plus=centre+step*direction
    minus=centre-step*direction
    out.update(admitted=True,centre=centre,plus=plus,minus=minus,direction_norm=norm,
        step=step,predicted_pair_delta=sigma*norm,
        equality_error=max(float(np.max(np.abs(moments(q)-rhs))) for q in (plus,minus)),
        psd_min_eigenvalue=min(float(np.linalg.eigvalsh(q).min()) for q in (plus,minus)))
    return out
