"""Small mathematical filters; no dataset, model training, or resource allocation."""
from pathlib import Path
import json
import numpy as np


def logloss(z, gold=0):
    m = z.max()
    return float(m + np.log(np.exp(z-m).sum()) - z[gold])


def main():
    rng = np.random.default_rng(413)
    eps = 1e-6
    maximum_derivative_error = 0.
    maximum_vertex_gap = 0.
    interior_loss_witness = None
    # Physical qubit states and rank-one effects, not arbitrary probability pairs.
    for trial in range(512):
        psi = rng.normal(size=(4,2)) + 1j*rng.normal(size=(4,2))
        psi /= np.linalg.norm(psi,axis=-1,keepdims=True)
        phi = rng.normal(size=(3,2)) + 1j*rng.normal(size=(3,2))
        phi /= np.linalg.norm(phi,axis=-1,keepdims=True)
        coherent = np.abs(phi.conj() @ psi.T)**2
        diagonal = np.abs(phi)**2 @ (np.abs(psi)**2).T
        utility = rng.normal(size=(3,4))
        bias = rng.normal(size=3)
        A = ((coherent+eps)*utility).sum(-1)
        B = ((diagonal-coherent)*utility).sum(-1)
        C = (coherent+eps).sum(-1)
        D = (diagonal-coherent).sum(-1)
        def score(lam):
            return (A+lam*B)/(C+lam*D)+bias
        lam=.413; step=1e-5
        numeric=(score(lam+step)-score(lam-step))/(2*step)
        derivative=(B*C-A*D)/(C+lam*D)**2
        maximum_derivative_error=max(maximum_derivative_error,float(np.max(np.abs(numeric-derivative))))
        grid=np.array([score(x) for x in np.linspace(0,1,101)])
        maximum_vertex_gap=max(maximum_vertex_gap,float(np.max(grid.max(0)-np.maximum(score(0),score(1)))))
        if interior_loss_witness is None and logloss(score(.5)) < min(logloss(score(0)),logloss(score(1)))-.01:
            interior_loss_witness=dict(trial=trial,psi_real=psi.real.tolist(),psi_imag=psi.imag.tolist(),
                phi_real=phi.real.tolist(),phi_imag=phi.imag.tolist(),utility=utility.tolist(),bias=bias.tolist(),
                eps=eps,losses={str(x):logloss(score(x)) for x in (0.,.5,1.)},
                logits={str(x):score(x).tolist() for x in (0.,.5,1.)})
    assert maximum_derivative_error<1e-8 and maximum_vertex_gap<1e-12
    assert interior_loss_witness is not None
    H=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    psi=H@np.array([1.,0.]);rho=np.outer(psi,psi)
    final=H@rho@H.T;middle=H@np.diag(np.diag(rho))@H.T
    effect=np.diag([0.,1.])
    witness=dict(coherent=float(np.trace(effect@final)),
        output_dephased=float(np.trace(effect@np.diag(np.diag(final)))),
        interlayer_dephased=float(np.trace(effect@middle)))
    assert abs(witness['coherent'])<1e-12 and abs(witness['output_dephased'])<1e-12
    assert abs(witness['interlayer_dephased']-.5)<1e-12
    result=dict(status='PASS',scope='Synthetic mathematics only, not task utility or novelty',trials=512,
        scalar_row_derivative_max_error=maximum_derivative_error,scalar_row_grid_above_endpoint_max=maximum_vertex_gap,
        physical_interior_multiclass_CE_witness=interior_loss_witness,
        noncommuting_channel_placement_witness=witness,
        conclusions=['Fixed scalar output-dephasing has monotone individual row utility at fixed parameters',
            'Individual row monotonicity does not rule out an interior multiclass CE improvement',
            'Interlayer dephasing need not equal a blend of coherent and output-dephased endpoints',
            'All are standard channel/algebra facts; none certifies natural-text utility or new T'])
    p=Path(__file__).with_name('static-channel-check.json')
    p.write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k!='physical_interior_multiclass_CE_witness'}))
    print('interior CE losses:',interior_loss_witness['losses'])


if __name__=='__main__':main()
