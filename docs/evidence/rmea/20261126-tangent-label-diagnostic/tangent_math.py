"""Fixed-C initial slopes; no label input, parameter update, or finite query step."""
import numpy as np

NAMES = ('g', 'variance', 'self', 'upper')

def slopes_torch(detail, mask):
    import torch
    psi, phi, p, u, alpha = (detail[k] for k in ('psi','phi','probability','utilities','alpha'))
    valid = mask[:,None,:]
    D = detail['normalizer'].squeeze(-1)
    z = (alpha*u).sum(-1)
    delta = (u-z[:,:,None])*valid
    a = torch.einsum('btd,rd->brt', psi.conj(), phi)
    # Full projected token vectors, retained only for this minibatch.
    v = a[:,:,:,None]*psi[:,None,:,:]-p[:,:,:,None]*phi[None,:,None,:]
    v = v*valid[:,:,:,None]
    r = (delta[:,:,:,None]*v).sum(-2)
    g = 2*r.abs().square().sum(-1)/D
    vn = v.abs().square().sum(-1)
    self_ = 2*(delta.square()*vn).sum(-1)/D
    variance = (alpha*delta.square()).sum(-1)
    gamma = (vn/(p+1e-6)).sum(-1)
    upper = 2*gamma*variance
    out = dict(g=g, variance=variance, self=self_, upper=upper, gamma=gamma,
               cross=g-self_, z=z, residual=r)
    return out

def operator_numpy(psi, phi, u, eps=1e-6):
    """Separate dense A/B/C implementation for a single unpadded case."""
    rho = np.einsum('ti,tj->tij',psi,psi.conj())
    B = rho.sum(0)+len(psi)*eps*np.eye(8)
    A = np.einsum('rt,tij->rij',u,rho)+eps*u.sum(-1)[:,None,None]*np.eye(8)
    D = np.einsum('ri,ij,rj->r',phi.conj(),B,phi).real
    z = np.einsum('ri,rij,rj->r',phi.conj(),A,phi).real/D
    C = A-z[:,None,None]*B
    residual = np.einsum('rij,rj->ri',C,phi)
    g = 2*np.sum(abs(residual)**2,-1)/D
    return dict(g=g,z=z,residual=residual,A=A,B=B,C=C,D=D)

def statistics(scores, logits, target):
    """Strongest-other fixed native rival; index order resolves exact ties."""
    n,r = logits.shape
    assert r == 40 and scores.shape == (n,40,4)
    correct = logits.argmax(-1)==target
    rival_logits = logits.copy(); rival_logits[np.arange(n),target] = -np.inf
    rival = rival_logits.argmax(-1)
    delta = scores[np.arange(n),target]-scores[np.arange(n),rival]
    tol = 1e-10*np.maximum(1,np.max(abs(scores),axis=1))
    positive,negative = delta>tol,delta<-tol
    metrics = {}
    for k,name in enumerate(NAMES):
        def group(take):
            count=int(take.sum())
            return dict(n=count,positive_count=int(positive[take,k].sum()),negative_count=int(negative[take,k].sum()),
                positive_fraction=float(positive[take,k].mean()) if count else None,
                negative_fraction=float(negative[take,k].mean()) if count else None,
                mean_gold_minus_rival=float(delta[take,k].mean()) if count else None)
        metrics[name] = dict(all=group(np.ones(n,bool)),error=group(~correct),correct=group(correct),
                            per_gold_class=[group(target==c) for c in range(40)])
    return metrics,dict(correct=correct,rival=rival,delta=delta,tolerance=tol)

def decide(metrics, numerical):
    g=metrics['g']; controls=[metrics[k] for k in NAMES[1:]]
    sufficient = g['error']['n']>=50 and g['correct']['n']>=50
    advantage = None if not sufficient else g['error']['positive_fraction']-max(x['error']['positive_fraction'] for x in controls)
    harm = None if not sufficient else g['correct']['negative_fraction']-min(x['correct']['negative_fraction'] for x in controls)
    gates=dict(numerical=bool(numerical),counts=sufficient,
               error_advantage=advantage is not None and advantage>=.02,
               correct_harm=harm is not None and harm<=.01)
    return dict(gates=gates,error_advantage=advantage,excess_correct_harm=harm,
                decision='RETAIN_EXPLORATORY_DIRECTION' if all(gates.values()) else 'STOP_CURRENT_LABEL_DIRECTION')
