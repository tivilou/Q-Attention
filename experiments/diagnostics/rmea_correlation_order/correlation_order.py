"""Three-qubit correlation-order channel; established depolarizing primitives."""
import itertools,math
import torch


def reduced_density(state,keep):
    """Wire0 is MSB, matching native circuit. State shape [...,8]."""
    if state.shape[-1]!=8:raise ValueError('Three qubits required')
    keep=tuple(keep);rest=tuple(i for i in range(3) if i not in keep)
    if not keep or len(set(keep))!=len(keep) or any(i not in (0,1,2) for i in keep):raise ValueError('Bad subsystem')
    n=state.ndim-1
    v=state.reshape(state.shape[:-1]+(2,2,2)).permute(tuple(range(n))+tuple(n+i for i in keep+rest))
    v=v.reshape(state.shape[:-1]+(2**len(keep),2**len(rest)))
    return v@v.mH


def order_terms(psi,phi):
    """Return native p plus sum over weight1,2,3 Pauli products C_k.

    psi: [B,T,8], phi:[R,8]. The expansion is p=(1+C1+C2+C3)/8.
    No label, utility or optimizer state enters this operation.
    """
    if psi.ndim!=3 or phi.ndim!=2:raise ValueError('Expected B,T,8 and R,8')
    p=torch.einsum('rd,btd->brt',phi.conj(),psi).abs().square()
    sums=[]
    for size in (1,2):
        overlaps=[]
        for keep in itertools.combinations(range(3),size):
            a=reduced_density(psi,keep);b=reduced_density(phi,keep)
            overlaps.append(torch.einsum('rtu,bstu->brs',b.conj(),a).real)
        sums.append(torch.stack(overlaps).sum(0))
    c1=2*sums[0]-3;c2=4*sums[1]-3-2*c1;c3=8*p-1-c1-c2
    return p,torch.stack((c1,c2,c3),-1),sums


def channel_scores(psi,phi,time):
    """Fixed semigroup time t>=0, local D_exp(-t)^tensor3 versus purity-matched isotropic D_kappa."""
    if not math.isfinite(time) or time<0:raise ValueError('Nonnegative finite time required')
    p,c,sums=order_terms(psi,phi);lam=math.exp(-time)
    # Explicit positive mixture over retained subsets, avoiding Pauli cancellation.
    local=lam**3*p+(1-lam)*lam**2*sums[1]/2+(1-lam)**2*lam*sums[0]/4+(1-lam)**3/8
    _,self_c,_=order_terms(phi[None],phi)
    weights=self_c[0].diagonal(dim1=0,dim2=1).T
    # For pure phi, sum_k sum_{wt(P)=k} <P>_phi^2 = 7.
    if float(weights.min()) < -1e-9 or float((weights.sum(-1)-7).abs().max())>1e-9:raise ValueError('Pure-effect correlation expansion failed')
    k=torch.arange(1,4,device=phi.device,dtype=phi.real.dtype)
    kappa=((weights*torch.exp(-2*time*k)).sum(-1)/7).sqrt()
    isotropic=kappa[None,:,None]*p+(1-kappa[None,:,None])/8
    derivative_local=-(c*k).sum(-1)/8
    kappa_prime=-(weights*k).sum(-1)/7
    derivative_isotropic=kappa_prime[None,:,None]*(p-1/8)
    purity_local=(1+(weights*torch.exp(-2*time*k)).sum(-1))/8
    purity_isotropic=(1+7*kappa.square())/8
    return dict(native=p,local=local,isotropic=isotropic,derivative_local=derivative_local,
                derivative_isotropic=derivative_isotropic,purity_error=(purity_local-purity_isotropic).abs().max(),
                order_terms=c,effect_order_weights=weights,kappa=kappa)
