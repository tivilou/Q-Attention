"""One frozen rule. No labels, fitting, coefficient search or checkpoint edits."""
import numpy as np


def unit_rows(x):
    x=np.asarray(x,np.complex128)
    norms=np.linalg.norm(x,axis=-1)
    if not np.isfinite(x).all() or not np.isfinite(norms).all() or (norms<=1e-12).any():
        return None
    return x/norms[...,None]


def density(x):
    return np.einsum('ti,tj->ij',x,x.conj())/len(x)


def projector(states,roles,mask):
    """Cluster full top eigenspaces; reject numerical rank transition bands."""
    groups=[];gaps=[]
    for column in [2,3]:
        role=mask&(roles[:,column]!=0)
        if not role.any():return None,'missing_role',{}
        vals,vecs=np.linalg.eigh(density(states[role]))
        distances=vals[-1]-vals
        # Near the cluster boundary, abstain rather than pick an unstable rank.
        if ((distances>=.5e-8)&(distances<=1.5e-8)).any():
            return None,'role_cluster_boundary',{}
        keep=distances<=1e-8
        groups.append(vecs[:,keep]);gaps.append(float(distances[~keep].min()) if (~keep).any() else None)
    U,s,_=np.linalg.svd(np.concatenate(groups,axis=1),full_matrices=False)
    if ((s>=1e-7)&(s<=1e-5)).any():return None,'union_rank_boundary',{}
    keep=s>1e-5
    P=U[:,keep]@U[:,keep].conj().T
    assert np.max(abs(P-P.conj().T))<1e-12 and np.max(abs(P@P-P))<1e-10
    return P,'ok',dict(rank=int(keep.sum()),role_excluded_gaps=gaps,
        minimum_retained_singular=float(s[keep].min()))


def attention(states,effects,mask):
    q=abs(np.einsum('rd,td->rt',effects.conj(),states))**2
    raw=(q+1e-6)*mask[None,:]
    alpha=raw/raw.sum(-1,keepdims=True)
    assert np.isfinite(alpha).all()
    return alpha.astype(np.float32)


def transformed_effects(F,effects):
    filtered=np.einsum('ij,rj->ri',F,effects)
    result=unit_rows(filtered)
    assert result is not None
    return result


def route(states,effects,roles,mask,original_alpha):
    """All variants share eligibility/fallback; numeric identity is separately exposed."""
    mask=np.asarray(mask,bool);roles=np.asarray(roles)
    assert mask.any() and states.shape==(len(mask),8) and effects.shape==(40,8)
    assert original_alpha.shape==(40,len(mask)) and original_alpha.dtype==np.float32
    # Ignore padding norms. Its replacement vector has zero weight under mask.
    native=np.asarray(states,np.complex128).copy();native[~mask]=np.eye(8)[0]
    psi=unit_rows(native);phi=unit_rows(effects)
    if psi is None or phi is None:
        return {k:original_alpha.copy() for k in ['identity','protected','unprotected']},dict(reason='zero_or_nonfinite_state',rank=None)
    identity=attention(psi,phi,mask)
    result=dict(identity=identity)
    context=mask&(roles[:,2]==0)&(roles[:,3]==0)
    P,reason,stats=projector(psi,roles,mask)
    if not context.any():reason='no_context'
    if reason!='ok':
        result.update(protected=original_alpha.copy(),unprotected=original_alpha.copy())
        return result,dict(reason=reason,rank=None)
    C=density(psi[context]);Q=np.eye(8)-P
    K=Q@C@Q;K=(K+K.conj().T)/2
    I=np.eye(8)
    for name,N in [('protected',K),('unprotected',C)]:
        F=np.linalg.solve(I+8*N,I)
        w=transformed_effects(F,phi)
        result[name]=attention(psi,w,mask)
    F=np.linalg.solve(I+8*K,I)
    eigen=np.linalg.eigvalsh(F)
    assert eigen.min()>=1/9-1e-10 and eigen.max()<=1+1e-10
    assert np.max(abs(F@P-P))<1e-9
    # This is protected-subspace Born mass, not entity-token attention or utility.
    w=transformed_effects(F,phi)
    before=np.einsum('ri,ij,rj->r',phi.conj(),P,phi).real
    after=np.einsum('ri,ij,rj->r',w.conj(),P,w).real
    assert (after>=before-1e-9).all()
    l1=np.abs(result['protected']-original_alpha).sum(-1)
    return result,dict(reason='ok',**stats,context_tokens=int(context.sum()),
        protected_mass_delta_mean=float((after-before).mean()),
        attention_l1_mean=float(l1.mean()),filter_eigen_min=float(eigen.min()))
