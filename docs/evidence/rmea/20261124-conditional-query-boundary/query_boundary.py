"""Small deterministic algebra fixture; no task data or model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np

r=Path(__file__).resolve().parent
rng=np.random.default_rng(20261124)
normalize=lambda x:x/np.linalg.norm(x,axis=-1,keepdims=True)
q=normalize(rng.normal(size=(40,8))+1j*rng.normal(size=(40,8)))
k=normalize(rng.normal(size=(11,8))+1j*rng.normal(size=(11,8)))
u,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
q_context=q@u.T
k_folded=k@u.conj()
born=lambda a,b:abs(a.conj()@b.T)**2
diag=lambda a,b:abs(a)**2@(abs(b)**2).T
p=born(q_context,k)
folded=born(q,k_folded)
mask=np.array([True]*9+[False]*2)
attention=lambda x:((x+1e-6)*mask)/((x+1e-6)*mask).sum(-1,keepdims=True)
values=rng.normal(size=(11,64))
decoder=rng.normal(size=(40,64))
utilities=decoder@values.T
logit=lambda x:(attention(x)*utilities).sum(-1)
naive_diag=diag(q,k_folded)
original_diag=diag(q_context,k)
# Fold the channel as well: D_U(X)=U^dagger D(U X U^dagger) U.
covariant=np.empty_like(p)
for j,key in enumerate(k_folded):
    key_matrix=np.outer(key,key.conj())
    reconstructed=u@key_matrix@u.conj().T
    rotated_channel=u.conj().T@np.diag(np.diag(reconstructed))@u
    covariant[:,j]=np.einsum('ri,ij,rj->r',q.conj(),rotated_channel,q).real
result=dict(profile='R0-shared-context-query-unitary-boundary-v1',seed=20261124,
    source_sha256=hashlib.sha256((r.parent/'full-protocol-fair-reference-20261124/source/baselines/retacred/rmea.py').read_bytes()).hexdigest(),
    fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    probability_max_error=float(abs(p-folded).max()),
    masked_eps_attention_max_error=float(abs(attention(p)-attention(folded)).max()),
    signed_value_logit_max_error=float(abs(logit(p)-logit(folded)).max()),
    naive_dephasing_fold_max_difference=float(abs(original_diag-naive_diag).max()),
    channel_covariant_dephasing_max_error=float(abs(original_diag-covariant).max()),
    task_data_accessed=False,training_updates=0,
    scope='Known state-port unitary identity only. Not shallow-circuit closure, capacity equality, task utility, theorem novelty, or native checkpoint replay.')
assert result['probability_max_error']<1e-12
assert result['masked_eps_attention_max_error']<1e-12
assert result['signed_value_logit_max_error']<1e-10
assert result['channel_covariant_dephasing_max_error']<1e-12
assert result['naive_dephasing_fold_max_difference']>1e-3
(r/'query-boundary-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
