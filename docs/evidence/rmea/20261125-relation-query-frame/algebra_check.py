import json,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
P=np.array([[1.,0.],[np.sqrt(3)/2,.5],[np.sqrt(3)/2,-.5]])
K=np.array([[1.,0.],[0.,1.],[2**-.5,2**-.5]])
eps=1e-8
F=np.einsum('ri,rj->rij',P,P)+eps*np.eye(2)[None]
S=F.sum(0);ev,U=np.linalg.eigh(S);B=(U/np.sqrt(ev))@U.T
E=np.einsum('ij,rjk,kl->ril',B,F,B)
raw=np.einsum('ti,rij,tj->rt',K,E,K)
q=(P@K.T)**2+eps
norm=lambda x:x/x.sum(-1,keepdims=True)
native=norm(q);competitive=norm(q/q.sum(0,keepdims=True));relation_frame=norm(raw)
assert np.all(raw>=0) and np.max(abs(E.sum(0)-np.eye(2)))<1e-12
assert np.max(abs(relation_frame-native))>1e-3
assert np.max(abs(relation_frame-competitive))>1e-3
# The previously inspected filtered-state identity is different: it changes the key too.
filtered=(K@np.linalg.inv(B));filtered/=np.linalg.norm(filtered,axis=-1,keepdims=True)
filtered_prob=np.einsum('ti,rij,tj->rt',filtered,E,filtered)
assert np.max(abs(filtered_prob-q/q.sum(0,keepdims=True)))<1e-12
pdf=Path('E:/Claude.Code/Paper.Writing/Q-Attention/.codex-artifacts/qattn-l5-research-20261115/literature/pgm-fulltext/55c3403ba4ebdf9a/99ef00741dfeff7b21f88016555a72c10990bf1e616a8035178dd884a5ab7214.pdf')
assert hashlib.sha256(pdf.read_bytes()).hexdigest()=='99ef00741dfeff7b21f88016555a72c10990bf1e616a8035178dd884a5ab7214'
v=dict(status='PASS_ALGEBRA_ONLY',povm_completeness_error=float(np.max(abs(E.sum(0)-np.eye(2)))),different_from_native=float(np.max(abs(relation_frame-native))),different_from_scalar_competition=float(np.max(abs(relation_frame-competitive))),filtered_state_competition_error=float(np.max(abs(filtered_prob-q/q.sum(0,keepdims=True)))),native=native.tolist(),competitive=competitive.tolist(),relation_frame=relation_frame.tolist(),source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),utility_proven=False,novelty_proven=False,original_circuit_reachability_proven=False,new_task_rows=0,new_updates=0)
(r/'verification.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v[k] for k in ('status','different_from_native','different_from_scalar_competition','filtered_state_competition_error')}))
