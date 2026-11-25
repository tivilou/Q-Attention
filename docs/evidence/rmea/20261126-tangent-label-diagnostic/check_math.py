"""Synthetic numerical and target-boundary preflight, not task data."""
from pathlib import Path
import hashlib,json,tempfile
import numpy as np
import torch
from tangent_math import slopes_torch,operator_numpy,statistics,decide,NAMES
import trace_io
r=Path(__file__).resolve().parent;cfg=json.loads((r/'design.json').read_bytes())
torch.set_num_threads(1);rng=np.random.default_rng(20261126);maxerr=0.;bound=0.
for n in (1,2,7,31):
    psi=rng.normal(size=(3,n,8))+1j*rng.normal(size=(3,n,8));psi/=np.linalg.norm(psi,axis=-1,keepdims=True)
    phi=rng.normal(size=(40,8))+1j*rng.normal(size=(40,8));phi/=np.linalg.norm(phi,axis=-1,keepdims=True)
    u=rng.normal(size=(3,40,n));mask=np.ones((3,n),bool)
    if n>1:mask[1,-1]=False
    p=abs(np.einsum('rd,btd->brt',phi.conj(),psi))**2;raw=(p+1e-6)*mask[:,None,:];D=raw.sum(-1,keepdims=True)
    d={k:torch.from_numpy(v) for k,v in dict(psi=psi,phi=phi,utilities=u,probability=p,alpha=raw/D,normalizer=D).items()}
    s={k:v.numpy() for k,v in slopes_torch(d,torch.from_numpy(mask)).items()}
    for b in range(3):
        ref=operator_numpy(psi[b,mask[b]],phi,u[b,:,mask[b]].T)
        for k in ('g','residual','z'):maxerr=max(maxerr,float(np.max(abs(s[k][b]-ref[k])/(1+abs(ref[k])))))
    bound=max(bound,float(np.max(np.maximum(s['g']-s['upper'],0))))
assert maxerr<1e-12 and bound<1e-12,(maxerr,bound)
# Controlled, outcome-known sentinel checks the population gate direction.
scores=np.zeros((200,40,4));logits=np.zeros((200,40));labels=np.zeros(200,np.int64)
logits[:100,1]=1;logits[100:,0]=1;scores[:,0,0]=1
m,a=statistics(scores,logits,labels);gate=decide(m,True)
assert gate['decision']=='RETAIN_EXPLORATORY_DIRECTION'
scores[:,0,0]=0;m,_=statistics(scores,logits,labels);assert decide(m,True)['decision']=='STOP_CURRENT_LABEL_DIRECTION'
with tempfile.TemporaryDirectory() as t:
    arrays=dict(h=np.zeros((2,1024)),roles=np.zeros((2,4)),mask=np.ones(2,bool),angles=np.zeros((2,2,3,2)),effect_angles=np.zeros((40,2,3,2)),
        psi=np.tile(np.eye(8)[0],(2,1)).astype(complex),phi=np.tile(np.eye(8)[0],(40,1)).astype(complex),
        values=np.zeros((2,64)),utilities=np.zeros((40,2)),probability=np.ones((40,2)),alpha=np.full((40,2),.5),normalizer=np.full((40,1),2+2e-6),
        decoder=np.zeros((40,64)),bias=np.zeros(40),logits=np.zeros(40),residual=np.zeros((40,8),complex),target=np.array([0]))
    arrays.update({k:np.zeros(40) for k in ('g','variance','self','upper','gamma','cross','z')})
    samples=[trace_io.capture(t,split,j,j,dict(tokens=['a','b'],id_hash='synthetic'),arrays,'0'*64) for split in ('fit','assessment') for j in range(4)]
    trace=trace_io.close(t,samples,cfg,trace_io.sha(r/'design.json'),r/'source')
assert not trace['structural_errors'] and not trace['warnings']
files={n:trace_io.sha(r/n) for n in cfg['code_hashes']}
for n,h in files.items():assert h==cfg['code_hashes'][n]
out=dict(status='PASS_LOCAL_PREFLIGHT',synthetic_cases=12,max_scaled_operator_error=maxerr,bound_violation=bound,
    gate_sentinels=2,trace_synthetic_cases=8,trace_coordinate_reads=trace['coordinate_reads'],files=files,
    external_research_review=False,native_head_calls=0,task_rows_decoded=0,GPU_allocations=0)
(r/'local-preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
