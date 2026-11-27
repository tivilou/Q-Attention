from pathlib import Path
import hashlib, json, time
import numpy as np

P=Path(__file__).parent
protocol=json.loads((P/'probe-protocol.json').read_text(encoding='utf-8'))
assert protocol['seed']==20261127 and protocol['synthetic_lengths']==[4,8,37,96]
start=time.perf_counter()
rng=np.random.default_rng(protocol['seed'])
EPS=protocol['epsilon']

def gate(y,z):
    c,s=np.cos(y/2),np.sin(y/2)
    return np.array([[np.exp(-.5j*z)*c,-np.exp(-.5j*z)*s],
                     [np.exp(.5j*z)*s,np.exp(.5j*z)*c]])

def states(angles, dense=False):
    answer=[]
    for a in angles:
        state=np.zeros(8, dtype=complex);state[0]=1
        for layer in range(2):
            for wire in range(3):
                g=gate(*a[layer,wire])
                if dense:
                    op=np.ones((1,1))
                    for k in range(3):op=np.kron(op,g if k==wire else np.eye(2))
                    state=op@state
                else:
                    pairs=state.reshape(2**wire,2,2**(2-wire))
                    state=np.einsum('ij,ajb->aib',g,pairs).reshape(8)
            for control,target in ((0,1),(1,2),(2,0)):
                perm=[i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
                if dense: state=np.eye(8)[perm]@state
                else: state=state[perm]
        answer.append(state)
    return np.stack(answer)

Wq=rng.normal(size=(1028,12))/np.sqrt(1028)
Wk=rng.normal(size=(1028,12))/np.sqrt(1028)
bq=rng.normal(size=12)*.1;bk=rng.normal(size=12)*.1
V=rng.normal(size=(1024,64))/32;vb=rng.normal(size=64)*.1
O=rng.normal(size=(64,64))/8;ob=rng.normal(size=64)*.1
C=rng.normal(size=(128,40))/np.sqrt(128);cb=rng.normal(size=40)*.1
effects=states(rng.normal(size=(40,2,3,2)))

def prepare(x,W,b):return states((np.pi*np.tanh(x@W+b)).reshape(-1,2,3,2))

def forward(x,mask,mode='q'):
    x=x[mask]
    q,k=prepare(x,Wq,bq),prepare(x,Wk,bk)
    score=abs(q.conj()@k.T)**2 if mode=='q' else abs(q)**2@(abs(k)**2).T
    alpha=(score+EPS)/(score+EPS).sum(axis=1,keepdims=True)
    y=alpha@(x[:,:1024]@V+vb)
    hidden=np.tanh(y@O+ob)
    subj=x[:,1026]>0;obj=x[:,1027]>0
    assert subj.any() and obj.any()
    readout=np.r_[hidden[subj].mean(axis=0),hidden[obj].mean(axis=0)]
    return {'q':q,'k':k,'score':score,'alpha':alpha,'hidden':hidden,'logits':readout@C+cb}

records=[]
for n in protocol['synthetic_lengths']:
    x=rng.normal(size=(n,1028));x[:,1024:]=0
    x[:,1024]=np.linspace(-1,1,n);x[:,1025]=np.linspace(1,-1,n)
    x[0,1026]=1;x[-1,1027]=1
    mask=np.ones(n,bool)
    base=forward(x,mask)
    d=forward(x,mask,'d')
    q,k=base['q'],base['k']
    qd=states((np.pi*np.tanh(x@Wq+bq)).reshape(-1,2,3,2),True)
    kd=states((np.pi*np.tanh(x@Wk+bk)).reshape(-1,2,3,2),True)
    rhoq=q[:,:,None]*q[:,None,:].conj();rhok=k[:,:,None]*k[:,None,:].conj()
    density=np.einsum('iab,jba->ij',rhoq,rhok).real
    re=q.real@k.real.T+q.imag@k.imag.T
    im=q.real@k.imag.T-q.imag@k.real.T
    real16=re**2+im**2
    diagonal=np.einsum('ia,ja->ij',np.diagonal(rhoq,axis1=1,axis2=2).real,
                       np.diagonal(rhok,axis1=1,axis2=2).real)
    padded=np.r_[x,np.full((7,1028),np.nan)]
    padding=forward(padded,np.r_[mask,np.zeros(7,bool)])
    perm=rng.permutation(n)
    permuted=forward(x[perm],mask)
    norm=max(abs(np.sum(abs(q)**2,axis=1)-1).max(),abs(np.sum(abs(k)**2,axis=1)-1).max())
    errors=dict(state_norm=float(norm),dense=float(max(abs(q-qd).max(),abs(k-kd).max())),
        density=float(abs(base['score']-density).max()),real16=float(abs(base['score']-real16).max()),
        dephased=float(abs(d['score']-diagonal).max()),
        rowsum=float(abs(base['alpha'].sum(axis=1)-1).max()),
        padding=float(abs(base['logits']-padding['logits']).max()),
        cached_pair_permutation=float(abs(base['logits']-permuted['logits']).max()))
    assert max(errors.values())<1e-12
    assert np.isfinite(base['logits']).all() and np.isfinite(d['logits']).all()
    xp=x.copy();xm=x.copy();xp[1,0]+=1e-5;xm[1,0]-=1e-5
    plus=forward(xp,mask);minus=forward(xm,mask)
    derivative=(plus['score'][0,1]-minus['score'][0,1])/2e-5
    # Native raw score of unchanged word0 is local at the fixed cached-input
    # boundary. Its normalized attention/final logits are NOT claimed local.
    native_plus=abs(effects.conj()@plus['k'][0])**2
    native_minus=abs(effects.conj()@minus['k'][0])**2
    native_change=float(abs(native_plus-native_minus).max())
    assert native_change==0 and abs(derivative)>1e-8
    records.append(dict(words=n,errors=errors,raw_pair_derivative=float(derivative),
                        native_unchanged_word_raw_change=native_change,
                        q_d_logit_difference=float(abs(base['logits']-d['logits']).max())))

plus=np.zeros(8,complex);plus[:2]=[1,1];plus/=np.sqrt(2)
minus=plus.copy();minus[1]*=-1
phase_q=[float(abs(np.vdot(plus,s))**2) for s in (plus,minus)]
phase_d=[float(abs(plus)**2@abs(s)**2) for s in (plus,minus)]
assert np.allclose(phase_q,[1,0],atol=1e-12) and np.allclose(phase_d,[.5,.5],atol=1e-12)

A=rng.normal(size=(40,8))+1j*rng.normal(size=(40,8))
frame=np.linalg.qr(A)[0]
effects40=np.einsum('ai,aj->aij',frame.conj(),frame)
assert abs(effects40.sum(0)-np.eye(8)).max()<1e-12
rho=[]
for _ in range(2):
    t=rng.normal(size=(8,3))+1j*rng.normal(size=(8,3));s=t@t.conj().T;rho.append(s/np.trace(s))
prob=lambda s:np.einsum('aij,ji->a',effects40,s).real
affine=float(abs(prob((rho[0]+rho[1])/2)-(prob(rho[0])+prob(rho[1]))/2).max())
assert affine<1e-12 and abs(prob(rho[0]).sum()-1)<1e-12

out=dict(status='PASS_SYNTHETIC_ROLE_INTERFACE_ONLY',seconds=time.perf_counter()-start,
    seed=protocol['seed'],records=records,phase_example=dict(q=phase_q,d=phase_d),
    joint40POVM_affine_error=affine,parameter_count=99616,
    schema='private exploratory check; not a scientific implementation',
    boundaries=['known identities not newT','not whole-family separation',
      'not native init or task reachability','cached-input derivative not raw-text causality',
      'not external independent review','no task forward,fit,GPU,heldout or admission'],
    protocol_sha256=hashlib.sha256((P/'probe-protocol.json').read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(P/'probe-result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
