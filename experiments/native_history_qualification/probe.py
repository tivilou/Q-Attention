"""Frozen synthetic equation check ONLY. No dataset, plugin factory or runner."""
from pathlib import Path
import ast, hashlib, itertools, json, math, os, time
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
import torch
torch.set_num_threads(1)
H=Path(__file__).resolve().parent
protocol=json.loads((H/'protocol.json').read_bytes())
start=time.perf_counter()
tol=protocol['tolerance'];dtype=torch.float64
source=Path('D:/CodexHome/task-artifacts/qattn-joint-computation-role-20261126/comparison/sequence_memory.py')
native=Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='de82d4bd5bbcdfe9303e19b3706d70d3e29519d0afc5ee4201d3c059b0f25c5e'
assert hashlib.sha256(native.read_bytes()).hexdigest()=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
scope={'torch':torch,'math':math,'DEPTH':2,'QUBITS':3}
for path,names in [(source,{'apply_native','native_unitary'}),(native,{'circuit'})]:
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
native_unitary=scope['native_unitary'];circuit=scope['circuit']
I=torch.eye(8,dtype=torch.complex128);IN=np.eye(8,dtype=complex)
mask=torch.tensor([[1,1,1,1,0],[1,0,1,1,0]],dtype=torch.bool)
def ar(n):return torch.arange(n,dtype=dtype)
parameters={
 'prepare_weight':.025*torch.sin(ar(12336).reshape(12,1028)/17),
 'prepare_bias':.1*torch.cos(ar(12)/7),
 'effect':.2*torch.sin(ar(480).reshape(40,2,3,2)/13),
 'value_weight':.012*torch.cos(ar(65536).reshape(64,1024)/23),
 'value_bias':.01*torch.sin(ar(64)),
 'decoder':.15*torch.sin(ar(2560).reshape(40,64)/11),
 'bias':.01*torch.cos(ar(40)),
 'transition_weight':.1*torch.sin(ar(768).reshape(64,12)/31),
 'transition_bias':.1*torch.cos(ar(64)/9),
 'terminal_classical':.2*torch.sin(ar(320).reshape(40,8)/5)}
for p in parameters.values():p.requires_grad_(True)
h=(.3*torch.cos(ar(2*5*1024).reshape(2,5,1024)/19)).masked_fill(~mask[...,None],float('nan')).requires_grad_()
roles=torch.nn.functional.one_hot(torch.arange(5)%4,4).to(dtype).expand(2,-1,-1).clone().masked_fill(~mask[...,None],float('nan')).requires_grad_()

def dense_np(a):
    """Independent dense Kronecker construction, not the source reshape port."""
    U=IN.copy()
    for layer in range(2):
        for wire in range(3):
            y,z=a[layer,wire];c,s=np.cos(y/2),np.sin(y/2)
            G=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])@np.array([[c,-s],[s,c]])
            mats=[G if k==wire else np.eye(2) for k in range(3)]
            U=np.kron(np.kron(mats[0],mats[1]),mats[2])@U
        for control,target in [(0,1),(1,2),(2,0)]:
            C=np.zeros((8,8))
            for j in range(8):C[j^((1<<(2-target)) if j&(1<<(2-control)) else 0),j]=1
            U=C@U
    return U

def clean_inputs(h,roles,m):
    if m.dtype!=torch.bool or not m.any(-1).all():raise ValueError('Every row needs a valid word')
    return torch.where(m[...,None],h,torch.zeros_like(h)),torch.where(m[...,None],roles,torch.zeros_like(roles))

def matrices(h,roles,m,p):
    hh,rr=clean_inputs(h,roles,m)
    z=torch.tanh(torch.nn.functional.linear(torch.cat([hh,rr],-1),p['prepare_weight'],p['prepare_bias']))
    U=native_unitary((math.pi*z).reshape(h.shape[:2]+(2,3,2)))
    phi=circuit(math.pi*torch.tanh(p['effect']))
    E=(phi[:,:,None]*phi[:,None,:].conj()+I[None]/8)/2
    return hh,z,U,E

def smoothing(U,E,m,mode='Q',mixed=False):
    n=U.shape[1];rho=U.new_zeros(len(U),8,8)
    rho[:,0,0]=1
    if mixed:rho=I.expand_as(rho)/8
    states=[rho]
    for t in range(n):
        v=(rho+U[:,t]@rho@U[:,t].mH)/2
        if mode=='D':v=torch.diag_embed(v.diagonal(dim1=-2,dim2=-1))
        rho=torch.where(m[:,t,None,None],v,rho);states.append(rho)
    B=E[None].expand(len(U),-1,-1,-1);g=[];complements=[]
    for t in reversed(range(n)):
        u=U[:,t,None];f=states[t][:,None]
        selected=u@f@u.mH/2;skip=f/2
        if mode=='D':
            selected=torch.diag_embed(selected.diagonal(dim1=-2,dim2=-1))
            skip=torch.diag_embed(skip.diagonal(dim1=-2,dim2=-1))
        gt=torch.einsum('brij,brji->br',B,selected).real
        gc=torch.einsum('brij,brji->br',B,skip).real
        g.append(torch.where(m[:,t,None],gt,0));complements.append(torch.where(m[:,t,None],gc,0))
        BD=torch.diag_embed(B.diagonal(dim1=-2,dim2=-1)) if mode=='D' else B
        before=(BD+u.mH@BD@u)/2
        B=torch.where(m[:,t,None,None,None],before,B)
    g=torch.stack(g[::-1],-1);comp=torch.stack(complements[::-1],-1)
    terminal=torch.einsum('rij,bji->br',E,rho).real
    return g,g/g.sum(-1,keepdim=True),terminal,states,comp

def positive_smoothing(P,e,m):
    n=P.shape[1];mass=P.new_zeros(len(P),8);mass[:,0]=1;states=[mass]
    for t in range(n):
        nxt=(mass+(P[:,t]@mass[...,None])[...,0])/2
        mass=torch.where(m[:,t,None],nxt,mass);states.append(mass)
    b=e[None].expand(len(P),-1,-1);g=[]
    for t in reversed(range(n)):
        sel=(P[:,t]@states[t][...,None])[...,0]/2
        g.append(torch.where(m[:,t,None],torch.einsum('bri,bi->br',b,sel),0))
        before=(b+torch.einsum('brj,bji->bri',b,P[:,t]))/2
        b=torch.where(m[:,t,None,None],before,b)
    g=torch.stack(g[::-1],-1)
    return g,g/g.sum(-1,keepdim=True)

def evaluate(h,roles,m,p,mode):
    hh,z,U,E=matrices(h,roles,m,p)
    if mode=='C':
        scores=torch.nn.functional.linear(z,p['transition_weight'],p['transition_bias']).reshape(len(h),h.shape[1],8,8)
        P=torch.softmax(scores,dim=-2);e=(torch.softmax(p['terminal_classical'],-1)+1/8)/2
        g,a=positive_smoothing(P,e,m)
    else:g,a,*_=smoothing(U,E,m,mode)
    v=torch.nn.functional.linear(hh,p['value_weight'],p['value_bias'])
    v=torch.where(m[...,None],v,0)
    logits=torch.einsum('brt,btd,rd->br',a,v,p['decoder'])+p['bias']
    return logits,a,g,U,E

errors={};outs={}
for mode in ['Q','D','C']:
    outs[mode]=evaluate(h,roles,mask,parameters,mode)
qlog,alpha,g,U,E=outs['Q'];un=U.detach().numpy();en=E.detach().numpy()
angles=(math.pi*torch.tanh(torch.nn.functional.linear(torch.cat(clean_inputs(h,roles,mask),-1),parameters['prepare_weight'],parameters['prepare_bias']))).detach().numpy().reshape(2,5,2,3,2)
indU=np.array([[dense_np(a) for a in row] for row in angles])
errors['native_vs_independent_dense_unitary']=float(np.max(np.abs(un-indU)))
errors['native_zero_state_column']=float((U[...,0]-circuit(torch.tensor(angles))).abs().max())
errors['unitarity']=float(np.max(np.abs(indU@indU.conj().swapaxes(-1,-2)-IN)))
indphi=np.array([dense_np(a)[:,0] for a in (math.pi*torch.tanh(parameters['effect'])).detach().numpy()])
indE=(indphi[:,:,None]*indphi[:,None,:].conj()+IN[None]/8)/2
errors['effect_independent']=float(np.max(np.abs(en-indE)))
enumerated=np.zeros((2,40,5));enumD=np.zeros_like(enumerated);enumC=np.zeros_like(enumerated)
z=torch.tanh(torch.nn.functional.linear(torch.cat(clean_inputs(h,roles,mask),-1),parameters['prepare_weight'],parameters['prepare_bias']))
CP=torch.softmax(torch.nn.functional.linear(z,parameters['transition_weight'],parameters['transition_bias']).reshape(2,5,8,8),dim=-2).detach().numpy()
ce=((torch.softmax(parameters['terminal_classical'],-1)+1/8)/2).detach().numpy()
for b in range(2):
    ids=torch.where(mask[b])[0].tolist();n=len(ids)
    state_paths=np.array(list(itertools.product(range(8),repeat=n)),dtype=int)
    for coins in itertools.product([0,1],repeat=n):
        psi=np.eye(8,dtype=complex)[:,0];pD=np.eye(8)[0];weights=np.ones(len(state_paths))/(2**n);previous=np.zeros(len(state_paths),dtype=int)
        for k,(t,coin) in enumerate(zip(ids,coins)):
            if coin:
                psi=indU[b,t]@psi;pD=np.abs(indU[b,t])**2@pD
                weights*=CP[b,t,state_paths[:,k],previous]
            else:weights*=state_paths[:,k]==previous
            previous=state_paths[:,k]
        probs=np.einsum('i,rij,j->r',psi.conj(),indE,psi).real/(2**n)
        dprobs=np.einsum('rii,i->r',indE,pD).real/(2**n)
        cprobs=ce[:,state_paths[:,-1]]@weights
        for t,coin in zip(ids,coins):
            if coin:enumerated[b,:,t]+=probs;enumD[b,:,t]+=dprobs;enumC[b,:,t]+=cprobs
errors['Q_all_coin_histories']=float(np.max(np.abs(g.detach().numpy()-enumerated)))
errors['D_all_coin_population_histories']=float(np.max(np.abs(outs['D'][2].detach().numpy()-enumD)))
errors['C_all_coin_state_histories']=float(np.max(np.abs(outs['C'][2].detach().numpy()-enumC)))
gp,ap=positive_smoothing(U.abs().square(),E.diagonal(dim1=-2,dim2=-1).real,mask)
errors['D_density_vs_population']=float((gp-outs['D'][2]).abs().max())
gq,aq,pq,states,comp=smoothing(U,E,mask)
errors['marginal_selected_plus_skip']=float(torch.where(mask[:,None],gq+comp-pq[...,None],0).abs().max())
errors['states_trace']=max(float((f.diagonal(dim1=-2,dim2=-1).sum(-1)-1).abs().max()) for f in states)
assert min(float(torch.linalg.eigvalsh(f).min()) for f in states)>-tol
assert float(torch.linalg.eigvalsh(E).min())>=1/16-tol
for mode,(_log,a,gm,*_) in outs.items():
    assert torch.isfinite(_log).all() and torch.isfinite(a).all()
    assert float(a.masked_select(~mask[:,None].expand_as(a)).abs().max())==0
    assert float((a.sum(-1)-1).abs().max())<tol
    assert float(gm.masked_select(mask[:,None].expand_as(gm)).min())>=1/32-tol
    assert float(a.max())<=float((9/(mask.sum(-1)+8)).max())+tol
gm,am,*_=smoothing(U,E,mask,mixed=True)
errors['maxmixed_uniform']=float((am-mask[:,None].to(dtype)/mask.sum(-1)[:,None,None]).abs().max())
# Remove all padding, including the interior hole: the same valid sequence.
for b in range(2):
    keep=mask[b]
    ll,aa,*_=evaluate(h[b:b+1,keep],roles[b:b+1,keep],torch.ones(1,int(keep.sum()),dtype=torch.bool),parameters,'Q')
    errors[f'padding_logits_row{b}']=float((ll-qlog[b:b+1]).abs().max())
    errors[f'padding_alpha_row{b}']=float((aa-alpha[b:b+1,:,keep]).abs().max())
try:evaluate(h,roles,mask&False,parameters,'Q');raise AssertionError('Invalid mask accepted')
except ValueError:pass
changed=h.detach().clone();changed[0,3]+=.2*torch.cos(ar(1024)/27)
g2=evaluate(changed,roles,mask,parameters,'Q')[2]
ratio=float(g[0,0,0]/g[0,0,1]);ratio2=float(g2[0,0,0]/g2[0,0,1]);assert abs(ratio2-ratio)>1e-10
gradient_records={};fd_records=[]
targets={'Q':['prepare_weight','prepare_bias','effect','value_weight','value_bias','decoder','bias'],
         'D':['prepare_weight','prepare_bias','effect','value_weight','value_bias','decoder','bias'],
         'C':['prepare_weight','prepare_bias','transition_weight','transition_bias','terminal_classical','value_weight','value_bias','decoder','bias']}
def objective(mode,hh=h,rr=roles):
    logits=evaluate(hh,rr,mask,parameters,mode)[0]
    return (logits*torch.cos(ar(80).reshape(2,40)/3)).sum()/80
for mode in ['Q','D','C']:
    tensors=[h,roles]+[parameters[k] for k in targets[mode]]
    grads=torch.autograd.grad(objective(mode),tensors)
    assert all(torch.isfinite(t).all() for t in grads)
    assert float(grads[0].masked_select(~mask[...,None].expand_as(h)).abs().max())==0
    assert float(grads[1].masked_select(~mask[...,None].expand_as(roles)).abs().max())==0
    gradient_records[mode]={k:float(v.norm()) for k,v in zip(['h','roles']+targets[mode],grads)}
    if mode=='Q':qgrads=dict(zip(['h','roles']+targets[mode],grads))
for key,index in [('h',(0,1,7)),('roles',(0,1,2)),('prepare_weight',(0,7)),('prepare_bias',(0,)),('effect',(0,0,0,0)),('value_weight',(0,7)),('value_bias',(0,)),('decoder',(0,0)),('bias',(0,))]:
    tensor=h if key=='h' else roles if key=='roles' else parameters[key]
    old=float(tensor[index]);step=protocol['fd_step']
    with torch.no_grad():
        tensor[index]=old+step;plus=float(objective('Q'))
        tensor[index]=old-step;minus=float(objective('Q'))
        tensor[index]=old
    fd=(plus-minus)/(2*step);ad=float(qgrads[key][index]);err=abs(fd-ad)
    fd_records.append({'tensor':key,'index':index,'autograd':ad,'central_difference':fd,'absolute_error':err})
    assert err<protocol['gradient_absolute_tolerance']
hh=torch.where(mask[...,None],h,0).detach().numpy()
vv=hh@parameters['value_weight'].detach().numpy().T+parameters['value_bias'].detach().numpy()
vv=np.where(mask.numpy()[...,None],vv,0)
aa=enumerated/enumerated.sum(-1,keepdims=True)
indlog=np.einsum('brt,btd,rd->br',aa,vv,parameters['decoder'].detach().numpy())+parameters['bias'].detach().numpy()
errors['full_1024_value_40_logit_readout']=float(np.max(np.abs(indlog-qlog.detach().numpy())))
assert max(errors.values())<tol,errors
seconds=time.perf_counter()-start;assert seconds<protocol['budget']['CPU_compute_seconds'],seconds
qcount=sum(parameters[k].numel() for k in targets['Q']);ccount=sum(parameters[k].numel() for k in targets['C'])
assert qcount==81028 and ccount==81700
out={'status':'PASS_SYNTHETIC_NATIVE3Q_HISTORY_EQUATIONS_NOT_METHOD_ADMISSION','seconds':seconds,
 'protocol_sha256':hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest(),
 'environment':{'python':os.sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','threads':1,'packages_installed':0},
 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,native]},
 'errors':errors,'finite_difference':fd_records,'gradient_norms':gradient_records,
 'future_ratio_witness':{'class':0,'words':[0,1],'changed_word':3,'before':ratio,'after':ratio2,'delta':ratio2-ratio},
 'raw_parameters':{'Q':qcount,'D':qcount,'classical_positive_smoother':ccount,'equal_effective_capacity':False},
 'bounds':{'minimum_joint_mass':1/32,'maxalpha_by_valid_length':{'4':9/12,'3':9/11},'matrix_dimension':8},
 'task_rows':0,'task_forwards':0,'optimizer_updates':0,'GPU':0,'model_api':0,
 'limits':['Synthetic deterministic weights, not the saved remote native initialization or a real task minibatch.',
 'No production plugin or runner adapter; independent classical/D training remains prospective, not performed.',
 'Single fixture only: finite gradients and local finite differences do not establish long-sequence learnability or task utility.',
 'No novel theorem, broad literature clearance, L3 or L4-T. Dataset-qualified reconciliation and actual cost remain prerequisites.']}
(H/'probe.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({'status':out['status'],'seconds':seconds,'max_error':max(errors.values()),'max_fd_error':max(t['absolute_error'] for t in fd_records),'witness':out['future_ratio_witness'],'counts':out['raw_parameters']}))
