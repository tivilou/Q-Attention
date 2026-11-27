"""Changed plugin contract tests; synthetic only, no optimizer/data/GPU."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,itertools,json,math,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--native',action='store_true');a=ap.parse_args()
H=Path(__file__).resolve().parent
assert not a.output.exists(),'Inspect sealed result instead of overwrite'
sys.path[:0]=[str(H),str(a.source)]
import numpy as np
import torch
import rmea_history_posterior as p
from baselines.retacred.rmea import RMEAHead
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(state):
    d=hashlib.sha256()
    for n,t in sorted(state.items()):
        v=t.detach().cpu().numpy();d.update(n.encode());d.update(str(v.dtype).encode());d.update(str(v.shape).encode());d.update(v.tobytes())
    return d.hexdigest()
assert sha(a.source/'baselines/retacred/rmea.py')=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
assert sha(a.source/'baselines/retacred/rmea_matched_routers.py')=='3813f29b61dfc1819a7fd2a610889efb97e8ace950344aaac5ed2800c0921e4d'
assert sha(H/'legacy_moments.py')=='73ef2fb6e3016459d6a4df9629363e7dd4115533eff69fff8760f64aa67ffe1a'
torch.manual_seed(13);initial=RMEAHead('quantum').float().state_dict();native_id=identity(initial)
if a.native:
    assert torch.__version__.startswith('2.4.1') and native_id=='f79575534288a8e57fef61deeba24ba568af7e683e4c1ebc984dcc7cfb6e0781'
models=p.make_models(13,initial);ids={k:identity(m.state_dict()) for k,m in models.items()}
counts={k:p.real_parameters(m) for k,m in models.items()}
assert counts==dict(zip(p.MODES,[81028,81028,81700,82308]))
assert ids[p.MODES[0]]==ids[p.MODES[1]]==native_id
assert all(torch.equal(m.state_dict()[key],initial[key]) for m in models.values() for key in p.COMMON)
assert all(torch.equal(models[p.MODES[2]].state_dict()[key],initial[key]) for key in ['prepare.weight','prepare.bias'])
import legacy_moments
with torch.random.fork_rng(devices=[]):legacy=legacy_moments.initialize(initial,13)['gaussian_covariance'].double()
for k,v in legacy.head.state_dict().items():assert torch.equal(models[p.MODES[3]].state_dict()[k].double(),v)
start=time.perf_counter();torch.manual_seed(20261127)
mask=torch.tensor([[1,1,1,1,0,0,0],[1,0,1,1,1,1,1]],dtype=torch.bool)
h=torch.randn(2,7,1024,dtype=torch.float64);roles=torch.randn(2,7,4,dtype=torch.float64)*.2
h[~mask]=float('nan');roles[~mask]=float('nan')
input_h=h.clone().requires_grad_();input_roles=roles.clone().requires_grad_();rows={}
def dense_np(angles):
    u=np.eye(8,dtype=complex)
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer,wire];c,s=np.cos(y/2),np.sin(y/2)
            g=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])@np.array([[c,-s],[s,c]])
            ms=[g if j==wire else np.eye(2) for j in range(3)]
            u=np.kron(np.kron(ms[0],ms[1]),ms[2])@u
        for control,target in ((0,1),(1,2),(2,0)):
            C=np.zeros((8,8))
            for j in range(8):C[j^((1<<(2-target)) if j&(1<<(2-control)) else 0),j]=1
            u=C@u
    return u
def enumerate_population(P,e):
    """Explicit state/coin source->destination probabilities, no DP helper."""
    n=len(P);paths=np.array(list(itertools.product(range(8),repeat=n)),dtype=int)
    result=np.zeros((len(e),n))
    for coins in itertools.product([0,1],repeat=n):
        weights=np.full(len(paths),1/(2**n));source=np.zeros(len(paths),dtype=int)
        for t,apply in enumerate(coins):
            destination=paths[:,t]
            weights*=P[t,destination,source] if apply else (destination==source)
            source=destination
        terminal=e[:,paths[:,-1]]@weights
        for t,apply in enumerate(coins):
            if apply:result[:,t]+=terminal
    return result
for name,original in models.items():
    m=copy.deepcopy(original).double()
    det=m(input_h,input_roles,mask,details=True);out=det['logits'];alpha=det['alpha']
    assert out.shape==(2,40) and torch.isfinite(out).all() and torch.isfinite(alpha).all()
    assert (alpha.sum(-1)-1).abs().max()<1e-10 and alpha.masked_select(~mask[:,None].expand_as(alpha)).abs().max()==0
    loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,39]));loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    assert torch.isfinite(input_h.grad).all() and torch.isfinite(input_roles.grad).all()
    assert input_h.grad[~mask].abs().max()==0 and input_roles.grad[~mask].abs().max()==0
    input_h.grad=None;input_roles.grad=None
    maxpad=0
    for b in range(2):
        keep=mask[b];trim=m(h[b:b+1,keep],roles[b:b+1,keep],torch.ones(1,int(keep.sum()),dtype=torch.bool))
        maxpad=max(maxpad,float((trim-out[b:b+1]).abs().max()))
    assert maxpad<1e-10
    oracle_error=0
    if name in p.MODES[:2]:
        for b in range(2):
            hv=h[b,mask[b]].numpy();rv=roles[b,mask[b]].numpy();n=len(hv)
            z=np.tanh(np.concatenate([hv,rv],-1)@m.prepare.weight.detach().numpy().T+m.prepare.bias.detach().numpy())
            units=[dense_np(ang) for ang in (math.pi*z).reshape(n,2,3,2)]
            phi=np.array([dense_np(ang)[:,0] for ang in (math.pi*torch.tanh(m.effect)).detach().numpy()])
            E=(phi[:,:,None]*phi[:,None,:].conj()+np.eye(8)[None]/8)/2;g=np.zeros((40,n))
            for coins in itertools.product([0,1],repeat=n):
                if name==p.MODES[0]:
                    state=np.eye(8,dtype=complex)[:,0]
                    for u,coin in zip(units,coins):
                        if coin:state=u@state
                    prob=np.einsum('i,rij,j->r',state.conj(),E,state).real/(2**n)
                else:
                    pop=np.eye(8)[0]
                    for u,coin in zip(units,coins):
                        if coin:pop=(np.abs(u)**2)@pop
                    prob=np.einsum('rii,i->r',E,pop).real/(2**n)
                for t,coin in enumerate(coins):
                    if coin:g[:,t]+=prob
            oracle_error=max(oracle_error,float(np.max(abs(g-det['masses'][b,:,mask[b]].detach().numpy()))))
    elif name==p.MODES[2]:
        # One fixed original fixture row, four valid words; all16 coins/4096 states.
        hv=h[0:1,mask[0]];rv=roles[0:1,mask[0]]
        zz=torch.tanh(m.prepare(torch.cat((hv,rv),-1)))
        PP=m.transition(zz).reshape(4,8,8).softmax(-2).detach().numpy()
        ee=((m.terminal.softmax(-1)+1/8)/2).detach().numpy()
        expected=enumerate_population(PP,ee)
        oracle_error=float(np.max(abs(expected-det['masses'][0,:,mask[0]].detach().numpy())))
    elif name==p.MODES[3]:
        hh=torch.where(mask[...,None],h,0);rr=torch.where(mask[...,None],roles,0)
        # Nonzero fixed moment readout checks the retained covariance term too.
        with torch.no_grad():
            weights=.1*torch.sin(torch.arange(1280,dtype=torch.float64).reshape(40,32)/7)
            legacy.moment_decoder.copy_(weights);m.moment_decoder.copy_(weights)
        oracle_error=float((m(h,roles,mask)-legacy(hh,rr,mask)).abs().max())
        gl=torch.autograd.grad(legacy(hh,rr,mask).sum(),tuple(legacy.parameters()))
        gm=torch.autograd.grad(m(h,roles,mask).sum(),tuple(m.parameters()))
        # Compare by names after lifting legacy head.* paths, not constructor order.
        ld={k.removeprefix('head.'):g for (k,_),g in zip(legacy.named_parameters(),gl)}
        md={k:g for (k,_),g in zip(m.named_parameters(),gm)}
        oracle_error=max(oracle_error,max(float((md[k]-v).abs().max()) for k,v in ld.items()))
    assert oracle_error<1e-9,(name,oracle_error)
    # Independent D is exactly a fixed-Q intervention at the identical init.
    if name==p.MODES[1]:
        q=copy.deepcopy(models[p.MODES[0]]).double()
        assert (q.same_state_dephased_logits(h,roles,mask)-out).abs().max()<1e-10
    rows[name]={'parameters':counts[name],'initial_sha256':ids[name],'padding_error':maxpad,'independent_reference_error':oracle_error,'all_parameter_tensor_gradients_finite':True,'nonzero_gradient_tensor_names':[k for k,t in m.named_parameters() if torch.count_nonzero(t.grad)>0]}
# Fixed non-doubly-stochastic orientation witness, distinct for every word.
T=np.array([[[1+((i+2)*(j+3)+(t+1)*(i+1)**2)%17 for j in range(8)] for i in range(8)] for t in range(3)],dtype=float)
T/=T.sum(axis=1,keepdims=True)
ee=1+.2*np.sin(np.arange(320).reshape(40,8)/5);ee=(ee/ee.sum(-1,keepdims=True)+1/8)/2
expected=enumerate_population(T,ee)
observed=p.population_masses(torch.tensor(T)[None],torch.tensor(ee),torch.ones(1,3,dtype=torch.bool))[0].numpy()
transposed=p.population_masses(torch.tensor(T.transpose(0,2,1))[None],torch.tensor(ee),torch.ones(1,3,dtype=torch.bool))[0].numpy()
orientation_error=float(np.max(abs(observed-expected)));transpose_error=float(np.max(abs(transposed-expected)))
assert orientation_error<1e-10 and transpose_error>1e-5
assert np.max(abs(T.sum(-1)-1))>1e-3
# Reach original-length96 without expanding 2^96 histories. No training or tuning.
with torch.no_grad():
    long_h=torch.cos(torch.arange(96*1024,dtype=torch.float64).reshape(1,96,1024)/31)
    long_roles=torch.zeros(1,96,4,dtype=torch.float64);longmask=torch.ones(1,96,dtype=torch.bool)
    for name,m in models.items():
        val=m.double()(long_h,long_roles,longmask);assert torch.isfinite(val).all()
seconds=time.perf_counter()-start
out={'status':'PASS_SYNTHETIC_HISTORY_PLUGIN_CPU','native_torch241_identity_verified':a.native,'native_initial_identity':native_id,'initial_model_identities':ids,'results':rows,'seconds':seconds,
 'C_orientation_fixture':{'raw_mass_error':orientation_error,'transposed_transition_error':transpose_error,'n':3,'coins':8,'latent_state_paths':512,'column_sum_error':float(np.max(abs(T.sum(1)-1))),'nondoubly_row_sum_deviation':float(np.max(abs(T.sum(-1)-1)))},
 'torch':torch.__version__,'numpy':np.__version__,'code_hashes':{n:sha(H/n) for n in ['rmea_history_posterior.py','check_cpu.py','legacy_moments.py']},
 'task_rows':0,'task_forwards':0,'optimizer_updates':0,'GPU':0,'limits':['Synthetic plugin contract only. C uses8 classical states, not equal density degrees of freedom.','No efficacy/newT/L3/L4T; local Torch2.8 initialization is not production Torch2.4.1 unless native flag passed.']}
a.output.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps({'status':out['status'],'native_identity_verified':a.native,'seconds':seconds,'counts':counts,'max_reference_error':max(x['independent_reference_error'] for x in rows.values())}))
