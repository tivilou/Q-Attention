"""One <=10s artificialCPU math check. No optimizer, task rows or GPU."""
from pathlib import Path
import copy,hashlib,itertools,json,math,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
H=Path(__file__).resolve().parent
OLD=Path('D:/CodexHome/task-artifacts/qattn-history-plugin-20261127')
assert not (H/'math-result.json').exists()
sys.path[:0]=[str(H),str(OLD),str(OLD/'source-context')]
import numpy as np
import torch
import ordered_subsequence as dp
import rmea_history_posterior as native
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(OLD/'rmea_history_posterior.py')=='35d748d2618c1a313622534eb8f87cd64e1a7bea2ae540d2fb9c0f10560df592'
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(13)
start=time.perf_counter()

def error(a,b):return float((a.detach()-b.detach()).abs().max())
def state_hash(m):
    d=hashlib.sha256()
    for n,t in sorted(m.state_dict().items()):
        a=t.detach().numpy();d.update(n.encode());d.update(str(a.dtype).encode());d.update(str(a.shape).encode());d.update(a.tobytes())
    return d.hexdigest()

def enumerate_masses(U,E,mask,mode):
    # Independent state-vector enumeration, no DP/helper call.
    rows=[]
    for b in range(len(U)):
        idx=mask[b].nonzero(as_tuple=True)[0];u=U[b,idx];n,d,_=u.shape;k=min(3,n)
        paths=list(itertools.combinations(range(n),k));count=math.comb(n-1,k-1)
        values=[]
        for t in range(n):
            states=[]
            for path in paths:
                if t not in path:continue
                state=u.new_zeros(d);state[0]=1
                for j in path:state=u[j]@state
                states.append(state)
            assert len(states)==count
            if mode=='D':g=sum(torch.einsum('i,rij,j->r',v.conj(),E,v).real for v in states)/count**2
            else:
                v=sum(states)/count;g=torch.einsum('i,rij,j->r',v.conj(),E,v).real
            values.append(g)
        g=torch.stack(values,-1);rows.append(g.new_zeros(E.shape[0],mask.shape[1]).index_copy(1,idx,g))
    return torch.stack(rows)

mask=torch.tensor([[1,1,1,1,0,0,0],[1,0,1,1,1,1,1]],dtype=torch.bool)
h=torch.randn(2,7,1024,dtype=torch.float64);roles=torch.randn(2,7,4,dtype=torch.float64)*.2
h[~mask]=float('nan');roles[~mask]=float('nan')
weights=torch.linspace(-.8,1.1,80,dtype=torch.float64).reshape(2,40)
model=native.HistoryHead('quantum_history').double()
classical=dp.SignedRealOperatorHead().double()

def components(m,h,r,mode):
    if mode=='C':return m.components(h,r,mask)
    hv,rv=native.valid_inputs(h,r,mask)
    angles=(math.pi*torch.tanh(m.prepare(torch.cat((hv,rv),-1)))).reshape(hv.shape[:2]+(2,3,2))
    U=native.native_unitary(angles)
    phi=native.circuit(math.pi*torch.tanh(m.effect))
    E=(phi[:,:,None]*phi[:,None,:].conj()+torch.eye(8,dtype=U.dtype)[None]/8)/2
    return U,E,hv,rv

def full(m,h,r,mode,enum=False):
    U,E,hv,rv=components(m,h,r,mode)
    mass=enumerate_masses(U,E,mask,mode) if enum else (dp.dephased_masses if mode=='D' else dp.coherent_masses)(U,E,mask)
    return dp.readout(mass,hv,mask,m.value,m.decoder,m.bias)

rows={};frozen={}
for mode,m in [('Q',model),('D',model),('C',classical)]:
    before=state_hash(m);h1=h.clone().requires_grad_();r1=roles.clone().requires_grad_()
    expected=full(m,h1,r1,mode,True)
    ga=torch.autograd.grad((expected['logits']*weights).sum(),[h1,r1,*m.parameters()])
    h2=h.clone().requires_grad_();r2=roles.clone().requires_grad_();actual=full(m,h2,r2,mode)
    gb=torch.autograd.grad((actual['logits']*weights).sum(),[h2,r2,*m.parameters()])
    assert actual['logits'].shape==(2,40) and actual['values'].shape==(2,7,64)
    le=error(expected['logits'],actual['logits']);me=error(expected['masses'],actual['masses']);ae=error(expected['alpha'],actual['alpha']);ge=max(error(x,y) for x,y in zip(ga,gb))
    assert max(le,me,ae)<1e-10 and ge<2e-9
    assert all(torch.isfinite(g).all() for g in gb)
    assert gb[0][~mask].abs().max()==0 and gb[1][~mask].abs().max()==0
    assert actual['alpha'].masked_select(~mask[:,None].expand_as(actual['alpha'])).abs().max()==0
    assert (actual['alpha'].sum(-1)-1).abs().max()<1e-10
    chosen='prepare.weight' if mode!='C' else 'operator.2.weight'
    named=dict(m.named_parameters());p=named[chosen];index=(0,0);gradient=gb[2+list(named).index(chosen)][index].item();delta=1e-6
    with torch.no_grad():
        center=p[index].item();p[index]=center+delta;plus=(full(m,h,roles,mode)['logits']*weights).sum().item()
        p[index]=center-delta;minus=(full(m,h,roles,mode)['logits']*weights).sum().item();p[index]=center
    fd=abs((plus-minus)/(2*delta)-gradient);assert fd<2e-8
    assert state_hash(m)==before
    frozen[mode]={k:v.detach() for k,v in actual.items()}
    rows[mode]=dict(initial_sha256=before,real_parameter_count=sum(p.numel() for p in m.parameters()),mass_error=me,alpha_error=ae,logit_error=le,all_input_parameter_gradient_error=ge,finite_difference_parameter=chosen,finite_difference_index=index,finite_difference_gradient=gradient,finite_difference_error=fd,padding_input_gradient_zero=True,state_unchanged=True)

with torch.no_grad():
    U,E,hv,rv=components(model,h,roles,'Q')
    angles=(math.pi*torch.tanh(model.prepare(torch.cat((hv,rv),-1)))).reshape(hv.shape[:2]+(2,3,2))
    native_column_error=error(U[...,0],native.circuit(angles))
    unity_error=error(U.mH@U,torch.eye(8,dtype=U.dtype).expand_as(U));assert max(native_column_error,unity_error)<1e-12
    # Collapse holes with the same input and parameter state, not a second initialization.
    trims={}
    for mode,m in [('Q',model),('D',model),('C',classical)]:
        A,F,_,_=components(m,h,roles,mode);gap=0
        for b in range(2):
            good=mask[b];mini=torch.ones(1,int(good.sum()),dtype=torch.bool)
            masses=(dp.dephased_masses if mode=='D' else dp.coherent_masses)(A[b:b+1,good],F,mini)
            out=dp.readout(masses,h[b:b+1,good],mini,m.value,m.decoder,m.bias)
            gap=max(gap,error(out['logits'],frozen[mode]['logits'][b:b+1]))
        assert gap<1e-10;trims[mode]=gap
    native_shape_contrast=error(frozen['Q']['alpha'],frozen['D']['alpha']);native_logit_contrast=error(frozen['Q']['logits'],frozen['D']['logits'])
    assert native_shape_contrast>1e-5 and native_logit_contrast>1e-6
    nulls=[]
    for n in [1,2,3]:
        good=torch.ones(1,n,dtype=torch.bool);u=U[:1,:n]
        q=dp.coherent_masses(u,E,good);d=dp.dephased_masses(u,E,good)
        gap=max(float((dp.attention(q,good)-1/n).abs().max()),float((dp.attention(d,good)-1/n).abs().max()))
        assert gap<1e-12;nulls.append(dict(words=n,uniform_attention_error=gap,mass_Q_D_error=error(q,d)))
    # Prespecified abstractphase case. Not a claimU2=-U1 is native-parameter attainable.
    v=torch.eye(2,dtype=torch.complex128);phase=torch.stack([v,-v,v,v])[None]
    ee=(torch.diag(torch.tensor([1.,0.]))+torch.eye(2)/2)[None].to(torch.complex128)/2
    good=torch.ones(1,4,dtype=torch.bool);e=.75
    q=dp.coherent_masses(phase,ee,good);d=dp.dephased_masses(phase,ee,good)
    q_ref=torch.tensor([e/9,e,e/9,e/9],dtype=torch.float64)[None,None];d_ref=torch.full((1,1,4),e/3,dtype=torch.float64)
    phase_error=max(error(q,q_ref),error(d,d_ref));assert phase_error<1e-12
    phase_record=dict(mass_error=phase_error,Q_masses=q.tolist(),D_masses=d.tolist(),Q_alpha=dp.attention(q,good).tolist(),D_alpha=dp.attention(d,good).tolist(),native_parameter_attainability=False)

    # Independent NumPy exact states and explicit pathrecord density blocks (nonzeroD).
    u=U[1,mask[1]].numpy();ev=E.numpy();n=len(u);k=3;count=math.comb(n-1,k-1)
    nq=np.zeros((40,n));nd=np.zeros((40,n));bad_conj=np.zeros((40,n));bad_repeat=np.zeros((40,n))
    for t in range(n):
        states=[]
        for path in itertools.combinations(range(n),k):
            if t not in path:continue
            state=np.eye(8,dtype=complex)[:,0]
            for j in path:state=u[j].dot(state)
            states.append(state)
        vectors=np.stack(states);coherent=vectors.sum(0)/count
        rho_signal=np.einsum('si,sj->ij',vectors,vectors.conj())/count**2
        nq[:,t]=np.einsum('i,rij,j->r',coherent.conj(),ev,coherent).real
        nd[:,t]=np.einsum('rij,ji->r',ev,rho_signal).real
        bad_conj[:,t]=np.einsum('i,rij,j->r',coherent,ev,coherent).real
        repeat=[]
        for path in itertools.combinations_with_replacement(range(n),k):
            if t not in path:continue
            state=np.eye(8,dtype=complex)[:,0]
            for j in path:state=u[j]@state
            repeat.append(state)
        vv=sum(repeat)/len(repeat);bad_repeat[:,t]=np.einsum('i,rij,j->r',vv.conj(),ev,vv).real
    ng=torch.ones(1,n,dtype=torch.bool)
    actualq=dp.coherent_masses(U[1:2,mask[1]],E,ng)[0].numpy();actuald=dp.dephased_masses(U[1:2,mask[1]],E,ng)[0].numpy()
    oracle_errors=dict(Q=float(np.max(np.abs(actualq-nq))),D=float(np.max(np.abs(actuald-nd))))
    assert max(oracle_errors.values())<1e-12 and np.min(nd)>1e-5
    reversed_mass=dp.coherent_masses(U.flip(1),E,mask.flip(1)).flip(-1)
    mutants=dict(reversed_word_order=error(reversed_mass,frozen['Q']['masses']),missing_Born_conjugation=float(np.max(np.abs(nq-bad_conj))),with_replacement_paths=float(np.max(np.abs(nq-bad_repeat))),D_wrong_C_inverse_not_squared=float(np.max(np.abs(nd*count-nd))))
    A,F,_,_=components(classical,h,roles,'C');transposed=dp.coherent_masses(A.transpose(-1,-2),F,mask)
    mutants['classical_wrong_transpose']=error(transposed,frozen['C']['masses'])
    assert min(mutants.values())>1e-7
    success={}
    for mode in ['Q','D']:
        g=frozen[mode]['masses'];mean=g.sum(-1)/mask.sum(-1)[:,None]
        total=(mean+dp.EPS)/(1+dp.EPS)
        valid=g[mask[:,None].expand_as(g)]
        success[mode]=dict(signal_probability_min=float(mean.min()),signal_probability_max=float(mean.max()),with_uniform_branch_min=float(total.min()),max_expected_attempts_at_fixture=float((1/total).max()),min_word_mass=float(valid.min()),max_word_mass=float(valid.max()))
        assert (mean>=0).all() and (mean<=1).all()

elapsed=time.perf_counter()-start;assert not torch.cuda.is_initialized()
result=dict(status='PASS_ORDERED_SUBSEQUENCE_SYNTHETIC_MATH_ONLY',compute_seconds=elapsed,torch_version=torch.__version__,numpy_version=np.__version__,python=sys.executable,device='CPU',new_packages=0,seed=13,source_sha256=sha(H/'ordered_subsequence.py'),checker_sha256=sha(H/'check.py'),native_source_sha256=sha(OLD/'rmea_history_posterior.py'),rows=rows,fullhead_shape=[2,7,1024,4,64,40],valid_word_counts=mask.sum(-1).tolist(),trim_padding_errors=trims,native_circuit_first_column_error=native_column_error,native_unitarity_error=unity_error,native_Q_D_alpha_contrast=native_shape_contrast,native_Q_D_logit_contrast=native_logit_contrast,small_n_nulls=nulls,abstract_phase_case=phase_record,numpy_enumeration_errors=oracle_errors,nonzero_D_oracle_min=float(np.min(nd)),mutant_discrepancies=mutants,fixture_success_probabilities=success,original_Torch241_initial_identity_verified=False,new_task_forwards=0,new_updates=0,new_GPU=0,production_plugin_installed=False,new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',limits=['ArtificialQT initialstate, not savednativeTorch2.4.1trajectory.','Fullheadandallgradients on2x7positions only; no longsequence runtime or actualGPUcost.','One specifiedparameterFD perQ/D/C, not allparameter finite differences.','Dremoves subset-pathcoherence only, not all intrapath quantum coherence.','AbstractV/-Vnativeparameterattainability unproved; separate fixednativefixtureQ/Dcontrast actually checked.','IndependentclassicalC artificialformula checked, not independently fitted or capacitycertified. KnownLCU/DP/phasefacts not newT.','No task utility/screen/GPUadmission; four oldtechnicalupdates remainconsumed.'])
(H/'math-result.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=result['status'],compute_seconds=elapsed,max_full_head_logit_error=max(x['logit_error'] for x in rows.values()),max_all_gradient_error=max(x['all_input_parameter_gradient_error'] for x in rows.values()),native_Q_D_alpha_contrast=native_shape_contrast,CUDA_initialized=False,new_taskcompute=0)))
