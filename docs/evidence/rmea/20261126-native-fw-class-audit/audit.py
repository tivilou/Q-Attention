from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time
import numpy as np
from independent_reference import circuit
import fw_trace
r=Path(__file__).resolve().parent;start=time.process_time();scope=json.loads((r/'scope.json').read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for pin in scope['pins']:assert sha(pin['path'])==pin['sha256']

def unpack(x):return np.asarray(x.get('values',x.get('real')),dtype=np.float64)+(1j*np.asarray(x['imag']) if 'imag' in x else 0)
def packed(x):
    x=np.asarray(x)
    if np.iscomplexobj(x):return dict(shape=list(x.shape),dtype=str(x.dtype),real=x.real.tolist(),imag=x.imag.tolist())
    return dict(shape=list(x.shape),dtype=str(x.dtype),values=x.tolist())
def coords(H):
    i,j=np.triu_indices(8,1)
    return np.concatenate([H.diagonal(axis1=-2,axis2=-1).real,np.sqrt(2)*H[...,i,j].real,np.sqrt(2)*H[...,i,j].imag],-1)

def run_fw(inputs,epsilon=1e-5):
    memory=np.zeros((8,8));gamma=np.full(8,math.sqrt(1+epsilon)/math.sqrt(7));beta=np.zeros(8);initial_hidden=np.zeros(8)
    a=dict(inputs=inputs.copy(),gamma=gamma,beta=beta,initial_hidden=initial_hidden,initial_memory=memory.copy())
    rows={k:[] for k in ('preliminary','memory_before','preLN','centered','variance','hidden','memory_after')}
    for b in inputs:
        h0=np.maximum(b,0);y=b+memory@h0;c=y-y.mean();var=np.mean(c*c);h=np.maximum(gamma*c/np.sqrt(var+epsilon)+beta,0)
        rows['preliminary'].append(h0);rows['memory_before'].append(memory.copy());rows['preLN'].append(y);rows['centered'].append(c);rows['variance'].append(var);rows['hidden'].append(h)
        memory=.95*memory+.5*np.outer(h,h);rows['memory_after'].append(memory.copy())
    a.update({k:np.asarray(v) for k,v in rows.items()});a['final_readout']=np.asarray([a['hidden'][-1,2]-a['hidden'][-1,0]])
    return a

e=np.eye(8);z0=np.full(8,-1/math.sqrt(7));z0[0]=math.sqrt(7);z2=np.full(8,-1/math.sqrt(7));z2[2]=math.sqrt(7)
repeated_second=z0.copy();repeated_second[0]/=1.5;q=(e[0]+e[2])/math.sqrt(2)
cases=dict(distinct=np.stack([z0,z2,q]),repeated=np.stack([z0,repeated_second,q]))
trace_cases=[];fw_results={};fw_errors={};fw_arrays={}
for name,inputs in cases.items():
    a=run_fw(inputs);fw_arrays[name]=a;desired=np.stack([e[0],e[2] if name=='distinct' else e[0]])
    error=float(np.max(abs(a['hidden'][:2]-desired)));assert error<1e-12
    M=.5*(.95*np.outer(e[0],e[0])+np.outer(desired[1],desired[1]));reference=q+M@q
    gamma=math.sqrt(1+1e-5)/math.sqrt(7);ref=np.maximum(gamma*(reference-reference.mean())/np.sqrt(np.var(reference)+1e-5),0)
    read_error=float(np.max(abs(a['hidden'][2]-ref)));assert read_error<1e-12
    mineig=float(min(np.linalg.eigvalsh(m).min() for m in a['memory_after']));assert mineig>=-1e-12 and np.min(a['hidden'])>=0
    fw_errors[name]=dict(write_state=error,independent_read=read_error,memory_min_eigenvalue=mineig)
    fw_results[name]=dict(final_readout=float(a['final_readout'][0]),final_hidden=a['hidden'][2].tolist(),preLN_gap=float(a['preLN'][2,2]-a['preLN'][2,0]))
    trace_cases.append(fw_trace.write_case(r,name,a))
assert fw_results['distinct']['final_readout']>0 and fw_results['repeated']['final_readout']<0
exact=dict(distinct_gap=F(1,2)*(1-F(19,20)),repeated_gap=-F(1,2)*(1+F(19,20)))
# Verify both queried coordinates survive ReLU without using floating-point signs.
aa=[1+F(1,2)*F(19,20),1+F(1,2)];bb=[1+F(1,2)*(1+F(19,20)),F(1)]
for row in (aa,bb):assert min(row)>sum(row)/8
assert exact['distinct_gap']==F(1,40) and exact['repeated_gap']==-F(39,40)
for name,values in [('distinct',aa),('repeated',bb)]:
    observed=fw_results[name]['preLN_gap'];assert abs(observed-float(values[1]-values[0])/math.sqrt(2))<1e-12

# Reuse actual saved native outputs; no model or circuit is called here.
stage=json.loads(Path(scope['pins'][2]['path']).read_bytes());native=stage['quantum']
psi=unpack(native['psi']);phi=unpack(native['phi']);U=np.eye(8);U[np.ix_([0,2],[0,2])]=np.array([[1,1],[1,-1]])/math.sqrt(2)
rotated=psi@U.T;query=phi@U.T
expected=np.stack([np.stack([e[0],e[0]]),np.stack([e[0],e[2]])]);assert np.max(abs(rotated-expected))<1e-12
assert np.max(abs(query-q))<1e-12
p0=abs(np.einsum('rd,btd->brt',phi.conj(),psi))**2;p1=abs(np.einsum('rd,btd->brt',query.conj(),rotated))**2
bridge_error=float(np.max(abs(p0-p1)));assert bridge_error<1e-12

errors=dict(operator=0.,hermitian64=0.,real16_probability=0.,real16_pool=0.,utility_shift=0.,permutation=0.)
algebra=[]
for T in (1,2,7):
    for kind in ('real','complex'):
        for variant in range(4):
            ti=np.arange(T)[:,None]+1;di=np.arange(8)[None,:]+1;ri=np.arange(40)[:,None]+1
            x=np.sin(ti*di*(variant+1)*.31)+.3*np.cos(ti+di)
            qq=np.cos(ri*di*(variant+1)*.17)+.2*np.sin(ri+di)
            if kind=='complex':x=x+1j*np.cos((ti+variant+1)*di*.23);qq=qq+1j*np.sin((ri+variant+1)*di*.19)
            x=x/np.linalg.norm(x,axis=-1,keepdims=True);qq=qq/np.linalg.norm(qq,axis=-1,keepdims=True)
            utility=np.sin((np.arange(40)[:,None]+1)*(np.arange(T)[None,:]+1)*.71)-.4*np.cos(np.arange(T)[None,:]+variant)
            p=abs(qq.conj()@x.T)**2;raw=p+1e-6;D=raw.sum(-1);direct=(raw*utility).sum(-1)/D
            rho=np.einsum('ti,tj->tij',x,x.conj());effect=np.einsum('ri,rj->rij',qq,qq.conj())+1e-6*np.eye(8)
            M=rho.sum(0);V=np.einsum('rt,tij->rij',utility,rho)
            baseline=utility.mean(-1); Vcenter=np.einsum('rt,tij->rij',utility-baseline[:,None],rho)
            operator=baseline+np.einsum('rij,rji->r',effect,Vcenter).real/np.einsum('rij,ji->r',effect,M).real
            hc=coords(effect);rc=coords(rho);dot=hc@rc.T;from64=(dot*utility).sum(-1)/dot.sum(-1)
            v16=np.concatenate([x.real,x.imag],-1);w16=np.concatenate([qq.real,qq.imag],-1);Jv=np.concatenate([-x.imag,x.real],-1)
            p16=(w16@v16.T)**2+(w16@Jv.T)**2;raw16=p16+1e-6;from16=(raw16*utility).sum(-1)/raw16.sum(-1)
            shifted=(raw*(utility+2.25)).sum(-1)/D
            perm=np.arange(T)[::-1];reordered=((p[:,perm]+1e-6)*utility[:,perm]).sum(-1)/(p[:,perm]+1e-6).sum(-1)
            for k,a,b in [('operator',operator,direct),('hermitian64',from64,direct),('real16_probability',p16,p),('real16_pool',from16,direct),('utility_shift',shifted,direct+2.25),('permutation',reordered,direct)]:
                error=float(np.max(abs(a-b)/(1+abs(b))));errors[k]=max(errors[k],error);assert error<=1e-12,(T,kind,variant,k,error)
            algebra.append(dict(T=T,kind=kind,variant=variant,arrays={k:packed(v) for k,v in dict(states=x,queries=qq,utility=utility,probability=p,raw=raw,D=D,direct=direct,operator=operator,hermitian64=from64,real16_probability=p16,real16_pool=from16).items()}))

angles=np.zeros((2,3,2));angles[0,0]=[math.pi/2,math.pi/2];other=angles.copy();other[0,0,1]*=-1
phi_c=circuit(angles);psi_c=circuit(other);complex_p=float(abs(np.vdot(phi_c,psi_c))**2)
naive_raw=float(np.dot(phi_c.real,psi_c.real)**2);naive_normalized=float(np.dot(phi_c.real/np.linalg.norm(phi_c.real),psi_c.real/np.linalg.norm(psi_c.real))**2)
assert complex_p<1e-28 and abs(naive_raw-.25)<1e-12 and abs(naive_normalized-1)<1e-12
(r/'algebra-trace.json').write_text(json.dumps(dict(cases=algebra,native_angle_counterexample=dict(query_angles=angles.tolist(),word_angles=other.tolist(),query=packed(phi_c),word=packed(psi_c),complex_probability=complex_p,discard_imaginary_raw=naive_raw,discard_imaginary_normalized=naive_normalized)),allow_nan=False)+'\n')
trace=fw_trace.close(r,trace_cases,sha(r/'scope.json'));cpu=time.process_time()-start;assert cpu<10
result=dict(status='PASS_SOURCE_BOUND_FUNCTION_CLASS_AUDIT',decision='NO_QUALIFIED_NEW_METHOD_FROM_THIS_AUDIT',FW=fw_results,FW_errors=fw_errors,
    exact_fraction_gaps={k:str(v) for k,v in exact.items()},native_bridge_error=bridge_error,native_bridge_context_map=dict(saved0='repeated',saved1='distinct'),
    algebra_cases=len(algebra),all_relations=40,errors=errors,native_angle_complex_counterexample=dict(complex_probability=complex_p,discard_imaginary_raw=naive_raw,discard_imaginary_normalized=naive_normalized),
    trace=trace,algebra_trace_sha256=sha(r/'algebra-trace.json'),CPU_seconds=cpu,numpy_version=np.__version__,
    resources=dict(task_rows=0,task_native_head=0,task_checkpoint=0,new_native_circuit=0,independent_dense_circuit=2,GPU=0,network=0,API=0,server_writes=0,COS=0),
    claim='Old matched native pair does not exclude this actual LN/ReLU/sustained-boundary FW instantiation; known exact classical controls verified, no general model containment or task utility claim',
    new_theory=False,method_admitted=False,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
