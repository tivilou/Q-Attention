from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
import numpy as np
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
def packed(x):
    x=np.asarray(x)
    return dict(shape=list(x.shape),real=x.real.tolist(),imag=x.imag.tolist()) if np.iscomplexobj(x) else dict(shape=list(x.shape),values=x.tolist())
cfg=read('scope.json');receipt=read('receipt.before-execution.json')
assert not (r/'results.json').exists()
assert sha((r/'scope.json').read_bytes())==receipt['scope_sha256'] and sha(Path(__file__).read_bytes())==receipt['code_sha256']
assert sha((r/'derivation.md').read_bytes())==receipt['derivation_sha256']
for p in cfg['source_pins']:assert sha(Path(p['path']).read_bytes())==p['sha256']
start=time.process_time();eps=cfg['EPS'];tol=cfg['absolute_tolerance'];rows=[];traces=[]
for case in cfg['cases']:
    psi=np.array(case['psi']['real'])+1j*np.array(case['psi']['imag']);phi=np.array(case['phi']['real'])+1j*np.array(case['phi']['imag'])
    u=np.array(case['u'],dtype=float);mask=np.array(case['mask'],bool)
    a=psi.conj()@phi;p=abs(a)**2;raw=(p+eps)*mask;D=raw.sum();alpha=raw/D;z=float(alpha@u);delta=u-z
    rhos=np.einsum('ti,tj->tij',psi,psi.conj());bts=(rhos+eps*np.eye(8))*mask[:,None,None]
    B=bts.sum(0);A=np.einsum('t,tij->ij',u,bts);C=A-z*B;residual=C@phi
    v=a[:,None]*(psi-a.conj()[:,None]*phi)*mask[:,None]
    tangent=delta@v;K=v.conj()@v.T
    gd=2*np.vdot(residual,residual).real/D;gt=2*np.vdot(tangent,tangent).real/D
    gram=2*(delta@K@delta).real/D;self_=2*np.sum(delta**2*np.einsum('ti,ti->t',v.conj(),v).real)/D
    cross=2*(delta@(K-np.diag(np.diag(K)))@delta).real/D
    variance=float(alpha@(delta**2));gamma=float(np.sum(np.einsum('ti,ti->t',v.conj(),v).real/(p+eps)))
    gamma_p=float(np.sum(p*(1-p)/(p+eps)*mask));upper=2*gamma*variance
    eta=float(gt/upper) if upper!=0 else None
    errors=dict(residual_identity=float(np.max(abs(residual-tangent))),derivative_identity=abs(gd-gram),
                split_identity=abs(gd-self_-cross),Gamma_probability=abs(gamma-gamma_p),
                unit_state=float(np.max(abs(np.sum(abs(psi)**2,axis=1)-1))),unit_query=abs(np.vdot(phi,phi).real-1),
                original_ratio=abs(np.vdot(phi,A@phi).real/np.vdot(phi,B@phi).real-z))
    ok=all(e<=tol for e in errors.values()) and gd>=-tol and gd<=upper+tol
    if upper==0:ok=ok and gd<=tol
    else:ok=ok and -tol<=eta<=1+tol
    row=dict(id=case['id'],g=float(gd),g_tangent=float(gt),g_self=float(self_),g_cross=float(cross),
             variance=variance,Gamma=gamma,upper_bound=upper,eta=eta,errors=errors,PASS=bool(ok))
    rows.append(row);traces.append(dict(id=case['id'],a=packed(a),p=packed(p),D=float(D),alpha=packed(alpha),z0=z,
        delta=packed(delta),A=packed(A),B=packed(B),C=packed(C),residual=packed(residual),v=packed(v),K=packed(K),summary=row))
by={x['id']:x for x in rows}
e=F(1,1000000);den=1+2*e;exact=[]
for label,sign in [('retained-native-A',1),('retained-native-B',-1)]:
    K=[[F(1,4),sign*F(1,4)],[sign*F(1,4),F(1,4)]];delta=[F(-1),F(1)]
    norm=sum(delta[i]*K[i][j]*delta[j] for i in range(2) for j in range(2))
    g=2*norm/den;self_=1/den;cross=g-self_;gamma=1/den;upper=2*gamma
    assert 0<=g<=upper and g==(F(0) if sign==1 else upper)
    exact.append(dict(id=label,g=str(g),g_self=str(self_),g_cross=str(cross),Gamma=str(gamma),variance='1',upper=str(upper),eta=str(g/upper)))
invariants={}
for label in ['word-gauge-nonuniform','permuted-masked-nonuniform']:
    invariants[label]=max(abs(by[label][k]-by['nonuniform'][k]) for k in ['g','g_self','g_cross','variance','Gamma','upper_bound','eta'])
gates=dict(numeric_rows=all(x['PASS'] for x in rows),invariances=all(x<=tol for x in invariants.values()),
           native_A_eta0=abs(by['retained-native-A']['eta'])<=tol,native_B_eta1=abs(by['retained-native-B']['eta']-1)<=tol,
           native_A_cross_negative=by['retained-native-A']['g_cross']<0,native_B_cross_positive=by['retained-native-B']['g_cross']>0,
           exact_certificates=len(exact)==2,source_hashes_unchanged=all(sha(Path(p['path']).read_bytes())==p['sha256'] for p in cfg['source_pins']))
cpu=time.process_time()-start;gates['CPU_budget']=cpu<=10
(r/'trace.json').write_bytes((json.dumps(dict(schema='tangent-control-algebra-trace.v1',rows=traces,exact_fraction=exact),separators=(',',':'))+'\n').encode())
out=dict(status='PASS_TANGENT_CONTROL_ALGEBRA' if all(gates.values()) else 'STOP_DERIVATIVE_CONTROL_CLAIM',
         scope_sha256=receipt['scope_sha256'],trace_sha256=sha((r/'trace.json').read_bytes()),rows=rows,exact_fraction=exact,
         invariance_errors=invariants,gates=gates,CPU_seconds=cpu,
         max_identity_error=max(max(x['errors'].values()) for x in rows),resources=cfg['resources'],
         new_theory=False,task_alignment=False,literature_clearance=False,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','exact_fraction']}))
raise SystemExit(0 if all(gates.values()) else 2)
