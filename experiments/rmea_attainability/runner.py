"""Read-only seven-case shared-effect assay; no Torch/model/optimizer/COS."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,subprocess,time,argparse
from pathlib import Path
import numpy as np
import intervals as iv
MODES=('uniform','relation_softmax','real_overlap','quantum','dephased')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
def pinned(root,ref):
    p=root/ref['path'];assert not Path(ref['path']).is_absolute() and '..' not in Path(ref['path']).parts and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) and sha(p)==ref['sha256'];return p
def basis(d):
    result=[]
    for j in range(d):
        for k in range(j+1,d):
            a=np.zeros((d,d),np.complex128);a[j,k]=a[k,j]=1/np.sqrt(2);result.append(a)
            a=np.zeros((d,d),np.complex128);a[j,k]=-1j/np.sqrt(2);a[k,j]=1j/np.sqrt(2);result.append(a)
    for k in range(1,d):
        a=np.zeros((d,d),np.complex128);a[np.arange(k),np.arange(k)]=1;a[k,k]=-k;result.append(a/np.sqrt(k*(k+1)))
    return np.array(result)
def witness(rhos,targets,eps,tau):
    d=rhos[0].shape[-1];F=basis(d);blocks=[];right=[]
    for rho,a in zip(rhos,targets):
        features=np.einsum('jab,tba->tj',F,rho).real;n=len(a)
        blocks.append(features-a[:,None]*features.sum(0)[None,:]);right.append((eps+1/d)*(n*a-1))
    A=np.concatenate(blocks);b=np.concatenate(right);U,S,V=np.linalg.svd(A,full_matrices=False);keep=S>max(A.shape)*np.finfo(float).eps*S[0]
    u=U[:,keep];sv=S[keep];projection=u.T@b;candidates=[b-u@projection,u@(projection/(sv*sv))]
    scores=[]
    for w in candidates:
        if np.linalg.norm(w)==0:w=np.eye(1,len(b))[0]
        w=w/np.linalg.norm(w);W=np.zeros((d,d),np.complex128);c=0.;bound=0.;offset=0
        for rho,a in zip(rhos,targets):
            n=len(a);ww=w[offset:offset+n];offset+=n;ss=ww@a
            W+=np.einsum('t,tab->ab',ww-ss,rho);c+=eps*(n*ss-ww.sum());bound+=tau*(1+eps)*n*np.abs(ww).sum()
        m=np.trace(W).real/d;score=abs(c-m)-np.sqrt(1-1/d)*np.linalg.norm(W-m*np.eye(d))-bound
        scores.append((float(score),w,W,float(c)))
    score,w,W,c=max(scores,key=lambda x:x[0])
    return dict(approximate_margin=score,rank=int(keep.sum()),rows=len(b),columns=d*d-1),w,W,c
def unit(eps,tau):
    from fractions import Fraction
    for a,b in [(1/3,-1/7),(-3.25,2.75),(1e-30,1e20)]:
        x,y=iv.I.f(a),iv.I.f(b)
        for z,q in [(x+y,Fraction(a)+Fraction(b)),(x*y,Fraction(a)*Fraction(b)),(x/y,Fraction(a)/Fraction(b))]:assert Fraction(z.lo)<=q<=Fraction(z.hi)
    psi=np.array([[1,0],[0,1],[2**-.5,2**-.5]],complex);rho=np.einsum('ti,tj->tij',psi,psi.conj())/(np.abs(psi)**2).sum(-1)[:,None,None]
    states=[[iv.density(x) for x in psi]];a=np.array([.05,.05,.90]);screen,w,*_=witness([rho],[a],eps,tau);bad=iv.certify(states,[a],w,iv.I(str(tau)),iv.I(str(eps)));assert bad['certified_infeasible']
    physical=iv.physical_allocation(psi,np.array([1,0],complex),iv.I(str(eps)));good=np.array([float((p.lo+p.hi)/2) for p in physical]);_,w,*_=witness([rho],[good],eps,tau);assert not iv.certify(states,[good],w,iv.I(str(tau)),iv.I(str(eps)))['certified_infeasible']
    assert not iv.certify(states,[np.full(3,1/3)],w,iv.I(str(tau)),iv.I(str(eps)))['certified_infeasible']
    return dict(status='PASS',cases=12,scope='Exact Fraction containment arithmetic; infeasible/feasible/uniform witness fixtures.')
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--output',type=Path,required=True);args=p.parse_args();src=Path(__file__).resolve().parent;repo=src.parents[1];cfg=json.loads((src/'contract.json').read_bytes());began=time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and np.__version__=='1.26.4'
    git=lambda *x:subprocess.check_output(['git','-C',str(repo),*x],stderr=subprocess.DEVNULL).decode().strip()
    assert not git('status','--porcelain=v1');out=args.output.resolve();assert not out.is_relative_to(repo) and not out.is_symlink();out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    eps,tau=cfg['epsilon'],cfg['target_coordinate_tolerance'];save(out/'unit.json',unit(eps,tau));root=Path(cfg['input_private_root']);closed=root.parent/'RUN_COMPLETE';assert sha(closed)==cfg['run_complete_sha256'];files=json.loads(closed.read_bytes())['files']
    for name,h in files.items():assert sha(pinned(root.parent,dict(path=name,sha256=h)))==h
    payload=json.loads((root/'sample-trace.json').read_bytes());assert len(payload['samples'])==35 and payload['experiment']['code_revision']==cfg['input_source_revision']
    selected=json.loads((root/'replay-manifest.json').read_bytes())['samples'];ids=[s['sample_id'] for s in selected];assert len(ids)==7 and all(s['split']=='valid' for s in selected)
    lookup={(s['roles']['provenance']['condition'],s['roles']['source_sample']['id']):s for s in payload['samples']};psis=[];targets={m:[] for m in MODES};lineage=[];phi=None
    for sid in ids:
        q=lookup[('quantum',sid)];ref=next(r for r in q['representations'] if r['id']=='psi')['artifact_refs'][0]
        with np.load(pinned(root,ref),allow_pickle=False) as f:psi=f['psi'][0].astype(np.complex128);effect=f['phi'].astype(np.complex128)
        assert psi.ndim==2 and psi.shape[1]==8 and effect.shape==(40,8) and np.isfinite(psi).all()
        if phi is None:phi=effect
        else:assert np.array_equal(phi,effect)
        psis.append(psi)
        for mode in MODES:
            sample=lookup[(mode,sid)];r=next(x for x in sample['representations'] if x['id']=='alpha')['artifact_refs'][0]
            with np.load(pinned(root,r),allow_pickle=False) as f:a=f['alpha'][0].astype(np.float64)
            assert a.shape==(40,len(psi)) and np.isfinite(a).all() and a.min()>=0 and np.max(np.abs(a.sum(-1)-1))<len(psi)*tau
            targets[mode].append(a);lineage.append(dict(sample_id=sid,condition=mode,artifact=r))
    rhos=[np.einsum('ti,tj->tij',p,p.conj())/(np.abs(p)**2).sum(-1)[:,None,None] for p in psis];exact_states=[[iv.density(v) for v in p] for p in psis]
    # Certify the declared resolution against all saved Q rows, not a chosen one.
    q_error=iv.I(0);uniform_error=iv.I(0)
    for i,psi in enumerate(psis):
        for r in range(40):
            physical=iv.physical_allocation(psi,phi[r],iv.I(str(eps)))
            for t,pred in enumerate(physical):
                error=(pred-iv.I.f(targets['quantum'][i][r,t])).absolute();q_error=iv.I(max(q_error.lo,error.lo),max(q_error.hi,error.hi))
    assert q_error.hi<iv.I(str(tau)).lo,'Positive Q control exceeds frozen resolution; no relaxation'
    for a in targets['uniform']:
        exact_uniform=iv.I(1)/a.shape[1]
        for v in a.flat:
            error=(exact_uniform-iv.I.f(v)).absolute();uniform_error=iv.I(max(uniform_error.lo,error.lo),max(uniform_error.hi,error.hi))
    assert uniform_error.hi<iv.I(str(tau)).lo,'Uniform input differs from maximally mixed positive control'
    results={};private=out/'private';private.mkdir();arrays={};certs={}
    for mode in MODES:
        records=[];best=None
        for r in range(40):
            a=[x[r] for x in targets[mode]];record,w,W,c=witness(rhos,a,eps,tau);record['relation']=r;records.append(record)
            if best is None or record['approximate_margin']>best[0]['approximate_margin']:best=(record,w,W,c)
        record,w,W,c=best;relation=record['relation'];cert=iv.certify(exact_states,[a[relation] for a in targets[mode]],w,iv.I(str(tau)),iv.I(str(eps)))
        if mode in ['quantum','uniform']:assert not cert['certified_infeasible'],'Feasible positive control rejected'
        arrays[mode+'_w']=w;arrays[mode+'_W']=W;arrays[mode+'_c']=np.array(c);certs[mode]=dict(cert,relation=relation,w_sha256=hashlib.sha256(w.tobytes()).hexdigest())
        results[mode]=dict(screened_relations=40,screen=records,strongest_relation=relation,interval_certificate=certs[mode],interpretation='At least one shared mixed effect impossible at frozen resolution' if cert['certified_infeasible'] else 'Not certified; no claim of PSD/pure feasibility')
    np.savez(private/'witnesses.npz',**arrays);save(private/'certificates.json',certs);save(private/'input-lineage.json',dict(input_root=str(root),run_complete_sha256=cfg['run_complete_sha256'],lineage=lineage))
    verification=dict(status='PASS',precision=iv.PRECISION,positive_q_max_coordinate_error=q_error.json(),positive_uniform_max_coordinate_error=uniform_error.json(),target_coordinate_tolerance=tau,interval_certificates=5,all40_relations_per_head_retained=True,source_lineage='Existing closed35-head numeric trace; no new model forward',scope='Fixed saved normalized states, shared independent binary effects, necessary norm-ball condition; no completeness or novelty claim')
    save(out/'certificate-validation.json',verification)
    summary=dict(status='PASS',schema='qattn.shared-effect-attainability.descriptive.v1',source_revision=git('rev-parse','HEAD'),contract_sha256=sha(src/'contract.json'),cases=7,token_count=sum(map(len,psis)),relations_per_head=40,results=results,epsilon=eps,target_coordinate_tolerance=tau,elapsed_seconds=time.monotonic()-began,training_runs=0,formal_runs=0,test_loaded=False,new_model_forward=0,new_cos_reads=0,l5_accepted=False,limitation=cfg['interpretation'])
    save(out/'summary.json',summary);closure={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()};assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<16*1024**2;save(out/'RUN_COMPLETE',dict(schema='qattn.attainability.closed.v1',files=closure));print(json.dumps(dict(status='COMPLETE',cases=7,training_runs=0,formal_runs=0)))
if __name__=='__main__':main()
