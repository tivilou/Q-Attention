"""Frozen TRAIN extraction and shared-query conflict diagnostic, not fitting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,math,sys,threading,time
import numpy as np
from pair_math import normalize,dual_search,TOL

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def save(path,v):
    tmp=path.with_suffix('.pending');tmp.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());tmp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    for k in ('root','design','output','progress'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    assert cfg['solver']['golden_iterations']==48 and cfg['solver']['anchors_per_class']==4 and cfg['solver']['tolerance']==TOL
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    source=a.root/cfg['source'];completed=a.root/cfg['completed'];cache=completed/'cache'
    assert digest(completed/'manifest.json')==cfg['manifest_sha256']
    for n,h in cfg['source_hashes'].items():assert digest(source/n)==h
    for n,v in cfg['train_cache'].items():assert digest(cache/n)==v['sha256']
    assert digest(a.root/cfg['labels'])==cfg['labels_sha256']
    labels=json.loads((a.root/cfg['labels']).read_bytes());assert len(labels)==40
    sys.path.insert(0,str(source));import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import circuit
    from baselines.retacred.rmea_matched_routers import Head
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and not a.output.exists();a.output.mkdir(parents=True)
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    y=np.asarray(data['targets']);assert y.shape==(58465,) and data['offsets'][-1]==len(data['h'])==len(data['roles'])
    begin=time.monotonic();cursor=[0,0];lock=threading.Lock()
    def progress(phase):
        with lock:save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase=phase,current=sum(cursor),total=2*len(y),unit='example',heartbeat_at=time.time(),observed_at=time.time()))
    progress('extract_fixed_native_constraints')
    def extract(lane,mode):
        torch.cuda.set_device(lane);device=torch.device('cuda',lane)
        pin=cfg['selected'][mode];path=completed/pin['path'];assert digest(path)==pin['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True);assert state_identity(state)==pin['state_sha256']
        model=Head(mode).float().to(device);model.load_state_dict(state,strict=True);model.eval();head=model.head
        d=8 if mode=='quantum' else 12;dtype=np.complex128 if mode=='quantum' else np.float64
        matrices=np.empty((len(y),d,d),dtype);logits=np.empty((len(y),40),np.float32)
        readout_error=0.;numpy_error=0.;hermitian_error=0.;native_expectation=np.empty(len(y),np.float64)
        with torch.inference_mode():
            for lo in range(0,len(y),128):
                assert time.monotonic()-begin<900
                hi=min(lo+128,len(y));lengths=np.diff(data['offsets'][lo:hi+1]);n=int(lengths.max())
                hh=np.zeros((hi-lo,n,1024),np.float32);rr=np.zeros((hi-lo,n,4),np.float32);mm=np.zeros((hi-lo,n),bool)
                for j,i in enumerate(range(lo,hi)):
                    l,r=map(int,data['offsets'][i:i+2]);hh[j,:r-l]=data['h'][l:r];rr[j,:r-l]=data['roles'][l:r];mm[j,:r-l]=True
                h,roles,mask=(torch.from_numpy(x).to(device) for x in (hh,rr,mm))
                z=model(h,roles,mask);logits[lo:hi]=z.cpu().numpy();gold=torch.from_numpy(np.array(y[lo:hi])).to(device)
                values=head.value(h);dec=head.decoder[gold]
                utility=(values.double()*dec[:,None].double()).sum(-1);bias=head.bias[gold].double()
                if mode=='quantum':
                    angles=(math.pi*torch.tanh(head.prepare(torch.cat((h,roles),-1)))).reshape(h.shape[:2]+(2,3,2))
                    keys=circuit(angles).to(torch.complex128);query=circuit(math.pi*torch.tanh(head.effect))[gold].to(torch.complex128)
                else:
                    keys=F.normalize(head.key(torch.cat((h,roles),-1)),dim=-1).double();query=F.normalize(head.query,dim=-1)[gold].double()
                outer=keys[..., :,None]*keys[...,None,:].conj();m=mask.double();eye=torch.eye(d,dtype=keys.dtype,device=device)
                B=(outer*m[...,None,None]).sum(1)+1e-6*m.sum(-1)[:,None,None]*eye
                A=(outer*(utility*m)[...,None,None]).sum(1)+1e-6*(utility*m).sum(-1)[:,None,None]*eye
                rivals=z.double().clone();rivals.scatter_(1,gold[:,None],-torch.inf);threshold=rivals.max(-1).values-bias
                C=A-threshold[:,None,None]*B
                hermitian_error=max(hermitian_error,float((C-C.mH).abs().max()));C=(C+C.mH)*.5
                matrices[lo:hi]=C.cpu().numpy()
                qnorm=query/torch.linalg.vector_norm(query,dim=-1,keepdim=True)
                ce=torch.einsum('bi,bij,bj->b',qnorm.conj(),C,qnorm).real
                native_expectation[lo:hi]=(ce/torch.linalg.matrix_norm(C).clamp_min(1e-300)).cpu().numpy()
                aq=torch.einsum('bi,bij,bj->b',query.conj(),A,query).real
                bq=torch.einsum('bi,bij,bj->b',query.conj(),B,query).real
                zz=aq/bq+bias;readout_error=max(readout_error,float((zz-z.double().gather(1,gold[:,None]).squeeze(1)).abs().max()))
                if lo==0:
                    for j in range(3):
                        kk=keys[j,mask[j]].cpu().numpy();uu=utility[j,mask[j]].cpu().numpy();theta=float(threshold[j])
                        bb=kk.T@kk.conj()+1e-6*len(kk)*np.eye(d)
                        aa=kk.T@(uu[:,None]*kk.conj())+1e-6*uu.sum()*np.eye(d)
                        numpy_error=max(numpy_error,float(abs(aa-theta*bb-matrices[lo+j]).max()))
                cursor[lane]=hi
                if lo%2048==0:progress('extract_fixed_native_constraints')
        pred=logits.argmax(-1);cm=np.bincount(40*y+pred,minlength=1600).reshape(40,40)
        assert cm.tolist()==cfg['prior_train_metrics'][mode]['confusion_matrix']
        assert readout_error<1e-4 and numpy_error<1e-8 and hermitian_error<1e-8
        assert np.isfinite(matrices).all() and np.isfinite(logits).all()
        assert time.monotonic()-begin<900
        return mode,dict(matrices=normalize(matrices),logits=logits,native_expectation=native_expectation,readout_error=readout_error,numpy_error=numpy_error,hermitian_error=hermitian_error,confusion_matrix=cm.tolist())
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(extract,0,'quantum'),pool.submit(extract,1,'real12')];parts=dict(j.result() for j in jobs)
    assert time.monotonic()-begin<900
    qwrong=parts['quantum']['logits'].argmax(-1)!=y;rwrong=parts['real12']['logits'].argmax(-1)!=y
    union=np.flatnonzero(qwrong|rwrong);common=~qwrong&~rwrong
    anchors={};pairs=[]
    for c in range(40):
        candidates=np.flatnonzero(common&(y==c)).tolist()
        candidates.sort(key=lambda i:(hashlib.sha256(('shared-query-pair-v1:'+str(i)).encode()).hexdigest(),i))
        anchors[c]=candidates[:4]
    for i in union:
        for j in anchors[int(y[i])]:pairs.append((int(i),j))
    pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2);assert len(pairs)>0
    private=dict(pairs=pairs,labels=y,union=union,quantum_original_wrong=qwrong,real12_original_wrong=rwrong)
    reports={}
    progress('solve_fixed_shared_query_pairs')
    for mode,part in parts.items():
        c=part['matrices'];left=c[pairs[:,0]];right=c[pairs[:,1]]
        all_individual_max=np.linalg.eigvalsh(c)[:,-1]
        lm=all_individual_max[pairs[:,0]];rm=all_individual_max[pairs[:,1]]
        weights,upper=dual_search(left,right)
        assert np.isfinite(weights).all() and np.isfinite(upper).all()
        category=np.full(len(pairs),3,np.int8)
        category[(lm < -TOL)|(rm < -TOL)]=0
        individual=(lm>TOL)&(rm>TOL)
        category[individual&(upper < -TOL)]=1
        category[individual&(upper>=-TOL)]=2
        conflict=category==1;independent_error=0.;negative=upper < 0
        for lo in range(0,int(negative.sum()),1024):
            ids=np.flatnonzero(negative)[lo:lo+1024]
            mix=weights[ids,None,None]*left[ids]+(1-weights[ids,None,None])*right[ids]
            check=torch.linalg.eigvalsh(torch.from_numpy(mix)).numpy()[:,-1]
            independent_error=max(independent_error,float(abs(check-upper[ids]).max()))
            assert np.all(check[upper[ids] < -TOL] < 0)
        assert independent_error<1e-10 and np.all((weights>=0)&(weights<=1))
        native_min=np.minimum(part['native_expectation'][pairs[:,0]],part['native_expectation'][pairs[:,1]])
        native_upper_violation=float(np.maximum(native_min-upper,0).max());assert native_upper_violation<1e-9
        wrong=qwrong if mode=='quantum' else rwrong
        participating=np.unique(pairs[conflict,0]);native_errors=np.flatnonzero(wrong)
        participating_errors=np.intersect1d(participating,native_errors)
        counts=lambda mask:np.bincount(category[mask],minlength=4).tolist()
        strata={'both_original_wrong':qwrong[pairs[:,0]]&rwrong[pairs[:,0]],'only_quantum_wrong':qwrong[pairs[:,0]]&~rwrong[pairs[:,0]],'only_real12_wrong':~qwrong[pairs[:,0]]&rwrong[pairs[:,0]]}
        original_error_individual_counts=dict(numerical_obstruction=int((all_individual_max[wrong] < -TOL).sum()),positive=int((all_individual_max[wrong]>TOL).sum()),unresolved=int((abs(all_individual_max[wrong])<=TOL).sum()))
        assert sum(original_error_individual_counts.values())==len(native_errors)
        reports[mode]=dict(pair_counts=counts(np.ones(len(pairs),bool)),individual_positive_pairs=int(individual.sum()),numerical_conflict_pairs=int(conflict.sum()),unique_union_rows_with_conflict=len(participating),original_errors=len(native_errors),original_error_individual_counts=original_error_individual_counts,unique_original_errors_with_conflict=len(participating_errors),independent_dual_max_error=independent_error,independently_verified_negative_duals=int(negative.sum()),native_upper_violation=native_upper_violation,
            readout_max_error=part['readout_error'],numpy_matrix_max_error=part['numpy_error'],hermitian_max_error=part['hermitian_error'],prior_confusion_exact=True,confusion_matrix=part['confusion_matrix'],
            strata={k:dict(pairs=int(v.sum()),pair_counts=counts(v)) for k,v in strata.items()},
            per_class=[dict(label=labels[k],original_errors=int(((y==k)&wrong).sum()),original_error_individual_counts=dict(numerical_obstruction=int(((y==k)&wrong&(all_individual_max < -TOL)).sum()),positive=int(((y==k)&wrong&(all_individual_max>TOL)).sum()),unresolved=int(((y==k)&wrong&(abs(all_individual_max)<=TOL)).sum())),anchor_count=len(anchors[k]),pair_count=int((y[pairs[:,0]]==k).sum()),pair_counts=counts(y[pairs[:,0]]==k),unique_original_errors_with_conflict=int(((y[participating_errors]==k)).sum())) for k in range(40)])
        private.update({mode+'_weights':weights,mode+'_upper':upper,mode+'_category':category,mode+'_matrices':c,mode+'_individual_max':all_individual_max,mode+'_logits':part['logits'],mode+'_native_expectation':part['native_expectation']})
        assert time.monotonic()-begin<900
    np.savez(a.output/'diagnostic.private.npz',**private)
    elapsed=time.monotonic()-begin;assert elapsed<900
    report=dict(profile=cfg['profile'],attempt=a.attempt,design_sha256=digest(a.design),code_hashes=cfg['code_hashes'],input_hashes=cfg['input_hashes'],rows=len(y),pairs=len(pairs),union_original_error_rows=len(union),both_wrong_rows=int((qwrong&rwrong).sum()),original_errors={'quantum':int(qwrong.sum()),'real12':int(rwrong.sum())},
        anchor_counts={labels[k]:len(anchors[k]) for k in range(40)},category_order=['individual_numerical_obstruction','individually_positive_joint_numerical_conflict','individually_positive_no_conflict_certificate','unresolved_individual'],results=reports,
        seconds=elapsed,optimizer_updates=0,valid_rows_evaluated=0,test_rows_evaluated=0,cos_writes=0,private_sha256=digest(a.output/'diagnostic.private.npz'),
        claim='Frozen computed TRAIN matrices, numerical necessary constraints with fixed native rivals; no interval certificate, global shareability, task gain, quantum cause or new T.')
    report['seconds']=time.monotonic()-begin;assert report['seconds']<900
    save(a.output/'summary.json',report);save(a.output/'RUN_COMPLETE',dict(summary_sha256=digest(a.output/'summary.json'),attempt=a.attempt))
    progress('completed');print(json.dumps(dict(seconds=elapsed,pairs=len(pairs),results={m:{k:v[k] for k in ('pair_counts','original_errors','unique_original_errors_with_conflict')} for m,v in reports.items()})))

if __name__=='__main__':main()
