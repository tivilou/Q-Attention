"""Original model: input-local channel vs unrestricted pointwise effect ranges."""
import argparse,hashlib,json,signal,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    t=p.with_suffix('.pending');t.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());t.replace(p)


def ranges(A,B,bias):
    """Hermitian whitening, with explicit numerical (not interval) allowance."""
    L=np.linalg.cholesky(B);inv=np.linalg.solve(L,np.eye(8))
    C=inv@A@inv.conj().T;C=(C+C.conj().swapaxes(-1,-2))/2
    w,Q=np.linalg.eigh(C)
    residual=np.linalg.norm(C@Q-Q*w[...,None,:],axis=(-2,-1))
    scale=1+np.max(np.abs(w),axis=-1)
    uncertainty=residual+64*np.finfo(float).eps*np.linalg.cond(B)*scale+1e-8*scale
    return np.stack((w[...,0]+bias,w[...,-1]+bias),-1),uncertainty


def margin(bound,unc,y):
    low=bound[:,0].copy();low[y]=-np.inf
    value=float(bound[y,1]-np.max(low))
    lower=bound[:,0]-unc;lower[y]=-np.inf
    upper=float(bound[y,1]+unc[y]-np.max(lower))
    return value,upper


def fixtures():
    eps=1e-6;I=np.eye(8);plus=np.zeros(8);plus[:2]=1/np.sqrt(2)
    minus=plus.copy();minus[1]*=-1
    rho=lambda v:np.outer(v,v.conj())
    values=[]
    for changed,c in ((plus,.2),(minus,.5)):
        pair=[]
        for diag in (False,True):
            q=rho(changed);q=np.diag(np.diag(q)) if diag else q
            A=np.stack([rho(plus)-q]+[np.zeros((8,8))]*39)
            B=rho(plus)+q+2*eps*I;bias=np.r_[0.,np.repeat(c,39)]
            b,u=ranges(A,B,bias);pair.append(margin(b,u,0)[0])
        values.append(pair)
    expected=[[-.2,1/(3+4*eps)-.2],[1/(1+2*eps)-.5,1/(3+4*eps)-.5]]
    error=float(np.max(np.abs(np.asarray(values)-expected)));assert error<1e-8
    assert values[0][0]<0<values[0][1] and values[1][1]<0<values[1][0]
    return dict(two_token_both_direction_error=error,observed_margins=values)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','reference','checkpoint','design','output','progress','allocation'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    alloc=json.loads(a.allocation.read_bytes())['allocation'];assert alloc['runner_attempt_id']==a.attempt and len(set(alloc['gpu_uuids']))==2
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    for n,h in cfg['source_hashes'].items():assert digest(a.source/n)==h
    assert digest(a.checkpoint)==cfg['checkpoint_sha256'] and digest(a.inputs/'discovery/manifest.json')==cfg['native_manifest_sha256']
    manifest=json.loads((a.inputs/'discovery/manifest.json').read_bytes());cache=a.inputs/'discovery/cache'
    for k in ('h','roles','offsets','targets'):
        name='train-'+k+'.npy';assert digest(cache/name)==manifest['cache'][name]['sha256']
    wp=a.reference/'train.private.json';assert digest(wp)==cfg['train_word_sha256']
    rows=json.loads(wp.read_bytes());n=cfg['train_count'];assert len(rows)==n==8192
    sys.path.insert(0,str(Path(__file__).parent));from qualify import regions
    masks=[];eligible=[]
    for row in rows:
        b,o,e=regions(len(row['tokens']),row['subject'],row['object']);masks.append((b|o) if e else np.zeros(len(b),bool));eligible.append(e)
    eligible=np.asarray(eligible,bool);assert int(eligible.sum())==7950
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    y=np.asarray(data['targets'][:n]);assert set(y).issubset(range(40))
    for i,row in enumerate(rows):
        lo,hi=map(int,data['offsets'][i:i+2]);assert hi-lo==len(row['tokens'])
        for span,col in ((row['subject'],2),(row['object'],3)):
            expected=np.zeros(hi-lo,bool);expected[slice(*span)]=True;assert np.array_equal(expected,np.asarray(data['roles'][lo:hi,col],bool))
    a.output.mkdir(exist_ok=True);assert not (a.output/'started.json').exists()
    write(a.output/'started.json',dict(attempt=a.attempt,design_sha256=digest(a.design)))
    started=time.time();signal.alarm(cfg['timeout_seconds']);unit=fixtures()
    import torch
    sys.path.insert(0,str(a.source));from baselines.retacred.rmea import RMEAHead,EPS
    assert EPS==1e-6;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.device_count()==2
    initial=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    logits=np.empty((2,n,40));moments=[];lock=threading.Lock();cursors=[0,0]
    def progress(lane,count):
        with lock:
            cursors[lane]=count
            write(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='selective_attainability_train',current=sum(cursors),total=n,unit='example',observed_at=time.time(),heartbeat_at=time.time(),gpu_lanes=cursors.copy()))
    def worker(lane,ids):
        device=torch.device('cuda',lane);torch.cuda.set_device(device);errors=[0.,0.];local=[]
        with torch.inference_mode():
            model=RMEAHead('quantum').double().to(device);model.load_state_dict(initial);model.eval()
            for left in range(0,len(ids),cfg['microbatch_size']):
                index=ids[left:left+cfg['microbatch_size']];width=int(np.diff(data['offsets'])[index].max());bs=len(index)
                h=np.zeros((bs,width,1024));roles=np.zeros((bs,width,4));valid=np.zeros((bs,width),bool);masked=valid.copy()
                for j,i in enumerate(index):
                    lo,hi=map(int,data['offsets'][i:i+2]);size=hi-lo;h[j,:size]=data['h'][lo:hi];roles[j,:size]=data['roles'][lo:hi];valid[j,:size]=True;masked[j,:size]=masks[i]
                H,R,V=(torch.from_numpy(v).to(device) for v in (h,roles,valid));mask=torch.from_numpy(masked).to(device)
                d=model(H,R,V,details=True);psi,phi,u=d['psi'],d['phi'],d['utilities'];rho=psi[..., :,None]*psi.conj()[...,None,:]
                altered=torch.where(mask[:,:,None,None],torch.diag_embed(torch.diagonal(rho,dim1=-2,dim2=-1)),rho)
                I=torch.eye(8,device=device,dtype=rho.dtype);z=[];As=[];Bs=[]
                for q in (rho,altered):
                    rv=q*V[:,:,None,None];B=rv.sum(1)+EPS*V.sum(1)[:,None,None]*I
                    A=torch.einsum('brt,btij->brij',u.to(rv.dtype),rv)+EPS*(u*V[:,None,:]).sum(-1)[:,:,None,None]*I
                    numerator=torch.einsum('ri,brij,rj->br',phi.conj(),A,phi).real
                    denominator=torch.einsum('ri,bij,rj->br',phi.conj(),B,phi).real
                    zz=numerator/denominator+model.bias;z.append(zz);As.append(A.cpu().numpy());Bs.append(B.cpu().numpy())
                    probability=torch.einsum('ri,btij,rj->brt',phi.conj(),q,phi).real
                    raw=(probability+EPS)*V[:,None,:];direct=(raw*u).sum(-1)/raw.sum(-1)+model.bias
                    err=float((direct-zz).abs().max());errors[1]=max(errors[1],err);assert err<1e-8
                err=float((z[0]-d['logits']).abs().max());errors[0]=max(errors[0],err);assert err<1e-8 and torch.equal(z[0].argmax(-1),d['logits'].argmax(-1))
                zz=torch.stack(z).cpu().numpy();assert np.isfinite(zz).all();logits[:,index]=zz
                flip=(zz[0].argmax(-1)==y[index])!=(zz[1].argmax(-1)==y[index])
                bias=model.bias.cpu().numpy()
                for j in np.where(flip)[0]:
                    assert eligible[index[j]];A=np.stack(As)[:,j];B=np.stack(Bs)[:,j];bd=[];un=[];mm=[]
                    for k in range(2):
                        bound,unc=ranges(A[k],B[k],bias);bd.append(bound);un.append(unc);mm.append(margin(bound,unc,int(y[index[j]])))
                        assert np.all(zz[k,j]>=bound[:,0]-unc-1e-7) and np.all(zz[k,j]<=bound[:,1]+unc+1e-7)
                    local.append((int(index[j]),A,B,np.asarray(bd),np.asarray(un),np.asarray(mm)))
                if left%128==0 or left+bs==len(ids):progress(lane,left+bs)
            for k,v in model.state_dict().items():assert torch.equal(v.cpu(),initial[k])
        with lock:moments.extend(local)
        return dict(lane=lane,rows=len(ids),native_error=errors[0],moment_response_error=errors[1])
    progress(0,0)
    with ThreadPoolExecutor(max_workers=2) as pool:checks=[f.result() for f in [pool.submit(worker,i,ids) for i,ids in enumerate(np.array_split(np.arange(n),2))]]
    assert np.max(np.abs(logits[:,~eligible]-logits[0,~eligible]))<1e-8
    moments.sort(key=lambda x:x[0]);flip_ids=np.array([q[0] for q in moments],dtype=np.int64)
    correct=logits.argmax(-1)==y;assert np.array_equal(flip_ids,np.where(correct[0]!=correct[1])[0])
    damage=correct[0,flip_ids];repair=~damage
    M=np.stack([q[5] for q in moments]) if moments else np.empty((0,2,2))
    dw=damage&(M[:,1,1]<-cfg['margin_tolerance']);rw=repair&(M[:,0,1]<-cfg['margin_tolerance'])
    gate='PASS_BIDIRECTIONAL_EXISTENCE_ONLY' if dw.any() and rw.any() else 'FAIL_CLOSE_BIDIRECTIONAL_PROFILE'
    def counts(mask):return np.bincount(y[flip_ids[mask]],minlength=40).tolist()
    result=dict(profile=cfg['profile'],gate=gate,examples=n,eligible_rows=int(eligible.sum()),damaged=int(damage.sum()),repaired=int(repair.sum()),damage_witnesses=int(dw.sum()),repair_witnesses=int(rw.sum()),damage_fraction=float(dw.sum()/damage.sum()) if damage.any() else None,repair_fraction=float(rw.sum()/repair.sum()) if repair.any() else None,per_class=dict(damage=counts(damage),repair=counts(repair),damage_witness=counts(dw),repair_witness=counts(rw)),maximum_endpoint_uncertainty=max((float(q[4].max()) for q in moments),default=0.),seconds=time.time()-started,optimizer_updates=0,valid_test_used=False,claim_scope=cfg['claim_scope'])
    private=a.output/'moments.private.npz'
    np.savez_compressed(private,logits=logits,labels=y,eligible=eligible,flip_ids=flip_ids,A=np.stack([q[1] for q in moments]) if moments else np.empty((0,2,40,8,8),complex),B=np.stack([q[2] for q in moments]) if moments else np.empty((0,2,8,8),complex),bounds=np.stack([q[3] for q in moments]) if moments else np.empty((0,2,40,2)),uncertainty=np.stack([q[4] for q in moments]) if moments else np.empty((0,2,40)),margins=M,bias=initial['bias'].numpy())
    result['present_gold_classes']=sorted(map(int,set(y)));result['absent_gold_classes']=sorted(set(range(40))-set(result['present_gold_classes']))
    result['all40_output_classes_used']=True
    result['private_sha256']=digest(private);result['design_sha256']=digest(a.design);assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    write(a.output/'summary.json',result);write(a.output/'fixtures.json',dict(status='PASS',unit=unit,lanes=checks,ineligible_identity=True,source_weights_unchanged=True))
    write(a.output/'RUN_COMPLETE',dict(files={n:digest(a.output/n) for n in ('summary.json','fixtures.json','moments.private.npz')}));print(json.dumps(result))


if __name__=='__main__':main()
